#!/usr/bin/env python3
"""Generate evaluation questions from a guideline tree.

    CLI:
        uv run python data/questions/generate_questions.py <guideline_id> \\
            --styles lookup scenario sdm \\
            --questions-per-node 2 \\
            --output data/questions/out/<id>/questions.jsonl
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv

# data/questions/generate_questions.py → data/ is one up, the repository root
# two. pipeline_core (the corpus library this reads trees with) sits under
# data/corpus/, so that directory goes on the path.
_HERE = Path(__file__).resolve().parent
_DATA_DIR = _HERE.parent
_REPO_ROOT = _DATA_DIR.parent
sys.path.insert(0, str(_DATA_DIR / "corpus"))
load_dotenv(_REPO_ROOT / ".env", override=False)  # fills OPENAI_API_KEY if unset

from pipeline_core.core.assemble import assemble  # noqa: E402
from pipeline_core.core.tree import SectionNode, TreeDocument  # noqa: E402

from openai import OpenAI  # noqa: E402

logger = logging.getLogger("generate_questions")

PROMPTS_DIR = _HERE / "prompts"
GUIDELINES_DIR = _DATA_DIR / "corpus" / "guidelines"
DEFAULT_OUTPUT_BASE = _HERE / "out"

# Tree levels for the two retrieval granularities the agent exposes.
CHAPTER_LEVEL = 2
SUBTOPIC_LEVEL = 3

FRONT_MATTER_PATTERNS = [
    "what is new",
    "take-home message",
    "top 10",
    "preamble",
    "introduction",
    "evidence gaps",
    "future directions",
    "conclusion",
    "abbreviation",
    "diabetes advocacy",
]

INCLUDE_CHAPTERS: dict[str, list[str]] = {
    "blood-pressure-2025": ["sec-6", "sec-7", "sec-8", "sec-9", "sec-10"],
    "dyslipidemia-2026": ["sec-6", "sec-7", "sec-8", "sec-9"],
    "cvd-prevention-2019": ["sec-5", "sec-6", "sec-7", "sec-8"],
    "diabetes-care-2026": [
        f"ch{i:02d}" for i in range(1, 17)
    ],  # ch01–ch16 (ch17 Advocacy excluded)
}

STYLE_PROMPTS = {
    "lookup": PROMPTS_DIR / "question_lookup.md",
    "scenario": PROMPTS_DIR / "question_scenario.md",
    "sdm": PROMPTS_DIR / "question_sdm.md",
}


@dataclass
class GuidelineMeta:
    guideline_id: str
    guideline_name: str
    guideline_dir: Path


# ── tree helpers ─────────────────────────────────────────────────────────────


def load_guideline_meta(guideline_id: str) -> GuidelineMeta:
    gdir = GUIDELINES_DIR / guideline_id
    yaml_path = gdir / "guideline.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"{yaml_path} not found")
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    return GuidelineMeta(
        guideline_id=guideline_id,
        guideline_name=cfg.get("guideline_name") or guideline_id,
        guideline_dir=gdir,
    )


def build_title_path(node: SectionNode, tree: TreeDocument) -> list[str]:
    """Breadcrumb of titles from root to self, dropping structural roots."""
    titles = []
    for sid in node.path:
        n = tree.nodes.get(sid)
        if n is None or n.section_type == "root":
            continue
        titles.append(n.title)
    return titles


def ancestor_id_at_level(node: SectionNode, tree: TreeDocument, level: int) -> str:
    """The id of the ancestor (or self) sitting at `level` on this node's path."""
    for sid in node.path:
        n = tree.nodes.get(sid)
        if n is not None and n.level == level:
            return sid
    return ""


def chapter_id_from_path(node: SectionNode, tree: TreeDocument) -> str:
    for sid in node.path:
        n = tree.nodes.get(sid)
        if n is not None and n.section_type == "chapter":
            return sid
    return ""


def _chapter_title(node: SectionNode, tree: TreeDocument) -> str:
    cid = chapter_id_from_path(node, tree)
    ch = tree.nodes.get(cid)
    return ch.title if ch else ""


def _rag_chapters(tree: TreeDocument) -> list[SectionNode]:
    return [
        n
        for n in tree.nodes.values()
        if n.level == CHAPTER_LEVEL and n.section_type == "chapter" and n.rag_eligible
    ]


def _l3_units_under(chapter: SectionNode, tree: TreeDocument) -> list[SectionNode]:
    """rag-eligible level-3 (non-abstract) units within a chapter, in doc order."""
    return [
        n
        for n in tree.nodes.values()
        if n.rag_eligible
        and n.level == SUBTOPIC_LEVEL
        and n.section_type != "abstract"
        and chapter.section_id in n.path
    ]


def collect_target_nodes(
    tree: TreeDocument,
    *,
    include_chapters: list[str] | None = None,
    skip_patterns: list[str] | None = None,
) -> tuple[list[SectionNode], list[str], list[str]]:
    """Select the retrieval units to anchor questions on, chapter by chapter."""
    patterns = [p.lower() for p in (skip_patterns or [])]
    chapters = _rag_chapters(tree)
    if include_chapters:
        wanted = set(include_chapters)
        allowed = [c for c in chapters if c.section_id in wanted]
    else:
        allowed = [
            c
            for c in chapters
            if not (patterns and any(p in c.title.lower() for p in patterns))
        ]

    units: list[SectionNode] = []
    fallback: list[str] = []
    for ch in allowed:
        l3 = _l3_units_under(ch, tree)
        if l3:
            units.extend(l3)
        else:
            # No subtopics — anchor on the chapter, assembled whole.
            units.append(ch)
            fallback.append(ch.section_id)

    # Restore document order (tree.nodes is keyed in doc order).
    order = {sid: i for i, sid in enumerate(tree.nodes.keys())}
    units.sort(key=lambda n: order[n.section_id])
    return units, [c.title for c in allowed], fallback


# ── LLM call + JSON parse ────────────────────────────────────────────────────


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # take content between the first pair of fences
        body = s.split("```", 2)
        if len(body) >= 2:
            s = body[1]
            if s.lstrip().lower().startswith("json"):
                s = s.lstrip()[4:]
        s = s.strip()
        # may have a trailing fence captured; drop anything after a closing fence
        if "```" in s:
            s = s.split("```", 1)[0].strip()
    return s


def generate_for_node(
    *,
    client: OpenAI,
    model: str,
    template: str,
    meta: GuidelineMeta,
    node: SectionNode,
    tree: TreeDocument,
    content: str,
    num_questions: int,
    max_tokens: int,
    temperature: float,
) -> list[dict]:
    title_path_str = " > ".join(build_title_path(node, tree))
    prompt = template.format(
        guideline_name=meta.guideline_name,
        title_path=title_path_str,
        content=content,
        num_questions=num_questions,
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "system",
                "content": "You generate evaluation questions for a clinical RAG system. Return only a raw JSON array, no markdown.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    raw = resp.choices[0].message.content or ""
    parsed = json.loads(_strip_fences(raw))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected JSON array, got {type(parsed).__name__}")
    out = []
    for item in parsed:
        if not isinstance(item, dict) or "question" not in item:
            continue
        q = (item["question"] or "").strip()
        if q:
            out.append({"question": q})
    # LLMs sometimes return more questions than asked; enforce the cap.
    return out[:num_questions]


# ── output record ────────────────────────────────────────────────────────────


def build_record(
    *,
    meta: GuidelineMeta,
    node: SectionNode,
    tree: TreeDocument,
    content: str,
    style: str,
    q_dict: dict,
    q_index: int,
) -> dict:
    return {
        "question_id": f"{meta.guideline_id}_{node.section_id}_{style}_q{q_index:02d}",
        "question": q_dict["question"],
        "question_style": style,
        "guideline_id": meta.guideline_id,
        "guideline_name": meta.guideline_name,
        "source": {
            "node_id": node.section_id,
            "level": node.level,
            "section_type": node.section_type,
            "path": list(node.path),
            "title_path": build_title_path(node, tree),
            # The two retrieval granularities the agent exposes — precomputed so
            # the evaluator scores hit-rate by a field lookup, not a tree walk.
            "chapter_id": chapter_id_from_path(node, tree),
            "subtopic_id": ancestor_id_at_level(node, tree, SUBTOPIC_LEVEL),
            "char_count": len(content),
        },
    }


# ── main ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate evaluation questions from a guideline tree, anchored at retrieval units.",
    )
    p.add_argument(
        "guideline", help="guideline_id (matches data/corpus/guidelines/<id>/)"
    )
    p.add_argument(
        "--styles",
        nargs="+",
        choices=list(STYLE_PROMPTS.keys()),
        default=list(STYLE_PROMPTS.keys()),
        help="Question style(s) to generate (default: all three).",
    )
    p.add_argument(
        "--questions-per-node",
        type=int,
        default=2,
        help="Questions per (node, style) combo (default: 2).",
    )
    p.add_argument(
        "--min-chars",
        type=int,
        default=400,
        help="Skip units whose assembled content is shorter than this (default: 400).",
    )
    p.add_argument(
        "--chapters",
        nargs="+",
        help="Explicit chapter ids to include (overrides the built-in INCLUDE_CHAPTERS whitelist).",
    )
    p.add_argument(
        "--skip-chapter-patterns",
        nargs="+",
        default=FRONT_MATTER_PATTERNS,
        help="Front-matter title substrings; only used when no chapter whitelist applies.",
    )
    p.add_argument(
        "--include-front-matter",
        action="store_true",
        help="Disable both the whitelist and the front-matter filter (keep all chapters).",
    )
    p.add_argument(
        "--sample",
        type=int,
        help="Evenly sample N units across the guideline (for cross-guideline balance).",
    )
    p.add_argument(
        "--limit", type=int, help="Cap on target units, first N (for smoke testing)."
    )
    p.add_argument("--model", default="gpt-4o-mini", help="OpenAI chat model.")
    p.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Parallel LLM requests (default 8; these calls are I/O-bound). Set 1 for serial.",
    )
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--max-tokens", type=int, default=1500)
    p.add_argument(
        "--output",
        help="Output JSONL path (default: result/datasets/<id>/questions.jsonl).",
    )
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite output if it exists."
    )
    p.add_argument("--dry-run", action="store_true", help="Plan only — no LLM calls.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    meta = load_guideline_meta(args.guideline)
    tree = TreeDocument.load(meta.guideline_dir)
    logger.info(
        "📚 %s (%s) — %d nodes",
        meta.guideline_id,
        meta.guideline_name,
        len(tree.nodes),
    )

    # Chapter selection: explicit --chapters > built-in whitelist > front-matter
    # pattern filter. --include-front-matter disables all of them.
    if args.include_front_matter:
        include_chapters, skip_patterns = None, None
    elif args.chapters:
        include_chapters, skip_patterns = args.chapters, None
    elif args.guideline in INCLUDE_CHAPTERS:
        include_chapters, skip_patterns = INCLUDE_CHAPTERS[args.guideline], None
    else:
        include_chapters, skip_patterns = None, args.skip_chapter_patterns

    targets, allowed_titles, fallback = collect_target_nodes(
        tree,
        include_chapters=include_chapters,
        skip_patterns=skip_patterns,
    )
    logger.info(
        "🎯 %d sub-topic retrieval units from %d chapters: %s",
        len(targets),
        len(allowed_titles),
        ", ".join(allowed_titles),
    )
    if fallback:
        logger.info(
            "↩️  %d childless chapter(s) anchored whole (no L3): %s",
            len(fallback),
            ", ".join(fallback),
        )

    # Pre-assemble and filter by min-chars. assemble(include_descendants=True)
    # returns the unit's whole subtree — exactly what read_section gives the agent.
    work: list[tuple[SectionNode, str]] = []
    for node in targets:
        try:
            content = assemble(
                meta.guideline_dir,
                node.section_id,
                include_descendants=True,
                resolve_resources=True,
                table_format="markdown",
            )
        except Exception as e:
            logger.warning("  ⚠️  assemble %s failed: %s", node.section_id, e)
            continue
        if len(content) < args.min_chars:
            continue
        work.append((node, content))
    logger.info("📏 %d units pass min_chars=%d", len(work), args.min_chars)
    if args.sample and args.sample < len(work):
        # Evenly-spaced (strided) sample across the guideline, so coverage spans
        # all chapters rather than the first N. Deterministic (resume-friendly).
        n = len(work)
        idx = (
            [round(i * (n - 1) / (args.sample - 1)) for i in range(args.sample)]
            if args.sample > 1
            else [0]
        )
        idx = sorted(set(idx))
        work = [work[i] for i in idx]
        logger.info("🎲 sampled %d units evenly across the guideline", len(work))
    if args.limit:
        work = work[: args.limit]
        logger.info("✂️  limited to first %d", len(work))

    output_path = (
        Path(args.output)
        if args.output
        else DEFAULT_OUTPUT_BASE / meta.guideline_id / "questions.jsonl"
    )
    expected = len(work) * len(args.styles) * args.questions_per_node
    logger.info(
        "📝 plan: %d units × %d styles × %d q/node = %d questions → %s",
        len(work),
        len(args.styles),
        args.questions_per_node,
        expected,
        output_path,
    )

    if args.dry_run:
        for node, content in work[:5]:
            tp = " > ".join(build_title_path(node, tree))
            logger.info(
                "  [DRY] %s  L%d  (%d chars)  %s",
                node.section_id,
                node.level,
                len(content),
                tp,
            )
        if len(work) > 5:
            logger.info("  [DRY] ... and %d more", len(work) - 5)
        return

    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists. Use --overwrite to replace.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    templates = {s: STYLE_PROMPTS[s].read_text(encoding="utf-8") for s in args.styles}

    client = OpenAI()

    # One task per (unit, style); fan out over a thread pool (I/O-bound calls).
    tasks = [
        (node, content, style) for (node, content) in work for style in args.styles
    ]
    n_total = len(tasks)

    def run_task(
        task: tuple[SectionNode, str, str],
    ) -> tuple[SectionNode, str, list[dict], Exception | None]:
        node, content, style = task
        try:
            questions = generate_for_node(
                client=client,
                model=args.model,
                template=templates[style],
                meta=meta,
                node=node,
                tree=tree,
                content=content,
                num_questions=args.questions_per_node,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
            records = [
                build_record(
                    meta=meta,
                    node=node,
                    tree=tree,
                    content=content,
                    style=style,
                    q_dict=q,
                    q_index=qi,
                )
                for qi, q in enumerate(questions, 1)
            ]
            return node, style, records, None
        except Exception as e:  # noqa: BLE001 — isolate per-task failures
            return node, style, [], e

    n_written = 0
    n_failed = 0
    done = 0
    with output_path.open("w", encoding="utf-8") as fout:
        with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
            for node, style, records, err in ex.map(run_task, tasks):
                done += 1
                if err is not None:
                    n_failed += 1
                    logger.warning(
                        "  ⚠️  [%d/%d] %s (%s) failed: %s",
                        done,
                        n_total,
                        node.section_id,
                        style,
                        err,
                    )
                    continue
                for record in records:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    n_written += 1
                fout.flush()
                logger.info(
                    "  ✓ [%d/%d] %s (%s) → %d q",
                    done,
                    n_total,
                    node.section_id,
                    style,
                    len(records),
                )

    logger.info(
        "✅ wrote %d questions to %s (failed combos: %d)",
        n_written,
        output_path,
        n_failed,
    )


if __name__ == "__main__":
    main()
