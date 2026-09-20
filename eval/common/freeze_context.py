#!/usr/bin/env python3
"""Phase 1 of the controlled-context arm: FREEZE the retrieval stage.

    uv run python eval/common/freeze_context.py \\
        --answers eval/result/answers/bench50_gpt54mini.jsonl \\
        --output  eval/result/frozen/bench50_context.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# eval/common/ -> repo root is two up; system/ imported the same way the CLI does.
# The sibling `scoring` module (this dir) owns the canonical JSONL loader.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_COMMON = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_COMMON)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from scoring import load_jsonl  # noqa: E402
from system.guidelines import GuidelineScope  # noqa: E402

logger = logging.getLogger("freeze_context")


def _resolve_sections(
    read_set: list[str], selected: list[str], granularity: str
) -> list[dict]:
    """Resolve each read `guideline_id:section_id` to the full payload the
    production `read_section` returned (via the SAME GuidelineScope.read)."""
    scope = GuidelineScope.of(selected, granularity)
    sections: list[dict] = []
    for doc_id in read_set:
        gid, _, sid = doc_id.partition(":")
        try:
            content = scope.read(gid, sid)
        except ValueError as e:
            logger.warning("  ⚠️  could not resolve %s: %s", doc_id, e)
            continue
        sections.append(
            {
                "doc_id": content.doc_id,
                "guideline_name": content.guideline_name,
                "title": content.title,
                "text": content.text,
            }
        )
    return sections


def freeze(records: list[dict], granularity: str) -> list[dict]:
    frozen: list[dict] = []
    for r in records:
        read_set = r.get("read_set", [])
        # A question may span several guidelines; default to its own guideline_id.
        selected = r.get("selected_guidelines") or [r["guideline_id"]]
        sections = _resolve_sections(read_set, selected, granularity)
        if not sections:
            logger.warning(
                "  ⚠️  %s has no resolvable sections (read_set=%s) — skipping",
                r.get("question_id"),
                read_set,
            )
            continue
        rec = {
            "question_id": r["question_id"],
            "question": r["question"],
            "guideline_id": r["guideline_id"],
            "retriever_model": r.get("model"),
            "granularity": granularity,
            "sections": sections,
        }
        if r.get("patient_profile"):
            rec["patient_profile"] = r["patient_profile"]
        frozen.append(rec)
    return frozen


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--answers",
        required=True,
        help="answers.jsonl from a FIXED-retriever run (main+helper = gpt-5.4-mini).",
    )
    p.add_argument("--output", required=True, help="Frozen-context artifact (JSONL).")
    p.add_argument(
        "--granularity",
        default="subtopic",
        choices=["chapter", "subtopic"],
        help="Must match the granularity the answers were retrieved at.",
    )
    p.add_argument("--limit", type=int, help="Cap on questions (smoke test).")
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite output if it exists."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    answers_path = Path(args.answers)
    if not answers_path.exists():
        raise SystemExit(f"{answers_path} not found")
    output_path = Path(args.output)
    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists. Use --overwrite to replace.")

    records = load_jsonl(answers_path)[: args.limit or None]
    logger.info("📥 %d answer records from %s", len(records), answers_path)
    frozen = freeze(records, args.granularity)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as fout:
        for rec in frozen:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_sec = sum(len(f["sections"]) for f in frozen)
    logger.info(
        "✅ froze %d questions (%d sections total) → %s",
        len(frozen),
        n_sec,
        output_path,
    )


if __name__ == "__main__":
    main()
