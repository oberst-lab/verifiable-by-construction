"""Shared candidate pool + hit-rate scoring for the retrieval benchmark."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

# eval/retrieval/retrievers/base.py → the repository root is three up. The
# corpus library and the question generator's pure tree helpers both live
# under data/.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_QUESTIONS_DIR = _REPO_ROOT / "data" / "questions"
for _p in (str(_REPO_ROOT / "data" / "corpus"), str(_QUESTIONS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline_core.core.assemble import assemble  # noqa: E402
from pipeline_core.core.tree import TreeDocument  # noqa: E402
import generate_questions as gq  # noqa: E402  (pure tree helpers)


@dataclass
class CandidateUnit:
    """One retrieval unit in the shared pool. `doc_id` is globally unique
    (`node_id` repeats across guidelines, so it must be qualified)."""

    node_id: str
    guideline_id: str
    chapter_id: str
    title_path: list[str]
    summary: str
    text: str

    @property
    def doc_id(self) -> str:
        return f"{self.guideline_id}:{self.node_id}"

    @property
    def chapter_doc_id(self) -> str:
        return f"{self.guideline_id}:{self.chapter_id}"

    @property
    def breadcrumb(self) -> str:
        return " > ".join(self.title_path)


def build_candidates(guideline_ids: list[str]) -> list[CandidateUnit]:
    """The shared pool every method ranks or selects over: every retrieval unit
    across the given guidelines (level-3 sub-topics plus the childless-chapter
    fallback), front matter included as distractors. This is the same set the
    system itself retrieves from.
    """
    out: list[CandidateUnit] = []
    for gid in guideline_ids:
        meta = gq.load_guideline_meta(gid)
        tree = TreeDocument.load(meta.guideline_dir)
        units, _allowed, _fallback = gq.collect_target_nodes(
            tree, include_chapters=None, skip_patterns=None
        )
        for node in units:
            try:
                text = assemble(
                    meta.guideline_dir,
                    node.section_id,
                    include_descendants=True,
                    resolve_resources=True,
                    table_format="markdown",
                )
            except Exception:
                text = ""
            out.append(
                CandidateUnit(
                    node_id=node.section_id,
                    guideline_id=gid,
                    chapter_id=gq.chapter_id_from_path(node, tree),
                    title_path=gq.build_title_path(node, tree),
                    summary=node.summary or "",
                    text=text,
                )
            )
    return out


# ── ground truth + scoring ────────────────────────────────────────────────────


def gt_doc_id(record: dict) -> str:
    return f"{record['guideline_id']}:{record['source']['node_id']}"


def gt_chapter_doc_id(record: dict) -> str:
    return f"{record['guideline_id']}:{record['source']['chapter_id']}"


def hits(
    record: dict,
    ranked_doc_ids: list[str],
    candidates_by_doc: dict[str, CandidateUnit],
    k: int,
) -> tuple[bool, bool]:
    """(subtopic_hit, chapter_hit) for the top-k of a ranked doc_id list."""
    topk = ranked_doc_ids[:k]
    sub = gt_doc_id(record) in topk
    gt_ch = gt_chapter_doc_id(record)
    chap = any(
        candidates_by_doc[d].chapter_doc_id == gt_ch
        for d in topk
        if d in candidates_by_doc
    )
    return sub, chap


# ── OpenAI per-model call quirks ───────────────────────────────────────────────


def _is_reasoning_model(model: str) -> bool:
    m = model.lower()
    return m.startswith(("gpt-5", "o1", "o3", "o4"))


def anthropic_rejects_temperature(model: str) -> bool:
    """The Claude 5-generation reasoning models reject the `temperature` param ('deprecated for
    this model') and run at provider-native defaults, exactly as the gpt-5/o-series do on
    the OpenAI path.
    """
    return bool(re.match(r"claude-[a-z]+-5(?:-|$)", model.lower()))


def token_kwargs(model: str, n: int) -> dict:
    """Reasoning models (gpt-5/o-series) use `max_completion_tokens`; others
    use `max_tokens`."""
    key = "max_completion_tokens" if _is_reasoning_model(model) else "max_tokens"
    return {key: n}


def sampling_kwargs(
    model: str,
    *,
    temperature: float | None = 0.0,
    reasoning_effort: str | None = None,
) -> dict:
    """What to send for sampling and reasoning, per model family."""
    if _is_reasoning_model(model):
        return {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    out: dict = {}
    if temperature is not None:
        out["temperature"] = temperature
    if reasoning_effort:
        out["reasoning_effort"] = reasoning_effort
    return out
