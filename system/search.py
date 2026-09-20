"""Retrieval backend for the agent's `search` tool — a pluggable seam."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel, Field

from .guidelines import GuidelineScope
from .helper_agent import helper_agent
from .model_registry import resolve_helper_model
from .models import SectionSummary

_DEBUG = os.environ.get("SEARCH_DEBUG", "").lower() in ("1", "true", "yes")


def _trace(msg: str) -> None:
    if _DEBUG:
        print(msg, flush=True)


class SearchBackend(Protocol):
    """Turns a natural-language query into a small ordered set of in-scope
    sections. Implementations may use an LLM, embeddings, BM25, etc. — the agent
    is blind to which."""

    async def search(
        self, query: str, scope: GuidelineScope
    ) -> list[SectionSummary]: ...


class _Selection(BaseModel):
    """Structured output of the semantic selector: a brief rationale plus the
    chosen section ids (validated against the candidate set by the caller)."""

    reasoning: str = Field(description="One brief sentence on why these sections.")
    section_ids: list[str] = Field(
        default_factory=list,
        description="Relevant section doc_ids, most relevant first.",
    )


_SYSTEM = (Path(__file__).parent / "prompts" / "retrieval_select.md").read_text(
    encoding="utf-8"
)


def _listing(candidates: list[SectionSummary]) -> str:
    """One line per candidate: [doc_id] Guideline › Title — summary."""
    lines = []
    for c in candidates:
        crumb = f"{c.guideline_name} › {c.title}"
        lines.append(f"[{c.doc_id}] {crumb} — {c.summary or '(no summary)'}")
    return "\n".join(lines)


class SemanticSearchBackend:
    """Default backend: one cheap helper-model call selects relevant sections
    from their summaries (no full-text read). Reasoning is forced off (helper
    agent runs with thinking off). On any failure it degrades to returning ALL
    in-scope sections, so the agent can still proceed."""

    def __init__(
        self,
        model_key: str | None = None,
        *,
        provider_model: str | None = None,
        usage=None,
    ):
        self._provider_model = provider_model or resolve_helper_model(model_key)
        self._usage = usage
        self._model_id = self._provider_model.split(":", 1)[
            -1
        ]  # bare name, for pricing

    def _record_usage(self, result) -> None:
        """Add one helper run's tokens to the usage sink, read defensively
        (pydantic_ai's usage shape varies by version)."""
        if self._usage is None:
            return
        try:
            u = result.usage
            if callable(u):  # older pydantic_ai exposed usage() as a method
                u = u()
            inp = getattr(u, "input_tokens", None)
            inp = getattr(u, "request_tokens", 0) if inp is None else inp
            out = getattr(u, "output_tokens", None)
            out = getattr(u, "response_tokens", 0) if out is None else out
            self._usage.record_tokens(self._model_id, inp, out, calls=1)
        except Exception:
            pass

    async def search(self, query: str, scope: GuidelineScope) -> list[SectionSummary]:
        candidates = scope.sections()  # in-scope sub-topic units
        if not candidates:
            return []
        by_id = {c.doc_id: c for c in candidates}
        prompt = f"QUESTION:\n{query}\n\nSECTIONS:\n{_listing(candidates)}"
        try:
            agent = helper_agent(
                self._provider_model,
                instructions=_SYSTEM,
                max_tokens=1200,
                output_type=_Selection,
            )
            result = await agent.run(prompt)
            self._record_usage(result)
            chosen = result.output.section_ids
            why = (result.output.reasoning or "").strip()
        except Exception as e:
            # Selector unavailable/failed: degrade to the full in-scope set rather
            # than returning nothing, so the turn can still be answered.
            _trace(
                f"[search] q={query!r} FAILED ({e}) → degraded to full set "
                f"({len(candidates)})"
            )
            return candidates
        # Keep valid ids in the model's order; dict.fromkeys dedupes preserving order.
        out = [by_id[sid] for sid in dict.fromkeys(chosen) if sid in by_id]
        # Debug trace (lands in the server log via stdout): the query, what it
        # picked, and the selector's own rationale — so a run's retrieval is
        # observable without re-running it offline.
        _trace(
            f"[search] q={query!r} → {len(out)} picked: "
            f"{', '.join(s.doc_id for s in out)} | why: {why[:160]}"
        )
        return out
