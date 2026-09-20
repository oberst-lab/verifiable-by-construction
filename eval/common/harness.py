#!/usr/bin/env python3
"""Answer harness — run the production agent over a question set and persist the
raw evaluation material for the citation-faithfulness track.

    uv run python eval/common/harness.py \\
        --questions eval/result/datasets/blood-pressure-2025/qa.jsonl \\
        --output    eval/result/answers/bp_haiku.jsonl \\
        --model claude-haiku-4-5 --limit 20
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv

# eval/common/ → repo root is two up; eval/ is one up. The system package is
# imported from the repo root exactly as the server does.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = Path(__file__).resolve().parents[1]
for _p in (
    str(_REPO_ROOT),
    str(_REPO_ROOT / "data" / "corpus"),
    str(_EVAL_DIR / "common"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)
load_dotenv(_REPO_ROOT / ".env", override=False)

import system.guidelines as G  # noqa: E402
from system.agent_factory import make_agent  # noqa: E402
from system.model_registry import DEFAULT_MODEL, resolve  # noqa: E402
from system.models import AgentDeps, UIState  # noqa: E402
from system.search import SemanticSearchBackend  # noqa: E402

from system.cite_parser import distinct_cited_docs, parse_citations  # noqa: E402
from model_endpoints import (  # noqa: E402
    build_agent_model,
    needs_custom_endpoint,
)
from release_meta import (  # noqa: E402
    dump_manifest,
    git_sha,
    model_slug,
    unpriced_cost_note,
)
from scoring import load_jsonl  # noqa: E402
from usage import UsageTracker, record_run  # noqa: E402

logger = logging.getLogger("harness")

DEFAULT_OUTPUT_BASE = _EVAL_DIR / "result" / "answers"
# Stage-3 releases: generation on a frozen context, blessed for reuse (--release).
_ANSWERS_RELEASES = _EVAL_DIR / "result" / "releases" / "answers"
# Frozen-context prompt: the controlled-context arm — retrieval is frozen (run
# once by a fixed retriever), the sections are handed to the model in the user
# message, and search/read are disabled. See freeze_context.py for Phase 1.
_FROZEN_PROMPT = _EVAL_DIR / "common" / "prompts" / "system_prompt_frozen_context.md"


def compose_frozen_prompt(question: str, sections: list[dict]) -> str:
    """The user message for the controlled arm: the frozen sections (each headed
    by its doc_id + guideline name + title, so the model can cite by doc_id and
    name the guideline in prose) followed by the question."""
    blocks = [
        f"[doc_id: {s['doc_id']}] {s['guideline_name']} — {s['title']}\n{s['text']}"
        for s in sections
    ]
    joined = "\n\n".join(blocks) if blocks else "(no sections retrieved)"
    return f"Guideline sections retrieved for this question:\n\n{joined}\n\nQuestion: {question}"


def _arm_label(frozen_context: dict | None) -> str:
    """The control arm a run belongs to: single source of truth for the `arm`
    field and the log line."""
    return "controlled" if frozen_context is not None else "end_to_end"


DEFAULT_MAX_TOKENS = 25000

_ANTHROPIC_MECHANISM: dict[str, str] = {
    "claude-opus-5": "output_config",
    "claude-sonnet-5": "output_config",
    "claude-sonnet-4-6": "output_config",
    "claude-haiku-4-5": "budget_tokens",
}
# Models whose thinking is OFF unless asked for, so "no parameter sent" genuinely means
# no thinking. Everything else in _ANTHROPIC_MECHANISM thinks by default, which is why
# `--effort off` has to send something explicit rather than just omitting the parameter.
_ANTHROPIC_THINKING_OFF_BY_DEFAULT = frozenset({"claude-haiku-4-5"})
# Effort levels `output_config.effort` accepts. There is no "minimal".
_ANTHROPIC_EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
# Thinking budgets for the extended-thinking models. Written out rather than borrowed
# from pydantic_ai's ANTHROPIC_THINKING_BUDGET_MAP so the number that lands in the
# disclosure record is one we chose and can cite, not one a library chose for us.
ANTHROPIC_BUDGET: dict[str, int] = {
    "minimal": 1024,
    "low": 2048,
    "medium": 10000,
    "high": 16384,
    "xhigh": 32768,
}


@dataclass(frozen=True)
class ReasoningPlan:
    """One model's resolved generation config, plus what actually reaches the API."""

    model: str
    max_tokens: int
    thinking_effort: str | bool | None
    model_settings: dict  # provider-specific raw keys, bypassing the profile
    sent: dict  # disclosure record, written into every answer + the MANIFEST


def _raw_thinking_effort(model: str, effort: str | None) -> str | bool:
    """Effort for a RAW `provider:model` escape-hatch (a model not in the registry catalog,
    e.g. ``openai:gpt-4.1-mini`` — used heavily by the retrieval experiments but not offered
    in the production picker).
    """
    provider, _, name = model.partition(":")
    non_reasoning = provider == "openai" and not name.lower().startswith(
        ("gpt-5", "o1", "o3", "o4")
    )
    if effort is None:
        effort = "off"
    # `none` / `false` -> off is deliberate and tested; see the docstring before editing.
    if non_reasoning or str(effort).lower() in {"off", "none", "false"}:
        return False
    return effort


def _reasoning_route(model: str, effort: str) -> tuple[str | bool | None, dict, dict]:
    """(thinking_effort, extra_model_settings, sent) for ONE model at ONE effort level."""
    provider, _, name = model.partition(":")
    off = effort == "off"

    if provider == "anthropic":
        mech = _ANTHROPIC_MECHANISM.get(name)
        if mech is None:
            raise SystemExit(
                f"{model}: no Anthropic reasoning mechanism recorded. Add it to "
                "_ANTHROPIC_MECHANISM after checking the vendor's per-model thinking "
                "table; guessing sends a field the model may reject."
            )
        if off:
            if name in _ANTHROPIC_THINKING_OFF_BY_DEFAULT:
                return (
                    None,
                    {},
                    {
                        "field": None,
                        "value": "thinking off",
                        "honored": "verified",
                        "vendor_note": "no parameter sent; this model's documented "
                        "default is thinking off, so omission genuinely disables it",
                    },
                )
            return (
                None,
                {"anthropic_thinking": {"type": "disabled"}},
                {
                    "field": "thinking.type",
                    "value": "disabled",
                    "honored": "verified",
                    "vendor_note": "sent explicitly because this model thinks by "
                    "default, so omitting the parameter would NOT disable it",
                },
            )
        if mech == "output_config":
            if effort not in _ANTHROPIC_EFFORT_LEVELS:
                raise SystemExit(
                    f"{model}: effort {effort!r} is not accepted by "
                    f"output_config.effort. Valid: {list(_ANTHROPIC_EFFORT_LEVELS)}"
                )
            return (
                None,  # no unified Thinking capability; the raw key carries it
                {"anthropic_effort": effort},
                {
                    "field": "output_config.effort",
                    "value": effort,
                    "honored": "verified",
                },
            )
        budget = ANTHROPIC_BUDGET.get(effort)
        if budget is None:
            raise SystemExit(
                f"{model}: no thinking budget recorded for effort {effort!r}. "
                f"Valid: {sorted(ANTHROPIC_BUDGET)}"
            )
        return (
            None,
            {"anthropic_thinking": {"type": "enabled", "budget_tokens": budget}},
            {
                "field": "thinking.budget_tokens",
                "value": budget,
                "honored": "verified",
            },
        )

    # OpenAI: reasoning.effort on the Responses API, reasoning_effort on chat.
    field = "reasoning.effort" if provider == "openai-responses" else "reasoning_effort"
    # `none` is a documented OpenAI value, and it is also where gpt-4.1* lands (no
    # reasoning channel, so _raw_thinking_effort forced it off), where it is a no-op on a
    # model that never reasons.
    value = "none" if off else effort
    return (
        None,
        {"openai_reasoning_effort": value},
        {"field": field, "value": value, "honored": "verified"},
    )


def resolve_model_config(
    model_arg: str, effort_arg: str | None, max_tokens_arg: int | None
) -> ReasoningPlan:
    """Resolve --model / --effort / --max-tokens into a ReasoningPlan. Shared by every
    answer-driver main() so the effort policy lives in ONE place (was duplicated, and
    one copy drifted).
    """
    if needs_custom_endpoint(model_arg) or ":" in model_arg:
        # A model given as a raw `provider:model` string rather than a registry
        # key. It bypasses the registry; the routing itself happens in
        # build_agent_model.
        model = model_arg
        max_tokens = max_tokens_arg or DEFAULT_MAX_TOKENS
        concrete: str | bool = _raw_thinking_effort(model, effort_arg)
    else:
        resolved = resolve(model_arg, effort_arg)
        model = resolved.model
        max_tokens = max_tokens_arg or resolved.max_tokens
        concrete = resolved.thinking_effort
        if (
            effort_arg is not None
            and concrete is not False
            and str(concrete) != str(effort_arg)
        ):
            raise SystemExit(
                f"{model_arg}: the registry does not carry effort {effort_arg!r} and "
                f"silently resolved it to {concrete!r}. Pass the raw provider:model form "
                f"({resolved.model!r}) to reach that level, which routes through "
                "_reasoning_route and validates against the vendor's own set."
            )

    if effort_arg is None:
        # Provider-native: send nothing. Recorded as a null field so a reader can tell
        # "no parameter was sent" apart from "a parameter was sent with this value".
        return ReasoningPlan(
            model=model,
            max_tokens=max_tokens,
            thinking_effort=None,
            model_settings={},
            sent={
                "requested": "provider-default",
                "field": None,
                "value": None,
                "honored": "not-applicable",
            },
        )

    level = "off" if concrete is False else str(concrete)
    thinking_effort, extra, sent = _reasoning_route(model, level)
    sent = {"requested": str(effort_arg), **sent}

    # Thinking counts toward max_tokens, so a budget at or above the ceiling truncates
    # the answer before it is written, or 400s. Fail here rather than after paying for
    # 222 questions of half-written output.
    budget = (extra.get("anthropic_thinking") or {}).get("budget_tokens")
    if budget is not None and max_tokens <= budget:
        raise SystemExit(
            f"{model}: max_tokens={max_tokens} does not exceed the thinking budget "
            f"{budget} for effort {concrete!r}. Thinking tokens count toward "
            f"max_tokens, so raise it (--max-tokens {budget * 2} or more)."
        )

    return ReasoningPlan(
        model=model,
        max_tokens=max_tokens,
        thinking_effort=thinking_effort,
        model_settings=extra,
        sent=sent,
    )


def _requested_label(sent: dict | None, effort: str | bool | None) -> str:
    """The effort the RUN ASKED FOR, as a stable string."""
    if sent and sent.get("requested"):
        return str(sent["requested"])
    return _effort_label(effort)


def _effort_label(effort: str | bool | None) -> str:
    """JSON-stable disclosure label for the effective reasoning config — always a
    string so the recorded field never mixes str and bool. None = no reasoning
    param sent (provider-native default); False = thinking explicitly disabled.
    """
    if effort is None:
        return "provider-default"
    if effort is False:
        return "off"
    return str(effort)


class _NoEvents:
    """No-op UIEvents sink — the emitter calls these; we discard them (the
    read/retrieval sets still accumulate in deps.state)."""

    def document(self, doc):
        return None

    def citation(self, citation):
        return None

    def calculator(self, calc):
        return None

    def search(self, retrieval):
        return None


class AnswerHarness:
    """Builds one shared agent and runs questions through it concurrently."""

    def __init__(
        self,
        guideline_ids: list[str],
        *,
        model: str,
        thinking_effort: str | bool,
        max_tokens: int,
        extra_model_settings: dict | None = None,
        reasoning_sent: dict | None = None,
        # Recorded as provenance. The retrieval unit is the sub-topic and there
        # is no other setting; a report that omitted it would be ambiguous to
        # anyone comparing it with one produced when there was.
        granularity: str = "subtopic",
        usage: UsageTracker | None = None,
        frozen_context: dict[str, list[dict]] | None = None,
        frozen_source: str | None = None,
        helper_model: str | None = None,
    ):
        # `model` / `thinking_effort` / `max_tokens` arrive already resolved by
        # main() — no model knowledge is hardcoded here. thinking_effort None =
        # provider-native default reasoning (the standard eval policy; see main()).
        self.granularity = granularity
        self.model = model  # resolved provider:model string
        self._model_id = model.split(":", 1)[-1]  # for billing
        self._thinking_effort = thinking_effort  # the label we ASKED for
        self._reasoning_sent = reasoning_sent
        self._max_tokens = max_tokens
        # Answers cut off by the output ceiling are not capability signals, they are
        # measurement artifacts, so the count travels with the run instead of being
        # rediscovered later from answer lengths.
        self.n_truncated = 0
        self.usage = usage
        self._frozen = frozen_context
        self._frozen_source = frozen_source
        self._arm = _arm_label(frozen_context)
        # The frozen arm disables retrieval and swaps the prompt; end-to-end uses
        # the system's own.
        retrieval_tools = frozen_context is None
        override = (
            _FROZEN_PROMPT.read_text(encoding="utf-8")
            if frozen_context is not None
            else None
        )

        self._num2id: dict[tuple[str, int], str] = {}
        if frozen_context is None:
            for gid in guideline_ids:
                for s in G.list_all_sections([gid], tree_level=G.SUBTOPIC_LEVEL):
                    self._num2id[(gid, s.section_number)] = s.section_id

        if helper_model and ":" in helper_model:
            backend = SemanticSearchBackend(provider_model=helper_model, usage=usage)
        else:
            backend = SemanticSearchBackend(model_key=helper_model, usage=usage)
        # `self.model` stays the STRING (record / slug / billing use it). Only the
        # pydantic_ai Agent gets the routed form, which for a natively routed
        # provider is the string unchanged. Built once, not per call.
        self._agent = make_agent(
            events=_NoEvents(),
            model=build_agent_model(model),
            max_tokens=max_tokens,
            thinking_effort=thinking_effort,
            extra_model_settings=extra_model_settings,
            search_backend=backend,
            instructions_override=override,
            retrieval_tools=retrieval_tools,
        )

    def _read_doc_id(self, guideline_id: str, section_number: int) -> str | None:
        nid = self._num2id.get((guideline_id, section_number))
        return f"{guideline_id}:{nid}" if nid else None

    async def run_one(
        self,
        *,
        question_id: str,
        question: str,
        guideline_id: str,
        selected_guidelines: list[str] | None = None,
        patient_context: str | None = None,
        extra: dict | None = None,
    ) -> dict:
        deps = AgentDeps(
            state=UIState(),
            selected_guidelines=selected_guidelines or [guideline_id],
            trim_prior_thinking=True,
            patient_context=patient_context,
        )

        if self._frozen is not None:
            # Controlled arm: inject the frozen sections + question; retrieval is
            # disabled, so read/retrieved sets ARE the frozen sections (identical
            # across generation models — that is the whole point of the arm).
            sections = self._frozen.get(question_id, [])
            run_input = compose_frozen_prompt(question, sections)
            result = await self._agent.run(run_input, deps=deps)
            read = list(dict.fromkeys(s["doc_id"] for s in sections))
            retrieved = list(read)
        else:
            result = await self._agent.run(question, deps=deps)
            # read set (sections opened), canonical + deduped, in read order.
            read = []
            for d in deps.state.documents:
                doc_id = self._read_doc_id(d.guideline_id, d.section_number)
                if doc_id:
                    read.append(doc_id)
            read = list(dict.fromkeys(read))
            # retrieved set: union of what `search` surfaced (already canonical).
            retrieved = []
            for r in deps.state.retrievals:
                retrieved.extend(r.doc_ids)
            retrieved = list(dict.fromkeys(retrieved))

        answer = result.output if isinstance(result.output, str) else str(result.output)

        served_model = None
        finish_reason = None
        reasons: list[str] = []
        try:
            for _m in result.all_messages():
                _mn = getattr(_m, "model_name", None)
                if _mn:
                    served_model = _mn
                _fr = getattr(_m, "finish_reason", None)
                if _fr:
                    reasons.append(str(_fr))
                    finish_reason = str(_fr)
        except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
            # Reset BOTH, not just served_model: a half-collected reason list would
            # under-report truncation, which is worse than reporting it as unknown.
            served_model, finish_reason, reasons = None, None, []
        _CUT = {"length", "max_tokens"}
        truncated = any(r in _CUT for r in reasons)
        if truncated:
            self.n_truncated += 1
            logger.warning(
                "⚠️  %s hit the output ceiling (max_tokens=%d, finish_reason=%s) — this "
                "answer is truncated and must not be read as a capability signal",
                question_id,
                self._max_tokens,
                finish_reason,
            )

        citations = parse_citations(answer)
        cited = distinct_cited_docs(citations)

        if self.usage is not None:
            self._record_usage(result)

        record = {
            "question_id": question_id,
            "question": question,
            "guideline_id": guideline_id,
            "model": self.model,
            # The snapshot the provider actually served (best-effort; None if the
            # route does not report it). `model` is the requested alias; this is what
            # really answered, so an alias silently repointed later is still traceable.
            "served_model": served_model,
            # Effective reasoning config, always a string (see _effort_label):
            # "provider-default" = no reasoning param sent (ran native default);
            # "off" = thinking explicitly disabled; else the override effort.
            "thinking_effort": _requested_label(
                self._reasoning_sent, self._thinking_effort
            ),
            "reasoning_sent": self._reasoning_sent,
            # max_tokens travels per answer because it has not been uniform across a
            # release: raw `provider:model` rows and registry keys took different
            # ceilings, and a ceiling is only interpretable next to the truncation flag.
            "max_tokens": self._max_tokens,
            # The FINAL response's reason, plus every reason in the loop: a cut at
            # an intermediate step is invisible in the final one.
            "finish_reason": finish_reason,
            "finish_reasons": reasons or None,
            "truncated": truncated,
            "granularity": self.granularity,
            "arm": self._arm,
            "answer": answer,
            "citations": [
                {
                    "doc_id": c.doc_id,
                    "quote": c.quote,
                    "guideline_id": c.guideline_id,
                    "section_id": c.section_id,
                }
                for c in citations
            ],
            "cited_set": cited,
            "read_set": read,
            "retrieved_set": retrieved,
            "n_citations": len(citations),
            "n_cited_docs": len(cited),
            "n_read": len(read),
        }
        # Controlled arm: record WHICH frozen context produced these answers, so the
        # jsonl self-identifies (model is already stamped; without this you'd know the
        # generation model but not the context it read).
        if self._frozen is not None:
            record["frozen_context"] = self._frozen_source
        # The applicability judge reads patient_profile off the record; the planted
        # labels (patient_id / planted_pitfall) ride along via `extra`.
        if patient_context is not None:
            record["patient_profile"] = patient_context
        if extra:
            record.update(extra)
        return record

    def _record_usage(self, result) -> None:
        # Delegated so the harness and the metric judges extract usage the same way.
        record_run(self.usage, self._model_id, result)


# ── IO ────────────────────────────────────────────────────────────────────────

_ANSWERS_MANIFEST_HEADER = (
    "# Answers release: generation on a frozen context, blessed for reuse.\n"
    "# READ-ONLY: to regenerate, re-run harness.py --release (--overwrite); do not\n"
    "# hand-edit. Every generation-quality scorer reads this one artifact, so the\n"
    "# answers are frozen once rather than each scorer re-running generation.\n"
    "# Produced by eval/common/harness.py --release.\n\n"
)


def answers_release_dir(
    frozen_path: Path, model: str, suffix: str | None = None
) -> Path:
    """Where a --release run lands: releases/answers/<frozen-source>[-suffix]/<model-slug>/.
    The frozen source is the release dir the --frozen-context file sits in.
    """
    parent = frozen_path.parent.name + (f"-{suffix}" if suffix else "")
    return _ANSWERS_RELEASES / parent / model_slug(model)


def _library_versions() -> dict[str, str | None]:
    """Versions of the clients that decide what actually reaches each vendor's API."""
    from importlib.metadata import PackageNotFoundError, version

    out: dict[str, str | None] = {}
    for pkg in ("pydantic-ai", "anthropic", "openai"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            out[pkg] = None
    return out


def write_answers_manifest(
    out_dir: Path,
    *,
    frozen_path: Path,
    model: str,
    thinking_effort: str | bool | None,
    max_tokens: int,
    granularity: str,
    n_questions: int,
    reasoning_sent: dict | None = None,
    n_truncated: int | None = None,
) -> None:
    """Write the answers-release MANIFEST beside the answers.jsonl the run already
    wrote, tracing the derivation chain (this run -> the frozen-context release it
    consumed -> that release's question set) so the release self-identifies end to
    end. Reads the run's usage from the answers.usage.json already on disk."""
    usage_dict = json.loads((out_dir / "answers.usage.json").read_text())
    frozen_src = frozen_path.parent.name
    # One more link up: which question set the frozen context derived from (best-
    # effort; the frozen release records it in its own MANIFEST).
    src_manifest = frozen_path.parent / "MANIFEST.yaml"
    src_meta = (
        yaml.safe_load(src_manifest.read_text()) if src_manifest.exists() else None
    )
    # Best-effort: tolerate a missing / empty / non-mapping MANIFEST rather than
    # crashing AFTER generation has already produced answers.jsonl.
    questions_release = (
        src_meta.get("derived_from") if isinstance(src_meta, dict) else None
    )
    manifest = {
        "name": f"{frozen_src}/{model_slug(model)}",
        "kind": "answers",
        "released": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_sha(),
        "arm": "controlled",
        "generation": {
            "model": model,
            "thinking_effort": _requested_label(reasoning_sent, thinking_effort),
            # The API field + value that actually carried the effort, and whether the
            # vendor documents that it is honored. `thinking_effort` above is only what
            # we requested; see ReasoningPlan for why the two are recorded separately.
            "reasoning_sent": reasoning_sent,
            "temperature": "provider-default",
            "max_tokens": max_tokens,
            # Answers cut off by max_tokens. Must be 0 for the run to be read as a
            # capability measurement; anything else means the metric partly measures the
            # ceiling. None for runs written before this was tracked.
            "n_truncated": n_truncated,
            "granularity": granularity,
            "n_questions": n_questions,
        },
        "libraries": _library_versions(),
        # The derivation chain, so the release self-identifies end to end.
        "frozen_context": frozen_src,
        "derived_from_questions": questions_release,
        "source_run": {"usage": usage_dict},
    }
    note = unpriced_cost_note(usage_dict)
    if note:
        manifest["source_run"]["cost_note"] = note
    dump_manifest(out_dir / "MANIFEST.yaml", _ANSWERS_MANIFEST_HEADER, manifest)


def _question_fields(r: dict) -> dict:
    """The pass-through question fields run_one needs. Shared by the plain
    question set and the frozen-context artifact, which carries the same keys."""
    return {k: r[k] for k in ("question_id", "question", "guideline_id")}


def _load_questions(path: Path, limit: int | None) -> list[dict]:
    # `limit or None`: a falsy limit (None or 0) means "all", matching the prior
    # `records[:limit] if limit else records` semantics.
    return [_question_fields(r) for r in load_jsonl(path)[: limit or None]]


async def _run_all(
    harness: AnswerHarness, records: list[dict], concurrency: int
) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    n = len(records)
    done = 0

    async def _task(rec: dict) -> dict | None:
        nonlocal done
        async with sem:
            try:
                out = await harness.run_one(**rec)
            except Exception as e:  # noqa: BLE001 — isolate per-question failures
                logger.warning("  ⚠️  %s failed: %s", rec["question_id"], e)
                return None
            done += 1
            logger.info(
                "  ✓ [%d/%d] %s  (%d cites / %d read)",
                done,
                n,
                rec["question_id"],
                out["n_citations"],
                out["n_read"],
            )
            return out

    results = await asyncio.gather(*(_task(r) for r in records))
    return [r for r in results if r is not None]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the agent over a question set and persist answers + citations."
    )
    p.add_argument(
        "--questions",
        help="qa.jsonl / questions.jsonl path. Required unless --frozen-context "
        "is given (which carries its own questions).",
    )
    p.add_argument(
        "--frozen-context",
        help="Controlled-context arm: a frozen-context artifact from "
        "freeze_context.py. Retrieval is disabled and the model answers from the "
        "frozen sections. Supplies the question set, so --questions is not needed.",
    )
    p.add_argument(
        "--output",
        help="Output answers.jsonl (default: result/answers/<questions-stem>.jsonl).",
    )
    p.add_argument(
        "--release",
        action="store_true",
        help="Bless the run straight into result/releases/answers/<frozen-source>/<model>/ "
        "(answers.jsonl + usage + MANIFEST), instead of scratch. Frozen-context arm "
        "only (the answers stage is generation on a frozen context). No manual copy.",
    )
    p.add_argument(
        "--release-suffix",
        help="Appended to the release directory name, so the same model can be blessed "
        "twice on the same frozen context under different settings (e.g. "
        "--effort high --release-suffix effort-high). Without it the release path is "
        "(frozen source, model) alone and a second configuration would collide with the "
        "arm it exists to be compared against.",
    )
    p.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Registry model KEY (default: {DEFAULT_MODEL}) OR a raw provider:model string for eval-only models, e.g. openai:gpt-4.1-mini.",
    )
    p.add_argument(
        "--helper-model",
        default=None,
        help="Selector (section-selection) model. Default = the production helper "
        "(gpt-5.4-mini). For a freeze run where the retriever should equal the main "
        "model, pass the SAME value as --model (registry key, or raw provider:model "
        "such as openai:gpt-4.1-mini).",
    )
    p.add_argument(
        "--effort",
        default=None,
        help="OVERRIDE reasoning effort (low/medium/high/off). Default (unset) = "
        "provider-native default reasoning for every model — the standard eval "
        "policy. Pass this only for a deliberate sweep.",
    )
    p.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Override the registry's per-model max_tokens.",
    )
    p.add_argument(
        "--concurrency", type=int, default=4, help="Parallel agent runs (default 4)."
    )
    p.add_argument("--limit", type=int, help="Cap on questions (smoke test).")
    p.add_argument(
        "--overwrite", action="store_true", help="Overwrite output if it exists."
    )
    p.add_argument("--dry-run", action="store_true", help="Plan only — no agent runs.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    # Surface the arm conflict here (before --dry-run / any run), not only in the
    # AnswerHarness constructor, which --dry-run never reaches.
    if args.release:
        if not args.frozen_context:
            raise SystemExit(
                "--release requires --frozen-context (answers = generation on a "
                "frozen context)"
            )
        if args.output:
            raise SystemExit("--release computes its own path; drop --output")
        if args.limit:
            raise SystemExit(
                "--release blesses a COMPLETE run; --limit would overwrite the "
                "release path with a truncated set. Smoke to scratch first (drop "
                "--release), then release without --limit."
            )
        frozen_ctx_root = (
            _EVAL_DIR / "result" / "releases" / "frozen-context"
        ).resolve()
        if frozen_ctx_root not in Path(args.frozen_context).resolve().parents:
            raise SystemExit(
                "--release needs a blessed frozen-context release (a context.jsonl.gz "
                "under result/releases/frozen-context/<name>/); the given path is outside "
                "that tree, so its provenance chain cannot be recorded."
            )

    # Two input modes: a plain question set (the end-to-end arm) or a
    # frozen-context artifact (controlled arm — it carries its own questions +
    # sections + the granularity they were retrieved at).
    if args.frozen_context:
        # Resolve up front so the frozen-source name (frozen_path.parent.name, used for
        # the release dir + the record's frozen_context stamp) is correct even for a
        # bare filename — matching the --release guard, which also resolves.
        frozen_path = Path(args.frozen_context).resolve()
        if not frozen_path.exists():
            raise SystemExit(f"{frozen_path} not found")
        frozen_records = load_jsonl(frozen_path)[: args.limit or None]
        if not frozen_records:
            raise SystemExit(f"{frozen_path} has no records")
        records = []
        for r in frozen_records:
            rec = _question_fields(r)
            if r.get("patient_profile"):
                rec["patient_context"] = r["patient_profile"]
            records.append(rec)
        frozen_context = {r["question_id"]: r["sections"] for r in frozen_records}
        n_patient = sum(1 for rec in records if rec.get("patient_context"))
        if n_patient:
            logger.info(
                "   patient context re-injected for %d/%d questions",
                n_patient,
                len(records),
            )
        # Granularity is a property of the frozen artifact, not a CLI choice.
        granularity = frozen_records[0].get("granularity", "subtopic")
        source, stem = frozen_path, frozen_path.stem
    else:
        if not args.questions:
            raise SystemExit("--questions is required unless --frozen-context is given")
        questions_path = Path(args.questions)
        if not questions_path.exists():
            raise SystemExit(f"{questions_path} not found")
        records = _load_questions(questions_path, args.limit)
        frozen_context = None
        granularity = "subtopic"
        source, stem = questions_path, questions_path.stem

    # Effort policy + registry-key/raw-model resolution live in one shared helper
    # (see resolve_model_config), so every answer-driver main() stays in step.
    # Resolved up front because --release derives its output dir from the model slug.
    plan = resolve_model_config(args.model, args.effort, args.max_tokens)
    model, thinking_effort, max_tokens = (
        plan.model,
        plan.thinking_effort,
        plan.max_tokens,
    )
    if args.release:
        output_path = (
            answers_release_dir(frozen_path, model, args.release_suffix)
            / "answers.jsonl"
        )
    elif args.output:
        output_path = Path(args.output)
    else:
        output_path = DEFAULT_OUTPUT_BASE / f"{stem}.jsonl"
    guideline_ids = sorted({r["guideline_id"] for r in records})
    arm = _arm_label(frozen_context)
    logger.info("📥 %d questions from %s → %s", len(records), source, output_path)
    logger.info(
        "   model=%s effort=%s granularity=%s arm=%s guidelines=%s",
        args.model,
        args.effort or "(default)",
        granularity,
        arm,
        guideline_ids,
    )

    if args.dry_run:
        for r in records[:5]:
            logger.info("  [DRY] %s  %s", r["question_id"], r["question"][:80])
        if len(records) > 5:
            logger.info("  [DRY] ... and %d more", len(records) - 5)
        return

    if output_path.exists() and not args.overwrite:
        raise SystemExit(f"{output_path} exists. Use --overwrite to replace.")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.release:
        # Keep each release self-contained + parallel-friendly: tee the run log into
        # the model's own dir (not a shared/scratch location), alongside its answers.
        fh = logging.FileHandler(output_path.parent / "run.log", encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
        logging.getLogger().addHandler(fh)

    # Log what actually goes on the wire, not just what was asked for, so a vendor
    # remap, folding `medium` into `high`, or an unverified passthrough is
    # visible in the run log before the run costs anything.
    logger.info(
        "   model → %s  max_tokens=%d  reasoning: %s",
        model,
        max_tokens,
        json.dumps(plan.sent),
    )
    if plan.sent.get("honored") == "unverified":
        logger.warning(
            "⚠️  %s: the reasoning parameter is forwarded but it is NOT established "
            "that the backend honors it. Do not report this as a controlled effort "
            "level without a separation check against an adjacent level.",
            model,
        )
    if plan.sent.get("vendor_note"):
        logger.warning("⚠️  %s: %s", model, plan.sent["vendor_note"])

    usage = UsageTracker()
    harness = AnswerHarness(
        guideline_ids,
        model=model,
        thinking_effort=thinking_effort,
        max_tokens=max_tokens,
        extra_model_settings=plan.model_settings,
        reasoning_sent=plan.sent,
        granularity=granularity,
        usage=usage,
        frozen_context=frozen_context,
        # The frozen-context release name (its dir), stamped into each record. For a
        # --release run the frozen path is guaranteed to sit in releases/frozen-context/
        # <name>/, so this is the clean release name; None outside the frozen arm.
        frozen_source=frozen_path.parent.name if frozen_context is not None else None,
        helper_model=args.helper_model,
    )
    results = asyncio.run(_run_all(harness, records, args.concurrency))

    with output_path.open("w", encoding="utf-8") as fout:
        for rec in results:
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")

    logger.info("✅ wrote %d answer records to %s", len(results), output_path)
    if harness.n_truncated:
        logger.warning(
            "⚠️  %d/%d answers hit the max_tokens ceiling. Raise --max-tokens and re-run: "
            "truncated answers make every downstream metric partly a measure of the "
            "ceiling rather than of the model.",
            harness.n_truncated,
            len(results),
        )
    usage.log_summary()
    usage.write(output_path.with_suffix(".usage.json"))

    if args.release:
        # answers.jsonl + answers.usage.json are already in the release dir (that is
        # output_path); add the MANIFEST so the release self-identifies + is blessed.
        write_answers_manifest(
            output_path.parent,
            frozen_path=frozen_path,
            model=model,
            thinking_effort=thinking_effort,
            max_tokens=max_tokens,
            granularity=granularity,
            n_questions=len(results),
            reasoning_sent=plan.sent,
            n_truncated=harness.n_truncated,
        )
        logger.info("📦 blessed answers release → %s", output_path.parent)


if __name__ == "__main__":
    main()
