#!/usr/bin/env python3
"""Claim Support Rate: does the cited quote actually SUPPORT the claim? Plus
Quote Fidelity, the read-out of why a claim fell short.

    uv run python eval/citation/claim_support.py \\
        --answers eval/result/answers/smoke_bp.jsonl \\
        --judge-model openai:gpt-4.1-mini --limit 5
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Literal, Optional, get_args

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# eval/citation/ → repo root is two up; the system package + common/ on the path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_EVAL_DIR / "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
load_dotenv(_REPO_ROOT / ".env", override=False)

from citation_locator import (  # noqa: E402
    DEFAULT_LOCATOR_MODEL,
    locate_citations,
)
from citation_locator import _version_for as _locator_version  # noqa: E402
from cache import cache_key, cache_read, cache_write  # noqa: E402
from model_endpoints import (  # noqa: E402
    build_agent_model,
    pacing_state,
    needs_custom_endpoint,
    limiter_for_model,
    resolve_route,
    round_seed,
    set_pacer,
)
from rate_limit import is_rate_limit_error  # noqa: E402
from scoring import assert_absent, fmt_pct, load_jsonl, write_report  # noqa: E402
from usage import UsageTracker, record_run  # noqa: E402
from segmenter import SEGMENTER_VERSION, attribute_citations  # noqa: E402

_PROMPT_PATH = Path(__file__).parent / "prompts" / "claim_support.yaml"
_SPEC = yaml.safe_load(_PROMPT_PATH.read_text(encoding="utf-8"))
PROMPT_VERSION = _SPEC["version"]


def _user_prompt(question: str, claim: str, quote: str) -> str:
    """The judge's user turn. QUESTION comes FIRST so the scope is read before the claim."""
    head = (
        f"QUESTION (the clinical question this answer was written for):\n{question}\n\n"
        if question.strip()
        else ""
    )
    return (
        f"{head}CLAIM:\n{claim}\n\n"
        f"QUOTE (verbatim from the cited guideline section):\n{quote}"
    )


_PROMPT_HASH = hashlib.sha256(
    json.dumps(
        {
            **{k: _SPEC[k] for k in ("version", "system", "schema_doc", "fields")},
            "user_turn": [
                _user_prompt("<Q>", "<CLAIM>", "<QUOTE>"),
                _user_prompt("", "<CLAIM>", "<QUOTE>"),
            ],
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()[:12]
DEFAULT_JUDGE_MODEL = "openai:gpt-4.1-mini"  # cheap; swap for the real judge later
_CACHE_DIR = _EVAL_DIR / "result" / ".judge_cache"

SUPPORT_LABELS = (
    "fully_supported",
    "partially_supported",
    "not_supported",
    "contradicted",
)
OVERSTATEMENT_AXIS = "overstatement"
UNSUPPORTED_ADDITION_AXIS = "unsupported_addition"
SHORTFALL_AXES = (OVERSTATEMENT_AXIS, UNSUPPORTED_ADDITION_AXIS)
OVERSTATEMENT_SUBTYPES = (
    "strength_inflation",  # absorbs the former dropped_hedge
    "scope_inflation",  # tightened: quote must state the narrower scope
    "causal_upgrade",  # tightened: quote must contain an association word
)
SHORTFALL_TYPES = (UNSUPPORTED_ADDITION_AXIS, *OVERSTATEMENT_SUBTYPES)
SHORTFALL_PRECEDENCE = (
    "causal_upgrade",
    "scope_inflation",
    "strength_inflation",
    UNSUPPORTED_ADDITION_AXIS,
)
assert set(SHORTFALL_PRECEDENCE) == set(SHORTFALL_TYPES), (
    "SHORTFALL_PRECEDENCE must rank exactly the SHORTFALL_TYPES vocabulary"
)


def shortfall_axis_of(shortfall_type: str | None) -> str | None:
    """Roll a v5 `shortfall_type` leaf up to its v4 axis. The three overstatement leaves
    map to `overstatement`; `unsupported_addition` is its own axis; null stays null. This
    is what keeps the v4 report vocabulary (axis_totals, the *_distribution fields,
    validate_judge) working unchanged after the merge."""
    if shortfall_type in OVERSTATEMENT_SUBTYPES:
        return OVERSTATEMENT_AXIS
    return shortfall_type if shortfall_type in SHORTFALL_AXES else None


def derive_axis_fields(verdict: dict) -> dict:
    """Add the DERIVED v4 fields (`shortfall_axis`, `overstatement_subtype`) to a v5
    verdict dict, in place. Every downstream consumer reads those two names, so deriving
    them here is what confines the v5 merge to this module."""
    st = verdict.get("shortfall_type")
    verdict["shortfall_axis"] = shortfall_axis_of(st)
    verdict["overstatement_subtype"] = st if st in OVERSTATEMENT_SUBTYPES else None
    return verdict


def axis_totals(axes: Counter) -> dict:
    """Per-axis shortfall totals, keyed `<axis>_total`, built FROM SHORTFALL_AXES so a renamed
    or added axis reaches the report automatically. Shared by both scorers (this module and
    claim_support_sentence.py) so the two cannot silently disagree on the axis vocabulary.
    """
    return {f"{ax}_total": axes.get(ax, 0) for ax in SHORTFALL_AXES}


class Verdict(BaseModel):
    # Descriptions and the schema docstring are injected from _SPEC (the yaml) so the
    # judge-facing rubric has one source. The Literal value sets are the STRUCTURE and
    # stay in code; they must match SUPPORT_LABELS / SHORTFALL_AXES / OVERSTATEMENT_SUBTYPES.
    support_reasoning: str = Field(description=_SPEC["fields"]["support_reasoning"])
    support_evidence: str = Field(description=_SPEC["fields"]["support_evidence"])
    support_verdict: Literal[
        "fully_supported",
        "partially_supported",
        "not_supported",
        "contradicted",
    ] = Field(description=_SPEC["fields"]["support_verdict"])
    shortfall_type: Optional[
        Literal[
            "unsupported_addition",
            "strength_inflation",
            "scope_inflation",
            "causal_upgrade",
        ]
    ] = Field(default=None, description=_SPEC["fields"]["shortfall_type"])
    is_numeric_claim: bool = Field(description=_SPEC["fields"]["is_numeric_claim"])


# pydantic sends a model's docstring as the schema's top-level description; set it
# from _SPEC so that too has its single source in the yaml.
Verdict.__doc__ = _SPEC["schema_doc"]


def _assert_literals_match() -> None:
    """Fail loudly at import if a Verdict Literal drifts from the module constants it is
    supposed to mirror. typing.Literal needs its members written inline, so the strings are
    necessarily duplicated; this is what keeps the duplication honest.
    """
    for field, expected in (
        ("support_verdict", SUPPORT_LABELS),
        ("shortfall_type", SHORTFALL_TYPES),
    ):
        ann = Verdict.model_fields[field].annotation
        got = {a for a in get_args(ann) if isinstance(a, str)}
        got |= {
            a
            for arg in get_args(ann)
            for a in get_args(arg)  # unwrap Optional[Literal[...]]
            if isinstance(a, str)
        }
        if got != set(expected):
            raise AssertionError(
                f"Verdict.{field} Literal {sorted(got)} != {sorted(expected)}"
            )


_assert_literals_match()

_SYSTEM = _SPEC["system"]


# ── caching ─────────────────────────────────────────────────────────────────────


def _cache_key(
    judge_model: str, question: str, claim: str, quote: str, round_index: int = 1
) -> str:
    """Cache key for one judged atom on one sampling round."""
    round_part = () if round_index == 1 else (f"r{round_index}",)
    return cache_key(
        judge_model, PROMPT_VERSION, _PROMPT_HASH, *round_part, question, claim, quote
    )


_DECODING_STAMP = _CACHE_DIR / ".decoding.json"
_decoding_checked = False


def _cache_is_populated() -> bool:
    """True if the directory already holds verdicts. The stamp file itself does not count."""
    return any(
        f.suffix == ".json" and f.name != _DECODING_STAMP.name
        for f in _CACHE_DIR.glob("*.json")
    )


def _write_stamp(settings: dict) -> None:
    """Write the stamp atomically, matching cache.cache_write."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = _DECODING_STAMP.with_name(f"{_DECODING_STAMP.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(settings, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, _DECODING_STAMP)


def assert_cache_matches_decoding() -> None:
    """Refuse to reuse a cache built under different judge decoding settings."""
    global _decoding_checked
    if _decoding_checked:
        return
    _decoding_checked = True
    current = judge_settings()

    if _DECODING_STAMP.exists():
        try:
            stamped = json.loads(_DECODING_STAMP.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 — unreadable, handled below
            stamped = None
        if stamped is None:
            # NOT "treat as absent and carry on": returning here would leave a corrupt stamp in
            # place and disable the guard for every future run, silently and permanently.
            raise SystemExit(
                f"{_DECODING_STAMP} is unreadable, so the settings behind the cached verdicts "
                f"cannot be established.\n"
                f"  Delete the cache directory and let it rebuild, or restore the stamp if you "
                f"know what produced those entries. Do not simply delete the stamp: that "
                f"re-certifies the existing verdicts under whatever settings run next."
            )
        if stamped != current:
            raise SystemExit(
                f"{_CACHE_DIR} holds verdicts produced under judge settings {stamped}, but this "
                f"run requests {current}. Serving those entries would mix two measurements.\n"
                f"  Pass --no-cache to compare settings without touching the cache, or move the "
                f"whole cache directory aside (it is regenerable, at full API cost).\n"
                f"  Do NOT delete {_DECODING_STAMP} on its own: that leaves every verdict in "
                f"place and re-stamps them under the new settings, which is the exact failure "
                f"this guard exists to prevent."
            )
        return

    if _cache_is_populated():
        raise SystemExit(
            f"{_CACHE_DIR} already holds cached verdicts but carries no {_DECODING_STAMP.name}, "
            f"so the settings that produced them are unknown.\n"
            f"  This run requests {current}. If you know the cache was built under exactly those "
            f"settings, write them to {_DECODING_STAMP} yourself to adopt it. Otherwise move the "
            f"directory aside and let it rebuild, or pass --no-cache."
        )
    _write_stamp(current)


def _cache_read(key: str) -> dict | None:
    assert_cache_matches_decoding()
    return cache_read(_CACHE_DIR, key)


def _cache_write(key: str, value: dict) -> None:
    assert_cache_matches_decoding()
    cache_write(_CACHE_DIR, key, value)


# ── judging ─────────────────────────────────────────────────────────────────────


JUDGE_TEMPERATURE = None
JUDGE_MAX_TOKENS = 4000
JUDGE_THINKING = "off"  # single source; build_judge_agent derives Thinking from this

JUDGE_OUTPUT_RETRIES = 3

METERED_JUDGE_RPM = 60.0


def judge_seed(judge_model: str, round_index: int) -> int | None:
    """Sampling seed for one voting round. Thin alias for `round_seed`, kept because the
    judge is where the requirement is most consequential and the name reads at the call
    site."""
    return round_seed(judge_model, round_index)


def set_judge_limiter(judge_model: str, rpm: float | None):
    """Install the process-wide pacer for this run, and return it."""
    if rpm is None:
        rpm = METERED_JUDGE_RPM if needs_custom_endpoint(judge_model) else 0.0
    return set_pacer(rpm)


def judge_pacing() -> dict | None:
    """Pacing disclosure for the report, or None when nothing was paced."""
    st = pacing_state()
    if st is None:
        return None
    # Stated together on purpose: `judge_vote_rounds: 3` beside an unseeded route
    # through a caching host would be a false claim, so the flag that makes it true
    # travels with the rate.
    st["seeded_rounds"] = True
    return st


JUDGE_USAGE = UsageTracker()


def reset_judge_usage() -> None:
    JUDGE_USAGE.reset()


def judge_settings() -> dict:
    """The judge decoding settings we REQUEST, for the report provenance block."""
    settings = {
        "temperature": (
            "provider-default" if JUDGE_TEMPERATURE is None else JUDGE_TEMPERATURE
        ),
        "max_tokens": JUDGE_MAX_TOKENS,
        "thinking": JUDGE_THINKING,
    }
    if JUDGE_THINKING != "off" and JUDGE_TEMPERATURE is not None:
        settings["temperature_honored"] = (
            "unverified: providers may ignore sampling params in thinking mode"
        )
    return settings


def build_judge_agent(
    judge_model: str, instructions: str, output_type: type, seed: int | None = None
):
    """Shared judge-agent builder for every metric's judge (claim-support, applicability, ...)."""
    from pydantic_ai import Agent
    from pydantic_ai.capabilities import Thinking
    from pydantic_ai.settings import ModelSettings

    effort = False if JUDGE_THINKING == "off" else JUDGE_THINKING
    settings = ModelSettings(max_tokens=JUDGE_MAX_TOKENS)
    if JUDGE_TEMPERATURE is not None:
        settings["temperature"] = JUDGE_TEMPERATURE
    extra: dict = {}
    # See METERED_JUDGE_RPM above: without a seed the voting rounds of a caching
    # host are one sample counted N times.
    if seed is not None:
        extra["seed"] = seed
    if extra:
        settings["extra_body"] = extra
    return Agent(
        build_agent_model(judge_model),
        instructions=instructions,
        capabilities=[Thinking(effort=effort)],
        model_settings=settings,
        output_type=output_type,
        retries=JUDGE_OUTPUT_RETRIES,
    )


_JUDGE_AGENTS: dict = {}


def _judge_agent(judge_model: str, seed: int | None = None):
    agent = _JUDGE_AGENTS.get((judge_model, seed))
    if agent is None:
        agent = build_judge_agent(judge_model, _SYSTEM, Verdict, seed=seed)
        _JUDGE_AGENTS[(judge_model, seed)] = agent
    return agent


JUDGE_VOTE_ROUNDS = 3
JUDGE_VOTE_MAX_ROUNDS = 5


def _plurality(labels: list[str]) -> str | None:
    """The label that appears strictly more often than every other, or None."""
    if not labels:
        return None
    counts = Counter(labels)
    (top, n), *rest = counts.most_common()
    return top if not rest or rest[0][1] < n else None


def _fold_winning_bloc(drawn: list[dict], winner: str) -> dict:
    """Build the atom's verdict from the rounds that voted with the majority."""
    bloc = [v for v in drawn if v.get("support_verdict") == winner]
    out = dict(bloc[0])
    if winner == "fully_supported":
        out["shortfall_type"] = None
    else:
        # A null is not an answer to "how does this claim fall short" once the bloc has
        # settled that it does, so it does not get a vote. Ties are broken by the rubric's
        # own precedence rather than by round order.
        named = [v.get("shortfall_type") for v in bloc if v.get("shortfall_type")]
        counts = Counter(named)
        top = max(counts.values(), default=0)
        out["shortfall_type"] = next(
            (t for t in SHORTFALL_PRECEDENCE if counts.get(t, 0) == top and top),
            None,
        )
    derive_axis_fields(out)
    out["judge_bloc_size"] = len(bloc)
    out["judge_typing_split"] = len({v.get("overstatement_subtype") for v in bloc}) > 1
    return out


async def judge_atom_voted(
    question: str, claim: str, quote: str, judge_model: str, use_cache: bool
) -> dict:
    """Judge one atom by majority over sampled rounds."""
    # Deterministic and free: `judge_atom` returns this before any API call, so drawing it
    # five times would be five identical answers at no cost and no information.
    if not claim.strip() or not quote.strip():
        return {"support_verdict": "skipped_empty"}

    # Rounds 1..3 concurrently: they are independent draws, and running them in sequence
    # would triple wall-clock for no benefit. Rounds 4 and 5 are conditional, so they are
    # only drawn one at a time, and only for the atoms that need them.
    first = await asyncio.gather(
        *(
            judge_atom(question, claim, quote, judge_model, use_cache, round_index=r)
            for r in range(1, JUDGE_VOTE_ROUNDS + 1)
        )
    )
    drawn = list(first)
    labels = [v["support_verdict"] for v in drawn]
    first3_labels = [x for x in labels if x != "not_parsable"]

    winner = _plurality([x for x in labels if x != "not_parsable"])
    r = JUDGE_VOTE_ROUNDS
    while winner is None and r < JUDGE_VOTE_MAX_ROUNDS:
        r += 1
        v = await judge_atom(
            question, claim, quote, judge_model, use_cache, round_index=r
        )
        drawn.append(v)
        labels.append(v["support_verdict"])
        winner = _plurality([x for x in labels if x != "not_parsable"])

    valid = [x for x in labels if x != "not_parsable"]
    if not valid:
        out = dict(drawn[0])
    elif winner is None:
        # Every round drawn and still no plurality. Take the most conservative label
        # present rather than an arbitrary one, and flag it so the count is disclosable.
        order = [
            "contradicted",
            "not_supported",
            "partially_supported",
            "fully_supported",
        ]
        out = dict(
            next(
                v for v in drawn if v["support_verdict"] == min(valid, key=order.index)
            )
        )
        out["judge_no_majority"] = True
    else:
        out = _fold_winning_bloc(drawn, winner)

    out["judge_votes"] = labels
    # Every round's shortfall typing, in round order, so a future change of aggregation is a
    # re-aggregation rather than a re-run. Cheap to keep and the reason this file had to be
    # re-scored once already: the report kept only the winner's typing.
    out["judge_vote_types"] = [
        (v.get("shortfall_type"), v.get("overstatement_subtype")) for v in drawn
    ]
    out["judge_rounds_used"] = len(labels)
    # Unanimity over the fixed first three, None when fewer than three survived.
    out["judge_first3_unanimous"] = (
        len(set(first3_labels)) == 1
        if len(first3_labels) == JUDGE_VOTE_ROUNDS
        else None
    )
    return out


async def judge_atom(
    question: str,
    claim: str,
    quote: str,
    judge_model: str,
    use_cache: bool,
    round_index: int = 1,
) -> dict:
    """Judge one (claim, quote) atom on one sampling round → a Verdict dict. Empty
    claim/quote and judge failures get sentinel verdicts (`skipped_empty` /
    `not_parsable`) so they never silently corrupt the denominator."""
    if not claim.strip() or not quote.strip():
        return {"support_verdict": "skipped_empty"}

    question = (question or "").strip()

    key = _cache_key(judge_model, question, claim, quote, round_index)
    if use_cache:
        cached = _cache_read(key)
        if cached is not None:
            JUDGE_USAGE.record_cache_hit()
            return derive_axis_fields(cached)

    agent = _judge_agent(judge_model, seed=judge_seed(judge_model, round_index))
    # Pacing sits BELOW the cache check on purpose: a cache hit puts nothing on the wire, so
    # pacing it would spend the run's rate budget on calls that never happen and make a
    # warm re-run as slow as a cold one. None unless THIS model is metered.
    limiter = limiter_for_model(judge_model)
    if limiter is not None:
        await limiter.acquire()
    try:
        result = await agent.run(_user_prompt(question, claim, quote))
        # Recorded on the real-call path only. A cache hit spends nothing, so counting it
        # would invent tokens and turn the cost field into fiction.
        record_run(JUDGE_USAGE, judge_model, result)
        verdict = derive_axis_fields(result.output.model_dump())
    except Exception as e:  # noqa: BLE001 — isolate per-atom judge failures
        # A 429 has to reach the limiter, or the next caller departs on the pre-throttle
        # schedule at the moment the endpoint is already rejecting.
        if limiter is not None and is_rate_limit_error(e):
            limiter.penalise()
        verdict = {"support_verdict": "not_parsable", "error": str(e)[:200]}

    if use_cache and verdict.get("support_verdict") != "not_parsable":
        _cache_write(key, verdict)
    return verdict


async def _atoms_for(
    record: dict, localizer: str, localizer_model: str, use_cache: bool
):
    """Pair each citation with the claim it backs, via the chosen localiser:
    'llm-verbatim' (extractive — verbatim span, DEFAULT), 'llm' (abstractive —
    rewrite to stand alone), or 'rules' (positional marker→sentence attribution)."""
    answer = record.get("answer", "")
    if localizer == "rules":
        return attribute_citations(answer)
    mode = "extractive" if localizer == "llm-verbatim" else "abstractive"
    return await locate_citations(answer, localizer_model, use_cache, mode=mode)


async def score_answers(
    records: list[dict],
    judge_model: str,
    *,
    localizer: str,
    localizer_model: str,
    concurrency: int,
    use_cache: bool,
) -> dict:
    """Judge every (claim, quote) atom across all records, then aggregate."""
    # Flatten to atoms, remembering which record each belongs to.
    atom_lists = await asyncio.gather(
        *(_atoms_for(r, localizer, localizer_model, use_cache) for r in records)
    )
    atoms: list[tuple[int, str, str, str]] = []  # (record_idx, question, claim, quote)
    for ri, alist in enumerate(atom_lists):
        for a in alist:
            atoms.append((ri, records[ri].get("question") or "", a.claim, a.quote))

    sem = asyncio.Semaphore(concurrency)

    async def _run(question: str, claim: str, quote: str) -> dict:
        async with sem:
            return await judge_atom(question, claim, quote, judge_model, use_cache)

    verdicts = await asyncio.gather(*(_run(q, c, qt) for _, q, c, qt in atoms))

    # Per-record + global tallies.
    per_record_atoms: dict[int, list[dict]] = {}
    for (ri, _question, claim, quote), v in zip(atoms, verdicts):
        per_record_atoms.setdefault(ri, []).append(
            {"claim": claim, "quote": quote, **v}
        )

    labels = Counter(v.get("support_verdict") for v in verdicts)
    axes = Counter(v.get("shortfall_axis") for v in verdicts if v.get("shortfall_axis"))
    subtypes = Counter(
        v.get("overstatement_subtype")
        for v in verdicts
        if v.get("overstatement_subtype")
    )

    def _rates(counter: Counter) -> dict:
        fully = counter.get("fully_supported", 0)
        partial = counter.get("partially_supported", 0)
        nots = counter.get("not_supported", 0)
        contra = counter.get("contradicted", 0)
        denom = (
            fully + partial + nots + contra
        )  # the 4 real verdicts; sentinels excluded
        return {
            "n_scored": denom,
            "support_rate_strict": (fully / denom) if denom else None,
            # Lenient counts a partial as a pass (no fractional credit) — the
            # strict/lenient gap then reads directly as the share of partials,
            # which is why both are reported side by side.
            "support_rate_lenient": ((fully + partial) / denom) if denom else None,
            "contradiction_rate": (contra / denom) if denom else None,
        }

    # Rates read only the 4 verdict keys; the shortfall axis/subtype are reported
    # separately (Quote Fidelity = the overstatement-subtype distribution).
    global_rates = _rates(labels)

    numeric = [v for v in verdicts if v.get("is_numeric_claim")]
    numeric_labels = Counter(v.get("support_verdict") for v in numeric)
    numeric_rates = _rates(numeric_labels)

    per_record = []
    for ri, r in enumerate(records):
        ratoms = per_record_atoms.get(ri, [])
        rlabels = Counter(a.get("support_verdict") for a in ratoms)
        per_record.append(
            {
                "question_id": r.get("question_id"),
                "n_atoms": len(ratoms),
                "labels": dict(rlabels),
                "support_rate_strict": _rates(rlabels)["support_rate_strict"],
                "atoms": ratoms,
            }
        )

    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": _PROMPT_HASH,
        "judge_model": judge_model,
        "judge_settings": judge_settings(),
        "judge_output_retries": JUDGE_OUTPUT_RETRIES,
        "localizer": localizer,
        "localizer_version": SEGMENTER_VERSION
        if localizer == "rules"
        else _locator_version(
            "extractive" if localizer == "llm-verbatim" else "abstractive"
        ),
        "localizer_model": None if localizer == "rules" else localizer_model,
        "n_answers": len(records),
        "n_atoms": len(atoms),
        **global_rates,
        **axis_totals(axes),
        "label_distribution": dict(labels),
        "shortfall_axis_distribution": dict(axes),
        "overstatement_subtype_distribution": dict(subtypes),
        "numeric": {"n": len(numeric), **numeric_rates},
        "per_record": per_record,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Claim Support Rate + Quote Fidelity (LLM judge) over a harness answer JSONL."
    )
    p.add_argument("--answers", required=True, help="Harness answers.jsonl path.")
    p.add_argument("--report", help="Optional path to write the full JSON report.")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --report to replace an existing file. Without it an existing "
        "report is left alone and the run refuses to start (archive, do not overwrite).",
    )
    p.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help=f"provider:model judge (default {DEFAULT_JUDGE_MODEL}).",
    )
    p.add_argument(
        "--localizer",
        choices=["llm", "llm-verbatim", "rules"],
        default="llm-verbatim",
        help="Claim↔citation localiser: 'llm-verbatim' (extractive verbatim span, "
        "DEFAULT), 'llm' (abstractive rewrite), or 'rules' (positional marker→sentence).",
    )
    p.add_argument(
        "--localizer-model",
        default=DEFAULT_LOCATOR_MODEL,
        help=f"provider:model for the LLM localiser (default {DEFAULT_LOCATOR_MODEL}).",
    )
    p.add_argument("--limit", type=int, help="Cap on answer records (smoke test).")
    p.add_argument(
        "--concurrency", type=int, default=8, help="Parallel judge calls (default 8)."
    )
    p.add_argument(
        "--rpm",
        type=float,
        default=None,
        help="Judge calls per minute. Only meaningful behind a metered host, where "
        f"it defaults to {METERED_JUDGE_RPM:g}; a natively routed judge is unpaced. "
        "0 disables it.",
    )
    p.add_argument(
        "--no-cache", action="store_true", help="Ignore the SHA-256 disk cache."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Before any API call: a guard that fires after the run has already paid for it.
    assert_absent(args.report, args.overwrite)
    # Early for the same reason as in claim_support_sentence.py: raised from inside
    # asyncio.gather this surfaces as a CancelledError traceback instead of a message.
    if not args.no_cache:
        assert_cache_matches_decoding()
    args.judge_model = resolve_route(args.judge_model)
    set_judge_limiter(args.judge_model, args.rpm)
    answers_path = Path(args.answers)
    if not answers_path.exists():
        raise SystemExit(f"{answers_path} not found")

    records = load_jsonl(answers_path)
    if args.limit:
        records = records[: args.limit]

    result = asyncio.run(
        score_answers(
            records,
            args.judge_model,
            localizer=args.localizer,
            localizer_model=args.localizer_model,
            concurrency=args.concurrency,
            use_cache=not args.no_cache,
        )
    )

    print(
        f"Claim Support Rate — judge {result['judge_model']}  prompt {result['prompt_version']}"
    )
    print(
        f"  localizer={result['localizer']} ({result['localizer_version']}"
        + (f", {result['localizer_model']}" if result["localizer_model"] else "")
        + ")"
    )
    print(
        f"  answers={result['n_answers']}  atoms={result['n_atoms']}  scored={result['n_scored']}"
    )
    print(f"  Claim Support Rate (strict)  = {fmt_pct(result['support_rate_strict'])}")
    print(f"  Claim Support Rate (lenient) = {fmt_pct(result['support_rate_lenient'])}")
    print(f"  Contradiction (safety) rate  = {fmt_pct(result['contradiction_rate'])}")
    print("  support labels:")
    for label in (*SUPPORT_LABELS, "skipped_empty", "not_parsable"):
        n = result["label_distribution"].get(label, 0)
        if n:
            print(f"    {label:20} {n:4}")
    print(
        f"  shortfall axis: overstatement {result['overstatement_total']}  "
        f"unsupported_addition {result['unsupported_addition_total']}"
    )
    print(
        f"  Quote Fidelity — overstatement subtypes (total {result['overstatement_total']}):"
    )
    for sub in OVERSTATEMENT_SUBTYPES:
        n = result["overstatement_subtype_distribution"].get(sub, 0)
        if n:
            print(f"    {sub:20} {n:4}")
    num = result["numeric"]
    if num["n"]:
        print(
            f"  numeric claims: n={num['n']}  support(strict)={fmt_pct(num['support_rate_strict'])}"
        )

    if args.report:
        write_report(args.report, result, args.overwrite)
        print(f"  full report → {args.report}")


if __name__ == "__main__":
    main()
