#!/usr/bin/env python3
"""Citation Coverage — what share of the answer's CLAIM sentences
are covered by a citation?

    uv run python eval/citation/coverage.py \\
        --answers eval/result/answers/smoke_bp.jsonl
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import fcntl
import hashlib
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_EVAL_DIR / "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
load_dotenv(_REPO_ROOT / ".env", override=False)

from cache import cache_key, cache_read, cache_write  # noqa: E402
from model_endpoints import (  # noqa: E402
    CUSTOM_ENDPOINT_PREFIXES,
    build_agent_model,
    needs_custom_endpoint,
    limiter_for_model,
    resolve_route,
    round_seed,
    thinking_off_kwargs,
)
from rate_limit import is_rate_limit_error  # noqa: E402
from release_meta import git_sha, rel_to_repo  # noqa: E402
from scoring import assert_absent, fmt_pct, load_jsonl, write_report  # noqa: E402
from usage import UsageTracker, record_run  # noqa: E402
from segmenter import (  # noqa: E402
    CITE_RE,
    CLAIM_VALIDITY_VERSION,
    SEGMENTER_VERSION,
    _normalize_markdown,
    segment_marker_map_with_validity,
)
from vccr import Tiers  # noqa: E402  the verbatim gate's tier check

from system.helper_agent import helper_agent  # noqa: E402

# Sentences shorter than this are treated as fragments, not claim sentences.
MIN_SENTENCE_CHARS = 15

DEFAULT_WINDOW = 1
BRACKET_WINDOWS = (0, 2, 3, 4, 5)

_FILTER_PROMPT_PATH = Path(__file__).parent / "prompts" / "claim_filter.yaml"
_FILTER_SPEC = yaml.safe_load(_FILTER_PROMPT_PATH.read_text(encoding="utf-8"))
CLAIM_FILTER_VERSION = _FILTER_SPEC["version"]
_FILTER_SYSTEM = _FILTER_SPEC["system"]
_FILTER_PROMPT_HASH = hashlib.sha256(_FILTER_SYSTEM.encode("utf-8")).hexdigest()[:12]
DEFAULT_FILTER_MODEL = "openai-responses:gpt-5.4-mini"
_FILTER_CACHE_DIR = _EVAL_DIR / "result" / ".coverage_filter_cache"


class _ClaimFlags(BaseModel):
    # Keep this description operational only. The claim definition lives ONCE in
    # prompts/claim_filter.yaml; restating it here risks two drifting sources of truth.
    claim_numbers: list[int] = Field(
        description="The 1-based numbers of the sentences that ARE claims, "
        "per the definition in the instructions."
    )


FILTER_SETTINGS = {
    "temperature": "provider-default",
    "max_tokens": 500,
    "thinking": "off",
}

CLAIM_FILTER_ROUNDS = 3
_ROUND_RETRIES = 2


def _majority_threshold(surviving: int) -> int:
    """Votes needed for a sentence to count as a claim, given how many rounds survived."""
    return surviving // 2 + 1


FILTER_STATS: dict[str, Any] = {
    # rounds actually voted over -> how many records. A histogram rather than a number, so an
    # inhomogeneous run is visible instead of being flattened to its last record.
    "rounds_hist": {},
    "unanimous": 0,
    "split": 0,
    "degraded_records": 0,
    "failed_records": 0,
}


# Filter spend, accumulated across a run and surfaced in the report. Same module-level
# rationale as FILTER_STATS, and reset alongside it.
FILTER_USAGE = UsageTracker()


def _reset_filter_stats() -> None:
    """Called once per scoring run so one process scoring several models cannot pool them."""
    FILTER_STATS.update(
        rounds_hist={}, unanimous=0, split=0, degraded_records=0, failed_records=0
    )
    FILTER_USAGE.reset()


_FILTER_AGENTS: dict = {}


def _filter_agent(model: str, seed: int | None):
    agent = _FILTER_AGENTS.get((model, seed))
    if agent is not None:
        return agent

    thinking_off = FILTER_SETTINGS["thinking"] == "off"
    effort = False if thinking_off else FILTER_SETTINGS["thinking"]
    if not needs_custom_endpoint(model):
        agent = helper_agent(
            model,
            instructions=_FILTER_SYSTEM,
            max_tokens=FILTER_SETTINGS["max_tokens"],
            output_type=_ClaimFlags,
            effort=effort,
        )
    else:
        from pydantic_ai import Agent
        from pydantic_ai.capabilities import Thinking
        from pydantic_ai.settings import ModelSettings

        extra: dict = {}
        if thinking_off:
            extra.update(thinking_off_kwargs(model).get("extra_body", {}))
        if seed is not None:
            extra["seed"] = seed
        settings = ModelSettings(max_tokens=FILTER_SETTINGS["max_tokens"])
        if extra:
            settings["extra_body"] = extra
        agent = Agent(
            build_agent_model(model),
            instructions=_FILTER_SYSTEM,
            capabilities=[Thinking(effort=effort)],
            model_settings=settings,
            output_type=_ClaimFlags,
        )
    _FILTER_AGENTS[(model, seed)] = agent
    return agent


# How long a caller will wait for whoever else is computing the same filter round, and how
# often it re-checks. Generous because the thing being waited for is one API call, and the
# alternative to waiting is duplicating it.
_FILTER_LOCK_TIMEOUT_S = 180.0
_FILTER_LOCK_POLL_S = 0.4


def _try_lock(key: str):
    """Take the inter-process lock for one filter round, or return None if another process
    holds it. Never blocks. The caller MUST pass the handle to `_release`.
    """
    path = _FILTER_CACHE_DIR / f"{key}.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    fh = path.open("w")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return None
    return fh


def _release(fh) -> None:
    if fh is None:
        return
    with contextlib.suppress(OSError):
        fcntl.flock(fh, fcntl.LOCK_UN)
    fh.close()


async def _claim_round(key: str):
    """Wait until either the value appears in the cache or this caller may compute it."""
    deadline = asyncio.get_running_loop().time() + _FILTER_LOCK_TIMEOUT_S
    while True:
        fh = _try_lock(key)
        if fh is not None:
            # Re-read UNDER the lock: the holder we were queued behind may have written the
            # value while we waited, and recomputing it is the duplication this prevents.
            cached = cache_read(_FILTER_CACHE_DIR, key)
            if cached is not None:
                _release(fh)
                return None, cached
            return fh, None
        cached = cache_read(_FILTER_CACHE_DIR, key)
        if cached is not None:
            return None, cached
        if asyncio.get_running_loop().time() >= deadline:
            # Do not fail a run over a lock. Proceeding unlocked costs one duplicated call
            # in a rare case; refusing would lose the round and change the vote rule.
            return None, None
        await asyncio.sleep(_FILTER_LOCK_POLL_S)


async def _classify_once(
    sentences: list[str], model: str, use_cache: bool, round_index: int
) -> list[bool] | None:
    """One sampling round, retried on error. None once the retries are exhausted."""
    key = cache_key(
        model, CLAIM_FILTER_VERSION, _FILTER_PROMPT_HASH, f"r{round_index}", *sentences
    )

    def _from_cached(cached: list[int]) -> list[bool]:
        FILTER_USAGE.record_cache_hit()
        keep = set(cached)
        return [(i in keep) for i in range(1, len(sentences) + 1)]

    lock_fh = None
    if use_cache:
        cached = cache_read(_FILTER_CACHE_DIR, key)
        if cached is not None:
            return _from_cached(cached)
        # Cold. Before spending the call, find out whether another PROCESS is already
        # spending it: see _claim_round for the race and what it cost. The lock, when we get
        # one, is held until after cache_write in the finally below.
        lock_fh, cached = await _claim_round(key)
        if cached is not None:
            return _from_cached(cached)

    numbered = "\n".join(f"{i}. {s}" for i, s in enumerate(sentences, 1))
    agent = _filter_agent(model, round_seed(model, round_index))
    # Pacing sits BELOW the cache check: a hit puts nothing on the wire, so pacing it would
    # spend the rate budget on calls that never happen and make a warm re-run as slow as a
    # cold one. None for a direct-API filter, which is how every scored model so far ran.
    limiter = limiter_for_model(model)
    try:
        for _ in range(_ROUND_RETRIES + 1):
            if limiter is not None:
                await limiter.acquire()
            try:
                result = await agent.run(numbered)
            except Exception as e:  # noqa: BLE001 — retry; a lost round changes the vote rule
                # A 429 must reach the pacer, or the retry departs on the pre-throttle
                # schedule at the moment the endpoint is already rejecting.
                if limiter is not None and is_rate_limit_error(e):
                    limiter.penalise()
                continue
            # Recorded on the real-call path only, and BEFORE the parse: a retried round
            # still spent its tokens, so charging only the successful attempt would
            # understate the run.
            record_run(FILTER_USAGE, model, result)
            claim_nums = [
                n for n in result.output.claim_numbers if 1 <= n <= len(sentences)
            ]
            if use_cache:
                cache_write(_FILTER_CACHE_DIR, key, sorted(set(claim_nums)))
            keep = set(claim_nums)
            return [(i in keep) for i in range(1, len(sentences) + 1)]
        return None
    finally:
        _release(lock_fh)


async def _classify_claims(
    sentences: list[str], model: str, use_cache: bool, rounds: int | None = None
) -> list[bool]:
    """Label each sentence claim (True) / not-claim (False), by strict majority over `rounds`
    independent samples. On total failure or empty input, degrade to keeping every sentence so
    the denominator never shrinks silently.
    """
    if not sentences:
        return []
    rounds = CLAIM_FILTER_ROUNDS if rounds is None else rounds

    # Gathered, not awaited in sequence: the rounds are independent, and running them serially
    # would triple this stage's wall clock for no reason.
    results = await asyncio.gather(
        *(_classify_once(sentences, model, use_cache, i) for i in range(rounds))
    )
    per_round = [r for r in results if r is not None]

    hist = FILTER_STATS["rounds_hist"]
    hist[str(len(per_round))] = hist.get(str(len(per_round)), 0) + 1
    if not per_round:  # every round failed, after retries
        FILTER_STATS["failed_records"] += 1
        return [True] * len(sentences)
    if len(per_round) < rounds:
        FILTER_STATS["degraded_records"] += 1

    need = _majority_threshold(len(per_round))
    votes = [sum(rd[i] for rd in per_round) for i in range(len(sentences))]
    for v in votes:
        FILTER_STATS["unanimous" if v in (0, len(per_round)) else "split"] += 1
    return [v >= need for v in votes]


def _assert_filter_model_samples_independently(model: str) -> None:
    """Refuse a filter whose voting rounds would not be independent draws."""
    if not (
        needs_custom_endpoint(model)
        or model.lower().startswith(CUSTOM_ENDPOINT_PREFIXES)
    ):
        return
    # range(CLAIM_FILTER_ROUNDS), matching _classify_claims: the filter's rounds are
    # 0-based while the judge's are 1-based. Checking 1..N here would verify seeds the
    # run never uses, which is the same class of defect as not checking at all.
    seeds = [round_seed(model, r) for r in range(CLAIM_FILTER_ROUNDS)]
    if None in seeds or len(set(seeds)) != len(seeds):
        raise SystemExit(
            f"{model}: refusing to use this model as the claim filter. Its "
            f"{CLAIM_FILTER_ROUNDS} voting rounds would send bodies its host treats "
            f"as identical (seeds {seeds}), and such a host replays a stored response, "
            "so the vote would be one sample reported as three. Either give round_seed "
            "a case for this route, or use a natively routed model."
        )


def answer_markers(answer: str, n_expected: int) -> list[dict]:
    """The `{{cite:doc_id|quote}}` payloads of an answer, in document order, index-aligned
    with `segment_marker_map_with_validity`'s marker list.
    """
    markers = [
        {"doc_id": m.group(1).strip(), "quote": m.group(2)}
        for m in CITE_RE.finditer(_normalize_markdown(answer or ""))
    ]
    if len(markers) != n_expected:
        raise SystemExit(
            f"marker scan disagreement: {len(markers)} markers in the normalized text "
            f"against {n_expected} from the segmenter. Marker indices would address "
            f"different citations in the two scans, so the verbatim gate is unsafe here."
        )
    return markers


def window_pairs(
    claim_flags: list[bool], marker_sents: list[int | None], window: int
) -> dict[int, list[int]]:
    """{claim sentence idx: [marker idx, ...]} under a territorial window of `window`
    claims. Same territory rule as `claim_support_sentence.pair_claims` (a marker group
    at sentence i with the previous group at p reaches only (p, i]); window=0 credits
    only the marker's own attributed sentence.
    """
    groups: list[tuple[int, list[int]]] = []
    for m, si in enumerate(marker_sents):
        if si is None:
            continue
        if groups and groups[-1][0] == si:
            groups[-1][1].append(m)
        else:
            groups.append((si, [m]))

    pairs: dict[int, list[int]] = {}
    if window == 0:
        for si, ms in groups:
            if claim_flags[si]:
                pairs.setdefault(si, []).extend(ms)
        return pairs
    prev = -1
    for si, ms in groups:
        territory = [j for j in range(prev + 1, si + 1) if claim_flags[j]]
        for j in territory[-window:]:
            pairs.setdefault(j, []).extend(ms)
        prev = max(prev, si)
    return pairs


def window_credited(
    claim_flags: list[bool], marker_sents: list[int | None], window: int
) -> set[int]:
    """Claim-sentence indices covered by >= 1 marker. A thin view over `window_pairs`,
    so the gated and ungated counts can never come from two different traversals."""
    return set(window_pairs(claim_flags, marker_sents, window))


async def citation_coverage(
    record: dict,
    model: str,
    use_cache: bool,
    claim_filter: str,
    window: int,
    tiers: Tiers | None = None,
) -> dict:
    texts, marker_sents, malformed = segment_marker_map_with_validity(
        record.get("answer", "")
    )
    cand_idx = [i for i, s in enumerate(texts) if len(s) >= MIN_SENTENCE_CHARS]

    if claim_filter == "llm" and cand_idx:
        flags = await _classify_claims([texts[i] for i in cand_idx], model, use_cache)
    else:
        flags = [True] * len(cand_idx)
    claim_flags = [False] * len(texts)
    for i, keep in zip(cand_idx, flags):
        claim_flags[i] = keep

    n_malformed_claims = sum(
        1 for i in range(len(texts)) if malformed[i] and claim_flags[i]
    )
    for i in range(len(texts)):
        if malformed[i]:
            claim_flags[i] = False

    n = sum(claim_flags)
    pairs = {
        k: window_pairs(claim_flags, marker_sents, k)
        for k in {window, *BRACKET_WINDOWS}
    }
    covered = {k: len(p) for k, p in pairs.items()}

    compliant_cites = None
    covered_ok: dict[int, int] | None = None
    if tiers is not None:
        markers = answer_markers(record.get("answer", ""), len(marker_sents))
        allowed = {
            m
            for m, mk in enumerate(markers)
            if tiers.passes(tiers.of(mk["doc_id"], mk["quote"]))
        }
        compliant_cites = len(allowed)
        covered_ok = {
            k: sum(1 for ms in p.values() if any(m in allowed for m in ms))
            for k, p in pairs.items()
        }
    return {
        "question_id": record.get("question_id"),
        "n_sentences": len(texts),
        "n_candidate_sentences": len(cand_idx),  # over the length floor
        "n_claim_sentences": n,  # denominator (after the claim filter AND the gate)
        # len(cand_idx) - n would fold the two exclusions together; they are separate
        # instruments and are reported separately, as claim support reports them.
        "n_dropped_non_claim": len(cand_idx) - n - n_malformed_claims,
        "n_malformed_claims": n_malformed_claims,
        "window": window,
        "n_covered_sentences": covered[window],
        "coverage": (covered[window] / n) if n else None,
        # Same shape as the ungated fields, and None (not 0) when the gate did not run, so
        # "not measured" never reads as "nothing was compliant".
        "n_citations": len(marker_sents),
        "n_compliant_citations": compliant_cites,
        "n_covered_compliant": covered_ok[window] if covered_ok else None,
        "coverage_compliant": (
            (covered_ok[window] / n) if (covered_ok and n) else None
        ),
        "bracket": {
            f"k{k}": {
                "n_covered": covered[k],
                "coverage": (covered[k] / n) if n else None,
                "n_covered_compliant": covered_ok[k] if covered_ok else None,
                "coverage_compliant": (
                    (covered_ok[k] / n) if (covered_ok and n) else None
                ),
            }
            for k in BRACKET_WINDOWS
        },
    }


async def score_answers(
    records: list[dict],
    model: str,
    use_cache: bool,
    claim_filter: str,
    window: int = DEFAULT_WINDOW,
    verbatim_gate: bool = True,
) -> dict:
    # ONE Tiers for the whole run, so its section cache is shared across answers rather
    # than re-reading and re-normalizing the same section once per answer.
    tiers = Tiers() if verbatim_gate else None
    rows = await asyncio.gather(
        *(
            citation_coverage(r, model, use_cache, claim_filter, window, tiers)
            for r in records
        )
    )
    scored = [r for r in rows if r["coverage"] is not None]
    empty = [r for r in rows if r["coverage"] is None]
    tot_claim = sum(r["n_claim_sentences"] for r in scored)

    def _agg(covered_of) -> dict:
        mean = (
            sum(covered_of(r) / r["n_claim_sentences"] for r in scored) / len(scored)
            if scored
            else None
        )
        pooled = sum(covered_of(r) for r in scored) / tot_claim if tot_claim else None
        return {"mean_coverage": mean, "pooled_coverage": pooled}

    return {
        "verbatim_criteria": tiers.cr.version if tiers is not None else None,
        "segmenter_version": SEGMENTER_VERSION,
        "claim_filter": claim_filter,
        "claim_filter_version": CLAIM_FILTER_VERSION if claim_filter == "llm" else None,
        "claim_filter_prompt_hash": (
            _FILTER_PROMPT_HASH if claim_filter == "llm" else None
        ),
        "claim_filter_model": model if claim_filter == "llm" else None,
        "usage": FILTER_USAGE.as_dict(),
        "claim_filter_vote": (
            {
                "rounds_requested": CLAIM_FILTER_ROUNDS,
                "rounds_hist": dict(sorted(FILTER_STATS["rounds_hist"].items())),
                "degraded_records": FILTER_STATS["degraded_records"],
                "failed_records": FILTER_STATS["failed_records"],
                "unanimous": FILTER_STATS["unanimous"],
                "split": FILTER_STATS["split"],
                "split_rate": round(
                    FILTER_STATS["split"]
                    / max(1, FILTER_STATS["unanimous"] + FILTER_STATS["split"]),
                    4,
                ),
            }
            if claim_filter == "llm"
            else None
        ),
        "weighting": "unweighted (all claim sentences = 1)",
        "min_sentence_chars": MIN_SENTENCE_CHARS,
        "window": window,
        "n_answers": len(records),
        "n_scored": len(scored),
        "n_dropped_non_claim": sum(r["n_dropped_non_claim"] for r in scored),
        "claim_validity_version": CLAIM_VALIDITY_VERSION,
        "n_malformed_claims_excluded": sum(r["n_malformed_claims"] for r in rows),
        # headline: territorial window of `window` claims per citation
        **_agg(lambda r: r["n_covered_sentences"]),
        # manual-CI bracket, always reported next to the headline
        "bracket": {
            f"k{k}": _agg(lambda r, k=k: r["bracket"][f"k{k}"]["n_covered"])
            for k in BRACKET_WINDOWS
        },
        # The same headline and ladder over the citations that pass the verbatim check.
        # Keyed apart rather than folded in, so a reader of the report can never take a
        # gated number for an ungated one.
        "verbatim_gate": verbatim_gate,
        "n_citations": sum(r["n_citations"] for r in scored),
        "n_compliant_citations": (
            sum(r["n_compliant_citations"] for r in scored) if verbatim_gate else None
        ),
        "compliant": (
            {
                **_agg(lambda r: r["n_covered_compliant"]),
                "bracket": {
                    f"k{k}": _agg(
                        lambda r, k=k: r["bracket"][f"k{k}"]["n_covered_compliant"]
                    )
                    for k in BRACKET_WINDOWS
                },
            }
            if verbatim_gate
            else None
        ),
        # Answers dropped from the mean because they have zero standalone claims
        # (all claim content is inside markers). Surfaced so the headline mean,
        # computed over n_scored not n_answers, is not read as a full-set figure.
        "diagnostics": {
            "n_empty_claim_answers": len(empty),
            "empty_claim_share": (len(empty) / len(records)) if records else None,
            "empty_claim_question_ids": [r["question_id"] for r in empty],
        },
        "per_record": rows,
    }


def build_meta(
    answers_path: Path, judge_model: str | None, metric: str = "citation_coverage"
) -> dict:
    """Provenance block so a leaf report.json is self-describing (matches the report
    convention). Records the parent answers release (read best-effort from the sibling
    MANIFEST.yaml), the judge model (the claim filter, or None when the filter is
    deterministic), the git sha and run time.
    """
    release = None
    manifest = answers_path.parent / "MANIFEST.yaml"
    if manifest.exists():
        try:
            release = (yaml.safe_load(manifest.read_text()) or {}).get("name")
        except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
            release = None
    return {
        "metric": metric,
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_sha(),
        "judge_model": judge_model,
        "answers": rel_to_repo(answers_path),
        "answers_release": release,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score (unweighted) Citation Coverage over a harness answer JSONL."
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
        "--claim-filter",
        choices=["llm", "none"],
        default="llm",
        help="Denominator filter: 'llm' (default — drop non-claim sentences via an "
        "LLM classifier) or 'none' (count every sentence over the length floor; "
        "fully deterministic, the old behaviour).",
    )
    p.add_argument(
        "--filter-model",
        default=DEFAULT_FILTER_MODEL,
        help=f"provider:model for the claim filter (default {DEFAULT_FILTER_MODEL}).",
    )
    p.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Territorial window: claims covered per citation (0 = strict "
        f"placement, no rescue; default {DEFAULT_WINDOW}). The k=0..5 bracket "
        "is always reported alongside.",
    )
    p.add_argument("--no-cache", action="store_true", help="Ignore the SHA-256 cache.")
    p.add_argument(
        "--no-verbatim-gate",
        action="store_true",
        help="Skip the parallel counts over verbatim-compliant citations only. They "
        "need the corpus; everything else in this metric does not.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Before any API call: a guard that fires after the run has already paid for it.
    assert_absent(args.report, args.overwrite)
    answers_path = Path(args.answers)
    if not answers_path.exists():
        raise SystemExit(f"{answers_path} not found")

    records = load_jsonl(answers_path)
    args.filter_model = resolve_route(args.filter_model)
    if args.claim_filter == "llm":
        _assert_filter_model_samples_independently(args.filter_model)
    # Counters are module-level, so one process scoring several answer sets must not pool them.
    _reset_filter_stats()
    result = asyncio.run(
        score_answers(
            records,
            args.filter_model,
            not args.no_cache,
            args.claim_filter,
            window=args.window,
            verbatim_gate=not args.no_verbatim_gate,
        )
    )
    judge = args.filter_model if args.claim_filter == "llm" else None
    result = {"meta": build_meta(answers_path, judge), **result}

    print(
        f"Citation Coverage (unweighted, window={result['window']}) — "
        f"segmenter {result['segmenter_version']}"
    )
    flt = result["claim_filter"]
    print(
        f"  claim filter = {flt}"
        + (
            f" ({result['claim_filter_version']}, {result['claim_filter_model']})"
            if flt == "llm"
            else ""
        )
    )
    print(f"  answers={result['n_answers']}  scored={result['n_scored']}")
    diag = result["diagnostics"]
    print(
        f"  empty-claim answers (excluded from mean) = {diag['n_empty_claim_answers']}"
        f" ({fmt_pct(diag['empty_claim_share'])})"
    )
    print(
        f"  non-claim sentences dropped from denominator = {result['n_dropped_non_claim']}"
    )
    print(
        f"  mid-clause collapse claims dropped ({result['claim_validity_version']}) = "
        f"{result['n_malformed_claims_excluded']}"
    )
    print(f"  mean coverage (per-question) = {fmt_pct(result['mean_coverage'])}")
    print(f"  pooled coverage (total)      = {fmt_pct(result['pooled_coverage'])}")
    if result["compliant"]:
        print(
            f"  verbatim-compliant citations = {result['n_compliant_citations']}"
            f" / {result['n_citations']}"
        )
        print(
            "  pooled coverage, compliant   = "
            f"{fmt_pct(result['compliant']['pooled_coverage'])}"
        )
    for k, agg in result["bracket"].items():
        print(
            f"  bracket {k:3s}: mean {fmt_pct(agg['mean_coverage'])}  "
            f"pooled {fmt_pct(agg['pooled_coverage'])}"
        )

    if args.report:
        write_report(args.report, result, args.overwrite)
        print(f"  full report → {args.report}")


if __name__ == "__main__":
    main()
