#!/usr/bin/env python3
"""Read-set lookup — where the evidence for an unsupported addition actually is.

uv run python eval/recovery/readset_probe.py         --judge gpt-5.4-mini --bucket-judge gpt-5.4-mini         --report eval/result/section_lookup/readset.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = Path(__file__).resolve().parents[1]
for _p in (
    str(_REPO_ROOT),
    str(_EVAL_DIR / "common"),
    str(_EVAL_DIR / "citation"),
    str(Path(__file__).resolve().parent),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

load_dotenv(_REPO_ROOT / ".env")

from cache import cache_key, cache_read, cache_write  # noqa: E402
from claim_support import build_judge_agent, judge_settings  # noqa: E402
from model_endpoints import needs_custom_endpoint, round_seed  # noqa: E402
from probe import (  # noqa: E402
    CRITERIA,
    _answers_path,
    METERED_DEFAULT_RPM,
    _free_path,
    GEN_MODELS,
    JUDGE_IDS,
    LOOKUP_VOTE_ROUNDS,
    MAX_DROPPED_ROUND_FRACTION,
    collect,
    questions,
)
from rate_limit import RateLimiter, is_rate_limit_error  # noqa: E402
from scoring import assert_absent, write_report  # noqa: E402
from vccr import _classify  # noqa: E402

from system.guidelines import CORPUS_ROOT  # noqa: E402

from pipeline_core.core.assemble import assemble  # noqa: E402  (path set by probe)

PROMPT_VERSION = "readset-lookup-v2"  # v2: the judge sees the QUESTION
_CACHE_DIR = _EVAL_DIR / "result" / ".readset_lookup_cache"
MAX_EVIDENCE_CHARS = 380_000

_SYSTEM = """You are checking whether clinical guideline text contains support for a \
claim.

You are given the QUESTION an answer was written for, a CLAIM taken from that answer, \
and the GUIDELINE TEXT the assistant had available when it wrote that answer. Determine \
whether the GUIDELINE TEXT contains text that supports the CLAIM.

Hard rules:
- Use the GUIDELINE TEXT only. Do not use outside or world knowledge.
- The QUESTION is CONTEXT, NEVER EVIDENCE. It sometimes carries the case (the population, \
the patient's lab values, the setting, and so on), so read it to see what the claim is \
talking about and who it is about, and do not charge the claim for a condition the question \
already provides.
- Copy the supporting text VERBATIM from the GUIDELINE TEXT. Do not paraphrase, \
summarise, or repair it.
- Answer true only if the copied text supports the WHOLE claim. Answer false if the \
GUIDELINE TEXT supports only part of what the claim asserts, or contains no such text.
- Each entry in `evidence` is ONE CONTIGUOUS passage copied from the GUIDELINE TEXT, \
with nothing added, removed, or joined on. Give two entries rather than joining two \
passages. Give an empty list when the answer is false."""


class ReadsetLookup(BaseModel):
    """Evidence BEFORE the boolean -- a field after it cannot inform it."""

    evidence: list[str] = Field(
        description="Passages COPIED VERBATIM from the GUIDELINE TEXT that together "
        "support the CLAIM, one contiguous passage per entry. Empty if there is none."
    )
    found: bool = Field(
        description="True only if every entry in `evidence` is text copied from the "
        "GUIDELINE TEXT and together they support the whole CLAIM."
    )


def _user_prompt(question: str, claim: str, evidence: str) -> str:
    """QUESTION first, and a blank one drops the block entirely. Mirrors probe.py, whose
    docstring carries the reasoning; the two prompts must stay the same shape or the two
    rungs of the same ladder stop being comparable."""
    head = (
        f"QUESTION (the clinical question this answer was written for):\n{question}\n\n"
        if question.strip()
        else ""
    )
    return f"{head}CLAIM:\n{claim}\n\nGUIDELINE TEXT:\n{evidence}"


# System prompt AND user-turn template hashed into the cache key; see probe.py.
_PROMPT_HASH = hashlib.sha256(
    json.dumps(
        {
            "system": _SYSTEM,
            "schema_doc": ReadsetLookup.model_json_schema(),
            "user_turn": [
                _user_prompt("<Q>", "<CLAIM>", "<TEXT>"),
                _user_prompt("", "<CLAIM>", "<TEXT>"),
            ],
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()[:12]


# Leniency order of the pass tiers, for picking a representative when a row's passages
# verify at different tiers. Not a score: it only decides which label the row reports.
_PASS_RANK = {"exact": 0, "normalized": 1, "elided": 2}

_SEC: dict[str, tuple[str, str]] = {}


def section(doc_id: str) -> tuple[str, str]:
    """(raw, normalized) full text of one read-set entry, descendants INCLUDED —
    this is what the agent had in context, so the whole subtree counts."""
    if doc_id not in _SEC:
        gid, _, sid = doc_id.partition(":")
        try:
            raw = (
                assemble(
                    CORPUS_ROOT / gid,
                    sid,
                    include_descendants=True,
                    resolve_resources=True,
                )
                or ""
            )
        except Exception:  # noqa: BLE001
            raw = ""
        _SEC[doc_id] = (raw, CRITERIA.normalize(raw) if raw else "")
    return _SEC[doc_id]


def read_sets(models=GEN_MODELS, bucket_judge: str = "gpt-5pt4-mini") -> dict:
    """(gen_model, question_id) -> the sections that model actually opened."""
    out: dict[tuple[str, str], list[str]] = {}
    for gm in models:
        with _answers_path(bucket_judge, gm).open(encoding="utf-8") as fh:
            for line in fh:
                # Guarded like probe.questions: a stray blank line in any of the fourteen
                # answer files would otherwise raise before a single API call.
                if not line.strip():
                    continue
                r = json.loads(line)
                out[(gm, r["question_id"])] = r["read_set"]
    return out


_AGENT = {}
# Real API calls, not cache hits. Cross-checked against the limiter's own count so an
# unwired limiter cannot pass for a fully cached run. See probe._CALLS.
_CALLS = Counter()


async def lookup(
    question: str,
    claim: str,
    evidence: str,
    judge_model: str,
    use_cache: bool,
    round_index: int = 1,
    limiter: RateLimiter | None = None,
) -> dict:
    """One draw. `round_index` enters the cache key so sampled rounds do not collapse onto
    one cached answer; round 1 is keyless for it, as in probe.py and claim_support."""
    question = (question or "").strip()
    round_part = () if round_index == 1 else (f"r{round_index}",)
    key = cache_key(
        judge_model,
        PROMPT_VERSION,
        _PROMPT_HASH,
        *round_part,
        question,
        claim,
        evidence,
    )
    if use_cache:
        hit = cache_read(_CACHE_DIR, key)
        if hit is not None:
            return hit
    seed = round_seed(judge_model, round_index)
    if (judge_model, seed) not in _AGENT:
        _AGENT[(judge_model, seed)] = build_judge_agent(
            judge_model, _SYSTEM, ReadsetLookup, seed=seed
        )
    # Same hard refusal as probe.lookup: an unpaced run against a metered host does
    # not fail, it returns a biased number, so a missing limiter must stop the run
    # rather than warn about it.
    if limiter is None and needs_custom_endpoint(judge_model):
        raise SystemExit(
            f"refusing to call {judge_model} unpaced: its host meters by the minute "
            f"and rejects with 429. Pass a RateLimiter (readset_probe.py --rpm) or "
            f"use a natively routed model."
        )
    # After the cache read, so a hit does not spend a paced slot. See probe.lookup.
    _CALLS["api"] += 1
    if limiter is not None:
        await limiter.acquire()
    try:
        res = await _AGENT[(judge_model, seed)].run(
            _user_prompt(question, claim, evidence)
        )
        out = res.output.model_dump()
    except Exception as e:  # noqa: BLE001
        if limiter is not None and is_rate_limit_error(e):
            limiter.penalise()
        return {"error": str(e)[:500]}
    if use_cache:
        cache_write(_CACHE_DIR, key, out)
    return out


async def lookup_voted(
    question: str,
    claim: str,
    evidence: str,
    judge_model: str,
    use_cache: bool,
    sem: asyncio.Semaphore | None = None,
    limiter: RateLimiter | None = None,
    rounds: int | None = None,
) -> dict:
    """Majority over `rounds` draws (default LOOKUP_VOTE_ROUNDS). Mirrors probe.lookup_voted,
    including the rule that the returned span is the first majority-side round's, chosen
    BEFORE the offline re-check so the selection cannot inflate the span-verified rate, and
    that a tie among surviving rounds breaks toward NOT found.
    """

    async def _one(r: int) -> dict:
        if sem is None:
            return await lookup(
                question,
                claim,
                evidence,
                judge_model,
                use_cache,
                round_index=r,
                limiter=limiter,
            )
        async with sem:
            return await lookup(
                question,
                claim,
                evidence,
                judge_model,
                use_cache,
                round_index=r,
                limiter=limiter,
            )

    n_rounds = LOOKUP_VOTE_ROUNDS if rounds is None else rounds
    drawn = await asyncio.gather(*(_one(r) for r in range(1, n_rounds + 1)))
    valid = [v for v in drawn if not v.get("error")]
    if not valid:
        out = dict(drawn[0])
        out.update(
            {
                "vote_rounds_used": 0,
                "vote_rounds_dropped": n_rounds,
                "votes": [],
                "vote_unanimous": None,
            }
        )
        return out
    votes = [bool(v.get("found")) for v in valid]
    winner = sum(votes) * 2 > len(votes)
    out = dict(next(v for v in valid if bool(v.get("found")) is winner))
    out["found"] = winner
    out["votes"] = votes
    out["vote_rounds_used"] = len(valid)
    out["vote_rounds_dropped"] = n_rounds - len(valid)
    # None, not True, when a round was lost: one surviving vote is trivially "unanimous",
    # which would make self-consistency look best exactly when the run had most trouble.
    # Same convention as claim_support's judge_first3_unanimous over a fixed n.
    out["vote_unanimous"] = (
        # A single-round run has no unanimity to report: one vote is trivially
        # "unanimous", which would make self-consistency look perfect exactly where it was
        # never measured. None, not True.
        (len(set(votes)) == 1 if len(valid) == n_rounds else None)
        if n_rounds > 1
        else None
    )
    return out


#: The rungs, in the order the evidence widens. One read-set answer lands in exactly one.
RUNGS = ("node", "sectionrest", "othersection", "unattributed", "nowhere")


def _rung(row: dict) -> str:
    """Which rung a judged row belongs to, from the offline localisation alone."""
    if row["span_tier"] == "error":
        return "error"
    if row["span_tier"] not in CRITERIA.pass_kinds:
        return "nowhere"
    if row.get("evidence_in_node"):
        return "node"
    cited = row.get("evidence_is_cited_section")
    if cited is None:
        # Verifies against the read set as a whole but against no single section: a span
        # spliced across two sections, so it is attributable to neither.
        return "unattributed"
    return "sectionrest" if cited else "othersection"


async def run(
    judge: str,
    concurrency: int,
    use_cache: bool,
    limit,
    models=GEN_MODELS,
    rpm: float | None = None,
    bucket_judge: str = "gpt-5pt4-mini",
    rounds: int | None = None,
) -> dict:
    """One round over the FULL population, not the residual of a node round."""
    n_rounds = LOOKUP_VOTE_ROUNDS if rounds is None else rounds
    items, stats = collect(bucket_judge, models)
    # `_section` is the NODE text, which this round does not judge: the evidence here is
    # the read set. It is kept for the offline node test at the bottom of `one()`.
    if limit:
        items = items[:limit]
    RS = read_sets(models, bucket_judge)
    QS = questions(models, bucket_judge)
    judge_model = JUDGE_IDS[judge]
    sem = asyncio.Semaphore(concurrency)
    _CALLS.clear()
    limiter = RateLimiter(rpm) if rpm else None
    if limiter:
        print(f"  pacing at {limiter.rpm:g} calls/min", flush=True)

    async def one(r: dict) -> dict:
        ids = RS[(r["gen_model"], r["question_id"])]
        parts, truncated = [], False
        for d in ids:
            raw, _ = section(d)
            if raw:
                parts.append(f"[{d}]\n{raw}")
        blob = "\n\n---\n\n".join(parts)
        if len(blob) > MAX_EVIDENCE_CHARS:
            blob, truncated = blob[:MAX_EVIDENCE_CHARS], True
        # No `async with sem` here: lookup_voted takes the same semaphore per round, and
        # holding it around the call would deadlock. See probe.py.
        q = r.get("question") or QS.get((r["gen_model"], r["question_id"]), "")
        if not q.strip():
            raise SystemExit(
                f"no question text for {r['gen_model']}/{r['question_id']}: a blank question "
                f"drops the QUESTION block and would judge a v1-shaped prompt under the v2 "
                f"label. Re-run probe.py so its rows carry `question`."
            )
        out = await lookup_voted(
            q, r["claim"], blob, judge_model, use_cache, sem, limiter, n_rounds
        )

        row = {
            "gen_model": r["gen_model"],
            "question_id": r["question_id"],
            "question": q,
            "claim": r["claim"],
            "cited_doc_id": r["cited_doc_id"],
            "node_id": r["node_id"],
            "quote": r.get("quote"),
            "read_set": ids,
            "evidence_chars": len(blob),
            "evidence_truncated": truncated,
            **out,
        }
        raw_ev = out.get("evidence")
        if isinstance(raw_ev, str):  # a judge that ignored the schema
            raw_ev = [raw_ev]
        spans = [x.strip() for x in (raw_ev or []) if isinstance(x, str) and x.strip()]
        spans = [x for x in spans if x != "NOTHING_FOUND"]
        ev = "\n".join(spans)
        row["evidence"] = ev
        row["evidence_spans"] = spans
        row["n_spans"] = len(spans)
        if out.get("error"):
            row["span_tier"] = "error"
        elif not out.get("found"):
            row["span_tier"] = "n/a"
        elif not spans:
            row["span_tier"] = "empty"
        else:
            tiers = [
                _classify(x, blob, CRITERIA.normalize(blob), CRITERIA) for x in spans
            ]
            row["span_tiers"] = tiers
            bad = [t for t in tiers if t not in CRITERIA.pass_kinds]
            row["span_tier"] = (
                bad[0] if bad else max(tiers, key=lambda t: _PASS_RANK.get(t, 0))
            )
        row["evidence_section"] = None
        row["evidence_is_cited_section"] = None
        row["evidence_in_node"] = None
        row["span_sections"] = None
        row["span_in_node"] = None
        if row["span_tier"] in CRITERIA.pass_kinds:
            # `cited_doc_id` is a LIST under set adjudication (a claim may cite several
            # sections at once), so membership, not equality.
            cited = r["cited_doc_id"]
            cited = [cited] if isinstance(cited, str) else cited
            placed: list[str | None] = []
            for x in spans:
                here = None
                for d in ids:
                    raw, nrm = section(d)
                    if _classify(x, raw, nrm, CRITERIA) in CRITERIA.pass_kinds:
                        here = d
                        break
                placed.append(here)
            row["span_sections"] = placed
            nraw = r["_section"]
            nnrm = CRITERIA.normalize(nraw) if nraw else ""
            row["span_in_node"] = [
                bool(nraw) and _classify(x, nraw, nnrm, CRITERIA) in CRITERIA.pass_kinds
                for x in spans
            ]
            # A passage that matches no single section leaves the row unattributed, which
            # is what `_rung` reads a None `evidence_is_cited_section` as. Only when every
            # passage is placed can the row name a section, and then the widest one wins.
            if None not in placed:
                row["evidence_in_node"] = all(row["span_in_node"])
                row["evidence_is_cited_section"] = all(d in cited for d in placed)
                outside = [d for d in placed if d not in cited]
                row["evidence_section"] = outside[0] if outside else placed[0]
        return row

    rows = await asyncio.gather(*(one(r) for r in items))
    ver = [r for r in rows if r["span_tier"] in CRITERIA.pass_kinds]
    # `judged` excludes calls that ERRORED: they are not a verdict, and leaving them in
    # would put failed HTTP requests in the denominator of every rate here. They stay in
    # `rows` so the report still carries them, and `n_errors` says how many.
    judged = [r for r in rows if r["span_tier"] != "error"]
    return {
        "probe": "readset-lookup",
        "prompt_version": PROMPT_VERSION,
        "judge": judge_model,
        # Requested decoding settings, from claim_support's single source; see probe.py.
        "judge_settings": judge_settings(),
        "vote_rounds": n_rounds,
        "criteria_version": CRITERIA.version,
        "bucket": "unsupported_addition",
        "bucket_judge": bucket_judge,
        "population": "every unsupported_addition item; NOT the residual of a node round",
        # From collect(), so the table can state the denominators it drops and why, the same
        # counters the node probe reported.
        "n_addition_pairs": stats["pairs"],
        "n_quote_not_locatable": stats["quote_not_locatable"],
        "n_doc_id_unresolvable": stats["doc_id_unresolvable"],
        "n_duplicate_claim_node": stats["duplicate"],
        "n_judged": len(judged),
        "n_errors": sum(1 for r in rows if r.get("error")),
        "n_truncated_evidence": sum(1 for r in rows if r["evidence_truncated"]),
        "n_found": sum(1 for r in rows if r.get("found")),
        "n_found_span_verified": len(ver),
        "found_rate_span_verified": len(ver) / len(judged) if judged else None,
        "n_vote_split": sum(1 for r in rows if r.get("vote_unanimous") is False),
        # Denominator is items with a COMPLETE round set, not all items: an item that lost a
        # round has no defined unanimity and must not dilute the rate.
        "vote_split_rate": (
            sum(1 for r in rows if r.get("vote_unanimous") is False)
            / sum(1 for r in rows if r.get("vote_unanimous") is not None)
            if any(r.get("vote_unanimous") is not None for r in rows)
            else None
        ),
        "n_rounds_dropped": sum(r.get("vote_rounds_dropped", 0) for r in rows),
        "rate_limit": limiter.stats() if limiter else None,
        "n_api_calls": _CALLS["api"],
        "concurrency": concurrency,
        "span_tier_distribution": dict(Counter(r["span_tier"] for r in rows)),
        "evidence_in_cited_section": sum(
            1 for r in ver if r["evidence_is_cited_section"]
        ),
        "evidence_in_other_read_section": sum(
            1 for r in ver if r["evidence_is_cited_section"] is False
        ),
        "rungs": dict(Counter(_rung(r) for r in judged)),
        "per_gen_model": {
            gm: {
                "n": sum(1 for r in judged if r["gen_model"] == gm),
                "n_errors": sum(
                    1
                    for r in rows
                    if r["gen_model"] == gm and r["span_tier"] == "error"
                ),
                "n_found": sum(
                    1 for r in judged if r["gen_model"] == gm and r.get("found")
                ),
                "n_found_verified": sum(1 for r in ver if r["gen_model"] == gm),
                "rungs": dict(
                    Counter(_rung(r) for r in judged if r["gen_model"] == gm)
                ),
            }
            for gm in models
        },
        "rows": rows,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    # No --input. This round draws its own population with collect(), so it does not run
    # behind a node round and cannot be pointed at one's leftovers. See run()'s docstring.
    p.add_argument("--judge", default="gpt-5pt4-mini", choices=sorted(JUDGE_IDS))
    p.add_argument(
        "--bucket-judge",
        default="gpt-5pt4-mini",
        help="Claim-support judge whose unsupported_addition bucket is the population. "
        "Must be the judge whose verdicts the funnel folds, or recovery is measured "
        "against a different judge's shortfall set than `certified`.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(GEN_MODELS),
        help="Generation models to draw the bucket for.",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=LOOKUP_VOTE_ROUNDS,
        help=f"Sampled judgements per item, majority-voted (default "
        f"{LOOKUP_VOTE_ROUNDS}). ROUNDS ARE CACHED INDIVIDUALLY, so a 1-round run can be "
        f"topped up to 3 later for exactly the two extra rounds -- round 1 is re-read from "
        f"cache, not re-paid. At 1 round there is no majority and no self-consistency "
        f"number, and every lost call is a lost ITEM rather than a lost round, so "
        f"--max-dropped-fraction bites much harder.",
    )
    p.add_argument(
        "--out-root",
        default="eval/result/section_lookup/readset",
        help="Directory the report is filed under, as <out-root>/judge-<judge>/r<rounds>/"
        "report.json. Structured by ROUND so a later top-up lands beside the earlier run "
        "instead of colliding with it: the two are different measurements and the 1-round "
        "one stays readable as its own record. Ignored when --report is given.",
    )
    p.add_argument(
        "--report",
        help="Explicit output path, overriding the --out-root layout.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --report to replace an existing file (default: refuse).",
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument(
        "--rpm",
        type=float,
        help="Calls per minute, paced client-side. Defaults to "
        f"{METERED_DEFAULT_RPM:g} for a judge behind a metered host, unpaced "
        f"otherwise. "
        "Pass 0 to opt out. See rate_limit.py for why concurrency is not this.",
    )
    p.add_argument(
        "--max-dropped-fraction", type=float, default=MAX_DROPPED_ROUND_FRACTION
    )
    p.add_argument("--no-cache", action="store_true")
    a = p.parse_args()
    if a.rounds < 1:
        raise SystemExit("--rounds must be >= 1")
    if a.rounds % 2 == 0:
        # An even vote has no majority to break except by the tie rule, which points at
        # NOT found, so an even round count buys sampling noise and biases the rate down.
        raise SystemExit(
            f"--rounds must be odd (got {a.rounds}): an even vote is decided by the "
            f"tie-break, which points at NOT found, so it biases found_rate downward."
        )
    report = a.report or str(
        Path(a.out_root) / f"judge-{a.judge}" / f"r{a.rounds}" / "report.json"
    )
    Path(report).parent.mkdir(parents=True, exist_ok=True)
    # Before any API call: a guard that fires after the run has already paid for it.
    assert_absent(report, a.overwrite)
    if a.rpm is not None and a.rpm < 0:
        raise SystemExit("--rpm must be >= 0 (0 means unpaced)")

    rpm = a.rpm
    if rpm is None:
        rpm = METERED_DEFAULT_RPM if needs_custom_endpoint(JUDGE_IDS[a.judge]) else 0.0

    res = asyncio.run(
        run(
            a.judge,
            a.concurrency,
            not a.no_cache,
            a.limit,
            tuple(a.models),
            rpm or None,
            a.bucket_judge,
            a.rounds,
        )
    )
    print(f"readset-lookup ({PROMPT_VERSION})  judge {res['judge']}")
    print(
        f"  addition pairs {res['n_addition_pairs']}  "
        f"quote not locatable {res['n_quote_not_locatable']}  "
        f"dup (claim,node) {res['n_duplicate_claim_node']}  -> judged {res['n_judged']}"
    )
    print(
        f"  errors {res['n_errors']}  evidence truncated {res['n_truncated_evidence']}"
    )
    print(
        f"  found = {res['n_found']}   span-VERIFIED = {res['n_found_span_verified']} "
        f"({100 * res['found_rate_span_verified']:.1f}%)"
    )
    print(f"  span tiers: {res['span_tier_distribution']}")
    print(f"  rungs: {res['rungs']}")
    for gm, d in res["per_gen_model"].items():
        pct = 100 * d["n_found_verified"] / d["n"] if d["n"] else 0
        print(
            f"    {gm:18s} n={d['n']:4d}  found(verified) {d['n_found_verified']:4d} ({pct:.1f}%)"
        )
    if res["rate_limit"]:
        print(f"  pacing: {res['rate_limit']}  api calls {res['n_api_calls']}")
        if res["n_api_calls"] and not res["rate_limit"]["calls_paced"]:
            raise SystemExit(
                f"\nREJECTED: {res['n_api_calls']} calls reached the endpoint but the "
                f"limiter paced 0 of them. Wiring bug, not a result."
            )

    # Same guard as probe.py, for the same reason: a dropped round is a push toward NOT
    # found, so losing many makes the rate wrong rather than noisy.
    total_rounds = res["n_judged"] * res["vote_rounds"]
    dropped = res["n_rounds_dropped"]
    share = dropped / total_rounds if total_rounds else 0.0
    if share > a.max_dropped_fraction:
        # See probe._free_path: a fixed name plus overwrite=True would have the second
        # rejection destroy the first one's rows, and these files are untracked.
        out = _free_path(Path(report).with_suffix(".rejected.json"))
        write_report(out, res, overwrite=False)
        raise SystemExit(
            f"\nREJECTED: {dropped} of {total_rounds} voting rounds dropped "
            f"({100 * share:.1f}%, limit {100 * a.max_dropped_fraction:.1f}%). "
            f"Diagnostic report -> {out}. Lower --rpm and re-run; cached rounds are kept."
        )

    write_report(report, res, a.overwrite)
    print(f"  report -> {report}")


if __name__ == "__main__":
    main()
