#!/usr/bin/env python3
"""Section-lookup triage for the `unsupported_addition` bucket.

uv run python eval/recovery/probe.py         --judge gpt-5.4-mini         --report eval/result/reports/section_lookup.json
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
for _p in (str(_REPO_ROOT), str(_EVAL_DIR / "common"), str(_EVAL_DIR / "citation")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

load_dotenv(_REPO_ROOT / ".env")

from cache import cache_key, cache_read, cache_write  # noqa: E402
from claim_support import build_judge_agent, judge_settings  # noqa: E402
from model_endpoints import needs_custom_endpoint, round_seed  # noqa: E402
from rate_limit import RateLimiter, is_rate_limit_error  # noqa: E402
from scoring import assert_absent, write_report  # noqa: E402
from vccr import _classify, load_criteria  # noqa: E402

from system.guidelines import (  # noqa: E402
    CORPUS_ROOT,
    _ensure_pipeline_core_on_path,
)

PROMPT_VERSION = "section-lookup-v2"  # v2: the judge sees the QUESTION
LOOKUP_VOTE_ROUNDS = 3

METERED_DEFAULT_RPM = 18.0

MAX_DROPPED_ROUND_FRACTION = 0.01

_CACHE_DIR = _EVAL_DIR / "result" / ".section_lookup_cache"
# The gated claim-support reports: the population, the verdicts and the recovery
# branch all come from this one source rather than from three.
_RESULTS = _REPO_ROOT / "eval" / "result" / "claim_support_gated"
GEN_MODELS = ("gpt-5pt4-mini", "claude-haiku-4-5", "claude-sonnet-5")
JUDGE_IDS = {
    "gpt-5pt4-mini": "openai-responses:gpt-5.4-mini",
    "claude-haiku-4-5": "anthropic:claude-haiku-4-5",
}


class Lookup(BaseModel):
    """Evidence BEFORE the boolean — a field after it cannot inform it."""

    evidence: str = Field(
        description="The text COPIED VERBATIM from the SECTION that supports the "
        "CLAIM, or NOTHING_FOUND."
    )
    found: bool = Field(
        description="True only if `evidence` is text copied from the SECTION and it "
        "supports the whole CLAIM."
    )


_SYSTEM = """You are checking whether a clinical guideline section contains support \
for a claim.

You are given the QUESTION an answer was written for, a CLAIM taken from that answer, \
and one SECTION of a clinical guideline. Determine whether the SECTION contains text that \
supports the CLAIM.

Hard rules:
- Use the SECTION only. Do not use outside or world knowledge.
- The QUESTION is CONTEXT, NEVER EVIDENCE. It sometimes carries the case (the population, \
the patient's lab values, the setting, and so on), so read it to see what the claim is \
talking about and who it is about, and do not charge the claim for a condition the question \
already provides.
- Copy the supporting text VERBATIM from the SECTION. Do not paraphrase, summarise, \
or repair it.
- Answer true only if the copied text supports the WHOLE claim. Answer false if the \
SECTION supports only part of what the claim asserts, or contains no such text.
- When the answer is false, write NOTHING_FOUND in the evidence field."""


def _user_prompt(question: str, claim: str, section: str) -> str:
    """QUESTION comes FIRST so the scope is read before the claim. A BLANK question drops
    the block entirely rather than emitting an empty heading, which would read as "this
    answer had no question" instead of "we did not supply one"."""
    head = (
        f"QUESTION (the clinical question this answer was written for):\n{question}\n\n"
        if question.strip()
        else ""
    )
    return f"{head}CLAIM:\n{claim}\n\nSECTION:\n{section}"


_PROMPT_HASH = hashlib.sha256(
    json.dumps(
        {
            "system": _SYSTEM,
            "schema_doc": Lookup.model_json_schema(),
            "user_turn": [
                _user_prompt("<Q>", "<CLAIM>", "<SECTION>"),
                _user_prompt("", "<CLAIM>", "<SECTION>"),
            ],
        },
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
).hexdigest()[:12]


# ── corpus: locate the smallest node that actually contains the quote ────────────

_ensure_pipeline_core_on_path()
from pipeline_core.core.assemble import assemble  # noqa: E402

CRITERIA = load_criteria()
_TREE: dict[str, dict] = {}
_TEXT: dict[tuple[str, str], tuple[str, str]] = {}


def _tree(gid: str) -> dict:
    """Section id -> node for one guideline. An EMPTY dict when the guideline does not resolve,
    which happens because a marker's doc_id need not be a real `guideline:section` pair: one
    model emitted two citing `FIGURE ch10-f2`. That used to raise FileNotFoundError, which
    never fired on a small model set and would have killed a larger run partway through.
    """
    if gid not in _TREE:
        p = _REPO_ROOT / "data" / "corpus" / "guidelines" / gid / "tree.json"
        if not p.exists():
            _TREE[gid] = {}
            return _TREE[gid]
        try:
            _TREE[gid] = {
                n["section_id"]: n for n in json.loads(p.read_text())["nodes"]
            }
        except (FileNotFoundError, NotADirectoryError):
            _TREE[gid] = {}
    return _TREE[gid]


def _node_text(gid: str, sid: str) -> tuple[str, str]:
    """(raw, normalized) own text of one node — descendants EXCLUDED."""
    key = (gid, sid)
    if key not in _TEXT:
        try:
            raw = (
                assemble(
                    CORPUS_ROOT / gid,
                    sid,
                    include_descendants=False,
                    resolve_resources=True,
                )
                or ""
            )
        except Exception:  # noqa: BLE001 — a bad node contributes no text
            raw = ""
        _TEXT[key] = (raw, CRITERIA.normalize(raw) if raw else "")
    return _TEXT[key]


def _descendants(gid: str, sid: str) -> list[str]:
    nodes = _tree(gid)
    out, stack = [], [sid]
    while stack:
        cur = stack.pop()
        if cur in nodes:
            out.append(cur)
            stack.extend(nodes[cur].get("children", []))
    return out


def unresolvable(doc_id: str) -> bool:
    """True when a marker's doc_id names no guideline we hold, so no node can be searched.
    Distinguished from "the quote is in no node" because the causes differ: this is a
    malformed or unresolvable citation, the other is a quote the model did not copy."""
    return not _tree(doc_id.partition(":")[0])


def locate(doc_id: str, quote: str) -> tuple[str, str, str] | None:
    """(section_id, raw_text, match_tier) of the SMALLEST node whose own text
    contains `quote` under the VCCR criteria, or None if no node does."""
    gid, _, sid = doc_id.partition(":")
    hits = []
    for node in _descendants(gid, sid):
        raw, norm = _node_text(gid, node)
        tier = _classify(quote, raw, norm, CRITERIA)
        if tier in CRITERIA.pass_kinds:
            hits.append((len(raw), node, raw, tier))
    if not hits:
        return None
    _, node, raw, tier = min(hits)
    return node, raw, tier


# ── judging ──────────────────────────────────────────────────────────────────────

_AGENT = {}
# Real API calls, as opposed to cache hits. Read against the limiter's own `calls_paced`
# so a limiter that was built, announced and then never threaded through cannot look
# like a run that was fully cached. It has looked exactly like that.
_CALLS = Counter()


async def lookup(
    question: str,
    claim: str,
    section: str,
    judge_model: str,
    use_cache: bool,
    round_index: int = 1,
    limiter: RateLimiter | None = None,
) -> dict:
    """One draw. `round_index` enters the cache key, so sampled rounds do not collapse onto
    one cached answer; round 1 is keyless for it, matching claim_support._cache_key."""
    question = (question or "").strip()
    round_part = () if round_index == 1 else (f"r{round_index}",)
    key = cache_key(
        judge_model, PROMPT_VERSION, _PROMPT_HASH, *round_part, question, claim, section
    )
    if use_cache:
        hit = cache_read(_CACHE_DIR, key)
        if hit is not None:
            return hit
    seed = round_seed(judge_model, round_index)
    if (judge_model, seed) not in _AGENT:
        _AGENT[(judge_model, seed)] = build_judge_agent(
            judge_model, _SYSTEM, Lookup, seed=seed
        )
    if limiter is None and needs_custom_endpoint(judge_model):
        raise SystemExit(
            f"refusing to call {judge_model} unpaced: its host meters by the minute "
            f"and rejects with 429. Pass a RateLimiter (probe.py --rpm) or use a "
            f"natively routed model."
        )
    # AFTER the cache read, deliberately: a cache hit is not a request and must not consume
    # a paced slot, or a mostly-cached re-run would crawl for no reason.
    _CALLS["api"] += 1
    if limiter is not None:
        await limiter.acquire()
    try:
        res = await _AGENT[(judge_model, seed)].run(
            _user_prompt(question, claim, section)
        )
        out = res.output.model_dump()
    except Exception as e:  # noqa: BLE001 — isolate per-item failures
        # 500 chars, not 200: a shorter truncation throws away the response headers,
        # leaving no way to tell from an archived report whether the host sent a
        # `retry-after` the SDK could have honoured.
        if limiter is not None and is_rate_limit_error(e):
            limiter.penalise()
        return {"error": str(e)[:500]}
    if use_cache:
        cache_write(_CACHE_DIR, key, out)
    return out


async def lookup_voted(
    question: str,
    claim: str,
    section: str,
    judge_model: str,
    use_cache: bool,
    sem: asyncio.Semaphore | None = None,
    limiter: RateLimiter | None = None,
    rounds: int | None = None,
) -> dict:
    """Majority over `rounds` (default LOOKUP_VOTE_ROUNDS) independent draws."""

    async def _one(r: int) -> dict:
        if sem is None:
            return await lookup(
                question,
                claim,
                section,
                judge_model,
                use_cache,
                round_index=r,
                limiter=limiter,
            )
        async with sem:
            return await lookup(
                question,
                claim,
                section,
                judge_model,
                use_cache,
                round_index=r,
                limiter=limiter,
            )

    # `n_rounds`, not the module constant, everywhere below: a 1-round run must report
    # rounds_used 1 and rounds_dropped 0, and reading the constant would have it claim two
    # rounds were lost.
    n_rounds = LOOKUP_VOTE_ROUNDS if rounds is None else rounds
    drawn = await asyncio.gather(*(_one(r) for r in range(1, n_rounds + 1)))
    valid = [v for v in drawn if not v.get("error")]
    if not valid:
        out = dict(drawn[0])
        out["vote_rounds_used"] = 0
        out["vote_rounds_dropped"] = n_rounds
        out["votes"] = []
        out["vote_unanimous"] = None
        return out

    votes = [bool(v.get("found")) for v in valid]
    # Binary vote: with an odd number of survivors a majority always exists. With an even
    # number (a round errored) a tie is possible, and it is broken toward NOT found, the
    # conservative side, so a tie never manufactures evidence.
    winner = sum(votes) * 2 > len(votes)
    out = dict(next(v for v in valid if bool(v.get("found")) is winner))
    out["found"] = winner
    out["votes"] = votes
    out["vote_rounds_used"] = len(valid)
    # Unanimity over the surviving rounds, the probe's own self-consistency number. Reported
    # rather than hidden: at the provider default temperature it is the only evidence that a
    # single-draw version of this probe would have been stable.
    out["vote_rounds_dropped"] = n_rounds - len(valid)
    # None, not True, when a round was lost: one surviving vote is trivially "unanimous",
    # which would make self-consistency look best exactly when the run had most trouble.
    # Same convention as claim_support's judge_first3_unanimous over a fixed n.
    out["vote_unanimous"] = len(set(votes)) == 1 if len(valid) == n_rounds else None
    return out


def _answers_path(judge: str, gen_model: str) -> Path:
    """The answers file a model's claim support report was scored on."""
    rep = _RESULTS / f"judge-{judge}" / gen_model / "report.json"
    if not rep.is_file():
        raise SystemExit(
            f"{rep} not found, so the answers file behind {gen_model} cannot be resolved. "
            f"Run claim support for it under judge-{judge} first."
        )
    meta = json.loads(rep.read_text(encoding="utf-8")).get("meta") or {}
    rel = meta.get("answers")
    if not rel:
        raise SystemExit(
            f"{rep} records no `meta.answers`, so which answers it scored is unknown. "
            f"Re-run it under the current scorer, which records that field."
        )
    path = _REPO_ROOT / rel
    if not path.is_file():
        raise SystemExit(f"{path} (from {rep}'s meta.answers) does not exist.")
    return path


def questions(
    models: tuple[str, ...] = GEN_MODELS, judge: str = "gpt-5pt4-mini"
) -> dict[tuple[str, str], str]:
    """(gen_model, question_id) -> question text, from the answers each model's claim
    support report was scored on. Keyed by both because a question_id is only unique
    within a model's answer file."""
    out: dict[tuple[str, str], str] = {}
    for gm in models:
        with _answers_path(judge, gm).open(encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    out[(gm, r["question_id"])] = r.get("question", "")
    return out


def _report_question_ids(judge: str, gen_model: str) -> list[str]:
    """The question ids one claim support report references, for the pre-flight check."""
    rep = _RESULTS / f"judge-{judge}" / gen_model / "report.json"
    if not rep.is_file():
        return []
    return [r["question_id"] for r in json.loads(rep.read_text())["per_record"]]


def collect(bucket_judge: str, models: tuple[str, ...] = GEN_MODELS) -> list[dict]:
    """Every unsupported_addition item in this judge's own reports, deduplicated to
    unique (claim, containing nodes).
    """
    items: dict[tuple, dict] = {}
    stats = Counter()
    # The bucket_judge's reports are both the source of the bucket AND the record of which
    # answers each model was scored on, so the questions are resolved through the same
    # report rather than from a path built out of the model's directory name.
    qmap = questions(models, bucket_judge)
    # Indexed, not .get(..., ""): a blank question makes _user_prompt drop the QUESTION
    # block, so a release/id mismatch would judge and cache v1-shaped prompts under the v2
    # label, which is exactly the defect v2 exists to remove. Fail on the mismatch instead.
    missing = {
        (gm, qid)
        for gm in models
        for qid in _report_question_ids(bucket_judge, gm)
        if (gm, qid) not in qmap or not qmap[(gm, qid)].strip()
    }
    if missing:
        raise SystemExit(
            f"{len(missing)} (model, question_id) pairs in the claim support reports have no "
            f"question text in the answers release, e.g. {sorted(missing)[:3]}. The reports "
            f"and the release have diverged; do not run with a blank question."
        )
    for gm in models:
        rep = json.loads(
            (_RESULTS / f"judge-{bucket_judge}" / gm / "report.json").read_text()
        )
        for rec in rep["per_record"]:
            for claim in rec["claims"]:
                for pair in claim["pairs"]:
                    if pair.get("shortfall_type") != "unsupported_addition":
                        continue
                    stats["pairs"] += 1
                    docs = pair["doc_id"]
                    docs = [docs] if isinstance(docs, str) else list(docs)
                    quotes = str(pair["quote"]).split("\n\n")
                    # Every (doc, quote) the entry mentions; a quote may belong to any of
                    # the declared sections, so try each until one locates it.
                    nodes: dict[str, str] = {}
                    tiers: list[str] = []
                    for q in quotes:
                        if not q.strip():
                            continue
                        for d in docs:
                            loc = locate(d, q)
                            if loc is not None:
                                node, raw, tier = loc
                                nodes.setdefault(node, raw)
                                tiers.append(tier)
                                break
                        else:
                            if docs and all(unresolvable(d) for d in docs):
                                stats["doc_id_unresolvable"] += 1
                            else:
                                stats["quote_not_locatable"] += 1
                    if not nodes:
                        continue
                    node_ids = sorted(nodes)
                    k = (
                        qmap[(gm, rec["question_id"])],
                        claim["text"],
                        tuple(node_ids),
                    )
                    if k in items:
                        stats["duplicate"] += 1
                        continue
                    raw = "\n\n".join(nodes[n] for n in node_ids)
                    items[k] = {
                        "gen_model": gm,
                        "question_id": rec["question_id"],
                        "question": qmap[(gm, rec["question_id"])],
                        "claim": claim["text"],
                        "quote": pair["quote"],
                        "cited_doc_id": docs,
                        "node_id": node_ids,
                        "node_chars": len(raw),
                        "quote_match_tier": tiers,
                        "round1_verdict": pair.get("support_verdict"),
                        "_section": raw,
                    }
    return list(items.values()), stats


async def run(
    judge: str,
    concurrency: int,
    use_cache: bool,
    limit: int | None,
    models: tuple[str, ...] = GEN_MODELS,
    bucket_judge: str | None = None,
    rpm: float | None = None,
    max_dropped: float = MAX_DROPPED_ROUND_FRACTION,
    rounds: int | None = None,
) -> dict:
    """`judge` searches for evidence; `bucket_judge` is whose claim support run supplied the
    unsupported_addition bucket. They default to the same name and usually are, but they are
    different roles: comparing two SEARCH routes requires holding the bucket fixed, and there
    is no claim support tree for a route that has only ever been a lookup judge."""
    judge_model = JUDGE_IDS[judge]
    n_rounds = LOOKUP_VOTE_ROUNDS if rounds is None else rounds
    bucket_judge = bucket_judge or judge
    items, stats = collect(bucket_judge, models)
    if limit:  # smoke test — the summary counts then describe the SUBSET, not the run
        items = items[:limit]
    sem = asyncio.Semaphore(concurrency)
    _CALLS.clear()
    limiter = RateLimiter(rpm) if rpm else None
    if limiter:
        eta = len(items) * n_rounds / limiter.rpm
        print(
            f"  pacing at {limiter.rpm:g} calls/min: "
            f"{len(items)} items x {n_rounds} rounds, "
            f"up to {eta / 60:.1f}h if nothing is cached",
            flush=True,
        )
    done = Counter()

    async def one(it: dict) -> dict:
        out = await lookup_voted(
            it["question"],
            it["claim"],
            it["_section"],
            judge_model,
            use_cache,
            sem,
            limiter,
            rounds=n_rounds,
        )
        section, norm = it.pop("_section"), None
        it.update(out)
        # Offline check: is the returned span REALLY in the section? Same matcher as
        # VCCR, so "the judge quoted the source" means the same thing in both places.
        ev = (out.get("evidence") or "").strip()
        if out.get("error"):
            it["span_tier"] = "error"
        elif not out.get("found"):
            it["span_tier"] = "n/a"
        elif not ev or ev == "NOTHING_FOUND":
            it["span_tier"] = "empty"
        else:
            norm = CRITERIA.normalize(section)
            it["span_tier"] = _classify(ev, section, norm, CRITERIA)
        done["items"] += 1
        done["errors"] += bool(it.get("error"))
        done["rounds_dropped"] += it.get("vote_rounds_dropped") or 0
        # Did the judge just hand back the quote round 1 already rejected?
        if it["span_tier"] in CRITERIA.pass_kinds:
            nq, ne = CRITERIA.normalize(it["quote"]), CRITERIA.normalize(ev)
            it["span_is_original_quote"] = nq in ne or ne in nq
        else:
            it["span_is_original_quote"] = None
        return it

    async def progress() -> None:
        """A live drop-rate readout. One throttling failure was visible in the final
        report and nowhere before it, so four hours ran with nothing on screen to say the
        run had already stopped being a measurement."""
        while True:
            await asyncio.sleep(60)
            n = done["items"] or 1
            share = done["rounds_dropped"] / (n_rounds * n)
            extra = f"  429 penalties {limiter.n_penalties}" if limiter else ""
            print(
                f"    {done['items']}/{len(items)} items  "
                f"all-rounds-lost {done['errors']}  "
                f"rounds dropped {done['rounds_dropped']} "
                f"({100 * share:.1f}%, reject above {100 * max_dropped:.1f}%)"
                f"{extra}",
                flush=True,
            )

    watcher = asyncio.create_task(progress())
    try:
        rows = await asyncio.gather(*(one(i) for i in items))
    finally:
        watcher.cancel()

    verified = [r for r in rows if r["span_tier"] in CRITERIA.pass_kinds]
    found = [r for r in rows if r.get("found")]
    return {
        "probe": "section-lookup",
        "prompt_version": PROMPT_VERSION,
        "judge": judge_model,
        "judge_settings": judge_settings(),
        # What THIS run drew, not the module default: a 1-round report must not be
        # readable as a 3-round one, and the rounds a later run adds are separate draws
        # under separate cache keys.
        "vote_rounds": n_rounds,
        # What the client was allowed to send, beside what the endpoint did about it. A
        # report with no pacing recorded cannot be told apart from one that was throttled.
        "rate_limit": limiter.stats() if limiter else None,
        # Cache hits are excluded, so this is what actually reached the endpoint. A report
        # whose rate_limit says 0 paced calls while this is nonzero was NOT paced.
        "n_api_calls": _CALLS["api"],
        "concurrency": concurrency,
        "criteria_version": CRITERIA.version,
        "verbatim_criteria": CRITERIA.version,
        "bucket_judge": bucket_judge,
        "source_reports": (
            f"eval/result/claim_support/judge-{bucket_judge}/*/report.json"
        ),
        "bucket": "unsupported_addition",
        "n_addition_pairs": stats["pairs"],
        "n_quote_not_locatable": stats["quote_not_locatable"],
        # A marker whose doc_id names no guideline in the corpus. Reported apart from
        # quote_not_locatable because it is a malformed citation, not a missing quote.
        "n_doc_id_unresolvable": stats["doc_id_unresolvable"],
        "n_duplicate_claim_node": stats["duplicate"],
        "n_judged": len(rows),
        "n_errors": sum(1 for r in rows if r.get("error")),
        # Self-consistency, over the surviving rounds of each item. The probe's own answer
        # to "would a single draw have been stable"; claim support's comparable figure is
        # judge_split_rate_first3.
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
        "n_found": len(found),
        "n_found_span_verified": len(verified),
        "found_rate": len(found) / len(rows) if rows else None,
        "found_rate_span_verified": len(verified) / len(rows) if rows else None,
        "span_tier_distribution": dict(Counter(r["span_tier"] for r in rows)),
        "n_span_is_original_quote": sum(
            1 for r in rows if r.get("span_is_original_quote")
        ),
        "round1_verdict_distribution": dict(Counter(r["round1_verdict"] for r in rows)),
        "per_gen_model": {
            gm: {
                "n": sum(1 for r in rows if r["gen_model"] == gm),
                "n_found": sum(
                    1 for r in rows if r["gen_model"] == gm and r.get("found")
                ),
                "n_found_verified": sum(
                    1
                    for r in rows
                    if r["gen_model"] == gm and r["span_tier"] in CRITERIA.pass_kinds
                ),
            }
            for gm in models
        },
        "rows": rows,
    }


def _free_path(p: Path) -> str:
    """`p`, or the first `p` with a numeric suffix that does not exist yet."""
    if not p.exists():
        return str(p)
    for i in range(2, 1000):
        cand = p.with_name(f"{p.stem}-{i}{p.suffix}")
        if not cand.exists():
            return str(cand)
    raise SystemExit(f"too many rejected reports beside {p}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--judge", default="gpt-5pt4-mini", choices=sorted(JUDGE_IDS))
    p.add_argument(
        "--bucket-judge",
        choices=sorted(JUDGE_IDS),
        help="Whose claim support run supplies the unsupported_addition bucket. Defaults "
        "to --judge. Set it to hold the bucket fixed while varying the search route.",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=list(GEN_MODELS),
        help="Generation models to triage (directory names under the judge tree).",
    )
    p.add_argument("--report", required=True)
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --report to replace an existing file (default: refuse).",
    )
    p.add_argument("--limit", type=int)
    p.add_argument("--concurrency", type=int, default=16)
    p.add_argument(
        "--rpm",
        type=float,
        help="Calls per minute, paced client-side. Defaults to "
        f"{METERED_DEFAULT_RPM:g} for a judge behind a metered host and to unpaced otherwise. "
        "Pass 0 to opt out explicitly. A concurrency cap is NOT a rate cap: see "
        "rate_limit.py.",
    )
    p.add_argument(
        "--max-dropped-fraction",
        type=float,
        default=MAX_DROPPED_ROUND_FRACTION,
        help="Refuse to write the report above this share of dropped voting rounds "
        f"(default {MAX_DROPPED_ROUND_FRACTION}). Raise it only to keep a diagnostic run.",
    )
    p.add_argument(
        "--rounds",
        type=int,
        default=None,
        help=f"Voting rounds to draw (default {LOOKUP_VOTE_ROUNDS}). Use 1 for a first "
        "number and raise it later: the round index is part of the cache key, so a "
        "3-round run after a 1-round one re-uses round 1 and pays only for rounds 2 and 3. "
        "`vote_rounds` in the report always states what that run drew.",
    )
    p.add_argument("--no-cache", action="store_true")
    a = p.parse_args()
    # Before any API call: a guard that fires after the run has already paid for it.
    assert_absent(a.report, a.overwrite)
    if a.rpm is not None and a.rpm < 0:
        raise SystemExit("--rpm must be >= 0 (0 means unpaced)")
    if not 0 <= a.max_dropped_fraction <= 1:
        raise SystemExit("--max-dropped-fraction must be a share between 0 and 1")

    # Paced by default on the metered route, because the failure mode of forgetting is a
    # silently biased report rather than an error. `--rpm 0` is how you say you mean it.
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
            a.bucket_judge,
            rpm or None,
            a.max_dropped_fraction,
            a.rounds,
        )
    )
    print(f"section-lookup ({PROMPT_VERSION})  judge {res['judge']}")
    print(
        f"  addition pairs {res['n_addition_pairs']}  "
        f"quote not locatable {res['n_quote_not_locatable']}  "
        f"dup (claim,node) {res['n_duplicate_claim_node']}  -> judged {res['n_judged']}"
    )
    print(f"  errors {res['n_errors']}")
    print(
        f"  found = {res['n_found']} ({100 * res['found_rate']:.1f}%)   "
        f"span-VERIFIED found = {res['n_found_span_verified']} "
        f"({100 * res['found_rate_span_verified']:.1f}%)"
    )
    print(f"  span tiers: {res['span_tier_distribution']}")
    print(f"  span == the original quote: {res['n_span_is_original_quote']}")
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
                f"limiter paced 0 of them, so the run was not throttled at all and its "
                f"error count says nothing about the rate ceiling. This is a wiring bug, "
                f"not a result."
            )

    total_rounds = res["n_judged"] * res["vote_rounds"]
    dropped = res["n_rounds_dropped"]
    share = dropped / total_rounds if total_rounds else 0.0
    if share > a.max_dropped_fraction:
        out = _free_path(Path(a.report).with_suffix(".rejected.json"))
        write_report(out, res, overwrite=False)
        raise SystemExit(
            f"\nREJECTED: {dropped} of {total_rounds} voting rounds were dropped "
            f"({100 * share:.1f}%, limit {100 * a.max_dropped_fraction:.1f}%).\n"
            f"  Majority voting breaks a tie toward NOT found, so this biases found_rate "
            f"DOWN and the run is not a measurement.\n"
            f"  Diagnostic report -> {out}\n"
            f"  Lower --rpm and re-run; cached rounds are kept, so a re-run only pays for "
            f"what is missing."
        )

    write_report(a.report, res, a.overwrite)
    print(f"  report -> {a.report}")


if __name__ == "__main__":
    main()
