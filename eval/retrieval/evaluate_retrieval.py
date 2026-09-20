#!/usr/bin/env python3
"""Stage 3 — retrieval benchmark driver.

    uv run python eval/retrieval/evaluate_retrieval.py \\
        --methods vector bm25 semantic --k 1 3 5 10 \\
        --semantic-model gpt-5-nano --embed-model text-embedding-3-small
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures as cf
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_TRACK_DIR = Path(__file__).resolve().parent  # eval/retrieval/ — holds retrievers/
_EVAL_DIR = Path(__file__).resolve().parents[1]  # eval/ — holds common/ and result/
for _p in (
    str(_TRACK_DIR),
    str(_EVAL_DIR / "common"),
):  # retrievers + usage (shared util)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv  # noqa: E402
from retrievers.base import build_candidates, hits, gt_doc_id  # noqa: E402
from release_meta import git_sha  # noqa: E402
from scoring import (  # noqa: E402
    assert_absent,
    write_report,
    write_text_guarded,
)
from rate_limit import SyncRateLimiter  # noqa: E402
from model_endpoints import needs_custom_endpoint  # noqa: E402
from usage import UsageTracker  # noqa: E402

# Self-load the repo .env so the retrievers' OpenAI()/Anthropic() clients find their
# keys without the caller pre-exporting them (matches the citation drivers). The value
# is never read here — the SDK clients pick it up from the environment.
load_dotenv(_EVAL_DIR / ".." / ".env", override=False)

logger = logging.getLogger("evaluate_retrieval")

DATASETS_DIR = (
    _EVAL_DIR
    / "releases"
    / "questions"
    / "4-guidelines-qa-222-4o-mini-balanced"
    / "data"
)
REPORTS_DIR = _EVAL_DIR / "result" / "reports"
_REPO_ROOT = _EVAL_DIR.parent


def _fmt_cost(cost: float | None) -> str:
    """`$x.xxxx`, or `n/a` when unpriced. UsageTracker returns None whenever no rate
    is recorded, and formatting None with %f/:.4f would crash."""
    return "n/a" if cost is None else f"${cost:.4f}"


def dataset_meta(paths: list[Path]) -> dict:
    """Best-effort provenance for the dataset under evaluation, so every report is
    self-identifying about which frozen set produced it (paths alone drift).
    """
    rels: list[str] = []
    releases: set[str] = set()
    for p in paths:
        try:
            rels.append(str(p.relative_to(_REPO_ROOT)))
        except ValueError:
            rels.append(str(p))
        if any(anc.name == "releases" for anc in p.parents):
            for anc in p.parents:
                if anc.name == "data":
                    releases.add(anc.parent.name)
                    break
    release: str | list[str] | None = (
        next(iter(releases)) if len(releases) == 1 else (sorted(releases) or None)
    )
    return {"release": release, "paths": sorted(rels)}


# ── data loading ──────────────────────────────────────────────────────────────


def load_records(dataset_paths: list[Path]) -> list[dict]:
    records: list[dict] = []
    for p in dataset_paths:
        with p.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def default_datasets() -> list[Path]:
    return sorted(
        p for p in DATASETS_DIR.glob("*/qa.jsonl") if p.parent.name != "bench50"
    )


# ── metric aggregation ──────────────────────────────────────────────────────────


def _rate(flags: list[bool]) -> float:
    return round(sum(flags) / len(flags), 4) if flags else 0.0


def aggregate(per_q: list[dict], ks: list[int]) -> dict:
    """per_q items: {sub@k, chap@k (dicts), n_returned}. Returns hit-rates."""
    out = {
        "n": len(per_q),
        "avg_returned": round(sum(q["n_returned"] for q in per_q) / len(per_q), 2)
        if per_q
        else 0.0,
    }
    for k in ks:
        out[f"subtopic@{k}"] = _rate([q["sub"][k] for q in per_q])
        out[f"chapter@{k}"] = _rate([q["chap"][k] for q in per_q])
    # set-hit = GT anywhere in the returned list (meaningful for selection methods)
    out["subtopic_set"] = _rate([q["sub_set"] for q in per_q])
    out["chapter_set"] = _rate([q["chap_set"] for q in per_q])
    return out


def score_question(record, ranked, cand_by_doc, ks) -> dict:
    sub, chap = {}, {}
    for k in ks:
        s, c = hits(record, ranked, cand_by_doc, k)
        sub[k], chap[k] = s, c
    s_set, c_set = hits(record, ranked, cand_by_doc, len(ranked) or 1)
    return {
        "sub": sub,
        "chap": chap,
        "sub_set": s_set,
        "chap_set": c_set,
        "n_returned": len(ranked),
    }


# ── per-method run ──────────────────────────────────────────────────────────────


def _results(scored: list[dict], ks: list[int]) -> dict:
    """Overall + by-guideline/style/chapter breakdowns from scored questions
    (each carrying `_rec`)."""

    def grouped(keyfn):
        groups = defaultdict(list)
        for sc in scored:
            groups[keyfn(sc["_rec"])].append(sc)
        return {g: aggregate(v, ks) for g, v in sorted(groups.items())}

    return {
        "overall": aggregate(scored, ks),
        "breakdowns": {
            "by_guideline": grouped(lambda r: r["guideline_id"]),
            "by_style": grouped(lambda r: r["question_style"]),
            "by_chapter": grouped(
                lambda r: f"{r['guideline_id']}:{r['source']['chapter_id']}"
            ),
        },
    }


async def _gather_ranks(retriever, records, concurrency) -> list[tuple]:
    """Async path: rank all questions in ONE event loop, bounded by a semaphore."""
    sem = asyncio.Semaphore(concurrency)
    out: list[tuple | None] = [None] * len(records)
    done = 0

    async def one(idx, rec):
        nonlocal done
        failed = None
        async with sem:
            try:
                ranked = await retriever.arank(rec["question"])
            except Exception as e:  # noqa: BLE001
                logger.warning("  ⚠️  %s rank failed: %s", rec.get("question_id"), e)
                ranked, failed = [], str(e)[:300]
        out[idx] = (rec, ranked, failed)
        done += 1
        if done % 25 == 0 or done == len(records):
            logger.info("    %s: %d/%d", retriever.name, done, len(records))

    await asyncio.gather(*(one(i, rec) for i, rec in enumerate(records)))
    return out


def run_method(
    retriever, records, cand_by_doc, ks, concurrency, reuse: dict | None = None
) -> tuple[dict, list[dict]]:
    """Rank every question (parallel), score, aggregate + breakdowns."""

    def work(rec):
        failed = None
        try:
            ranked = retriever.rank(rec["question"])
        except Exception as e:  # noqa: BLE001
            logger.warning("  ⚠️  %s rank failed: %s", rec.get("question_id"), e)
            ranked, failed = [], str(e)[:300]
        return rec, ranked, failed

    todo = [r for r in records if not (reuse and r["question_id"] in reuse)]
    if reuse:
        logger.info(
            "    resume: %d of %d questions reused, %d to re-rank",
            len(records) - len(todo),
            len(records),
            len(todo),
        )
    if hasattr(retriever, "arank"):
        fresh = asyncio.run(_gather_ranks(retriever, todo, concurrency))
    else:
        with cf.ThreadPoolExecutor(max_workers=concurrency) as ex:
            fresh = list(ex.map(work, todo))
    by_id = {r["question_id"]: (r, ranked, failed) for r, ranked, failed in fresh}
    pairs = [
        by_id.get(r["question_id"], (r, (reuse or {}).get(r["question_id"], []), None))
        for r in records
    ]

    scored: list[dict] = []
    detail: list[dict] = []
    failures: list[dict] = []
    n_empty = 0
    for rec, ranked, failed in pairs:
        if failed is not None:
            failures.append({"question_id": rec.get("question_id"), "error": failed})
        elif not ranked:
            n_empty += 1
        sc = score_question(rec, ranked, cand_by_doc, ks)
        sc["_rec"] = rec
        scored.append(sc)
        detail.append(
            {
                "question_id": rec["question_id"],
                "ranked_top": ranked[: max(ks)],
                "gt": gt_doc_id(rec),
                "n_returned": len(ranked),
            }
        )
    res = _results(scored, ks)
    # In the method's own block, beside its rates, so a consumer reading one method cannot
    # miss it. Always present, so absent means the run predates this and is unknown rather
    # than clean.
    res["n_rank_failed"] = len(failures)
    res["rank_failures"] = failures[:20]
    res["n_empty_selection"] = n_empty
    if n_empty:
        logger.warning(
            "  %s: %d of %d questions returned an EMPTY selection (no exception). "
            "These score as misses; check whether the model declined the tool.",
            retriever.name,
            n_empty,
            len(records),
        )
    if failures:
        logger.error(
            "  ❌ %s: %d of %d questions FAILED to rank and were scored as misses. "
            "This report is not a measurement.",
            retriever.name,
            len(failures),
            len(records),
        )
    return res, detail


# ── reporting ──────────────────────────────────────────────────────────────────


def _free_dir(d: Path) -> Path:
    """`d`, or the first numbered sibling that does not exist. The expected sequence after
    a rejection is "fix and re-run", so a fixed name would have the second rejection
    destroy the first one's diagnostic, and these directories are untracked."""
    if not d.exists():
        return d
    for i in range(2, 1000):
        cand = d.with_name(f"{d.name}-{i}")
        if not cand.exists():
            return cand
    raise SystemExit(f"too many rejected reports beside {d}")


def summary_md(report: dict, ks: list[int]) -> str:
    lines = ["# Retrieval benchmark — summary", ""]
    m = report.get("meta", {})
    if m:
        ds = m.get("dataset", {})
        models = m.get("models", {})
        lines += [
            f"- **run:** {m.get('run_at', '?')}  (tag `{m.get('run_tag', '?')}`)",
            f"- **dataset:** `{ds.get('release', '?')}` — "
            f"{ds.get('n_questions', '?')} questions across "
            f"{len(ds.get('guidelines', []))} guideline(s)",
            f"- **models:** embed={models.get('embed', '?')} · "
            f"semantic={models.get('semantic', '?')}",
            "",
        ]
    hdr = (
        "| method | n | avg ret | "
        + " | ".join(f"sub@{k}" for k in ks)
        + " | "
        + " | ".join(f"chap@{k}" for k in ks)
        + " | $ |"
    )
    sep = "|" + "---|" * (3 + 2 * len(ks) + 1)
    lines += ["## Overall", "", hdr, sep]
    for m, r in report["methods"].items():
        o = r["overall"]
        row = (
            f"| {m} | {o['n']} | {o['avg_returned']} | "
            + " | ".join(f"{o[f'subtopic@{k}']:.3f}" for k in ks)
            + " | "
            + " | ".join(f"{o[f'chapter@{k}']:.3f}" for k in ks)
            + f" | {_fmt_cost(report['cost'].get(m))} |"
        )
        lines.append(row)
    lines += [
        "",
        f"_set-hit (GT anywhere in returned set) reported in report.json; "
        f"k set = {ks}; candidate pool = {report['n_candidates']} units._",
    ]
    return "\n".join(lines) + "\n"


# ── main ────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Retrieval benchmark (Stage 3).")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["vector", "bm25", "semantic"],
        choices=["vector", "bm25", "semantic"],
        help="Which methods to run.",
    )
    p.add_argument(
        "--datasets",
        nargs="+",
        help="qa.jsonl paths (default: all under result/datasets).",
    )
    p.add_argument("--k", nargs="+", type=int, default=[1, 3, 5, 10])
    p.add_argument("--embed-model", default="text-embedding-3-small")
    p.add_argument(
        "--semantic-model",
        default="gpt-5.4-mini",
        help="Bare model id for the semantic selector (no provider: prefix).",
    )
    p.add_argument(
        "--semantic-temperature",
        default="0",
        help="Temperature regime for the semantic selector: a float (pinned, e.g. "
        "'0' = the greedy temp-0 regime) or 'default' to omit the param and run the "
        "provider-native default (the temp-default regime, matching production). "
        "Reasoning / Claude-5 models reject the param and run provider-native either way.",
    )
    p.add_argument(
        "--rpm",
        type=float,
        help="Calls per minute for the semantic selector, paced client-side. Defaults to "
        "25 for a model behind a metered host and to unpaced otherwise. A CONCURRENCY "
        "cap is not a rate cap: --concurrency 8 on a few-second call offers about "
        "95/min, several times what a metered host admits. Pass 0 to opt out.",
    )
    p.add_argument(
        "--resume",
        help="Path to a `.rejected` report directory. Its successfully-ranked questions "
        "are reused and only the ones listed in `rank_failures` are asked again, so a "
        "round is repaired for the cost of its failures instead of re-run in full.",
    )
    p.add_argument(
        "--allow-rank-failures",
        action="store_true",
        help="Write the report even when questions failed to rank. Off by default, "
        "because a failed ranking scores as a miss and quietly lowers the hit rate.",
    )
    p.add_argument(
        "--semantic-reasoning-effort",
        default="default",
        help="Reasoning effort for the semantic selector. 'default' sends nothing (the "
        "vendor's own default, which for several vendors is thinking ON); 'none' "
        "turns thinking OFF on the vendors that honour the field, matching the "
        "interactive retriever, which runs Thinking(effort=False). Sent verbatim "
        "otherwise. NOT effective for a host that ignores this field and needs "
        "chat_template_kwargs; see retrievers/base.sampling_kwargs.",
    )
    p.add_argument(
        "--semantic-seed",
        type=int,
        default=None,
        help="Sampling seed for the semantic selector, sent only on the "
        "OpenAI-compatible path and only when given. REQUIRED, with a different value "
        "per round, for any model behind a caching host: such a host keys on the "
        "request body, so repeating a round otherwise replays the stored response and "
        "the rounds are one sample rather than three. Measured once: an identical "
        "request returned 11x faster and three rounds were byte-identical. "
        "Cache-Control is ignored; seed is honoured. Omit for a natively routed model "
        "so its existing runs stay reproducible.",
    )
    p.add_argument(
        "--semantic-max-tokens",
        type=int,
        default=8000,
        help="Token budget for the semantic selector. Must cover BOTH the hidden "
        "reasoning and the tool-call output on reasoning models (gpt-5/o-series); "
        "1200 was too tight and truncated their selection. Non-reasoning models are "
        "budget-invariant (they emit a short tool call regardless).",
    )
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--limit", type=int, help="Cap questions (smoke test).")
    p.add_argument(
        "--timestamp",
        default="manual",
        help="Tag for the report dir (no clock in-process).",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow the report dir's files to be replaced. Without it an existing "
        "report is left alone and the run refuses to start (archive, do not overwrite). "
        "The default --timestamp is a fixed tag, so back-to-back runs collide by design.",
    )
    p.add_argument(
        "--round",
        type=int,
        default=None,
        help="Round index for repeated stochastic runs (LLM methods). Recorded in "
        "meta so the three rounds of a model are self-identifying.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Build pool + print plan; no API, no scoring.",
    )
    args = p.parse_args()
    if args.rpm is not None:
        if args.rpm < 0:
            raise SystemExit("--rpm must be >= 0 (use --rpm 0 for an unpaced run)")
        if "semantic" not in args.methods:
            raise SystemExit(
                "--rpm paces the SEMANTIC selector only, and no semantic method was "
                "requested. Refusing rather than ignoring it."
            )
    return args


def main() -> None:
    args = parse_args()
    # Computed once and reused at the write site: a guard checking a path nothing writes
    # is worse than no guard, because it reads as protection.
    out_dir = REPORTS_DIR / f"retrieval_{args.timestamp}"
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    dataset_paths = (
        [Path(p) for p in args.datasets] if args.datasets else default_datasets()
    )
    records = load_records(dataset_paths)
    if args.limit:
        records = records[: args.limit]
    guideline_ids = sorted({r["guideline_id"] for r in records})
    logger.info(
        "📥 %d questions from %d guideline(s): %s",
        len(records),
        len(guideline_ids),
        ", ".join(guideline_ids),
    )

    logger.info("🧱 building candidate pool …")
    candidates = build_candidates(guideline_ids)
    cand_by_doc = {c.doc_id: c for c in candidates}
    logger.info(
        "🧱 %d candidate units across %d guideline(s)",
        len(candidates),
        len(guideline_ids),
    )

    # Sanity: every GT must be in the pool, else it can never be hit.
    missing = sorted({gt_doc_id(r) for r in records} - set(cand_by_doc))
    if missing:
        logger.warning(
            "⚠️  %d GT units NOT in candidate pool (will always miss): %s",
            len(missing),
            ", ".join(missing[:8]),
        )

    logger.info(
        "📝 plan: methods=%s · k=%s · embed=%s · semantic=%s · concurrency=%d",
        args.methods,
        args.k,
        args.embed_model,
        args.semantic_model,
        args.concurrency,
    )
    if args.dry_run:
        logger.info("✅ dry-run: pool built, no API calls made.")
        return

    # Before any API call, and after the dry-run return: a dry run writes nothing, so
    # refusing one over an existing report would be a false alarm. The default
    # --timestamp is a constant, so the second real run of the day collides by design.
    for _name in ("report.json", "summary.md"):
        assert_absent(out_dir / _name, args.overwrite)

    # Load the reusable rankings BEFORE any client is built, so a bad --resume path fails
    # before the run spends anything.
    reuse: dict[str, dict] = {}
    if args.resume:
        prev = Path(args.resume)
        prev_report = prev / "report.json" if prev.is_dir() else prev
        if not prev_report.is_file():
            raise SystemExit(f"--resume: {prev_report} not found")
        pr = json.loads(prev_report.read_text(encoding="utf-8"))
        for name, block in pr.get("methods", {}).items():
            failed = {f["question_id"] for f in block.get("rank_failures", [])}
            det = pr.get("details", {}).get(name) or []
            # A question is reusable when it did not fail. An EMPTY selection is reused
            # too: it is the model's answer, not a failure, and re-asking it would quietly
            # resample one arm of the run while leaving the rest fixed.
            reuse[name] = {
                d["question_id"]: d.get("ranked_top") or []
                for d in det
                if d["question_id"] not in failed
            }
            logger.info(
                "  --resume %s: %d reusable, %d failed",
                name,
                len(reuse[name]),
                len(failed),
            )

    from retrievers.bm25 import BM25Retriever
    from retrievers.vector import VectorRetriever
    from retrievers.semantic import SemanticRetriever

    # Temperature regime, parsed once: "default"/"none" → None (omit the param, run
    # provider-native); otherwise a pinned float. Recorded in meta so a leaf report is
    # self-describing (the earlier gap: temperature was applied but never recorded).
    sem_temp: float | None = (
        None
        if str(args.semantic_temperature).lower() in {"default", "none"}
        else float(args.semantic_temperature)
    )
    sem_effort = (
        None
        if str(args.semantic_reasoning_effort).lower() in {"default", ""}
        else str(args.semantic_reasoning_effort)
    )

    report = {
        # Provenance block: everything needed to identify this run after the fact
        # (which dataset, when, which models). --timestamp is the human folder tag;
        # run_at is the real wall clock.
        "meta": {
            "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "git_commit": git_sha(),
            "run_tag": args.timestamp,
            "round": args.round,
            "dataset": {
                **dataset_meta(dataset_paths),
                "n_questions": len(records),
                "guidelines": guideline_ids,
            },
            "methods": args.methods,
            "k": args.k,
            "models": {
                "embed": args.embed_model,
                "semantic": args.semantic_model,
                "semantic_max_tokens": args.semantic_max_tokens,
            },
            "n_candidates": len(candidates),
            "limit": args.limit,
            "concurrency": args.concurrency,
        },
        "n_candidates": len(candidates),
        "guidelines": guideline_ids,
        "k": args.k,
        "methods": {},
        "cost": {},
        "details": {},
    }

    def _log_overall(label, res, cost):
        o = res["overall"]
        kk = args.k[-1]
        logger.info(
            "   %s — sub@%d=%.3f chap@%d=%.3f avg_ret=%.1f %s",
            label,
            kk,
            o[f"subtopic@{kk}"],
            kk,
            o[f"chapter@{kk}"],
            o["avg_returned"],
            _fmt_cost(cost),
        )

    for m in args.methods:
        usage = UsageTracker()
        logger.info("▶️  method: %s", m)
        if m == "bm25":
            retriever = BM25Retriever(candidates)
            # Preprocessing and k1/b change the numbers by several points, so they
            # belong in the provenance block, not only in the source file.
            report["meta"]["bm25"] = retriever.config
        elif m == "vector":
            retriever = VectorRetriever(candidates, model=args.embed_model, usage=usage)
        elif m == "semantic":
            rpm = args.rpm
            if rpm is None:
                rpm = 25.0 if needs_custom_endpoint(args.semantic_model) else 0.0
            # client_max_retries=0 mirrors what the retriever sets on the paced
            # client, so the report's pacing block discloses that no retry sat
            # below the pacer to absorb a 429 uncounted.
            sem_limiter = SyncRateLimiter(rpm, client_max_retries=0) if rpm else None
            if sem_limiter:
                logger.info("    pacing semantic at %g calls/min", sem_limiter.rpm)
            retriever = SemanticRetriever(
                candidates,
                model=args.semantic_model,
                max_tokens=args.semantic_max_tokens,
                temperature=sem_temp,
                reasoning_effort=sem_effort,
                seed=args.semantic_seed,
                usage=usage,
                limiter=sem_limiter,
            )
            # null = no seed sent. A model behind a caching host with null here
            # cannot be trusted to have produced independent rounds; see
            # --semantic-seed.
            report["meta"]["models"]["semantic_seed"] = args.semantic_seed
            # Record how the selector tool was elicited (forced vs auto) — a
            # comparability caveat for OpenAI-compatible vendors (see SemanticRetriever).
            report["meta"]["models"]["semantic_tool_choice"] = (
                retriever.tool_choice_mode
            )
            # Record the temperature that ACTUALLY reached the API (not the requested
            # value): reasoning / Claude-5 models omit it and record "default" even when
            # a float was passed. "default" == provider-native (temp omitted).
            report["meta"]["models"]["semantic_temperature"] = (
                "default"
                if retriever.effective_temperature is None
                else retriever.effective_temperature
            )
            # Reasoning effort that actually applied ("off" for the thinking-free
            # Anthropic selector, an effort string for a pinned reasoning model, else
            # "default" = provider-native). Completes the sampling-config provenance.
            report["meta"]["models"]["semantic_reasoning_effort"] = (
                retriever.effective_reasoning_effort
            )
        else:
            continue
        res, detail = run_method(
            retriever,
            records,
            cand_by_doc,
            args.k,
            args.concurrency,
            reuse=reuse.get(m),
        )
        if m == "semantic":
            report["meta"]["models"]["semantic_rate_limit"] = (
                sem_limiter.stats() if sem_limiter else None
            )
            report["meta"]["models"]["semantic_served_model"] = retriever.served_model
            report["meta"]["models"]["semantic_system_fingerprint"] = getattr(
                retriever, "system_fingerprint", None
            )
            report["meta"]["models"]["semantic_reasoning_observed"] = (
                retriever.observed_reasoning
            )
        cost = usage.as_dict()["total_cost_usd"]
        report["methods"][m] = res
        report["cost"][m] = cost
        report["details"][m] = detail
        usage.log_summary()
        _log_overall(m, res, cost)

    failed = {
        name: res["n_rank_failed"]
        for name, res in report["methods"].items()
        if res.get("n_rank_failed")
    }
    if failed and not args.allow_rank_failures:
        rej = _free_dir(out_dir.with_name(out_dir.name + ".rejected"))
        write_report(rej / "report.json", report, overwrite=False)
        write_text_guarded(rej / "summary.md", summary_md(report, args.k), False)
        raise SystemExit(
            f"\nREJECTED: questions failed to rank and were scored as misses: "
            f"{failed}. The hit rate is biased down and this is not a measurement.\n"
            f"  Diagnostic report -> {rej}\n"
            f"  Fix the cause and re-run, or pass --allow-rank-failures to keep a run "
            f"whose failures you have accounted for."
        )

    # Through the helper so the guard is re-checked here too: `eval/result/reports/`
    # is untracked, and two runs sharing a --timestamp both clear the early check.
    write_report(out_dir / "report.json", report, args.overwrite)
    write_text_guarded(
        out_dir / "summary.md", summary_md(report, args.k), args.overwrite
    )
    logger.info("✅ report → %s", out_dir)
    print("\n" + summary_md(report, args.k))


if __name__ == "__main__":
    main()
