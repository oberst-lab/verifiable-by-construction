#!/usr/bin/env python3
"""Assemble the section-lookup funnel from the read-set probe's report.

uv run python eval/recovery/funnel.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import median

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (
    str(_REPO_ROOT),
    str(_REPO_ROOT / "evaluation" / "common"),
    str(_REPO_ROOT / "evaluation" / "citation"),
    str(Path(__file__).resolve().parent),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from probe import CRITERIA  # noqa: E402  (read-only reuse)
from readset_probe import RUNGS as READSET_RUNGS  # noqa: E402
from readset_probe import section  # noqa: E402  (read-only reuse)
from scoring import write_report  # noqa: E402

RESULTS = _REPO_ROOT / "eval" / "result" / "section_lookup"
PASS = set(CRITERIA.pass_kinds)

RUNGS = READSET_RUNGS


def null_control(read: dict) -> dict:
    """What does a rejected span's similarity to its own read set actually mean?"""
    from rapidfuzz import fuzz

    def blob(read_set) -> str:
        parts = [f"[{d}]\n{section(d)[0]}" for d in read_set if section(d)[0]]
        return CRITERIA.normalize("\n\n---\n\n".join(parts))

    def score(x, read_set) -> float:
        # Min over passages: the weakest passage speaks for the row, so a row cannot look
        # well-grounded because one of its two passages was.
        return min(
            fuzz.partial_ratio(CRITERIA.normalize(s), blob(read_set))
            for s in spans_of(x)
        )

    # The 30-character floor is per row on the total, as before: a handful of very short
    # spans match anything and would dominate a median of ratios.
    rejected = [
        x
        for x in read["rows"]
        if x["span_tier"] not in PASS and sum(len(s) for s in spans_of(x)) >= 30
    ]
    passed = [
        x
        for x in read["rows"]
        if x["span_tier"] in PASS and sum(len(s) for s in spans_of(x)) >= 30
    ]
    if not rejected:
        return {}

    own, other = [], []
    for i, x in enumerate(rejected):
        own.append(score(x, x["read_set"]))
        for j in range(1, len(rejected)):
            cand = rejected[(i + j) % len(rejected)]
            if not set(cand["read_set"]) & set(x["read_set"]):
                other.append(score(x, cand["read_set"]))
                break
    ceiling = [score(x, x["read_set"]) for x in passed]

    def med(xs):
        return round(median(xs), 1) if xs else None

    return {
        "_about": "median partial_ratio of a returned span against read-set text",
        "rejected_vs_own_read_set": med(own),
        "rejected_vs_unrelated_read_set": med(other),
        "verified_vs_own_read_set": med(ceiling),
        "n_rejected": len(own),
        "n_verified": len(ceiling),
    }


def spans_of(r: dict) -> list[str]:
    """The passages a row rests on. `evidence_spans` is the verification unit from The
    `evidence` STRING is the older shape, where it held one passage.
    """
    spans = r.get("evidence_spans")
    if spans is None:
        ev = (r.get("evidence") or "").strip()
        return [ev] if ev else []
    return [x.strip() for x in spans if isinstance(x, str) and x.strip()]


def span_overlap(read: dict) -> dict:
    """How much of a `Quote's node` find is the quote the support judge already rejected?"""
    per: dict[str, dict[str, int]] = {}
    for r in read["rows"]:
        # The node rung, which is the column this breakdown qualifies: a verified span
        # that landed inside the node the citation pointed at.
        if r.get("span_tier") not in PASS or not r.get("evidence_in_node"):
            continue
        d = per.setdefault(
            r["gen_model"], {"echo": 0, "quotectx": 0, "disjoint": 0, "n": 0}
        )
        evs = [CRITERIA.normalize(x) for x in spans_of(r)]
        if r.get("quote") is None:
            raise SystemExit(
                f"{r['gen_model']}/{r.get('question_id')}: the probe report carries no "
                f"`quote` for a node find, so the span/quote overlap cannot be classified. "
                f"Re-run readset_probe.py, which records `quote`."
            )
        q = CRITERIA.normalize(r.get("quote") or "")
        if evs and q and all(x in q for x in evs):
            d["echo"] += 1
        elif evs and q and not any(x in q or q in x for x in evs):
            d["disjoint"] += 1
        elif evs and q:
            d["quotectx"] += 1
        else:
            d["disjoint"] += 1
        d["n"] += 1
    for slug, d in per.items():
        assert d["echo"] + d["quotectx"] + d["disjoint"] == d["n"], (slug, d)
    return per


def build(read_report: Path) -> dict:
    """The funnel from ONE read-set report."""
    read = json.loads(read_report.read_text())
    if read.get("probe") != "readset-lookup":
        raise SystemExit(
            f"expected a readset-lookup report, got probe={read.get('probe')!r}. A "
            f"section-lookup (node) report has the same row schema but judges only the "
            f"cited quote's node, so every rung but `node` would come out zero and the "
            f"partition assertion below would be the only thing to notice."
        )
    per: dict[str, dict[str, int]] = {}
    for slug, d in read["per_gen_model"].items():
        per[slug] = dict.fromkeys(RUNGS, 0) | {
            "cases": d["n"],
            "facelocated": d["n_found"],
            **d["rungs"],
        }

    for slug, d in per.items():
        located = sum(d[k] for k in RUNGS if k != "nowhere")
        d["located"] = located
        # The rungs must exhaust the denominator; a funnel that does not partition is a
        # bug, not a rounding artifact.
        assert located + d["nowhere"] == d["cases"], (slug, d)
        assert d["facelocated"] <= d["cases"], (slug, d)
        # Face value can only be the looser of the two: every verified find was also a
        # find the judge asserted. A violation would mean the localisation lost a row.
        assert d["facelocated"] >= located, (slug, d)

    return {
        "probe": "section-lookup-funnel",
        "null_control": null_control(read),
        "judge": read["judge"],
        "criteria_version": read["criteria_version"],
        "inputs": [read_report.name],
        "population": read["population"],
        "counts": "the rungs and `located` are span-verified: a find whose span fails "
        "the verbatim re-check counts as not found. `facelocated` is the judge's own "
        "answer with no re-check. The two bound the truth from opposite sides.",
        "rungs": list(RUNGS),
        "per_gen_model": per,
        "span_overlap": span_overlap(read),
    }


def main() -> None:
    # The output path is hardcoded, so every run targets the same file. The CLI exists
    # only to make the refusal opt-out-able, matching every other scorer.
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--report",
        default="eval/result/section_lookup/readset/judge-gpt-5pt4-mini/r3/report.json",
        help="The readset_probe report to assemble. Defaults to the 3-round run under the "
        "structured layout; point it at r1/ to build the funnel from a single-round run.",
    )
    ap.add_argument(
        "--out",
        default=str(RESULTS / "funnel.json"),
        help="Where the funnel JSON is written. Defaults beside the probe reports, which "
        "is where the table exporter looks for it.",
    )
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow the funnel JSON to be replaced (default: refuse).",
    )
    args = ap.parse_args()

    src = Path(args.report)
    if not src.is_file():
        raise SystemExit(
            f"{src} not found. Run readset_probe.py first; its default layout is "
            f"eval/result/section_lookup/readset/judge-<judge>/r<rounds>/report.json."
        )
    out = build(src)
    p = Path(args.out).resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    write_report(p, out, args.overwrite)
    # `resolve()` above, and `relative_to` guarded: an --out under another root (a scratch
    # directory, say) used to write the file and THEN raise on this line, so the script
    # exited non-zero on a run that had fully succeeded.
    try:
        shown = p.relative_to(_REPO_ROOT)
    except ValueError:
        shown = p
    print(f"wrote {shown}")
    for slug, d in out["per_gen_model"].items():
        print(
            f"  {slug:20s} "
            + "  ".join(f"{k}={d[k]}" for k in RUNGS)
            + f"  | located={d['located']} facelocated={d['facelocated']}"
        )


if __name__ == "__main__":
    main()
