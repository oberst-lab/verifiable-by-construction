#!/usr/bin/env python3
"""Quote length — a descriptive profile of HOW a model cites, not how well.

    uv run python eval/citation/quote_length.py \\
        --answers eval/result/answers/<release>/<model>/answers.jsonl \\
        --report eval/result/quote_length/<model>/report.json
"""

from __future__ import annotations

import argparse
import statistics as st
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_EVAL_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.release_meta import git_sha, rel_to_repo  # noqa: E402
from common.scoring import assert_absent, load_jsonl, write_report  # noqa: E402

BANDS: tuple[tuple[int, str], ...] = (
    (100, "1-100"),
    (200, "101-200"),
    (300, "201-300"),
    (400, "301-400"),
    (10**9, "401+"),
)
BAND_LABELS: tuple[str, ...] = tuple(label for _, label in BANDS)


def band_of(n: int) -> str:
    """The band a length falls in. Bands are tried in order, first fit wins."""
    return next(label for upper, label in BANDS if n <= upper)


def _pct(sorted_vals: list[int], q: float) -> int:
    """Nearest-rank percentile. Deliberately not interpolated: the value reported is
    a length some real quote actually had, which is what a reader will look for."""
    return sorted_vals[min(len(sorted_vals) - 1, int(q * len(sorted_vals)))]


def score_answers(records: list[dict]) -> dict:
    """Quote-length profile over every citation in every answer record."""
    lengths: list[int] = []
    bands: Counter = Counter()
    per_record = []
    for r in records:
        rec_lengths = [len(c["quote"]) for c in r.get("citations", [])]
        for n in rec_lengths:
            lengths.append(n)
            bands[band_of(n)] += 1
        per_record.append(
            {
                "question_id": r.get("question_id"),
                "n_citations": len(rec_lengths),
                "median_chars": st.median(rec_lengths) if rec_lengths else None,
                "max_chars": max(rec_lengths) if rec_lengths else None,
            }
        )
    s = sorted(lengths)
    return {
        "unit": "characters of the quote as emitted, before normalization",
        "bands": [{"label": label, "max_chars": upper} for upper, label in BANDS],
        "n_answers": len(records),
        "n_citations": len(s),
        # None rather than 0 for an empty set: a model that emitted no citation at all
        # has no median, and a 0 here would be read as "its quotes are zero characters".
        "median_chars": st.median(s) if s else None,
        "mean_chars": st.mean(s) if s else None,
        "p90_chars": _pct(s, 0.90) if s else None,
        "p99_chars": _pct(s, 0.99) if s else None,
        "min_chars": s[0] if s else None,
        "max_chars": s[-1] if s else None,
        "band_counts": {label: bands.get(label, 0) for label in BAND_LABELS},
        "band_shares": (
            {label: bands.get(label, 0) / len(s) for label in BAND_LABELS} if s else {}
        ),
        "per_record": per_record,
    }


def build_meta(answers_path: Path) -> dict:
    """Provenance so a leaf report.json is self-describing, matching the
    eval/result/ convention. No judge is involved, so judge_model is None."""
    release = None
    manifest = answers_path.parent / "MANIFEST.yaml"
    if manifest.exists():
        try:
            import yaml

            release = (yaml.safe_load(manifest.read_text()) or {}).get("name")
        except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
            release = None
    return {
        "metric": "quote_length",
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_sha(),
        "judge_model": None,
        "answers": rel_to_repo(answers_path),
        "answers_release": release,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile citation quote lengths over a harness answer JSONL."
    )
    p.add_argument("--answers", required=True, help="Harness answers.jsonl path.")
    p.add_argument("--report", help="Optional path to write the full JSON report.")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --report to replace an existing file. Without it an existing "
        "report is left alone and the run refuses to start (archive, do not overwrite).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert_absent(args.report, args.overwrite)
    answers_path = Path(args.answers)
    if not answers_path.exists():
        raise SystemExit(f"{answers_path} not found")

    result = {
        "meta": build_meta(answers_path),
        **score_answers(load_jsonl(answers_path)),
    }

    n = result["n_citations"]
    print(f"quote length — {result['n_answers']} answers, {n} citations")
    if n:
        print(
            f"  median {result['median_chars']:.0f}  mean {result['mean_chars']:.1f}  "
            f"p90 {result['p90_chars']}  p99 {result['p99_chars']}  max {result['max_chars']}"
        )
        print("  bands (chars):")
        for label in BAND_LABELS:
            c = result["band_counts"][label]
            print(f"    {label:9} {c:5}  {100 * result['band_shares'][label]:5.1f}%")

    if args.report:
        write_report(args.report, result, args.overwrite)
        print(f"  wrote {args.report}")


if __name__ == "__main__":
    main()
