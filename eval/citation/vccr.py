#!/usr/bin/env python3
"""VCCR — Verbatim Compliance Rate; NCR = 1 - VCCR.

    uv run python eval/citation/vccr.py \\
        --answers eval/result/answers/smoke_bp.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from datetime import datetime
from pathlib import Path

# eval/citation/ → repo root is two up; the system package is imported from the root.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = Path(__file__).resolve().parents[1]
for _p in (
    str(_REPO_ROOT),
    str(_REPO_ROOT / "data" / "corpus"),
    str(_EVAL_DIR / "common"),
):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from release_meta import git_sha, rel_to_repo  # noqa: E402
from scoring import assert_absent, fmt_pct, load_jsonl, write_report  # noqa: E402
from system.guidelines import section_full_text  # noqa: E402

_CRITERIA_PATH = Path(__file__).parent / "verbatim_criteria.json"


class Criteria:
    """The frozen verbatim criteria, loaded from verbatim_criteria.json."""

    def __init__(self, cfg: dict):
        self.version: str = cfg["version"]
        self.pass_kinds: set[str] = set(cfg["pass_kinds"])
        # The exact tier removes the quotation marks a model wraps around its quote, and
        # nothing else. See the file's `exact._note`: the boundary is a category (outside
        # the span) rather than a threshold, which is what keeps it from being widened.
        self.wrapping_quotes: str = cfg.get("exact", {}).get(
            "strip_wrapping_quotes", ""
        )
        n = cfg["normalization"]
        self.nfkc: bool = n.get("nfkc", True)
        self.lowercase: bool = n.get("lowercase", True)
        self.fold: dict[str, str] = n.get("fold", {})
        fm = n.get("strip_footnote_markers")
        self.footnote_re = re.compile(fm) if fm else None
        self.collapse_ws: bool = n.get("collapse_whitespace", True)
        slb = n.get("strip_list_bullets")
        self.list_bullet_re = re.compile(slb) if slb else None
        sbp = n.get("strip_space_before_punct", "")
        self.space_punct_re = (
            re.compile(r"\s+([" + re.escape(sbp) + r"])") if sbp else None
        )
        self.apparatus: dict[str, str] = {
            name: r["pattern"]
            for name, r in cfg.get("apparatus", {}).get("rules", {}).items()
        }
        self.apparatus_re = (
            re.compile("|".join(self.apparatus.values())) if self.apparatus else None
        )
        # Punctuation at the excerpt's own edges, stripped from the QUOTE only and never
        # from the section: the excerpt is where the boundary is, so it is the only side
        # whose edge punctuation is the quoter's choice rather than the guideline's.
        b = cfg.get("boundary", {})
        self.boundary_punct: str = b.get("punctuation", "")
        self.boundary_keeps_ellipsis: bool = b.get("never_strip_ellipsis", False)
        e = cfg["elided"]
        self.ellipsis_re = re.compile(e["ellipsis_pattern"])
        self.min_fragment_len: int = e["min_fragment_len"]
        self.min_fragments: int = e["min_fragments"]
        self.tier_order: list[str] = cfg["tiers"]["_order"]

    def normalize(self, s: str) -> str:
        if self.nfkc:
            s = unicodedata.normalize("NFKC", s)
        if self.lowercase:
            s = s.lower()
        for a, b in self.fold.items():
            s = s.replace(a, b)
        if self.footnote_re is not None:
            s = self.footnote_re.sub("", s)
        # Before the collapse, because the pattern is anchored to a line start and the
        # collapse destroys line starts.
        if self.list_bullet_re is not None:
            s = self.list_bullet_re.sub("", s)
        if self.collapse_ws:
            s = re.sub(r"\s+", " ", s)
        if self.apparatus_re is not None:
            s = re.sub(r"\s+", " ", self.apparatus_re.sub(" ", s))
        # After the deletions, because it exists to undo the space they leave behind.
        if self.space_punct_re is not None:
            s = self.space_punct_re.sub(r"\1", s)
        return s.strip()


def load_criteria(path: Path = _CRITERIA_PATH) -> Criteria:
    return Criteria(json.loads(Path(path).read_text(encoding="utf-8")))


def _match_elided(nq: str, ns: str, cr: Criteria) -> bool:
    frags = [p.strip() for p in cr.ellipsis_re.split(nq) if p.strip()]
    long = [f for f in frags if len(f) >= cr.min_fragment_len]
    if len(long) < cr.min_fragments:
        return False
    pos = 0
    for f in long:
        i = ns.find(f, pos)
        if i == -1:
            return False
        pos = i + len(f)
    return True


def _classify(quote: str, raw: str, ns: str, cr: Criteria) -> str:
    """Classify a quote against a section's raw + pre-normalized text (best wins).
    Splitting out the pre-normalized source lets callers normalize each section
    once instead of once per quote."""
    if not quote or not raw:
        return "fail"
    if quote in raw:
        return "exact"
    if cr.wrapping_quotes:
        unwrapped = quote.strip(cr.wrapping_quotes + " ")
        if unwrapped and unwrapped != quote and unwrapped in raw:
            return "exact"
    nq = cr.normalize(quote).strip("'\"")
    if nq and nq in ns:
        return "normalized"
    if cr.boundary_punct:
        chars = cr.boundary_punct + " "
        nb = nq.strip(chars)
        if nb and nb != nq:
            removed = (
                nq[: len(nq) - len(nq.lstrip(chars))] + nq[len(nq.rstrip(chars)) :]
            )
            if not (cr.boundary_keeps_ellipsis and cr.ellipsis_re.search(removed)):
                if nb in ns:
                    return "normalized"
    if nq and _match_elided(nq, ns, cr):
        return "elided"
    return "fail"


def classify(quote: str, source_text: str, cr: Criteria) -> str:
    """Classify one quote against one section's full text into a tier (best wins).
    Normalizes the source per call — for batch scoring prefer the cached path in
    score_answers; this wrapper stays for one-off / unit use."""
    ns = cr.normalize(source_text) if source_text else ""
    return _classify(quote, source_text, ns, cr)


def tier_for_citation(citation: dict, cr: Criteria, cache: dict | None = None) -> str:
    """Resolve the cited section and classify the quote. `unresolved` (corpus
    can't find the section) is kept distinct from `fail` (quote not in section).
    `cache` memoizes (raw, normalized) text per (guideline, section), so a section
    cited N times is fetched + normalized once rather than N times."""
    gid, sid = citation["guideline_id"], citation["section_id"]
    if cache is not None and (gid, sid) in cache:
        raw, ns = cache[(gid, sid)]
    else:
        raw = section_full_text(gid, sid)
        ns = cr.normalize(raw) if raw is not None else None
        if cache is not None:
            cache[(gid, sid)] = (raw, ns)
    if raw is None:
        return "unresolved"
    return _classify(citation["quote"], raw, ns, cr)


class Tiers:
    """Verbatim tier per (node, quote), memoized. One definition, shared by every consumer that
    needs to ask "is this quote really in the guideline": the claim funnel's compliance
    stage and the claim support quote gate. It lives here beside the frozen criteria it
    uses, so those consumers cannot drift apart from the standalone metric or from each
    other.
    """

    def __init__(self):
        self.cr = load_criteria()
        self._sec: dict[tuple[str, str], tuple[str | None, str | None]] = {}

    def of(self, node: str, quote: str) -> str:
        gid, _, sid = node.partition(":")
        if (gid, sid) not in self._sec:
            raw = section_full_text(gid, sid)
            self._sec[(gid, sid)] = (raw, self.cr.normalize(raw) if raw else None)
        raw, ns = self._sec[(gid, sid)]
        if raw is None:
            return "unresolved"
        return _classify(quote, raw, ns, self.cr)

    def passes(self, tier: str) -> bool:
        return tier in self.cr.pass_kinds


def score_answers(records: list[dict], cr: Criteria) -> dict:
    """Score every citation across all answer records → VCCR/NCR + tier histogram."""
    tiers = Counter()
    per_record = []
    total = 0
    passed = 0
    cache: dict = {}  # (guideline, section) → (raw, normalized) text, fetched once
    for r in records:
        cites = r.get("citations", [])
        r_total = len(cites)
        r_pass = 0
        r_tiers = Counter()
        for c in cites:
            tier = tier_for_citation(c, cr, cache)
            tiers[tier] += 1
            r_tiers[tier] += 1
            total += 1
            if tier in cr.pass_kinds:
                passed += 1
                r_pass += 1
        per_record.append(
            {
                "question_id": r.get("question_id"),
                "n_citations": r_total,
                "n_pass": r_pass,
                "vccr": (r_pass / r_total) if r_total else None,
                "tiers": dict(r_tiers),
            }
        )
    vccr = (passed / total) if total else None
    return {
        "criteria_version": cr.version,
        "pass_kinds": sorted(cr.pass_kinds),
        "n_answers": len(records),
        "n_citations": total,
        "n_pass": passed,
        "vccr": vccr,
        "ncr": (1 - vccr) if vccr is not None else None,
        "tiers": dict(tiers),
        "per_record": per_record,
    }


def build_meta(answers_path: Path) -> dict:
    """Provenance block so a leaf report.json is self-describing (matches the
    report convention). VCCR is deterministic, so judge_model is
    always None. Records the parent answers release (read best-effort from the
    sibling MANIFEST.yaml), the git sha, and the run time."""
    release = None
    manifest = answers_path.parent / "MANIFEST.yaml"
    if manifest.exists():
        try:
            import yaml

            release = (yaml.safe_load(manifest.read_text()) or {}).get("name")
        except Exception:  # noqa: BLE001 — provenance is best-effort, never fatal
            release = None
    return {
        "metric": "verbatim_compliance",
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_sha(),
        "judge_model": None,
        "answers": rel_to_repo(answers_path),
        "answers_release": release,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Score VCCR/NCR over a harness answer JSONL."
    )
    p.add_argument("--answers", required=True, help="Harness answers.jsonl path.")
    p.add_argument("--report", help="Optional path to write the full JSON report.")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --report to replace an existing file. Without it an existing "
        "report is left alone and the run refuses to start (archive, do not overwrite).",
    )
    p.add_argument("--criteria", help="Override verbatim_criteria.json path.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Before any API call: a guard that fires after the run has already paid for it.
    assert_absent(args.report, args.overwrite)
    answers_path = Path(args.answers)
    if not answers_path.exists():
        raise SystemExit(f"{answers_path} not found")
    cr = load_criteria(Path(args.criteria)) if args.criteria else load_criteria()

    records = load_jsonl(answers_path)
    result = {"meta": build_meta(answers_path), **score_answers(records, cr)}

    print(
        f"VCCR report — criteria {result['criteria_version']}  pass_kinds={result['pass_kinds']}"
    )
    print(f"  answers={result['n_answers']}  citations={result['n_citations']}")
    print(f"  VCCR = {fmt_pct(result['vccr'])}   NCR = {fmt_pct(result['ncr'])}")
    print("  tier histogram:")
    for tier in cr.tier_order:
        n = result["tiers"].get(tier, 0)
        if n:
            mark = "PASS" if tier in cr.pass_kinds else "fail"
            print(f"    {tier:12} {n:4}  [{mark}]")

    if args.report:
        write_report(args.report, result, args.overwrite)
        print(f"  full report → {args.report}")


if __name__ == "__main__":
    main()
