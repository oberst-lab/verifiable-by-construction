#!/usr/bin/env python3
"""The claim funnel — every citation metric on ONE unit, the claim.

    uv run python eval/claim_funnel/funnel.py \\
        --answers eval/result/answers/<release>/<model>/answers.jsonl \\
        --claim-support eval/result/claim_support/judge-<j>/<model>/report.json \\
        --report eval/result/claim_funnel/judge-<j>/<model>/report.json
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_EVAL_DIR), str(_EVAL_DIR / "common")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from citation.vccr import Tiers  # noqa: E402  one definition, shared with the gate
from common.release_meta import git_sha, model_slug, rel_to_repo  # noqa: E402
from common.scoring import assert_absent, write_report  # noqa: E402

LENIENT = ("fully_supported", "partially_supported")


def _marker_indices(pair: dict) -> list[int]:
    """The marker indices a pair covers. Stored as a list under set adjudication and as
    a bare index otherwise; a JSON round trip can leave either as a string."""
    mk = pair["marker"]
    if isinstance(mk, str):
        mk = ast.literal_eval(mk)
    return list(mk) if isinstance(mk, (list, tuple)) else [mk]


_REQUIRED_ADJUDICATION = "set"

# Every per-record row carries every key, present or zero.
_RECORD_KEYS = (
    "sentences",
    "claims",
    "cited",
    "compliant",
    "certified",
    "certified_lenient",
    "compliant_citations_on_claims",
)

# Added only under --lookup; see the Lookup class and the module docstring.
_RECOVERY_KEYS = (
    "recovery_probed",
    "recovered",
    "recovered_verified",
    "recovery_bucket_unprobed",
)


class Lookup:
    """The section-lookup probe's verdicts, keyed the way a claim is identified here:
    (question_id, claim text). `probed` is every row the probe drew for this generation
    model, `verified` the subset whose returned span passed the verbatim matcher offline.
    Not-found rows carry span_tier "n/a", so the pass-tier test already implies found."""

    _MAX_DROPPED_ROUND_FRACTION = 0.01

    def __init__(self, report: dict, gen_model: str, pass_kinds, cs_judge: str | None):
        if report.get("probe") != "readset-lookup":
            raise SystemExit(
                f"--lookup expects a readset-lookup report, got "
                f"probe={report.get('probe')!r}. A section-lookup (node) report has the "
                f"same row schema but judges only the cited quote's node, so joining it "
                f"would make `supported_claims_rate` mean 'evidence beside the quote' "
                f"instead of 'evidence anywhere the model read', silently and with no "
                f"error. The node probe is retired."
            )
        self._check_health(report)
        # The bucket must come from the same judge whose verdicts the funnel folds, or the
        # recovery branch is drawn from one judge's `unsupported_addition` set while
        # `certified` comes from another's.
        bucket_judge = report.get("bucket_judge")
        if (
            cs_judge
            and bucket_judge
            and model_slug(bucket_judge) != model_slug(cs_judge)
        ):
            raise SystemExit(
                f"the lookup report drew its unsupported_addition bucket from judge "
                f"{bucket_judge!r}, but claim support here is {cs_judge!r}. Recovery would "
                f"be measured against a different judge's shortfall set than `certified`."
            )
        rows = [r for r in report["rows"] if r["gen_model"] == gen_model]
        if not rows and gen_model not in (report.get("per_gen_model") or {}):
            raise SystemExit(
                f"the lookup report covers {sorted(report.get('per_gen_model') or {})} "
                f"and does not mention gen_model {gen_model!r} at all, so it is the wrong "
                f"report for this model rather than an empty bucket."
            )
        live = [r for r in rows if r["span_tier"] != "error"]
        self.probed = {(r["question_id"], r["claim"]) for r in live}
        self.verified = {
            (r["question_id"], r["claim"]) for r in live if r["span_tier"] in pass_kinds
        }
        self.asserted = {(r["question_id"], r["claim"]) for r in live if r.get("found")}
        self.bucket = report.get("bucket")
        self.version = report.get("prompt_version")
        self.judge = report.get("judge")
        self.bucket_judge = bucket_judge
        self.n_rows = len(rows)
        self.n_errored = len(rows) - len(live)
        self.seen: set[tuple[str, str]] = set()

    def unmatched(self) -> set[tuple[str, str]]:
        return self.probed - self.seen

    @classmethod
    def _check_health(cls, report: dict) -> None:
        judged = report.get("n_judged") or 0
        rounds = report.get("vote_rounds") or 1
        dropped = report.get("n_rounds_dropped") or 0
        total = judged * rounds
        share = dropped / total if total else 0.0
        if share > cls._MAX_DROPPED_ROUND_FRACTION:
            raise SystemExit(
                f"the lookup report dropped {dropped} of {total} voting rounds "
                f"({100 * share:.1f}%). Its `found` verdicts are biased toward NOT found, "
                f"so a supported_claims_rate built on them is not a measurement. Re-run "
                f"the probe at a lower --rpm."
            )


def score(
    answers: dict[str, str],
    cs_report: dict,
    lookup: Lookup | None = None,
) -> dict:
    """Walk the claim support report's per-record claims, resolve each claim's markers
    against its answer text, and count the funnel."""
    adj = cs_report.get("adjudication")
    if adj != _REQUIRED_ADJUDICATION:
        raise SystemExit(
            f"claim support report has adjudication={adj!r}, expected "
            f"{_REQUIRED_ADJUDICATION!r}. The certified fold would compute a conjunction "
            f"over per-quote pairs where claim support computes a disjunction, so the "
            f"funnel's endpoint would not be the metric it is labelled as."
        )
    gate = cs_report.get("quote_gate", "all")
    if gate != "compliant":
        raise SystemExit(
            f"claim support report has quote_gate={gate!r}; the funnel needs 'compliant'. "
            f"Its verdicts must come from the run that withheld the non-compliant quotes "
            f"before judging, and an ungated report additionally carries no `gated_out` "
            f"flag, so the compliance stage would be unmeasurable. Point --claim-support "
            f"at the gated tree."
        )
    st = Counter()
    per_record = []
    # Homogeneous rows within a report, and no zero-valued recovery keys in a report that
    # never ran the probe: a 0 there would read as "nothing was recoverable" rather than
    # "this was not measured".
    keys = _RECORD_KEYS + (_RECOVERY_KEYS if lookup is not None else ())
    for rec in cs_report["per_record"]:
        qid = rec["question_id"]
        answer = answers.get(qid)
        if answer is None:
            raise SystemExit(
                f"{qid} is in the claim support report but not the answers"
            )
        r = Counter()
        r["sentences"] = rec["n_sentences"]
        for claim in rec["claims"]:
            if not (
                claim["is_claim"] and claim["over_floor"] and not claim["malformed"]
            ):
                continue
            r["claims"] += 1
            if lookup is not None:
                lookup.seen.add((qid, claim["text"]))
            gated_out = bool(claim.get("gated_out"))
            if not (claim["pairs"] or gated_out):
                continue
            r["cited"] += 1
            # THE GATE. A claim survives while ANY of its quotes is compliant; it fails
            # only when the gate emptied the set. See the module docstring for why this
            # is no longer a conjunction over the whole set.
            if gated_out:
                continue
            r["compliant"] += 1
            # Compliant citations only, because a gated report has already discarded the
            # others and cannot say how many there were. This is NOT comparable with the
            # citation counts of the standalone VCR metric, which sees every citation.
            r["compliant_citations_on_claims"] += sum(
                len(_marker_indices(pr)) for pr in claim["pairs"]
            )
            verdicts = [p["support_verdict"] for p in claim["pairs"]]
            certified = all(v == "fully_supported" for v in verdicts)
            if certified:
                r["certified"] += 1
            if all(v in LENIENT for v in verdicts):
                r["certified_lenient"] += 1
            key = (qid, claim["text"])
            in_scope = not any(
                v in ("not_supported", "contradicted") for v in verdicts
            ) and any(
                p["support_verdict"] == "partially_supported"
                and p.get("shortfall_type") == "unsupported_addition"
                for p in claim["pairs"]
            )
            # No `continue` here: this is not the last thing the claim loop does, and a
            # skip would silently drop whatever is added below it later.
            if lookup is not None and not certified and in_scope:
                r["recovery_probed"] += key in lookup.probed
                if key not in lookup.probed:
                    r["recovery_bucket_unprobed"] += 1
                r["recovered"] += key in lookup.asserted
                r["recovered_verified"] += key in lookup.verified
        st.update(r)
        # Fixed key list, not sorted(r): a Counter omits keys that stayed at zero, which
        # gave rows heterogeneous schemas (20 rows in one report carried only sentences and
        # claims) and would KeyError in any consumer indexing a stage directly.
        per_record.append({"question_id": qid, **{k: r[k] for k in keys}})

    def share(num: str, den: str) -> float | None:
        return (st[num] / st[den]) if st[den] else None

    recovery = {}
    if lookup is not None:
        stray = lookup.unmatched()
        if lookup.probed and len(stray) / len(lookup.probed) > 0.05:
            raise SystemExit(
                f"{len(stray)} of {len(lookup.probed)} probed lookup rows matched no "
                f"claim in the claim support report. The join key is (question_id, claim "
                f"text), so this is a mismatched pairing rather than a partial one: check "
                f"--lookup-model, and that both reports come from the same claim filter "
                f"version and the same scoring run."
            )
        st["supported"] = st["certified"] + st["recovered_verified"]
        recovery = {
            "recovery": {
                "bucket": lookup.bucket,
                "probe_version": lookup.version,
                "probe_judge": lookup.judge,
                # Which judge's shortfall set the branch draws from, distinct from who
                # searched. Absent here, a recovery measured against another judge's
                # `unsupported_addition` set would be indistinguishable from a matched one.
                "bucket_judge": lookup.bucket_judge,
                "rows_for_model": lookup.n_rows,
                "rows_errored": lookup.n_errored,
                "rows_unmatched": len(lookup.unmatched()),
                # Non-certified compliant claims the probe actually drew. The probe covers
                # one shortfall axis, so this is below `compliant - certified` by design.
                "probed": st["recovery_probed"],
                "eligible": st["compliant"] - st["certified"],
                # Of `eligible`, how many the probe SHOULD have covered and did not.
                # `eligible - probed` is not this number: most of that difference is the
                # other shortfall axes, which the probe does not cover on purpose.
                "bucket_unprobed": st["recovery_bucket_unprobed"],
                "recovered": st["recovered"],
                # The same count with the copied span re-checked. recovered is asserted,
                # so this is always the lower of the two.
                "recovered_verified": st["recovered_verified"],
                "recovered_given_probed": share("recovered", "recovery_probed"),
                "scope": "partially_supported with shortfall unsupported_addition",
            },
            "supported_claims": st["supported"],
            # Certified plus recovered, over every claim. NOT a laxer CCR: it answers
            # "is the claim true of the guideline", where CCR answers "can a reader
            # check it from the answer alone".
            "supported_claims_rate": share("supported", "claims"),
        }

    return {
        "unit": "claim sentence",
        "headline_k": cs_report.get("headline_k"),
        "quote_rule": "a claim keeps its compliant quotes and fails only with none",
        "stages": {
            "sentences": st["sentences"],
            "claims": st["claims"],
            "cited": st["cited"],
            "compliant": st["compliant"],
            "certified": st["certified"],
            "certified_lenient": st["certified_lenient"],
        },
        # Each conditional rate is over the PREVIOUS stage, so the chain multiplies out.
        "conditional": {
            "claims_per_sentence": share("claims", "sentences"),
            "cited_given_claim": share("cited", "claims"),
            "compliant_given_cited": share("compliant", "cited"),
            "certified_given_compliant": share("certified", "compliant"),
        },
        # The headline. Denominator is every claim, cited or not, which is the whole
        # point: a claim carrying no citation is not certifiable.
        "certified_claims_rate": share("certified", "claims"),
        "certified_claims_rate_lenient": share("certified_lenient", "claims"),
        "compliant_citations_per_compliant_claim": (
            st["compliant_citations_on_claims"] / st["compliant"]
            if st["compliant"]
            else None
        ),
        **recovery,
        "per_record": per_record,
    }


def build_meta(
    answers_path: Path, cs_path: Path, cs_report: dict, lookup_path: Path | None
) -> dict:
    return {
        "metric": "claim_funnel",
        "run_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "git_commit": git_sha(),
        # Inherited, not chosen here: the funnel reuses the verdicts already scored.
        "judge_model": cs_report.get("judge_model"),
        "judge_prompt_version": cs_report.get("judge_prompt_version"),
        "claim_filter_version": cs_report.get("claim_filter_version"),
        "adjudication": cs_report.get("adjudication"),
        "answers": rel_to_repo(answers_path),
        "claim_support_report": rel_to_repo(cs_path),
        "lookup_report": rel_to_repo(lookup_path) if lookup_path else None,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Chain the citation metrics on the claim.")
    p.add_argument("--answers", required=True)
    p.add_argument("--claim-support", required=True, help="claim_support report.json")
    p.add_argument(
        "--lookup",
        help="readset_probe report.json (probe: readset-lookup). Adds the recovery "
        "branch; without it the funnel ends at the certified stage. A section-lookup "
        "(node) report is refused: that probe is retired.",
    )
    p.add_argument(
        "--lookup-model",
        help="gen_model to select in the lookup report. Defaults to the answers "
        "directory name, which is how the release names a model.",
    )
    p.add_argument("--report", help="Path to write the funnel report JSON.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    assert_absent(args.report, args.overwrite)
    ans_path, cs_path = Path(args.answers), Path(args.claim_support)
    lk_path = Path(args.lookup) if args.lookup else None
    for p in (ans_path, cs_path, *(x for x in (lk_path,) if x)):
        if not p.exists():
            raise SystemExit(f"{p} not found")

    answers = {}
    for line in ans_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            answers[r["question_id"]] = r.get("answer", "")
    cs_report = json.loads(cs_path.read_text(encoding="utf-8"))

    tiers = Tiers()
    lookup = None
    if lk_path:
        lookup = Lookup(
            json.loads(lk_path.read_text(encoding="utf-8")),
            args.lookup_model or ans_path.parent.name,
            tiers.cr.pass_kinds,
            cs_report.get("judge_model"),
        )

    result = {
        "meta": build_meta(ans_path, cs_path, cs_report, lk_path),
        "verbatim_criteria": tiers.cr.version,
        **score(answers, cs_report, lookup),
    }

    s, c = result["stages"], result["conditional"]
    print(
        f"claim funnel — k={result['headline_k']}  judge {result['meta']['judge_model']}"
    )
    print(f"  sentences  {s['sentences']:5}")
    for stage, cond in (
        ("claims", "claims_per_sentence"),
        ("cited", "cited_given_claim"),
        ("compliant", "compliant_given_cited"),
        ("certified", "certified_given_compliant"),
    ):
        v = c[cond]
        print(
            f"  {stage:10} {s[stage]:5}   {'-' if v is None else f'{100 * v:5.1f}% of previous'}"
        )
    ccr, cl = result["certified_claims_rate"], result["certified_claims_rate_lenient"]

    def _p(x: float | None) -> str:
        return "-" if x is None else f"{100 * x:.1f}%"

    # Guarded because this print runs BEFORE write_report: a model whose filter accepted no
    # claim leaves both rates None, and formatting None here would abort the run after the
    # work was already done and write nothing.
    print(f"  Certified Claims Rate = {_p(ccr)}   (lenient {_p(cl)})")
    if "recovery" in result:
        rv = result["recovery"]
        print(
            f"  recovered  {rv['recovered']:5}   of {rv['probed']} probed "
            f"({rv['eligible']} non-certified compliant claims, bucket {rv['bucket']})"
        )
        print(f"  Supported Claims Rate = {_p(result['supported_claims_rate'])}")

    if args.report:
        write_report(args.report, result, args.overwrite)
        print(f"  wrote {args.report}")


if __name__ == "__main__":
    main()
