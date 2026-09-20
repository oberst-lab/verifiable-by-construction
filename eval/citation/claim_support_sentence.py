"""Claim support, sentence-based: do the model's cited quotes actually back the
claims it wrote?

    Usage (from repo root):
        uv run python eval/citation/claim_support_sentence.py \\
            --answers eval/result/answers/bench50_sonnet.jsonl \\
            --report eval/result/reports/out.json
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# eval/citation/ → repo root is two up; the system package + common/ on the path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_EVAL_DIR / "common"), str(Path(__file__).parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
load_dotenv(_REPO_ROOT / ".env", override=False)

from system.cite_parser import CITE_RE  # noqa: E402
from claim_support import (  # noqa: E402
    OVERSTATEMENT_SUBTYPES,
    PROMPT_VERSION,
    _PROMPT_HASH,
    DEFAULT_JUDGE_MODEL,
    axis_totals,
    JUDGE_USAGE,
    JUDGE_VOTE_MAX_ROUNDS,
    JUDGE_VOTE_ROUNDS,
    JUDGE_OUTPUT_RETRIES,
    METERED_JUDGE_RPM,
    assert_cache_matches_decoding,
    judge_atom_voted,
    judge_pacing,
    judge_settings,
    reset_judge_usage,
    set_judge_limiter,
)
from coverage import (  # noqa: E402
    CLAIM_FILTER_VERSION,
    FILTER_USAGE,
    _assert_filter_model_samples_independently,
    _reset_filter_stats,
    DEFAULT_FILTER_MODEL,
    MIN_SENTENCE_CHARS,
    _FILTER_PROMPT_HASH,
    _classify_claims,
    build_meta,
)
from model_endpoints import needs_custom_endpoint, resolve_route  # noqa: E402
from vccr import Tiers  # noqa: E402  the quote gate's verbatim check
from scoring import assert_absent, fmt_pct, load_jsonl, write_report  # noqa: E402
from segmenter import (  # noqa: E402
    CLAIM_VALIDITY_VERSION,
    SEGMENTER_VERSION,
    _normalize_markdown,
    segment_marker_map_with_validity,
)

# Territorial bound: a quote never reaches past the previous marker group,
# fallback included. Block reach is capped at MAX_BLOCK_CLAIMS claims nearest
# the citation.
PAIRING_VERSION = "pairing-v2"
PAIRINGS = ("nearest", "block")
MAX_BLOCK_CLAIMS = 5
LADDER_KS = (1, 3, 5)
HEADLINE_K = 1

QUOTE_GATES = ("compliant", "all")
DEFAULT_QUOTE_GATE = "compliant"

ADJUDICATIONS = ("set", "any-of")
DEFAULT_ADJUDICATION = "set"

_STRICT_OK = {"fully_supported"}
_LENIENT_OK = {"fully_supported", "partially_supported"}


@dataclass
class ClaimRow:
    """One sentence of the answer, with its filter status and pairing outcome."""

    idx: int
    text: str
    over_floor: bool
    is_claim: bool
    malformed: bool = (
        False  # excluded as a mid-clause citation collapse (glass-box audit)
    )
    pairs: list[dict] = field(default_factory=list)  # judged (quote) pairs
    gated_out: bool = False


def pair_claims(
    sent_texts: list[str],
    claim_flags: list[bool],
    marker_sents: list[int | None],
    pairing: str,
) -> dict[int, list[int]]:
    """Deterministic claim↔quote pairing. Returns {claim sentence idx: [marker
    idx, ...]}. `marker_sents[m]` is the sentence each marker attributes to
    (Coverage rule: in-span → that sentence, gap → preceding).
    """
    out: dict[int, list[int]] = {}

    # Group markers by attributed sentence, preserving document order.
    groups: list[tuple[int, list[int]]] = []  # (sentence idx, [marker idx])
    for m, si in enumerate(marker_sents):
        if si is None:
            continue
        if groups and groups[-1][0] == si:
            groups[-1][1].append(m)
        else:
            groups.append((si, [m]))

    # `pairing` is a territorial window k = how many claims (nearest the citation,
    # within (p, i]) one marker group covers. String aliases: nearest = k=1,
    # block = k=MAX_BLOCK_CLAIMS. k=1/3/5 is the reported manual-CI ladder.
    k = _pairing_k(pairing)
    prev = -1
    for si, markers in groups:
        territory = [j for j in range(prev + 1, si + 1) if claim_flags[j]]
        if territory:
            for j in territory[-k:]:
                out.setdefault(j, []).extend(markers)
        prev = max(prev, si)
    return out


def _pairing_k(pairing: str | int) -> int:
    """Map a pairing spec to its territorial window k. nearest → 1, block →
    MAX_BLOCK_CLAIMS, an int → itself."""
    if isinstance(pairing, int):
        return pairing
    return {"nearest": 1, "block": MAX_BLOCK_CLAIMS}[pairing]


def decompose(
    answer: str,
) -> tuple[list[str], list[bool], list[int | None], list[dict], list[bool]]:
    """Coverage-identical decomposition of one answer. Returns (sentence texts,
    over-floor flags, marker→sentence attribution, markers as {doc_id, quote},
    malformed flags — mid-clause citation-collapse claims to exclude)."""
    src = _normalize_markdown(answer or "")
    markers = [
        {"doc_id": m.group(1).strip(), "quote": m.group(2)}
        for m in CITE_RE.finditer(src)
    ]
    texts, marker_sents, malformed = segment_marker_map_with_validity(answer or "")
    if len(markers) != len(marker_sents):
        raise SystemExit(
            f"marker scan disagreement: {len(markers)} markers in the normalized text "
            f"against {len(marker_sents)} from the segmenter. Marker indices address "
            f"different citations in the two scans, so quote selection and the quote "
            f"gate are both unsafe on this answer."
        )
    floors = [len(t) >= MIN_SENTENCE_CHARS for t in texts]
    return texts, floors, marker_sents, markers, malformed


async def score_record(
    record: dict,
    judge_model: str,
    filter_model: str,
    use_cache: bool,
    sem: asyncio.Semaphore,
    adjudication: str = DEFAULT_ADJUDICATION,
    quote_gate: str = DEFAULT_QUOTE_GATE,
    tiers: Tiers | None = None,
) -> dict:
    texts, floors, marker_sents, markers, malformed = decompose(
        record.get("answer", "")
    )

    cand_idx = [i for i, ok in enumerate(floors) if ok]
    flags_cand = await _classify_claims(
        [texts[i] for i in cand_idx], filter_model, use_cache
    )
    claim_flags = [False] * len(texts)
    for i, keep in zip(cand_idx, flags_cand):
        claim_flags[i] = keep

    n_dropped_non_claim = sum(1 for keep in flags_cand if not keep)

    n_malformed_claims = sum(
        1 for i in range(len(texts)) if malformed[i] and claim_flags[i]
    )
    for i in range(len(texts)):
        if malformed[i]:
            claim_flags[i] = False

    allowed: set[int] | None = None
    if quote_gate == "compliant":
        t = tiers if tiers is not None else Tiers()
        allowed = {
            m
            for m, mk in enumerate(markers)
            if t.passes(t.of(mk["doc_id"], mk["quote"]))
        }

    def _gate(kmap: dict[int, list[int]]) -> dict[int, list[int]]:
        """Drop non-compliant markers from a pairing, and claims thereby left with no evidence."""
        if allowed is None:
            return kmap
        out = {}
        for ci, ms in kmap.items():
            keep = [m for m in ms if m in allowed]
            if keep:
                out[ci] = keep
        return out

    superset_map = pair_claims(texts, claim_flags, marker_sents, MAX_BLOCK_CLAIMS)

    jobs: dict[tuple[int, int], asyncio.Task] = {}

    question = record.get("question", "")

    async def _judge(claim: str, quote: str) -> dict:
        async with sem:
            return await judge_atom_voted(
                question, claim, quote, judge_model, use_cache
            )

    def _quote_text(ci: int, ms: list[int]) -> str:
        """The quote material for one claim, as one string."""
        out: list[str] = []
        for m in ms:
            q = markers[m]["quote"]
            if q not in out:
                out.append(q)
        return "\n\n".join(out)

    seen: dict[tuple[str, str], asyncio.Task] = {}
    if adjudication == "set":
        # One call per (claim, quote-set). The ladder cannot re-select from a superset
        # here, because a different k forms a different set and so a different prompt;
        # every rung the caller asks for is materialised.
        for k in LADDER_KS:
            for ci, ms in _gate(
                pair_claims(texts, claim_flags, marker_sents, k)
            ).items():
                key = (texts[ci], _quote_text(ci, ms))
                if key not in seen:
                    seen[key] = asyncio.create_task(_judge(*key))
    else:
        for ci, ms in _gate(superset_map).items():
            for m in ms:
                key = (texts[ci], markers[m]["quote"])
                if key not in seen:
                    seen[key] = asyncio.create_task(_judge(*key))
                jobs[(ci, m)] = seen[key]
    if seen:
        await asyncio.gather(*seen.values())

    def _verdict(ci: int, m: int) -> dict:
        return jobs[(ci, m)].result()

    def _set_verdict(ci: int, ms: list[int]) -> dict:
        return seen[(texts[ci], _quote_text(ci, ms))].result()

    # Support counts at each ladder k (denominator handled by the caller).
    def _counts_at(k: int) -> dict:
        kmap = _gate(pair_claims(texts, claim_flags, marker_sents, k))
        n_paired = n_strict = n_lenient = 0
        for ci in range(len(texts)):
            if not claim_flags[ci]:
                continue
            ms = kmap.get(ci, [])
            if not ms:
                continue
            n_paired += 1
            if adjudication == "set":
                verdicts = {_set_verdict(ci, ms).get("support_verdict")}
            else:
                verdicts = {_verdict(ci, m).get("support_verdict") for m in ms}
            if verdicts & _STRICT_OK:
                n_strict += 1
            if verdicts & _LENIENT_OK:
                n_lenient += 1
        return {"paired": n_paired, "strict": n_strict, "lenient": n_lenient}

    ladder = {k: _counts_at(k) for k in LADDER_KS}

    # Per-record detail + verdict/failure distributions use the HEADLINE k (=1),
    # matching Quote Fidelity's headline choice. per_record.pairs stay at k=1.
    head_ungated = pair_claims(texts, claim_flags, marker_sents, HEADLINE_K)
    head_map = _gate(head_ungated)
    # What the gate removed at the headline k, so a gated report can always be read
    # against the ungated one without re-deriving it.
    gated_markers: set[int] = set()
    if allowed is not None:
        gated_markers = {m for ms in head_ungated.values() for m in ms} - allowed
    n_claims_gated_out = len(head_ungated) - len(head_map)
    n_claims_gated_partial = sum(
        1
        for ci, ms in head_map.items()
        if _quote_text(ci, ms) != _quote_text(ci, head_ungated[ci])
    )
    rows: list[ClaimRow] = []
    for i, t in enumerate(texts):
        row = ClaimRow(
            idx=i,
            text=t,
            over_floor=floors[i],
            is_claim=claim_flags[i],
            malformed=malformed[i],
            gated_out=i in head_ungated and i not in head_map,
        )
        ms = head_map.get(i, [])
        if adjudication == "set" and ms:
            # ONE entry, because there is one verdict. `quote` carries the joined text
            # the judge actually saw and `doc_id` every section it drew on, so a reader
            # of per_record can reconstruct the prompt without re-deriving the pairing.
            row.pairs.append(
                {
                    "marker": ms,
                    "doc_id": sorted({markers[m]["doc_id"] for m in ms}),
                    "quote": _quote_text(i, ms),
                    "n_quotes": len({markers[m]["quote"] for m in ms}),
                    **_set_verdict(i, ms),
                }
            )
        elif adjudication != "set":
            for m in ms:
                row.pairs.append(
                    {
                        "marker": m,
                        "doc_id": markers[m]["doc_id"],
                        "quote": markers[m]["quote"],
                        **_verdict(i, m),
                    }
                )
        rows.append(row)

    claims = [r for r in rows if r.is_claim]
    head = ladder[HEADLINE_K]

    return {
        "question_id": record.get("question_id"),
        "n_sentences": len(texts),
        "n_claim_sentences": len(claims),
        "n_malformed_claims": n_malformed_claims,
        "n_dropped_non_claim": n_dropped_non_claim,
        "n_paired_claims": head["paired"],
        "n_markers": len(markers),
        # Quote-gate disclosure at the headline k. All three are 0 under gate `all`.
        "n_markers_gated": len(gated_markers),
        # Claims that lost EVERY quote and left the denominator: they cited nothing the
        # corpus contains, so there was no support question to ask.
        "n_claims_gated_out": n_claims_gated_out,
        # Claims that kept some evidence and lost some. These are the only claims whose
        # judge prompt differs between the two gates.
        "n_claims_gated_partial": n_claims_gated_partial,
        # Markers whose territory held no claim pair with nothing (pairing-v2:
        # no cross-border rescue) — disclosed, not silently dropped. Computed on
        # the k=5 superset (the widest reach) so "unpaired" means truly unreachable.
        "n_markers_unpaired": len(markers)
        - len({m for ms in superset_map.values() for m in ms}),
        "n_pairs": sum(len(r.pairs) for r in rows),
        "ladder": ladder,
        "n_supported_strict": head["strict"],
        "n_supported_lenient": head["lenient"],
        "claims": [
            {
                "idx": r.idx,
                "text": r.text,
                "is_claim": r.is_claim,
                "over_floor": r.over_floor,
                "malformed": r.malformed,
                "gated_out": r.gated_out,
                "pairs": r.pairs,
            }
            for r in rows
        ],
    }


async def score_answers(
    records: list[dict],
    judge_model: str,
    filter_model: str,
    concurrency: int,
    use_cache: bool,
    adjudication: str = DEFAULT_ADJUDICATION,
    quote_gate: str = DEFAULT_QUOTE_GATE,
) -> dict:
    sem = asyncio.Semaphore(concurrency)
    # ONE Tiers for the whole run, so its section cache is shared across answers: the
    # corpus holds 242 sections against thousands of markers, and a per-answer instance
    # would re-read and re-normalize the same section for every answer citing it.
    tiers = Tiers() if quote_gate == "compliant" else None
    per_record = await asyncio.gather(
        *(
            score_record(
                r,
                judge_model,
                filter_model,
                use_cache,
                sem,
                adjudication,
                quote_gate,
                tiers,
            )
            for r in records
        )
    )

    n_claims = sum(r["n_claim_sentences"] for r in per_record)
    n_malformed = sum(r["n_malformed_claims"] for r in per_record)
    n_dropped_non_claim = sum(r["n_dropped_non_claim"] for r in per_record)

    # Aggregate the manual-CI ladder: for each k, strict/lenient shares over ALL
    # claim sentences (ALCE-style recall) and conditionally over paired claims.
    def _agg_k(k: int) -> dict:
        s = sum(r["ladder"][k]["strict"] for r in per_record)
        le = sum(r["ladder"][k]["lenient"] for r in per_record)
        pa = sum(r["ladder"][k]["paired"] for r in per_record)
        return {
            "n_paired_claims": pa,
            "support_all_strict": (s / n_claims) if n_claims else None,
            "support_all_lenient": (le / n_claims) if n_claims else None,
            "support_conditional_strict": (s / pa) if pa else None,
            "support_conditional_lenient": (le / pa) if pa else None,
        }

    ladder = {f"k{k}": _agg_k(k) for k in LADDER_KS}
    head = ladder[f"k{HEADLINE_K}"]

    labels: Counter = Counter()
    axes: Counter = Counter()
    subtypes: Counter = Counter()
    for r in per_record:
        for c in r["claims"]:
            for p in c["pairs"]:
                labels[p.get("support_verdict")] += 1
                if p.get("support_verdict") == "contradicted":
                    continue
                if p.get("shortfall_axis"):
                    axes[p["shortfall_axis"]] += 1
                if p.get("overstatement_subtype"):
                    subtypes[p["overstatement_subtype"]] += 1

    return {
        "metric": "claim-support-sentence",
        "pairing_version": PAIRING_VERSION,
        "segmenter_version": SEGMENTER_VERSION,
        "claim_validity_version": CLAIM_VALIDITY_VERSION,
        "n_malformed_claims_excluded": n_malformed,
        # Candidates the LLM claim filter rejected as non-claims (e.g. incomplete
        # fragments under claim-filter-v3). Disjoint from the malformed count above;
        # together they are the full non-claim exclusion tally.
        "n_dropped_non_claim": n_dropped_non_claim,
        "malformed_claim_rate": (n_malformed / (n_claims + n_malformed))
        if (n_claims + n_malformed)
        else None,
        "claim_filter_version": CLAIM_FILTER_VERSION,
        "claim_filter_prompt_hash": _FILTER_PROMPT_HASH,
        "claim_filter_model": filter_model,
        # Which CHANNEL served the filter. Same weights either way, but the request
        # layer differs: a metered host needs a per-round seed and its own thinking-off
        # lever, and a report naming only the model could not say which produced it.
        "claim_filter_route": "metered"
        if needs_custom_endpoint(filter_model)
        else "native",
        "judge_model": judge_model,
        "judge_settings": judge_settings(),
        # Beside judge_settings(), not inside it: retries cannot change a verdict that
        # validated, so folding it into that dict would invalidate the judge cache and
        # applicability's cache key for nothing. See JUDGE_OUTPUT_RETRIES.
        "judge_output_retries": JUDGE_OUTPUT_RETRIES,
        "rate_limit": judge_pacing(),
        # The judge's channel, for the same reason as the filter's above. The two are
        # independent: a metered filter with a natively routed judge is a legitimate run.
        "judge_route": "metered" if needs_custom_endpoint(judge_model) else "native",
        "judge_prompt_version": PROMPT_VERSION,
        # Rubric content fingerprint (see claim_support.py); tamper-evident twin of
        # judge_prompt_version, and gates the shared judge cache so it matches the
        # verdicts these numbers came from.
        "judge_prompt_hash": _PROMPT_HASH,
        "usage": {
            "judge": JUDGE_USAGE.as_dict(),
            "claim_filter": FILTER_USAGE.as_dict(),
        },
        "min_sentence_chars": MIN_SENTENCE_CHARS,
        "adjudication": adjudication,
        # Which quotes the judge was shown (see QUOTE_GATES). Recorded because the two
        # gates answer different questions on the same answers, so a rate is not
        # interpretable without it, and because the funnel accepts only `all`.
        "quote_gate": quote_gate,
        "verbatim_criteria": tiers.cr.version if tiers is not None else None,
        "n_markers_gated": sum(r["n_markers_gated"] for r in per_record),
        "n_claims_gated_out": sum(r["n_claims_gated_out"] for r in per_record),
        "n_claims_gated_partial": sum(r["n_claims_gated_partial"] for r in per_record),
        "n_answers": len(records),
        "n_claim_sentences": n_claims,
        "n_pairs": sum(r["n_pairs"] for r in per_record),
        "n_markers_unpaired": sum(r["n_markers_unpaired"] for r in per_record),
        # Manual-CI window ladder (k=1 headline ≡ nearest ≡ Coverage@k=1;
        # k=5 ≡ block permissive bound). Aligned with Coverage's k reporting.
        "headline_k": HEADLINE_K,
        "n_paired_claims": head["n_paired_claims"],
        "support_all_strict": head["support_all_strict"],
        "support_all_lenient": head["support_all_lenient"],
        "support_conditional_strict": head["support_conditional_strict"],
        "support_conditional_lenient": head["support_conditional_lenient"],
        "ladder": ladder,
        # Quote Fidelity side-channel (v4): shortfall axis + overstatement subtypes at
        # the headline k. Quote Fidelity = the overstatement (intrinsic) distribution;
        # unsupported_addition (extrinsic gap) is reported separately, not as honesty.
        "pair_label_distribution": dict(labels),
        "judge_vote_rounds": JUDGE_VOTE_ROUNDS,
        "judge_vote_max_rounds": JUDGE_VOTE_MAX_ROUNDS,
        "judge_unanimous_first3": _vote_stats(per_record)["unanimous"],
        "judge_split_first3": _vote_stats(per_record)["split"],
        "judge_split_rate_first3": _vote_stats(per_record)["split_rate"],
        "judge_undecidable_first3": _vote_stats(per_record)["undecidable"],
        "judge_extra_rounds_drawn": _vote_stats(per_record)["extra_rounds"],
        "n_no_majority": _vote_stats(per_record)["no_majority"],
        # The single-round range. See _per_round_rates: a judge-sampling spread, NOT a CI.
        **_per_round_rates(per_record),
        "pair_shortfall_axis_distribution": dict(axes),
        "pair_overstatement_subtype_distribution": dict(subtypes),
        "n_shortfall_unnamed": (
            labels.get("partially_supported", 0)
            + labels.get("not_supported", 0)
            - sum(axes.values())
        ),
        **axis_totals(axes),
        "per_record": list(per_record),
    }


def _per_round_rates(per_record) -> dict:
    """Strict and lenient conditional rates recomputed from EACH of the first three rounds
    alone, and the range they span.
    """
    rounds = {r: {"strict": 0, "lenient": 0, "n": 0} for r in range(JUDGE_VOTE_ROUNDS)}
    for rec in per_record:
        for c in rec.get("claims", []):
            for pr in c.get("pairs", []):
                votes = pr.get("judge_votes")
                if not votes:
                    continue
                for i in range(JUDGE_VOTE_ROUNDS):
                    if i >= len(votes) or votes[i] == "not_parsable":
                        continue
                    rounds[i]["n"] += 1
                    rounds[i]["strict"] += votes[i] in _STRICT_OK
                    rounds[i]["lenient"] += votes[i] in _LENIENT_OK
    strict = [v["strict"] / v["n"] for v in rounds.values() if v["n"]]
    lenient = [v["lenient"] / v["n"] for v in rounds.values() if v["n"]]
    if not strict:
        return {}
    return {
        "per_round_strict": strict,
        "per_round_lenient": lenient,
        "single_round_strict_lo": min(strict),
        "single_round_strict_hi": max(strict),
        "single_round_lenient_lo": min(lenient),
        "single_round_lenient_hi": max(lenient),
    }


def _vote_stats(per_record) -> dict:
    """Judge self-consistency over the first three rounds, plus what voting actually cost."""
    unanimous = split = undecidable = no_majority = extra = 0
    for rec in per_record:
        for c in rec.get("claims", []):
            for pr in c.get("pairs", []):
                u = pr.get("judge_first3_unanimous")
                if u is True:
                    unanimous += 1
                elif u is False:
                    split += 1
                elif "judge_votes" in pr:
                    undecidable += 1
                if pr.get("judge_no_majority"):
                    no_majority += 1
                extra += max(0, pr.get("judge_rounds_used", 0) - JUDGE_VOTE_ROUNDS)
    n = unanimous + split
    return {
        "unanimous": unanimous,
        "split": split,
        "undecidable": undecidable,
        "no_majority": no_majority,
        "extra_rounds": extra,
        "split_rate": (split / n) if n else None,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sentence-based Claim Support (Coverage-shared claim side, "
        "deterministic pairing, manual-CI variants) over a harness answer JSONL."
    )
    p.add_argument("--answers", required=True, help="Harness answers.jsonl path.")
    p.add_argument("--report", help="Optional path to write the full JSON report.")
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow --report to replace an existing file. Without it an existing "
        "report is left alone and the run refuses to start (archive, do not overwrite).",
    )
    # Both accept a route alias as well as a full provider:model id, so either stage
    # can be pointed at either channel per run. The route is part of the cache key, so
    # two channels never share cached verdicts.
    p.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    p.add_argument("--filter-model", default=DEFAULT_FILTER_MODEL)
    p.add_argument("--limit", type=int, help="Cap on answer records (smoke test).")
    p.add_argument(
        "--adjudication",
        choices=ADJUDICATIONS,
        default=DEFAULT_ADJUDICATION,
        help="How a claim's verdict is formed from its quotes (see ADJUDICATIONS).",
    )
    p.add_argument(
        "--quote-gate",
        choices=QUOTE_GATES,
        default=DEFAULT_QUOTE_GATE,
        help="Which of a claim's quotes the judge is shown (see QUOTE_GATES). "
        f"Default {DEFAULT_QUOTE_GATE!r}: quotes that fail the frozen verbatim criteria "
        "are withheld, and a claim left with none leaves the denominator. Pass 'all' for "
        "the ungated rate, which is what the claim funnel requires and what the "
        "literature reports.",
    )
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument(
        "--rpm",
        type=float,
        default=None,
        help="Judge calls per minute. Only meaningful for a judge behind a metered host, "
        f"it defaults to {METERED_JUDGE_RPM:g}; direct-API judges are unpaced (the "
        "providers have no per-minute cap this path has hit). 0 disables pacing.",
    )
    p.add_argument("--no-cache", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # Before any API call: a guard that fires after the run has already paid for it.
    assert_absent(args.report, args.overwrite)
    answers_path = Path(args.answers)
    if not answers_path.exists():
        raise SystemExit(f"{answers_path} not found")
    records = load_jsonl(answers_path)
    if args.limit:
        records = records[: args.limit]
    _reset_filter_stats()
    reset_judge_usage()
    # The filter votes over byte-identical bodies, so an unseeded route through a
    # caching host would replay one sample as three. The judge is safe there because
    # it seeds each round; see claim_support.judge_seed.
    args.judge_model = resolve_route(args.judge_model)
    args.filter_model = resolve_route(args.filter_model)
    # The filter votes the same way the judge does, so it faces the same replay
    # problem. It is allowed on such a route when round_seed varies its seed per round;
    # this checks that mechanism rather than trusting it.
    _assert_filter_model_samples_independently(args.filter_model)
    # One pacer for the process: a metered host meters the KEY, so a metered judge and
    # a metered filter must share a rate rather than each claim the whole of it.
    # Installed when EITHER stage is routed there.
    set_judge_limiter(
        args.judge_model
        if needs_custom_endpoint(args.judge_model)
        else args.filter_model,
        args.rpm,
    )
    # Before any API call: a cache built under different judge decoding is not reusable.
    if not args.no_cache:
        assert_cache_matches_decoding()

    result = asyncio.run(
        score_answers(
            records,
            judge_model=args.judge_model,
            filter_model=args.filter_model,
            concurrency=args.concurrency,
            use_cache=not args.no_cache,
            adjudication=args.adjudication,
            quote_gate=args.quote_gate,
        )
    )

    print(
        f"Claim Support (sentence-based, {PAIRING_VERSION})  "
        f"judge {result['judge_model']} ({PROMPT_VERSION})"
    )
    print(
        f"  answers={result['n_answers']}  claim sentences={result['n_claim_sentences']}"
        f"  judged pairs={result['n_pairs']}"
        f"  unpaired markers={result['n_markers_unpaired']}"
    )
    print(
        f"  quote gate={result['quote_gate']}  withheld quotes={result['n_markers_gated']}"
        f"  claims left unevidenced={result['n_claims_gated_out']}"
        f"  claims re-evidenced={result['n_claims_gated_partial']}"
    )
    print(f"  manual-CI window ladder (headline k={result['headline_k']}):")
    print(
        "    k    strict (end-to-end / conditional)   lenient (end-to-end / conditional)"
    )
    for kkey, agg in result["ladder"].items():
        head = "  ←headline" if kkey == f"k{HEADLINE_K}" else ""
        print(
            f"    {kkey:3s}  {fmt_pct(agg['support_all_strict'])} / "
            f"{fmt_pct(agg['support_conditional_strict'])}      "
            f"{fmt_pct(agg['support_all_lenient'])} / "
            f"{fmt_pct(agg['support_conditional_lenient'])}{head}"
        )
    print("  pair labels (headline k):", dict(result["pair_label_distribution"]))
    print(
        f"  shortfall axis (headline k): overstatement {result['overstatement_total']}  "
        f"unsupported_addition {result['unsupported_addition_total']}"
    )
    print(
        f"  Quote Fidelity — overstatement subtypes (total {result['overstatement_total']}):"
    )
    for sub in OVERSTATEMENT_SUBTYPES:
        n = result["pair_overstatement_subtype_distribution"].get(sub, 0)
        if n:
            print(f"    {sub:20} {n:4}")

    if args.report:
        report = {
            "meta": build_meta(
                answers_path, args.judge_model, metric="claim-support-sentence"
            ),
            **result,
        }
        write_report(args.report, report, args.overwrite)
        print(f"  full report → {args.report}")


if __name__ == "__main__":
    main()
