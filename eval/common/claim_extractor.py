#!/usr/bin/env python3
"""LLM-based claim extraction — the "decompose" stage of decompose-then-verify.

    # eyeball it / A-B against the rule engine on real answers:
    uv run python eval/common/claim_extractor.py \\
        --answers eval/result/answers/patient_smoke.jsonl --limit 3 --compare
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# eval/common/ → repo root is two up; system/ + this dir on the path.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = Path(__file__).resolve().parents[1]
_COMMON_DIR = Path(__file__).resolve().parent
for _p in (str(_REPO_ROOT), str(_COMMON_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
load_dotenv(_REPO_ROOT / ".env", override=False)

from system.cite_parser import CITE_RE  # noqa: E402 — single source of the {{cite}} grammar
from cache import cache_key, cache_read, cache_write  # noqa: E402

from system.helper_agent import helper_agent  # noqa: E402

EXTRACTOR_VERSION = "llm-v1"  # abstractive (default, current behaviour)
EXTRACTOR_VERSION_EXTRACTIVE = "llm-ext-v1"  # extractive (verbatim spans)
DEFAULT_EXTRACTOR_MODEL = "openai:gpt-4.1-mini"  # same family as the judges
_CACHE_DIR = _EVAL_DIR / "result" / ".extractor_cache"


def _version_for(mode: str) -> str:
    return EXTRACTOR_VERSION_EXTRACTIVE if mode == "extractive" else EXTRACTOR_VERSION


class Extraction(BaseModel):
    """The extractor's structured output: the answer's claims, in document order."""

    claims: list[str] = Field(
        description="The atomic, self-contained claims found in the answer, in the "
        "order they appear. Each is one checkable statement or recommendation. "
        "Empty list if the answer contains no substantive claim."
    )


_SYSTEM_ABSTRACTIVE = """You are a careful clinical text analyst. You are given an \
AI assistant's ANSWER to a nurse's question about cardiovascular / blood-pressure / \
dyslipidemia / diabetes care. Break the answer into a list of atomic CLAIMS.

A claim is a single, self-contained statement or recommendation that could be \
checked on its own. Follow these rules:

- ATOMIC: one piece of information per claim. If a sentence asserts several things, \
split it into several claims.
- SELF-CONTAINED (decontextualize): resolve pronouns and back-references so each \
claim stands alone. Carry enough context — the drug, the population, the threshold \
— that the claim is unambiguous on its own, but do NOT pad it with unrelated facts.
- FAITHFUL: use ONLY what the answer says. Do NOT add, infer, correct, or invent \
information, and do NOT merge in your own clinical knowledge. You are restructuring \
the text, not writing new content.
- COVER the substantive content: every recommendation, assessment, and factual \
assertion in the answer should appear in some claim. Drop nothing meaningful.
- IGNORE formatting: section headers, list numbering and bullets, bold/italic \
markup, links, and any "References"/citation boilerplate are NOT claims. Read the \
meaning underneath the markdown — never emit a formatting fragment as a claim.
- KEEP numbers, doses, thresholds, and dates attached to the claim they belong to.

Return the claims as a list of plain strings, in the order they appear."""


_SYSTEM_EXTRACTIVE = """You are a careful clinical text analyst. You are given an AI \
assistant's ANSWER to a nurse's question about cardiovascular / blood-pressure / \
dyslipidemia / diabetes care. Extract the CLAIMS in the answer as VERBATIM spans.

A claim is a single checkable assertion, recommendation, or factual statement that \
the answer makes. Follow these rules:

- VERBATIM: each claim MUST be an exact, contiguous substring of the answer — the \
answer's own words, character for character. Do NOT paraphrase, summarise, reorder, \
resolve pronouns, fix grammar, or add or drop any word. If you alter the wording in \
any way you have failed. (Pronouns and context are resolved later by the checker, \
not by you.)
- SELECT: not every sentence is a claim. SKIP pure background, framing sentences, \
transitions, content-free hedges, and meta-commentary. Return only spans that \
assert something checkable.
- ATOMIC: if one sentence bundles several checkable facts, return each as its own \
verbatim span (copy the exact clause for each). One assertion per claim.
- IGNORE formatting: do not emit headers, list markers, bold/italic markup, links, \
or "References" boilerplate as claims.
- KEEP numbers, doses, thresholds, and dates inside the span they belong to.

Return the claims as a list of verbatim strings, in the order they appear."""


def _user_prompt(answer: str, question: str | None) -> str:
    q = (
        "NURSE QUESTION (context for resolving references only — do NOT extract "
        f"claims from it):\n{question}\n\n"
        if question
        else ""
    )
    return f"{q}ANSWER:\n{answer}"


def _strip_cites(text: str) -> str:
    """Remove inline {{cite:...}} markers before extraction — their embedded quotes
    would otherwise be mistaken for answer content."""
    return CITE_RE.sub("", text or "")


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def verbatim_rate(answer: str, claims: list[str]) -> float | None:
    """Diagnostic: fraction of claims that are an exact substring of the answer, ignoring case,
    whitespace, and trailing sentence punctuation (the extractor re-punctuates atomic spans
    — trivial, not a content edit). A claim that still fails has real drift:
    inserted/reworded text. 1.0 = fully verbatim.
    """
    if not claims:
        return None
    hay = _norm(_strip_cites(answer))
    hits = sum(1 for c in claims if _norm(c).rstrip(" .,;:") in hay)
    return hits / len(claims)


# ── caching ─────────────────────────────────────────────────────────────────────


def _cache_key(model: str, answer: str, question: str, version: str) -> str:
    return cache_key(model, version, question, answer)


def _cache_read(key: str) -> list[str] | None:
    return cache_read(_CACHE_DIR, key)


def _cache_write(key: str, value: list[str]) -> None:
    cache_write(_CACHE_DIR, key, value)


# ── extraction ────────────────────────────────────────────────────────────────


async def extract_claims(
    answer: str,
    model: str,
    use_cache: bool,
    *,
    question: str | None = None,
    mode: str = "abstractive",
) -> list[str]:
    """Decompose one answer into claims. `mode` picks the style: "abstractive" (default)
    rewrites into atomic, decontextualized claims; "extractive" returns verbatim spans of
    the answer (no rewriting). Citation markers are stripped first.
    """
    cleaned = _strip_cites(answer)
    if not cleaned.strip():
        return []

    system = _SYSTEM_EXTRACTIVE if mode == "extractive" else _SYSTEM_ABSTRACTIVE
    key = _cache_key(model, cleaned, question or "", _version_for(mode))
    if use_cache:
        cached = _cache_read(key)
        if cached is not None:
            return cached

    agent = helper_agent(
        model, instructions=system, max_tokens=2000, output_type=Extraction
    )
    try:
        result = await agent.run(_user_prompt(cleaned, question))
        claims = [c.strip() for c in result.output.claims if c.strip()]
    except Exception:  # noqa: BLE001 — isolate per-answer extractor failures
        return []

    if use_cache:
        _cache_write(key, claims)
    return claims


# ── CLI: eyeball / A-B against the rule engine ──────────────────────────────────


async def _run_cli(
    records: list[dict], model: str, use_cache: bool, compare: bool, mode: str
):
    sem = asyncio.Semaphore(8)

    async def _one(r: dict, m: str) -> list[str]:
        async with sem:
            return await extract_claims(
                r.get("answer", ""),
                model,
                use_cache,
                question=r.get("question"),
                mode=m,
            )

    modes = ["abstractive", "extractive"] if mode == "both" else [mode]
    by_mode = {m: await asyncio.gather(*(_one(r, m) for r in records)) for m in modes}

    rule_claims = None
    if compare:
        from segmenter import SEGMENTER_VERSION, split_sentences  # noqa: E402

        rule_claims = [split_sentences(r.get("answer", "")) for r in records]

    for i, r in enumerate(records):
        qid = r.get("question_id", f"#{i}")
        print(f"\n=== {qid} ===")
        q = r.get("question")
        if q:
            print(f"Q: {q}")
        if compare:
            rc = rule_claims[i]
            print(f"\n  rule-based ({SEGMENTER_VERSION}) — {len(rc)} sentences:")
            for s in rc:
                print(f"    · {s}")
        for m in modes:
            lc = by_mode[m][i]
            vr = verbatim_rate(r.get("answer", ""), lc)
            vr_s = f"  [verbatim {vr:.0%}]" if vr is not None else ""
            print(f"\n  LLM {m} ({_version_for(m)}, {model}) — {len(lc)} claims{vr_s}:")
            for s in lc:
                print(f"    · {s}")

    print(f"\n— {len(records)} answers —")
    for m in modes:
        total = sum(len(c) for c in by_mode[m])
        # Aggregate verbatim rate: each answer's claims checked against THAT answer.
        hits = sum(
            round(
                (verbatim_rate(r.get("answer", ""), by_mode[m][i]) or 0)
                * len(by_mode[m][i])
            )
            for i, r in enumerate(records)
        )
        vr_s = f" (verbatim {hits / total:.0%})" if total else ""
        print(f"— {m}: {total} claims{vr_s}")
    if compare:
        total_rule = sum(len(c) for c in rule_claims)
        print(f"— rule-based: {total_rule} sentences —")


def main() -> None:
    p = argparse.ArgumentParser(
        description="LLM claim extractor — run/A-B over a harness answer JSONL."
    )
    p.add_argument("--answers", required=True, help="Harness answers.jsonl path.")
    p.add_argument(
        "--model",
        default=DEFAULT_EXTRACTOR_MODEL,
        help=f"provider:model extractor (default {DEFAULT_EXTRACTOR_MODEL}).",
    )
    p.add_argument("--limit", type=int, help="Cap on answer records (smoke test).")
    p.add_argument(
        "--mode",
        choices=["abstractive", "extractive", "both"],
        default="abstractive",
        help="abstractive (rewrite, default), extractive (verbatim spans), or both.",
    )
    p.add_argument(
        "--compare",
        action="store_true",
        help="Show the rule-based split beside the LLM claims.",
    )
    p.add_argument(
        "--no-cache", action="store_true", help="Ignore the SHA-256 disk cache."
    )
    args = p.parse_args()

    answers_path = Path(args.answers)
    if not answers_path.exists():
        raise SystemExit(f"{answers_path} not found")

    from scoring import load_jsonl

    records = load_jsonl(answers_path)
    if args.limit:
        records = records[: args.limit]

    asyncio.run(
        _run_cli(records, args.model, not args.no_cache, args.compare, args.mode)
    )


if __name__ == "__main__":
    main()
