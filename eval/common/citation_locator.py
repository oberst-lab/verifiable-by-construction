#!/usr/bin/env python3
"""LLM claim↔citation localiser — which claim is each citation attached to?

    uv run python eval/common/citation_locator.py \\
        --answers eval/result/answers/smoke_bp_sonnet.jsonl --limit 2 --compare
"""

from __future__ import annotations

import argparse
import asyncio
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

from system.cite_parser import CITE_RE  # noqa: E402
from cache import cache_key, cache_read, cache_write  # noqa: E402
from segmenter import CitationAtom, attribute_citations  # noqa: E402

from system.helper_agent import helper_agent  # noqa: E402

LOCATOR_VERSION = "llm-loc-v2"  # abstractive
LOCATOR_VERSION_EXTRACTIVE = "llm-loc-ext-v1"  # extractive (verbatim spans)
DEFAULT_LOCATOR_MODEL = "openai:gpt-4.1-mini"
_CACHE_DIR = _EVAL_DIR / "result" / ".locator_cache"


def _version_for(mode: str) -> str:
    return LOCATOR_VERSION_EXTRACTIVE if mode == "extractive" else LOCATOR_VERSION


class _Located(BaseModel):
    cite_id: int = Field(description="The placeholder number n from [CITE_n].")
    claim: str = Field(
        description="The atomic, self-contained claim this citation is attached to "
        "(normally the sentence, or one or two sentences, immediately before the "
        "placeholder). Answer's own wording; pronouns resolved; markdown ignored."
    )


class _Localization(BaseModel):
    citations: list[_Located] = Field(
        description="One entry per [CITE_n] placeholder, keyed by its number."
    )


_SYSTEM = """You are given an AI assistant's ANSWER to a clinical question. Every \
place where it cited a source has been replaced with a placeholder [CITE_1], \
[CITE_2], ... . For EACH placeholder, identify the CLAIM the citation is attached \
to — the specific statement the author is backing up at that point.

Hard rules:
- This is about PLACEMENT / INTENT, not support. Decide which statement the \
citation is attached to BY WHERE IT SITS — normally the sentence (or one or two \
sentences) immediately before the placeholder. You are NOT given the cited source \
text and you must NOT guess what it says or whether it supports anything.
- Make each claim ATOMIC and SELF-CONTAINED: if the cited sentence bundles several \
facts, return ONLY the one this placeholder is attached to; resolve pronouns and \
references so the claim stands alone; ignore markdown formatting (headers, bold, \
list markers, links).
- Use ONLY the answer's own wording — do not add, infer, or correct information.
- Return EXACTLY ONE entry for EVERY placeholder, from [CITE_1] through the last \
one, in order. Do not skip, merge, or omit any placeholder — if two placeholders \
sit on the same statement, return that statement for each of them."""


_SYSTEM_EXTRACTIVE = """You are given an AI assistant's ANSWER to a clinical \
question. Every place where it cited a source has been replaced with a placeholder \
[CITE_1], [CITE_2], ... . For EACH placeholder, identify the CLAIM the citation is \
attached to — the specific statement the author is backing up at that point.

Hard rules:
- This is about PLACEMENT / INTENT, not support. Decide which statement the \
citation is attached to BY WHERE IT SITS — normally the sentence (or one or two \
sentences) immediately before the placeholder. You are NOT given the cited source \
text and you must NOT guess what it says or whether it supports anything.
- VERBATIM: return the claim as an EXACT, contiguous substring of the answer — the \
answer's own words, character for character. Do NOT paraphrase, resolve pronouns, \
fix grammar, or add or drop any word. If the cited sentence bundles several facts, \
copy ONLY the exact clause this placeholder is attached to. (Pronouns and context \
are resolved later by the checker, not by you.) Ignore markdown formatting — do not \
copy header/list/bold markers.
- Return EXACTLY ONE entry for EVERY placeholder, from [CITE_1] through the last \
one, in order. Do not skip, merge, or omit any placeholder — if two placeholders \
sit on the same statement, return that statement for each of them."""


def _placeholderize(text: str) -> tuple[str, list[CitationAtom]]:
    """Replace each {{cite:...}} marker with `[CITE_n]` and return the placeholdered
    text plus a CitationAtom per marker (claim filled in later by the LLM). The quote
    is kept HERE but never shown to the localiser."""
    out: list[str] = []
    atoms: list[CitationAtom] = []
    last = 0
    for n, m in enumerate(CITE_RE.finditer(text or ""), start=1):
        out.append(text[last : m.start()])
        out.append(f"[CITE_{n}]")
        last = m.end()
        doc_id = m.group(1).strip()
        gid, _, sid = doc_id.partition(":")
        atoms.append(
            CitationAtom(
                claim="",  # filled from the LLM result, keyed by n
                quote=m.group(2),
                doc_id=doc_id,
                guideline_id=gid,
                section_id=sid,
            )
        )
    out.append(text[last:])
    return "".join(out), atoms


# ── caching ─────────────────────────────────────────────────────────────────────


def _cache_key(model: str, placeholdered: str, version: str) -> str:
    return cache_key(model, version, placeholdered)


def _cache_read(key: str) -> dict | None:
    return cache_read(_CACHE_DIR, key)


def _cache_write(key: str, value: dict) -> None:
    cache_write(_CACHE_DIR, key, value)


# ── localisation ────────────────────────────────────────────────────────────────


async def locate_citations(
    answer: str, model: str, use_cache: bool, *, mode: str = "abstractive"
) -> list[CitationAtom]:
    """Pair every `{{cite}}` marker with the claim it is attached to, via the LLM. `mode` picks
    the style: "abstractive" (default) rewrites the claim to stand alone; "extractive"
    returns it as a verbatim span of the answer (no rewriting). Output order matches
    `cite_parser.parse_citations` (document order), same as `segmenter.attribute_citations`.
    """
    placeholdered, atoms = _placeholderize(answer or "")
    if not atoms:
        return []

    system = _SYSTEM_EXTRACTIVE if mode == "extractive" else _SYSTEM
    key = _cache_key(model, placeholdered, _version_for(mode))
    claim_by_id: dict[int, str] | None = None
    if use_cache:
        cached = _cache_read(key)
        if cached is not None:
            claim_by_id = {int(k): v for k, v in cached.items()}

    if claim_by_id is None:
        agent = helper_agent(
            model, instructions=system, max_tokens=2000, output_type=_Localization
        )
        try:
            result = await agent.run(f"ANSWER:\n{placeholdered}")
            claim_by_id = {c.cite_id: c.claim.strip() for c in result.output.citations}
        except Exception:  # noqa: BLE001 — isolate per-answer localiser failures
            claim_by_id = {}
        if use_cache:
            _cache_write(key, {str(k): v for k, v in claim_by_id.items()})

    fallback = None
    for n, atom in enumerate(atoms, start=1):
        claim = claim_by_id.get(n, "")
        if not claim:
            if fallback is None:
                fallback = attribute_citations(answer or "")
            if n - 1 < len(fallback):
                claim = fallback[n - 1].claim
        atom.claim = claim
    return atoms


# ── CLI: eyeball / A-B against the rule engine ──────────────────────────────────


async def _run_cli(records: list[dict], model: str, use_cache: bool, compare: bool):
    sem = asyncio.Semaphore(8)

    async def _one(r: dict) -> list[CitationAtom]:
        async with sem:
            return await locate_citations(r.get("answer", ""), model, use_cache)

    llm_atoms = await asyncio.gather(*(_one(r) for r in records))

    rule_atoms = None
    if compare:
        from segmenter import attribute_citations

        rule_atoms = [attribute_citations(r.get("answer", "")) for r in records]

    for i, r in enumerate(records):
        print(f"\n=== {r.get('question_id', f'#{i}')} ===")
        for n, a in enumerate(llm_atoms[i], start=1):
            if compare and rule_atoms[i] and n <= len(rule_atoms[i]):
                ra = rule_atoms[i][n - 1]
                print(f"\n  [CITE_{n}] rule-based claim:")
                print(f"      “{ra.claim[:150]}”")
            print(f"  [CITE_{n}] LLM claim:")
            print(f"      “{a.claim[:150]}”")
            print(f"  [CITE_{n}] quote (verifies against the claim): “{a.quote[:110]}”")


def main() -> None:
    p = argparse.ArgumentParser(
        description="LLM claim↔citation localiser — run/A-B over a harness answer JSONL."
    )
    p.add_argument("--answers", required=True, help="Harness answers.jsonl path.")
    p.add_argument(
        "--model",
        default=DEFAULT_LOCATOR_MODEL,
        help=f"provider:model localiser (default {DEFAULT_LOCATOR_MODEL}).",
    )
    p.add_argument("--limit", type=int, help="Cap on answer records (smoke test).")
    p.add_argument(
        "--compare",
        action="store_true",
        help="Show the rule-based (attribute_citations) claim beside the LLM claim.",
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

    asyncio.run(_run_cli(records, args.model, not args.no_cache, args.compare))


if __name__ == "__main__":
    main()
