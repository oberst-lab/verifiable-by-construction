"""Deterministic sentence segmentation for answer text, citation-marker aware."""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

from system.cite_parser import (
    CITE_RE,
)  # single source of truth for the {{cite}} grammar

with warnings.catch_warnings():  # pysbd 0.3.4 ships old-style regex escapes
    warnings.simplefilter("ignore", SyntaxWarning)
    import pysbd

SEGMENTER_VERSION = "pysbd-v1"
CLAIM_VALIDITY_VERSION = "claim-validity-v2"

_PYSBD = pysbd.Segmenter(language="en", clean=False, char_span=True)

# Patch-1 gate: pySBD refuses to split inside long straight-quote passages,
# merging real sentences into 300+ char units. Such spans are re-split with the
# owned rule engine (kept below as the fallback).
_RESPLIT_MIN_CHARS = 300
_RESPLIT_CUE = re.compile(r'[.!?]["”]\s+\S')

# Tokens ending in '.' that do NOT end a sentence (general + clinical). Lowercased,
# trailing dots stripped before lookup. Decimals and single-letter initials are
# handled separately in _is_boundary.
_ABBREV = frozenset(
    {
        "e.g",
        "i.e",
        "eg",
        "ie",
        "vs",
        "etc",
        "cf",
        "al",
        "et al",
        "fig",
        "no",
        "eq",
        "approx",
        "dr",
        "mr",
        "mrs",
        "ms",
        "prof",
        "st",
        "ca",
        "vol",
        "ref",
        "incl",
        "approx",
        "max",
        "min",
    }
)

# Closing punctuation that can trail a sentence-final mark, e.g. (…). or "…".
_TRAILING = set(".\"'")


@dataclass
class Sentence:
    text: str  # the claim sentence, markers stripped, whitespace-collapsed
    n_citations: int  # citation markers attached to this sentence


@dataclass
class CitationAtom:
    """One `(claim sentence, cited quote)` pair — the unit the judge metrics score.
    `claim` is the sentence the marker attaches to (markers stripped, whitespace
    collapsed); `quote`/`doc_id` come verbatim from the marker."""

    claim: str
    quote: str
    doc_id: str  # "<guideline_id>:<section_id>"
    guideline_id: str
    section_id: str


def _is_boundary(text: str, i: int) -> bool:
    """True if `text[i]` (one of . ? !) ends a sentence. `?`/`!` always do; `.`
    does unless it is a decimal point or part of a protected abbreviation."""
    ch = text[i]
    if ch in "?!":
        return True
    # ch == '.': decimal like 0.5 — digit before AND digit after the dot.
    if i > 0 and text[i - 1].isdigit() and i + 1 < len(text) and text[i + 1].isdigit():
        return False
    # The alpha(.)-run ending right before the dot — protected abbreviation?
    k = i - 1
    while k >= 0 and (text[k].isalpha() or text[k] == "."):
        k -= 1
    token = text[k + 1 : i].lower().strip(".")
    if token in _ABBREV:
        return False
    if len(token) == 1 and token.isalpha():  # single-letter initial "A."
        return False
    return True


def _push_span(spans: list[tuple[int, int]], text: str, start: int, end: int) -> None:
    """Append `[start, end)` with leading whitespace trimmed off. Spans must NOT
    own the gap before their first word: a citation marker sitting in the
    inter-sentence gap has to fall OUTSIDE every span so `_attribute` can route it
    to the *preceding* sentence (the documented ALCE/WebCiteS rule).
    """
    while start < end and text[start].isspace():
        start += 1
    if start < end:
        spans.append((start, end))


def _rules_split_spans(text: str) -> list[tuple[int, int]]:
    """The owned rule engine (former rules-v4 boundary detector), retained as the patch-1
    fallback: split at `. ? !` (minus decimals/abbreviations, and only when followed by
    whitespace/EOT) and at newlines. Robust to clinical 'mm Hg' / numeric ranges by
    construction.
    """
    spans: list[tuple[int, int]] = []
    n = len(text)
    start = 0
    i = 0
    while i < n:
        ch = text[i]
        if ch == "\n":
            if text[start:i].strip():
                _push_span(spans, text, start, i)
            start = i + 1
            i += 1
            continue
        if ch in ".?!" and _is_boundary(text, i):
            j = i + 1
            while j < n and text[j] in _TRAILING:  # consume trailing )."' etc.
                j += 1
            if j >= n or text[j].isspace():  # real boundary only before space/EOT
                if text[start:j].strip():
                    _push_span(spans, text, start, j)
                start = j
                i = j
                continue
        i += 1
    if text[start:n].strip():
        _push_span(spans, text, start, n)
    return spans


def split_sentence_spans(text: str) -> list[tuple[int, int]]:
    """Sentence `[start, end)` spans via pySBD (Golden-Rules engine, char spans),
    with two owned adjustments:
    """
    spans: list[tuple[int, int]] = []
    for sp in _PYSBD.segment(text or ""):
        s, e = sp.start, sp.end
        seg = text[s:e]
        if len(seg) >= _RESPLIT_MIN_CHARS and _RESPLIT_CUE.search(seg):
            sub = _rules_split_spans(seg)
            if len(sub) > 1:  # fallback found real internal boundaries
                for a, b in sub:
                    _push_span(spans, text, s + a, s + b)
                continue
        _push_span(spans, text, s, e)
    return spans


def _strip_markers(text: str) -> tuple[str, list[int]]:
    """Remove {{cite:...}} markers; return (stripped_text, marker_offsets) where
    each offset is the position in stripped_text at which a marker had sat."""
    out: list[str] = []
    offsets: list[int] = []
    last = 0
    grown = 0  # length of stripped text accumulated so far
    for m in CITE_RE.finditer(text):
        chunk = text[last : m.start()]
        out.append(chunk)
        grown += len(chunk)
        offsets.append(grown)  # marker sat right here in the stripped text
        last = m.end()
    out.append(text[last:])
    return "".join(out), offsets


def _attribute(pos: int, spans: list[tuple[int, int]]) -> int | None:
    """Index of the sentence a marker at `pos` belongs to: the sentence whose span
    contains pos, else the last sentence ending at/before pos (the preceding
    claim), else the first sentence."""
    preceding = None
    for i, (start, end) in enumerate(spans):
        if start <= pos < end:
            return i
        if end <= pos:
            preceding = i
    if preceding is not None:
        return preceding
    return 0 if spans else None


def _collapse(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")  # [text](url) -> text
_MD_HEADING_LINE = re.compile(r"(?m)^[ \t]*#{1,6}[ \t]+.*$")  # "## Heading" line
_MD_LIST_PREFIX = re.compile(
    r"(?m)^[ \t]*(?:[-*+]|\d+[.)])[ \t]+"
)  # "- ", "1. ", "2) "
_MD_EMPHASIS = re.compile(r"[*`]")  # **bold** *italic* `code` emphasis runs
_MD_LABEL_LINE = re.compile(
    r"(?m)^[ \t]*[^.!?\n]*:[ \t]*$"
)  # "Recommendations:" lead-in


def _normalize_chunk(s: str) -> str:
    """Flatten markdown in one non-marker text chunk. Order matters twice: strip emphasis
    before list-prefix detection so a bold-wrapped numbered heading (`**1. Heading**`)
    reduces to `1.
    """
    s = _MD_LINK.sub(r"\1", s)
    s = _MD_HEADING_LINE.sub("", s)  # drop heading lines entirely
    s = _MD_EMPHASIS.sub("", s)  # strip emphasis chars
    s = _MD_LIST_PREFIX.sub("", s)  # strip bullet / enumeration markers
    s = _MD_LABEL_LINE.sub("", s)  # drop bold-only / colon-terminated lead-in lines
    return s


def _normalize_markdown(text: str) -> str:
    """Flatten markdown OUTSIDE every {{cite}} marker; markers are re-emitted
    byte-for-byte so their quotes and recorded positions are unchanged."""
    out: list[str] = []
    last = 0
    for m in CITE_RE.finditer(text):
        out.append(_normalize_chunk(text[last : m.start()]))
        out.append(m.group(0))  # marker untouched
        last = m.end()
    out.append(_normalize_chunk(text[last:]))
    return "".join(out)


def _collapse_spaces_with_offsets(
    text: str, offsets: list[int]
) -> tuple[str, list[int]]:
    """Collapse runs of spaces/tabs to a single space, remapping marker offsets.
    Stripping a marker joins its two surrounding chunks and leaves a double space
    at the seam (`.”␣␣Next`) — which blocks pySBD's quote-boundary rules. Newlines
    are preserved (they are boundaries in their own right)."""
    out: list[str] = []
    remap: list[int] = []
    j = 0  # next offset to remap (offsets are sorted)
    i = 0
    n = len(text)
    while i <= n:
        while j < len(offsets) and offsets[j] == i:
            # A marker mid-space-run must stay INSIDE the collapsed gap (on the
            # single space already emitted), not slide onto the next sentence's
            # first character — that would re-introduce the v3 drift.
            if i < n and text[i] in " \t" and out and out[-1] == " ":
                remap.append(len(out) - 1)
            else:
                remap.append(len(out))
            j += 1
        if i == n:
            break
        ch = text[i]
        if ch in " \t":
            if not (out and out[-1] == " "):
                out.append(" ")
            i += 1
            continue
        out.append(ch)
        i += 1
    return "".join(out), remap


def _segment(text: str) -> tuple[str, list[int], list[tuple[int, int]]]:
    """Shared decomposition core: normalize markdown, strip {{cite}} markers (keeping the
    offset each sat at), collapse spaces (remapping offsets), and split into sentence spans.
    Every claim decomposition starts here, so segment_marker_map / claim_malformed_flags /
    segment_with_citations all see byte-identical spans."""
    stripped, offsets = _strip_markers(_normalize_markdown(text or ""))
    stripped, offsets = _collapse_spaces_with_offsets(stripped, offsets)
    return stripped, offsets, split_sentence_spans(stripped)


def segment_with_citations(text: str) -> list[Sentence]:
    """Split `text` into claim sentences, each tagged with its citation count."""
    stripped, offsets, spans = _segment(text)
    counts = [0] * len(spans)
    for pos in offsets:
        idx = _attribute(pos, spans)
        if idx is not None:
            counts[idx] += 1
    return [
        Sentence(text=_collapse(stripped[s:e]), n_citations=counts[i])
        for i, (s, e) in enumerate(spans)
    ]


def segment_marker_map(text: str) -> tuple[list[str], list[int | None]]:
    """Sentence texts plus the sentence index each `{{cite}}` marker attributes
    to (in-span → that sentence; gap → preceding — the shared ALCE rule).
    The common decomposition behind Coverage's window credit and Claim
    Support's territorial pairing."""
    stripped, offsets, spans = _segment(text)
    texts = [_collapse(stripped[s:e]) for s, e in spans]
    return texts, [_attribute(pos, spans) for pos in offsets]


def segment_marker_map_with_validity(
    text: str,
) -> tuple[list[str], list[int | None], list[bool]]:
    """`segment_marker_map` PLUS per-sentence malformed flags, in ONE segmentation pass
    (avoids re-segmenting when a caller needs both — e.g. Claim Support's decompose).
    Returns (sentence texts, marker→sentence attribution, malformed flags)."""
    stripped, offsets, spans = _segment(text)
    texts = [_collapse(stripped[s:e]) for s, e in spans]
    marker_sents: list[int | None] = []
    flags = [False] * len(spans)
    for pos in offsets:
        idx = _attribute(pos, spans)
        marker_sents.append(idx)
        if idx is not None and _marker_midclause_collapse(
            stripped[:pos], stripped[pos:]
        ):
            flags[idx] = True
    return texts, marker_sents, flags


def split_sentences(text: str) -> list[str]:
    """Claim sentences only (markers stripped)."""
    return [s.text for s in segment_with_citations(text)]


_INCOMPLETE_BEFORE = frozenset(
    "because since as such based on including include includes namely comprising like than "
    "from to of for with by that which when while if the a an is are was were "
    "be being about into at via should must can may will shows showing demonstrating "
    "indicating evaluate assess assesses emphasize emphasizes highlight highlights reassure "
    "normalize note noting following ensure ensures targets target recommends".split()
)


def _clean_before_own(before: str) -> str:
    """Trim trailing whitespace and the model's own closing quotes/braces/asterisks so the
    real last content char (and any sentence-ending period) is exposed."""
    return re.sub(r"[”\"’'\)\*\|\]\s]+$", "", before)


def _after_continues(after: str) -> bool:
    """True only if content continues on the SAME line after the marker. Horizontal whitespace
    and a leftover comma/semicolon from a stripped adjacent marker are skipped, but a
    NEWLINE ends the line: a marker before it is terminal, so list-item and paragraph-end
    citations are kept (the terminal-keep invariant).
    """
    a = re.sub(r"^[ \t,;]+", "", after)
    return bool(a) and a[0] not in ".!?\r\n"


def _marker_midclause_collapse(before: str, after: str) -> bool:
    """Form-4 only: the stripped marker sat where a grammatical complement belongs
    (incomplete own-text before) AND the sentence continues depending on it."""
    b = _clean_before_own(before)
    if not b or b[-1] in ".!?":
        return False  # sentence already ended before the citation -> keep
    m = re.search(r"([A-Za-z][A-Za-z'/-]*)$", b)
    last_raw = m.group(1) if m else ""
    incomplete = (last_raw.islower() and last_raw in _INCOMPLETE_BEFORE) or b[-1] == ":"
    if not incomplete:
        return False  # complete clause + citation -> keep
    return _after_continues(after)  # continues -> collapse; terminal -> keep


def claim_malformed_flags(text: str) -> list[bool]:
    """Per-sentence malformed flags, aligned 1:1 with `segment_marker_map(text)[0]`. True = a
    mid-clause citation collapse makes the claim un-judgeable (exclude it). Standalone entry
    point for the flags alone; callers needing texts + attribution too should use
    `segment_marker_map_with_validity` (one segmentation pass).
    """
    stripped, offsets, spans = _segment(text)
    flags = [False] * len(spans)
    for pos in offsets:
        if _marker_midclause_collapse(stripped[:pos], stripped[pos:]):
            idx = _attribute(pos, spans)
            if idx is not None:
                flags[idx] = True
    return flags


def attribute_citations(text: str) -> list[CitationAtom]:
    """Pair every `{{cite}}` marker (in document order) with the claim sentence it attaches to.
    Same marker→sentence convention as `segment_with_citations` (ALCE/WebCiteS: the sentence
    containing the marker, else the preceding one) — so the claim-side unit is located
    deterministically. Output order matches `cite_parser.parse_citations`.
    """
    src = _normalize_markdown(text or "")
    chunks: list[str] = []
    markers: list[tuple[int, str, str]] = []  # (offset_in_stripped, doc_id, quote)
    last = 0
    grown = 0
    for m in CITE_RE.finditer(src):
        chunk = src[last : m.start()]
        chunks.append(chunk)
        grown += len(chunk)
        markers.append((grown, m.group(1).strip(), m.group(2)))
        last = m.end()
    chunks.append(src[last:])
    stripped = "".join(chunks)
    stripped, remapped = _collapse_spaces_with_offsets(
        stripped, [pos for pos, _, _ in markers]
    )
    markers = [
        (new_pos, doc_id, quote)
        for new_pos, (_, doc_id, quote) in zip(remapped, markers)
    ]

    spans = split_sentence_spans(stripped)
    sent_texts = [_collapse(stripped[s:e]) for s, e in spans]

    out: list[CitationAtom] = []
    for pos, doc_id, quote in markers:
        idx = _attribute(pos, spans)
        claim = sent_texts[idx] if idx is not None else ""
        gid, _, sid = doc_id.partition(":")
        out.append(
            CitationAtom(
                claim=claim,
                quote=quote,
                doc_id=doc_id,
                guideline_id=gid,
                section_id=sid,
            )
        )
    return out
