"""Parse the inline `{{cite:doc_id|quote}}` markers the agent emits."""

from __future__ import annotations

import re
from dataclasses import dataclass

# Mirror of the frontend CITE_RE: group 1 = doc_id (no '|' or '}'), group 2 = the
# quote (anything up to the closing '}}'; may be empty).
CITE_RE = re.compile(r"\{\{cite:([^|}]+)\|([^}]*)\}\}")


@dataclass(frozen=True)
class Citation:
    """One parsed citation marker."""

    doc_id: str  # "<guideline_id>:<section_id>"
    quote: str  # verbatim span, exactly as emitted (not stripped)
    guideline_id: str
    section_id: str


def parse_citations(text: str) -> list[Citation]:
    """Extract every `{{cite:...}}` marker from an answer, in document order."""
    out: list[Citation] = []
    for m in CITE_RE.finditer(text or ""):
        doc_id = m.group(1).strip()
        quote = m.group(2)
        guideline_id, _, section_id = doc_id.partition(":")
        out.append(
            Citation(
                doc_id=doc_id,
                quote=quote,
                guideline_id=guideline_id,
                section_id=section_id,
            )
        )
    return out


def distinct_cited_docs(citations: list[Citation]) -> list[str]:
    """The set of distinct cited doc_ids, in first-seen order (the CITED set)."""
    return list(dict.fromkeys(c.doc_id for c in citations))


def strip_citations(text: str) -> str:
    """The answer prose with markers removed — for reading / claim segmentation."""
    return CITE_RE.sub("", text or "")
