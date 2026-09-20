"""Format-agnostic helpers shared by epub_pipeline and html_pipeline renderers."""

from __future__ import annotations

import re
from typing import Iterable

from bs4 import Tag


_NUMBER_RE = re.compile(r"^(\d+(?:\.\d+)*)\.?\s+(.*)")


def parse_number_from_heading(text: str) -> tuple[str | None, str]:
    """Extract leading numeric prefix from a heading."""
    m = _NUMBER_RE.match(text.strip())
    if not m:
        return None, text.strip()
    number = m.group(1)
    first_part = number.split(".")[0]
    # Reject 4+ digit prefixes with no dot (e.g. publication years)
    if "." not in number and len(first_part) >= 3:
        return None, text.strip()
    return number, m.group(2).strip()


def table_html_to_markdown(table: Tag) -> str:
    """Naive table → markdown conversion. Sufficient for common grid layouts."""
    rows = []
    for tr in table.find_all("tr"):
        cells = []
        for cell in tr.find_all(["th", "td"]):
            txt = re.sub(r"\s+", " ", cell.get_text(" ", strip=True))
            txt = txt.replace("|", "\\|")
            cells.append(txt)
        if cells:
            rows.append(cells)
    if not rows:
        return ""

    out = []
    header = rows[0]
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows[1:]:
        if len(r) < len(header):
            r = r + [""] * (len(header) - len(r))
        elif len(r) > len(header):
            r = r[: len(header)]
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def classify_section(
    section_id: str, path: list[str], has_children: bool, has_text: bool
) -> str:
    """Assign a coarse section_type label for downstream filtering and display."""
    if not path:
        return "root"
    if section_id in ("frontmatter", "bodymatter", "backmatter"):
        return "root"
    if section_id.startswith("bibliography"):
        return "bibliography"
    if section_id.startswith("appendix"):
        return "appendix"
    if section_id in ("affiliations", "article-information"):
        return "metadata"
    if section_id.startswith("abs"):
        return "abstract"
    if len(path) <= 2:
        return "chapter"
    if has_children and not has_text:
        return "container"
    return "subsection"


def is_rag_eligible(
    section_id: str,
    path: list[str],
    title: str = "",
    *,
    exclude_subtrees: Iterable[str] = (),
    exclude_id_prefixes: Iterable[str] = (),
    exclude_ids: Iterable[str] = (),
    exclude_titles: Iterable[str] = (),
) -> bool:
    """Apply config-driven rules to decide whether this section is RAG content."""
    if section_id in set(exclude_ids):
        return False
    excluded_subtree_set = set(exclude_subtrees)
    if any(ancestor in excluded_subtree_set for ancestor in path):
        return False
    for prefix in exclude_id_prefixes:
        if section_id.startswith(prefix):
            return False
    if title and title in set(exclude_titles):
        return False
    return True
