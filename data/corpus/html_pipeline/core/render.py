"""HTML → plain text rendering for the html_pipeline."""

from __future__ import annotations

import re
from typing import Optional

from bs4 import NavigableString, Tag

# Reuse the placeholder format + table-to-markdown converter from
# pipeline_core so there's a single source of truth.
from pipeline_core.core.placeholders import format_placeholder  # noqa: F401
from pipeline_core.core.render import table_html_to_markdown  # noqa: F401


# ── legacy id parsing ────────────────────────────────────────────────────────


def legacy_id_runs(legacy_id: str) -> list[str]:
    """Split a Silverchair legacy section id into alternating alpha/digit runs."""
    if not legacy_id or not legacy_id.startswith("s") or len(legacy_id) < 2:
        return []
    body = legacy_id[1:]
    runs: list[str] = []
    cur = body[0]
    cur_is_digit = cur.isdigit()
    for ch in body[1:]:
        if ch.isdigit() == cur_is_digit:
            cur += ch
        else:
            runs.append(cur)
            cur = ch
            cur_is_digit = ch.isdigit()
    runs.append(cur)
    return runs


def parent_legacy_id(legacy_id: str, existing_ids: set[str]) -> Optional[str]:
    """Return the nearest existing ancestor legacy id, or None if at the top."""
    runs = legacy_id_runs(legacy_id)
    if len(runs) <= 1:
        return None
    runs = runs[:-1]
    while runs:
        candidate = "s" + "".join(runs)
        if candidate in existing_ids:
            return candidate
        runs = runs[:-1]
    return None


HEADING_LEVEL = {"h2": 1, "h3": 2, "h4": 3, "h5": 4}


def heading_depth(tag_name: str) -> int:
    """``h2`` → 1, ``h3`` → 2, ``h4`` → 3, ``h5`` → 4."""
    return HEADING_LEVEL.get(tag_name, 5)


def is_section_heading(tag: Tag) -> bool:
    """A heading is sectioning iff it's an h2–h5 with class ``section-title``."""
    if not isinstance(tag, Tag):
        return False
    if tag.name not in HEADING_LEVEL:
        return False
    return "section-title" in (tag.get("class") or [])


# ── DOM filtering ────────────────────────────────────────────────────────────


_SKIP_CLASS_TOKENS = {
    "fig-modal",  # duplicate modal version of a figure
    "table-modal",  # duplicate modal version of a table
    "ref-list",  # bibliography
    "backreferences-section-jumplink",
    "permissionstatement-section-wrapper",
    "article-metadata-panel",
    "article-metadata-standalone-panel",
    "figshare-wrapper",
    "metadata-author-listing",
    "author-affiliations",
    "fig-orig",  # "View large / Download slide" actions inside a figure
    "fig-link",  # outer anchor wrapper around the figure image
    "screenreader-text",
    "hidden",
    "sr-only",
}

_SKIP_TAGS = {"script", "style", "noscript", "svg"}


_SKIP_IDS = {
    "sr-fig-viewer-action",  # "Open figure viewer" overlay span
}


def is_skippable(node: Tag) -> bool:
    if not isinstance(node, Tag):
        return False
    if node.name in _SKIP_TAGS:
        return True
    if node.get("id") in _SKIP_IDS:
        return True
    if node.get("aria-hidden") == "true":
        return True
    classes = set(node.get("class") or [])
    return bool(classes & _SKIP_CLASS_TOKENS)


def is_figure_block(node: Tag) -> bool:
    """A figure container (the visible one, not the modal)."""
    if not isinstance(node, Tag) or node.name != "div":
        return False
    classes = set(node.get("class") or [])
    return "fig-section" in classes and "fig-modal" not in classes


def is_table_block(node: Tag) -> bool:
    """A table container (the visible one, not the modal)."""
    if not isinstance(node, Tag) or node.name != "div":
        return False
    classes = set(node.get("class") or [])
    return "table-overflow" in classes and "table-modal" not in classes


# ── prose rendering ──────────────────────────────────────────────────────────


def render_prose(
    elements: list[Tag],
    *,
    figure_ids_by_data_id: dict[str, str],
    table_counter: list[int],
    chapter_id: str,
) -> tuple[str, list[str], list[str]]:
    """Walk a list of sibling DOM blocks and return ``(text, figs, tables)``."""
    parts: list[str] = []
    fig_ids: list[str] = []
    tab_ids: list[str] = []

    def emit(s: str) -> None:
        if s:
            parts.append(s)

    def walk(node) -> None:
        if isinstance(node, NavigableString):
            txt = str(node)
            if txt.strip():
                emit(txt)
            return

        if not isinstance(node, Tag):
            return

        if is_skippable(node):
            return

        # Figure container → emit placeholder once, register figure id.
        if is_figure_block(node):
            data_id = (node.get("data-id") or "").strip()
            if not data_id:
                # fallback: assign a sequential id based on map size
                data_id = f"F{len(figure_ids_by_data_id) + 1}"
            rid = figure_ids_by_data_id.get(data_id)
            if rid is None:
                rid = f"{chapter_id}-f{len(figure_ids_by_data_id) + 1}"
                figure_ids_by_data_id[data_id] = rid
            if rid not in fig_ids:
                fig_ids.append(rid)
            emit(_block_break(parts))
            emit(format_placeholder("figure", rid))
            emit("\n")
            return

        # Table container → emit placeholder, assign sequential id.
        if is_table_block(node):
            table_counter[0] += 1
            rid = f"{chapter_id}-t{table_counter[0]}"
            if rid not in tab_ids:
                tab_ids.append(rid)
            emit(_block_break(parts))
            emit(format_placeholder("table", rid))
            emit("\n")
            return

        # Block-level elements add paragraph breaks.
        if node.name in _BLOCK_TAGS:
            if parts and not parts[-1].endswith("\n"):
                parts.append("\n")

        for child in node.children:
            walk(child)

        if node.name in _BLOCK_TAGS:
            parts.append("\n")

    for el in elements:
        walk(el)

    raw = "".join(parts)
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r" *\n *", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    return raw.strip(), fig_ids, tab_ids


_BLOCK_TAGS = {
    "p",
    "div",
    "li",
    "ul",
    "ol",
    "table",
    "tr",
    "td",
    "th",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "br",
    "blockquote",
}


def _block_break(parts: list[str]) -> str:
    if parts and not parts[-1].endswith("\n"):
        return "\n"
    return ""


# ── figure / table asset extraction ──────────────────────────────────────────

_LOCAL_IMG_RE = re.compile(r"\./(?P<dir>[^/]+_files)/(?P<file>[^?]+)")


def extract_figure_assets(
    fig_block: Tag,
) -> tuple[str, str, Optional[str]]:
    """Pull (label, caption, local_img_filename) from a fig-section block."""
    label_div = fig_block.find("div", class_="fig-label")
    label = label_div.get_text(" ", strip=True) if label_div else ""

    cap_div = fig_block.find("div", class_=lambda c: c and "fig-caption" in c)
    if cap_div is None:
        cap_div = fig_block.find("div", class_="caption")
    caption = cap_div.get_text(" ", strip=True) if cap_div else ""

    img = fig_block.find("img")
    local_name: Optional[str] = None
    if img:
        src = (img.get("src") or "").strip()
        if src:
            m = _LOCAL_IMG_RE.search(src)
            if m:
                local_name = m.group("file")
            else:
                # last-resort: take the basename of whatever src is
                local_name = src.split("/")[-1] or None

    return label, caption, local_name


def extract_table_assets(table_block: Tag) -> tuple[str, str, str]:
    """Pull (caption, raw_html, markdown) from a table-overflow block."""
    table_tag = table_block.find("table")
    html_str = str(table_tag) if table_tag else str(table_block)
    md = table_html_to_markdown(table_tag) if table_tag else ""

    caption = ""
    cap_node = table_block.find("caption") or table_block.find(
        "div", class_=lambda c: c and "table-caption" in c
    )
    if cap_node:
        caption = cap_node.get_text(" ", strip=True)
    else:
        # Try the immediately-preceding sibling — Silverchair often places
        # the caption right before the table-overflow div.
        prev = table_block.find_previous_sibling()
        if prev and isinstance(prev, Tag):
            classes = set(prev.get("class") or [])
            if {"table-caption", "caption"} & classes:
                caption = prev.get_text(" ", strip=True)

    return caption, html_str, md
