"""EPUB-specific DOM walker. Format-agnostic helpers live in pipeline_core.core.render."""

from __future__ import annotations

import re

from bs4 import NavigableString, Tag

from pipeline_core.core.placeholders import format_placeholder, resource_id_from_href


def get_section_heading(section: Tag) -> Tag | None:
    """Return the heading element that belongs DIRECTLY to this section."""
    for cand in section.find_all(["h1", "h2", "h3", "h4", "h5", "h6"]):
        blocked = False
        for p in cand.parents:
            if p is section:
                break
            if p.name == "section":
                blocked = True
                break
        if not blocked:
            return cand
    return None


def render_direct_prose(section: Tag) -> tuple[str, list[str], list[str]]:
    """Walk this section's DOM and produce plain text + ordered placeholder lists."""
    parts: list[str] = []
    figs: list[str] = []
    tabs: list[str] = []
    own_heading = get_section_heading(section)

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

        # 1) Skip the section's own heading
        if node is own_heading:
            return

        # 2) Skip nested sections — they get their own pass
        if node.name == "section":
            return

        if node.name == "figure":
            inner_anchor = node.find("a", href=True)
            if inner_anchor:
                rid = resource_id_from_href(inner_anchor.get("href", ""))
                if rid:
                    kind, rid_str = rid
                    seen = rid_str in (figs if kind == "figure" else tabs)
                    if not seen:
                        if kind == "figure":
                            figs.append(rid_str)
                        else:
                            tabs.append(rid_str)
                        emit(format_placeholder(kind, rid_str))
            return

        # 4) <a href="...xhtml">: convert to placeholder if it's a resource link
        if node.name == "a":
            href = node.get("href", "")
            rid = resource_id_from_href(href)
            if rid:
                kind, rid_str = rid
                if kind == "figure":
                    if rid_str not in figs:
                        figs.append(rid_str)
                else:
                    if rid_str not in tabs:
                        tabs.append(rid_str)
                placeholder = format_placeholder(kind, rid_str)
                # collapse repeated placeholders separated by whitespace only
                if parts and parts[-1].rstrip().endswith(placeholder):
                    return
                emit(placeholder)
                return
            # other anchors: take inner text, drop the link wrapper
            for child in node.children:
                walk(child)
            return

        # 5) Block-level elements add paragraph breaks
        is_block = node.name in {
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
        }
        if is_block and parts and not parts[-1].endswith("\n"):
            parts.append("\n")

        for child in node.children:
            walk(child)

        if is_block:
            parts.append("\n")

    for child in section.children:
        walk(child)

    raw = "".join(parts)
    # normalize whitespace
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r" *\n *", "\n", raw)
    raw = re.sub(r"\n{3,}", "\n\n", raw)
    raw = raw.strip()
    return raw, figs, tabs
