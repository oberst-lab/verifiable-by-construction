"""Single source of truth for resource placeholder syntax."""

from __future__ import annotations

import re

PLACEHOLDER_RE = re.compile(r"\[\[(figure|table):([\w-]+)\]\]")
RESOURCE_HREF_RE = re.compile(r"^(at|tu|t|f)(\d+)\.xhtml$")


def format_placeholder(kind: str, resource_id: str) -> str:
    """Build the inline placeholder string."""
    return f"[[{kind}:{resource_id}]]"


def resource_id_from_href(href: str) -> tuple[str, str] | None:
    """Map an in-EPUB anchor href to (kind, resource_id)."""
    m = RESOURCE_HREF_RE.match(href)
    if not m:
        return None
    prefix, num = m.group(1), m.group(2)
    kind = "figure" if prefix == "f" else "table"
    return kind, f"{prefix}{num}"
