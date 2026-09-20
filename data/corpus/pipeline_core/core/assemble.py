"""Section assembler: reconstruct a coherent chapter/subtree text on demand."""

from __future__ import annotations

from pathlib import Path

from .placeholders import PLACEHOLDER_RE
from .resources import load_meta
from .tree import TreeDocument


def heading_line(node, max_level: int = 6) -> str:
    hashes = "#" * min(node.level, max_level)
    if node.number:
        return f"{hashes} {node.number}. {node.title}"
    return f"{hashes} {node.title}"


def resolve_placeholders(
    text: str,
    guideline_dir: Path,
    *,
    table_format: str = "markdown",
) -> str:
    """Replace [[figure:fN]] / [[table:tN]] with formatted resource content."""

    def repl(match):
        kind = match.group(1)
        rid = match.group(2)
        try:
            meta = load_meta(guideline_dir, kind, rid)
        except FileNotFoundError:
            return match.group(0)

        label = "FIGURE" if kind == "figure" else "TABLE"
        body_parts = [f"[{label} {rid}] {meta.caption}".rstrip()]

        # Prefer real markdown table content if present (only applies to tables
        # with an actual <table> element). Otherwise fall back to the LLM-vision
        # description, which carries the content of image-only resources.
        body_added = False
        if kind == "table" and table_format == "markdown" and meta.markdown_file:
            md_path = guideline_dir / meta.markdown_file
            if md_path.exists():
                body_parts.append(md_path.read_text(encoding="utf-8").strip())
                body_added = True

        if not body_added and meta.description_file:
            desc_path = guideline_dir / meta.description_file
            if desc_path.exists():
                body_parts.append(desc_path.read_text(encoding="utf-8").strip())

        return "\n\n" + "\n\n".join(body_parts) + "\n"

    return PLACEHOLDER_RE.sub(repl, text)


def assemble(
    guideline_dir: Path,
    section_id: str,
    *,
    include_descendants: bool = True,
    resolve_resources: bool = True,
    table_format: str = "markdown",
) -> str:
    """Assemble a section + (optionally) descendants into one coherent text block."""
    tree = TreeDocument.load(guideline_dir)
    if section_id not in tree.nodes:
        raise ValueError(f"Unknown section: {section_id}")

    section_ids = (
        list(tree.walk_subtree(section_id)) if include_descendants else [section_id]
    )

    parts: list[str] = []
    for sid in section_ids:
        node = tree.get(sid)
        parts.append(heading_line(node))
        if node.text_file:
            text = (guideline_dir / node.text_file).read_text(encoding="utf-8").strip()
            if resolve_resources:
                text = resolve_placeholders(
                    text, guideline_dir, table_format=table_format
                )
            parts.append(text)
        elif not include_descendants:
            parts.append("_(container node — content lives in subsections)_")
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"
