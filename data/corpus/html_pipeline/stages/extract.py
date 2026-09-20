"""Stage 1: extract a directory of locally-saved AHA/Silverchair HTML chapter
files into the same tree-based schema produced by ``epub_pipeline``.
"""

from __future__ import annotations

import logging
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup, Tag

# Shared infrastructure from epub_pipeline (format-agnostic).
from pipeline_core.core.render import classify_section, is_rag_eligible
from pipeline_core.core.resources import ResourceMeta, save_meta
from pipeline_core.core.tree import SectionNode, TreeDocument

# HTML-specific helpers.
from ..core.render import (
    extract_figure_assets,
    extract_table_assets,
    is_section_heading,
    parent_legacy_id,
    render_prose,
)

logger = logging.getLogger(__name__)


# ── public entry point ──────────────────────────────────────────────────────


def run(
    *,
    guideline_id: str,
    guideline_name: str,
    html_dir: Path,
    guideline_dir: Path,
    extract_config: dict,
    force: bool = False,
    clear_downstream: bool = False,
) -> None:
    """Extract a folder of HTML chapter files into ``guideline_dir``."""
    if not html_dir.exists() or not html_dir.is_dir():
        raise FileNotFoundError(f"HTML source directory not found: {html_dir}")

    sections_dir = guideline_dir / "sections"
    fig_dir = guideline_dir / "resources" / "figures"
    tab_dir = guideline_dir / "resources" / "tables"
    tree_json = guideline_dir / "tree.json"

    if tree_json.exists() and not force:
        raise FileExistsError(
            f"{tree_json} already exists; pass --force-stage extract to overwrite."
        )

    _clear_extract_outputs(
        sections_dir, fig_dir, tab_dir, clear_downstream=clear_downstream
    )
    sections_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    exclude_subtrees = extract_config.get("exclude_subtrees", [])
    exclude_id_prefixes = extract_config.get("exclude_id_prefixes", [])
    exclude_ids = extract_config.get("exclude_ids", ["bodymatter"])
    exclude_titles = extract_config.get("exclude_titles", ["References"])
    min_text_length = int(extract_config.get("min_text_length", 0))

    chapter_files = _discover_chapter_files(html_dir)
    if not chapter_files:
        raise FileNotFoundError(f"No HTML files found in {html_dir}")
    logger.info(
        "📚 Extracting %d HTML chapter file(s) from %s → %s",
        len(chapter_files),
        html_dir,
        guideline_dir,
    )

    nodes: dict[str, SectionNode] = {}
    figure_meta: dict[str, ResourceMeta] = {}
    table_meta: dict[str, ResourceMeta] = {}

    root_id = "bodymatter"
    nodes[root_id] = SectionNode(
        section_id=root_id,
        title=guideline_name,
        number=None,
        level=1,
        parent_id=None,
        path=[root_id],
    )

    for chapter_num, html_path in chapter_files:
        _process_chapter(
            chapter_num=chapter_num,
            html_path=html_path,
            html_dir=html_dir,
            root_id=root_id,
            nodes=nodes,
            figure_meta=figure_meta,
            table_meta=table_meta,
            sections_dir=sections_dir,
            fig_dir=fig_dir,
            tab_dir=tab_dir,
            min_text_length=min_text_length,
        )

    # Link parents → children.
    for sid, node in nodes.items():
        if node.parent_id and node.parent_id in nodes:
            nodes[node.parent_id].children.append(sid)

    # Classify + apply RAG eligibility.
    for sid, node in nodes.items():
        node.section_type = classify_section(
            sid,
            node.path,
            has_children=bool(node.children),
            has_text=bool(node.text_file),
        )
        node.rag_eligible = is_rag_eligible(
            sid,
            node.path,
            node.title,
            exclude_subtrees=exclude_subtrees,
            exclude_id_prefixes=exclude_id_prefixes,
            exclude_ids=exclude_ids,
            exclude_titles=exclude_titles,
        )

    # Persist meta files for every resource.
    for meta in figure_meta.values():
        _attach_existing_description(meta, fig_dir)
        save_meta(guideline_dir, meta)
    for meta in table_meta.values():
        _attach_existing_description(meta, tab_dir)
        save_meta(guideline_dir, meta)

    tree = TreeDocument(
        guideline_id=guideline_id,
        source=html_dir.name,
        nodes=nodes,
        resource_index_figures=sorted(figure_meta.keys(), key=_resource_sort_key),
        resource_index_tables=sorted(table_meta.keys(), key=_resource_sort_key),
    )
    tree.save(guideline_dir)

    _log_summary(nodes, figure_meta, table_meta, guideline_dir)


# ── chapter processing ──────────────────────────────────────────────────────


def _process_chapter(
    *,
    chapter_num: int,
    html_path: Path,
    html_dir: Path,
    root_id: str,
    nodes: dict[str, SectionNode],
    figure_meta: dict[str, ResourceMeta],
    table_meta: dict[str, ResourceMeta],
    sections_dir: Path,
    fig_dir: Path,
    tab_dir: Path,
    min_text_length: int,
) -> None:
    chapter_id = f"ch{chapter_num:02d}"
    soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")

    body = _find_article_body(soup)
    if body is None:
        logger.warning("  ! no widget-ArticleFulltext found in %s", html_path.name)
        return

    container = _find_section_container(body)
    if container is None:
        logger.warning(
            "  ! no widget-items container with section headings in %s", html_path.name
        )
        return

    chapter_title = _chapter_title_from(soup, html_path)
    chapter_number = str(chapter_num)

    children_in_order = [c for c in container.children if isinstance(c, Tag)]

    # Split children into: chapter intro elements + per-heading spans.
    intro_elements, spans = _split_into_spans(children_in_order)

    # Walk spans to collect the set of legacy ids that actually exist before
    # we try to infer parents (this matters when the source skips a depth).
    existing_legacy_ids = {h.get("data-legacyid", "") for h, _ in spans}
    existing_legacy_ids.discard("")

    figure_ids_by_data_id: dict[str, str] = {}
    table_counter = [0]
    chapter_fig_ids: list[str] = []
    chapter_tab_ids: list[str] = []
    files_dir = _files_sibling_dir(html_path)

    # Chapter node — direct prose = intro paragraphs before first heading.
    intro_text, intro_figs, intro_tabs = render_prose(
        intro_elements,
        figure_ids_by_data_id=figure_ids_by_data_id,
        table_counter=table_counter,
        chapter_id=chapter_id,
    )
    chapter_path = [root_id, chapter_id]
    chapter_node = SectionNode(
        section_id=chapter_id,
        title=chapter_title,
        number=chapter_number,
        level=2,
        parent_id=root_id,
        path=chapter_path,
        figures=list(intro_figs),
        tables=list(intro_tabs),
    )
    nodes[chapter_id] = chapter_node
    chapter_fig_ids.extend(intro_figs)
    chapter_tab_ids.extend(intro_tabs)
    _write_section_text(
        sections_dir, chapter_id, intro_text, chapter_node, min_text_length
    )

    # Per-heading subsection nodes.
    for heading_tag, span_elements in spans:
        legacy = (heading_tag.get("data-legacyid") or "").strip()
        if not legacy:
            continue
        raw_title = heading_tag.get_text("", strip=True) or heading_tag.get(
            "data-section-title", ""
        )
        title = re.sub(r"\s+", " ", raw_title).strip()
        section_id = f"{chapter_id}-{legacy}"
        parent_legacy = parent_legacy_id(legacy, existing_legacy_ids)
        if parent_legacy is None:
            parent_id = chapter_id
        else:
            parent_id = f"{chapter_id}-{parent_legacy}"

        parent_node = nodes.get(parent_id)
        if parent_node is None:
            # Defensive fallback: hang it off the chapter if its computed
            # parent is missing for any reason.
            parent_id = chapter_id
            parent_node = nodes[chapter_id]
        section_path = parent_node.path + [section_id]
        level = parent_node.level + 1

        text, figs, tabs = render_prose(
            span_elements,
            figure_ids_by_data_id=figure_ids_by_data_id,
            table_counter=table_counter,
            chapter_id=chapter_id,
        )
        node = SectionNode(
            section_id=section_id,
            title=title,
            number=None,
            level=level,
            parent_id=parent_id,
            path=section_path,
            figures=list(figs),
            tables=list(tabs),
        )
        nodes[section_id] = node
        chapter_fig_ids.extend(figs)
        chapter_tab_ids.extend(tabs)
        _write_section_text(sections_dir, section_id, text, node, min_text_length)

    # Populate resource metadata + assets for everything that this chapter
    # actually emitted a placeholder for.
    _populate_chapter_figures(
        container=container,
        figure_ids_by_data_id=figure_ids_by_data_id,
        files_dir=files_dir,
        fig_dir=fig_dir,
        figure_meta=figure_meta,
        section_owners=_section_owners(nodes, chapter_id, "figures"),
    )
    _populate_chapter_tables(
        container=container,
        chapter_id=chapter_id,
        table_count=table_counter[0],
        tab_dir=tab_dir,
        table_meta=table_meta,
        section_owners=_section_owners(nodes, chapter_id, "tables"),
    )


# ── chapter discovery ──────────────────────────────────────────────────────


_CHAPTER_NUM_RE = re.compile(r"^(\d+)\.\s")


def _discover_chapter_files(html_dir: Path) -> list[tuple[int, Path]]:
    """Find chapter HTML files and return them ordered by their leading number."""
    out: list[tuple[int, Path]] = []
    for p in html_dir.glob("*.html"):
        m = _CHAPTER_NUM_RE.match(p.name)
        if not m:
            continue
        out.append((int(m.group(1)), p))
    out.sort(key=lambda t: (t[0], t[1].name))
    return out


def _files_sibling_dir(html_path: Path) -> Optional[Path]:
    """Return the ``<basename>_files`` sibling directory if it exists."""
    candidate = html_path.with_name(html_path.stem + "_files")
    return candidate if candidate.is_dir() else None


# ── DOM navigation ─────────────────────────────────────────────────────────


def _find_article_body(soup: BeautifulSoup) -> Optional[Tag]:
    return soup.find("div", class_="widget-ArticleFulltext")


def _find_section_container(body: Tag) -> Optional[Tag]:
    """Return the ``widget-items`` div that actually contains the article
    section headings (Silverchair pages have a second ``widget-items`` at
    the top with author/metadata that should be ignored).
    """
    for wi in body.find_all("div", class_="widget-items", recursive=True):
        if wi.find(["h2", "h3", "h4", "h5"], class_="section-title"):
            return wi
    return None


def _split_into_spans(
    children: list[Tag],
) -> tuple[list[Tag], list[tuple[Tag, list[Tag]]]]:
    """Partition a flat sibling list at heading boundaries."""
    intro: list[Tag] = []
    spans: list[tuple[Tag, list[Tag]]] = []

    current_heading: Optional[Tag] = None
    current_content: list[Tag] = []

    for c in children:
        if is_section_heading(c):
            if current_heading is not None:
                spans.append((current_heading, current_content))
            elif current_content:
                intro.extend(current_content)
            current_heading = c
            current_content = []
        else:
            if current_heading is None:
                intro.append(c)
            else:
                current_content.append(c)
    if current_heading is not None:
        spans.append((current_heading, current_content))

    return intro, spans


def _chapter_title_from(soup: BeautifulSoup, html_path: Path) -> str:
    h1 = soup.find("h1", class_=lambda c: c and "wi-article-title" in c)
    if h1 is None:
        h1 = soup.find("h1")
    if h1:
        for tag in h1.find_all(["i", "span"], class_=True):
            tag.decompose()
        text = h1.get_text(" ", strip=True)
        text = re.sub(r":\s+Standards of Care.*$", "", text)
        text = re.split(r"\s*\|\s*", text)[0].strip()
        text = re.sub(r"^\d+\.\s+", "", text)
        return text
    return html_path.stem


# ── resource asset population ──────────────────────────────────────────────


def _section_owners(
    nodes: dict[str, SectionNode], chapter_id: str, kind: str
) -> dict[str, list[str]]:
    """Map ``resource_id`` → list of section_ids that placeholder-cite it,
    in document order (chapter first, then subsections in nodes-dict order).
    """
    owners: dict[str, list[str]] = {}
    for sid, node in nodes.items():
        if not (
            sid == chapter_id or node.parent_id == chapter_id or chapter_id in node.path
        ):
            continue
        for rid in getattr(node, kind):
            owners.setdefault(rid, []).append(sid)
    return owners


def _populate_chapter_figures(
    *,
    container: Tag,
    figure_ids_by_data_id: dict[str, str],
    files_dir: Optional[Path],
    fig_dir: Path,
    figure_meta: dict[str, ResourceMeta],
    section_owners: dict[str, list[str]],
) -> None:
    """For each figure block that the prose renderer cited, copy its image
    asset out of ``_files/`` and record caption + meta.
    """
    seen: set[str] = set()
    for fig_block in container.find_all(
        "div", class_=lambda c: c and "fig-section" in c
    ):
        classes = set(fig_block.get("class") or [])
        if "fig-modal" in classes:
            continue
        data_id = (fig_block.get("data-id") or "").strip()
        rid = figure_ids_by_data_id.get(data_id)
        if rid is None or rid in seen:
            continue
        seen.add(rid)

        label, caption, local_img_name = extract_figure_assets(fig_block)
        owners = section_owners.get(rid, [])
        meta = ResourceMeta(
            resource_id=rid,
            kind="figure",
            primary_section=owners[0] if owners else None,
            referencing_sections=list(owners),
            caption=(label + (": " if label and caption else "") + caption).strip(),
        )

        if local_img_name and files_dir is not None:
            src = files_dir / local_img_name
            if not src.exists():
                # Some browsers URL-encode characters; try a name-glob fallback.
                matches = list(files_dir.glob(local_img_name))
                if matches:
                    src = matches[0]
            if src.exists():
                ext = src.suffix.lower() or ".png"
                out_name = f"{rid}{ext}"
                shutil.copy2(src, fig_dir / out_name)
                meta.image_file = f"resources/figures/{out_name}"
            else:
                logger.warning(
                    "  ! figure asset missing for %s: %s", rid, local_img_name
                )
        if meta.caption:
            (fig_dir / f"{rid}.caption.txt").write_text(meta.caption, encoding="utf-8")
        figure_meta[rid] = meta


def _populate_chapter_tables(
    *,
    container: Tag,
    chapter_id: str,
    table_count: int,
    tab_dir: Path,
    table_meta: dict[str, ResourceMeta],
    section_owners: dict[str, list[str]],
) -> None:
    """Walk the chapter's table-overflow blocks in document order, assigning
    them the same sequential ids the prose renderer used.
    """
    table_blocks = [
        t
        for t in container.find_all("div", class_=lambda c: c and "table-overflow" in c)
        if "table-modal" not in set(t.get("class") or [])
    ]
    if len(table_blocks) != table_count:
        # Not fatal — but we log because it means there's a discrepancy
        # between what was placeholder'd and what we can scrape.
        logger.warning(
            "  ! table block count mismatch in %s: rendered=%d found=%d",
            chapter_id,
            table_count,
            len(table_blocks),
        )

    for idx, tb in enumerate(table_blocks, start=1):
        rid = f"{chapter_id}-t{idx}"
        owners = section_owners.get(rid, [])
        caption, html_str, md = extract_table_assets(tb)
        meta = ResourceMeta(
            resource_id=rid,
            kind="table",
            primary_section=owners[0] if owners else None,
            referencing_sections=list(owners),
            caption=caption,
        )

        (tab_dir / f"{rid}.html").write_text(html_str, encoding="utf-8")
        meta.html_file = f"resources/tables/{rid}.html"

        if md:
            (tab_dir / f"{rid}.markdown.md").write_text(md, encoding="utf-8")
            meta.markdown_file = f"resources/tables/{rid}.markdown.md"

        if caption:
            (tab_dir / f"{rid}.caption.txt").write_text(caption, encoding="utf-8")
        table_meta[rid] = meta


# ── shared helpers (kept in parity with epub_pipeline.stages.extract) ──────


_EXTRACT_OWNED_SUFFIXES = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".caption.txt",
    ".html",
    ".markdown.md",
    ".meta.json",
    ".txt",
)


def _clear_extract_outputs(
    sections_dir: Path, fig_dir: Path, tab_dir: Path, *, clear_downstream: bool = False
) -> None:
    for d in (sections_dir, fig_dir, tab_dir):
        if not d.exists():
            continue
        for f in d.iterdir():
            if not f.is_file():
                continue
            if clear_downstream:
                f.unlink()
                continue
            if any(f.name.endswith(s) for s in _EXTRACT_OWNED_SUFFIXES):
                f.unlink()


def _attach_existing_description(meta: ResourceMeta, resource_dir: Path) -> None:
    desc_path = resource_dir / f"{meta.resource_id}.description.md"
    if desc_path.exists():
        kind_dir = "figures" if meta.kind == "figure" else "tables"
        meta.description_file = (
            f"resources/{kind_dir}/{meta.resource_id}.description.md"
        )


def _write_section_text(
    sections_dir: Path,
    section_id: str,
    text: str,
    node: SectionNode,
    min_text_length: int,
) -> None:
    node.content_length = len(text)
    if not text.strip():
        return
    if min_text_length and len(text) < min_text_length:
        return
    file_text = text + "\n"
    (sections_dir / f"{section_id}.txt").write_text(file_text, encoding="utf-8")
    node.text_file = f"sections/{section_id}.txt"
    node.content_length = len(file_text)


def _resource_sort_key(rid: str) -> tuple:
    parts = rid.split("-")
    chapter = parts[0] if parts else rid
    tail = parts[-1] if len(parts) > 1 else rid
    letters = "".join(ch for ch in tail if not ch.isdigit())
    digits = "".join(ch for ch in tail if ch.isdigit())
    return (chapter, letters, int(digits) if digits else 0)


def _log_summary(
    nodes: dict[str, SectionNode],
    figs: dict[str, ResourceMeta],
    tabs: dict[str, ResourceMeta],
    guideline_dir: Path,
) -> None:
    by_depth = Counter(n.level for n in nodes.values())
    by_type = Counter(n.section_type for n in nodes.values())
    rag_yes = sum(1 for n in nodes.values() if n.rag_eligible)
    with_text = sum(1 for n in nodes.values() if n.text_file)
    multi_fig = sum(1 for m in figs.values() if len(m.referencing_sections) > 1)
    multi_tab = sum(1 for m in tabs.values() if len(m.referencing_sections) > 1)
    logger.info("✅ extract complete  →  %s", guideline_dir)
    logger.info("   sections:           %d  (with text: %d)", len(nodes), with_text)
    logger.info("   rag_eligible:       %d / %d", rag_yes, len(nodes))
    logger.info("   by depth:           %s", dict(sorted(by_depth.items())))
    logger.info("   by section_type:    %s", dict(by_type))
    logger.info(
        "   figures:            %d  (cited from >1 section: %d)", len(figs), multi_fig
    )
    logger.info(
        "   tables:             %d  (cited from >1 section: %d)", len(tabs), multi_tab
    )
