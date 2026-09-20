"""Stage 1: extract an EPUB into the new tree-based schema."""

from __future__ import annotations

import logging
import zipfile
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from pipeline_core.core.render import (
    classify_section,
    is_rag_eligible,
    parse_number_from_heading,
    table_html_to_markdown,
)
from pipeline_core.core.resources import ResourceMeta, save_meta
from pipeline_core.core.tree import SectionNode, TreeDocument

# EPUB-specific helpers
from ..core.render import get_section_heading, render_direct_prose

logger = logging.getLogger(__name__)

INDEX_XHTML_SUFFIX = "/xhtml/index.xhtml"


def run(
    *,
    guideline_id: str,
    epub_path: Path,
    guideline_dir: Path,
    extract_config: dict,
    force: bool = False,
    clear_downstream: bool = False,
) -> None:
    """Extract the EPUB into guideline_dir using the schema above."""
    if not epub_path.exists():
        raise FileNotFoundError(f"EPUB not found: {epub_path}")

    sections_dir = guideline_dir / "sections"
    fig_dir = guideline_dir / "resources" / "figures"
    tab_dir = guideline_dir / "resources" / "tables"
    tree_json = guideline_dir / "tree.json"

    if tree_json.exists() and not force:
        raise FileExistsError(
            f"{tree_json} already exists; pass --force-stage extract to overwrite."
        )

    # Clear stale outputs. By default we respect stage ownership: extract owns
    # the source-derived files (caption/html/markdown/image/meta + section
    # texts) and never touches downstream LLM artifacts like description.md.
    _clear_extract_outputs(
        sections_dir, fig_dir, tab_dir, clear_downstream=clear_downstream
    )
    sections_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    exclude_subtrees = extract_config.get("exclude_subtrees", ["backmatter"])
    exclude_id_prefixes = extract_config.get("exclude_id_prefixes", [])
    exclude_ids = extract_config.get(
        "exclude_ids", ["frontmatter", "bodymatter", "backmatter"]
    )
    exclude_titles = extract_config.get("exclude_titles", [])
    min_text_length = int(extract_config.get("min_text_length", 0))

    logger.info("📚 Extracting %s → %s", epub_path.name, guideline_dir)

    with zipfile.ZipFile(epub_path) as z:
        names = z.namelist()
        try:
            index_name = next(n for n in names if n.endswith(INDEX_XHTML_SUFFIX))
        except StopIteration:
            raise RuntimeError(f"No xhtml/index.xhtml found in {epub_path.name}")

        soup = BeautifulSoup(z.read(index_name).decode("utf-8"), "html.parser")

        # ── walk section tree ────────────────────────────────────────────────
        all_sections = soup.find_all("section")
        nodes: dict[str, SectionNode] = {}
        for sec in all_sections:
            sid = sec.get("id")
            if not sid:
                continue

            heading = get_section_heading(sec)
            heading_text = (
                heading.get_text(" ", strip=True) if heading else "(no heading)"
            )
            number, clean_title = parse_number_from_heading(heading_text)

            parent_id = None
            for p in sec.parents:
                if isinstance(p, Tag) and p.name == "section" and p.get("id"):
                    parent_id = p.get("id")
                    break

            # path = chain of section_id ancestors (root → self)
            path: list[str] = []
            cur = sec
            while True:
                if isinstance(cur, Tag) and cur.name == "section":
                    cid = cur.get("id")
                    if cid:
                        path.insert(0, cid)
                if cur.parent is None:
                    break
                cur = cur.parent

            nodes[sid] = SectionNode(
                section_id=sid,
                title=clean_title,
                number=number,
                level=len(path),
                parent_id=parent_id,
                path=path,
            )

        # link parents → children (in document order; find_all is document order)
        for sid, node in nodes.items():
            if node.parent_id and node.parent_id in nodes:
                nodes[node.parent_id].children.append(sid)

        # ── render each section's direct prose + placeholders ────────────────
        figure_meta: dict[str, ResourceMeta] = {}
        table_meta: dict[str, ResourceMeta] = {}
        for sec in all_sections:
            sid = sec.get("id")
            if not sid or sid not in nodes:
                continue

            text, figs, tabs = render_direct_prose(sec)
            node = nodes[sid]
            node.figures = figs
            node.tables = tabs
            node.content_length = len(text)

            if text.strip() and (min_text_length == 0 or len(text) >= min_text_length):
                tf = sections_dir / f"{sid}.txt"
                # Trailing newline keeps the file POSIX-friendly and prevents
                # pre-commit "fix end of files" hooks from drifting against
                # content_length. We record the length including the newline.
                file_text = text + "\n"
                tf.write_text(file_text, encoding="utf-8")
                node.text_file = f"sections/{sid}.txt"
                node.content_length = len(file_text)

            # accumulate referencing relations
            for fid in figs:
                _accumulate_ref(figure_meta, fid, "figure", sid)
            for tid in tabs:
                _accumulate_ref(table_meta, tid, "table", sid)

        # ── classify + apply RAG eligibility ────────────────────────────────
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

        # ── extract figure assets ────────────────────────────────────────────
        for fid, meta in figure_meta.items():
            _populate_figure(z, names, fid, meta, fig_dir)
            save_meta(guideline_dir, meta)

        # ── extract table assets ─────────────────────────────────────────────
        for tid, meta in table_meta.items():
            _populate_table(z, names, tid, meta, tab_dir)
            save_meta(guideline_dir, meta)

    # ── write tree.json ─────────────────────────────────────────────────────
    tree = TreeDocument(
        guideline_id=guideline_id,
        source=epub_path.name,
        nodes=nodes,
        resource_index_figures=sorted(figure_meta.keys(), key=_resource_sort_key),
        resource_index_tables=sorted(table_meta.keys(), key=_resource_sort_key),
    )
    tree.save(guideline_dir)

    _log_summary(nodes, figure_meta, table_meta, guideline_dir)


# ── helpers ──────────────────────────────────────────────────────────────────


# Files this stage owns; only these get cleared on re-extract by default.
_EXTRACT_OWNED_SUFFIXES = (
    ".jpg",
    ".jpeg",
    ".png",  # source images from EPUB
    ".caption.txt",  # caption text from EPUB
    ".html",  # source HTML fragment
    ".markdown.md",  # markdown converted from HTML <table>
    ".meta.json",  # resource metadata (rebuilt each run)
    ".txt",  # section text files (only inside sections/)
)


def _clear_extract_outputs(
    sections_dir: Path, fig_dir: Path, tab_dir: Path, *, clear_downstream: bool = False
) -> None:
    """Remove only files this stage owns. Preserves downstream LLM artifacts
    (e.g. .description.md from describe_visuals) so they survive a re-extract.
    """
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
    """Re-link a description.md that survived from a prior describe_visuals run."""
    desc_path = resource_dir / f"{meta.resource_id}.description.md"
    if desc_path.exists():
        kind_dir = "figures" if meta.kind == "figure" else "tables"
        meta.description_file = (
            f"resources/{kind_dir}/{meta.resource_id}.description.md"
        )


def _accumulate_ref(
    bag: dict[str, ResourceMeta], rid: str, kind: str, section_id: str
) -> None:
    meta = bag.get(rid)
    if meta is None:
        meta = ResourceMeta(
            resource_id=rid,
            kind=kind,
            primary_section=section_id,
        )
        bag[rid] = meta
    if section_id not in meta.referencing_sections:
        meta.referencing_sections.append(section_id)


def _populate_figure(
    z: zipfile.ZipFile, names: list[str], fid: str, meta: ResourceMeta, fig_dir: Path
) -> None:
    fxhtml = f"EPUB/xhtml/{fid}.xhtml"
    if fxhtml not in names:
        logger.warning("  ! figure descriptor missing: %s", fxhtml)
        return
    fsoup = BeautifulSoup(z.read(fxhtml).decode("utf-8"), "html.parser")
    cap_node = fsoup.find("figcaption") or fsoup.find("p")
    meta.caption = cap_node.get_text(" ", strip=True) if cap_node else ""

    img = fsoup.find("img")
    if img and img.get("src"):
        src = img["src"]
        basename = Path(src).name
        match = next((n for n in names if n.endswith(basename)), None)
        if match:
            ext = Path(match).suffix
            out_name = f"{fid}{ext}"
            (fig_dir / out_name).write_bytes(z.read(match))
            meta.image_file = f"resources/figures/{out_name}"

    if meta.caption:
        (fig_dir / f"{fid}.caption.txt").write_text(meta.caption, encoding="utf-8")

    _attach_existing_description(meta, fig_dir)


def _populate_table(
    z: zipfile.ZipFile, names: list[str], tid: str, meta: ResourceMeta, tab_dir: Path
) -> None:
    """Populate a table resource."""
    txhtml = f"EPUB/xhtml/{tid}.xhtml"
    if txhtml not in names:
        logger.warning("  ! table descriptor missing: %s", txhtml)
        return
    raw = z.read(txhtml).decode("utf-8")
    tsoup = BeautifulSoup(raw, "html.parser")

    cap_node = tsoup.find("caption") or tsoup.find(["figcaption", "p"])
    meta.caption = cap_node.get_text(" ", strip=True) if cap_node else ""

    (tab_dir / f"{tid}.html").write_text(raw, encoding="utf-8")
    meta.html_file = f"resources/tables/{tid}.html"

    table_tag = tsoup.find("table")
    if table_tag:
        md = table_html_to_markdown(table_tag)
        if md:
            (tab_dir / f"{tid}.markdown.md").write_text(md, encoding="utf-8")
            meta.markdown_file = f"resources/tables/{tid}.markdown.md"
    else:
        # Image-only table: extract the referenced JPG so a vision-LLM stage
        # can transcribe it later.
        img = tsoup.find("img")
        if img and img.get("src"):
            basename = Path(img["src"]).name
            match = next((n for n in names if n.endswith(basename)), None)
            if match:
                ext = Path(match).suffix
                out_name = f"{tid}{ext}"
                (tab_dir / out_name).write_bytes(z.read(match))
                meta.image_file = f"resources/tables/{out_name}"

    if meta.caption:
        (tab_dir / f"{tid}.caption.txt").write_text(meta.caption, encoding="utf-8")

    _attach_existing_description(meta, tab_dir)


def _resource_sort_key(rid: str):
    # split prefix letters and trailing number for natural ordering
    letters = "".join(ch for ch in rid if not ch.isdigit())
    digits = "".join(ch for ch in rid if ch.isdigit())
    return (letters, int(digits) if digits else 0)


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
