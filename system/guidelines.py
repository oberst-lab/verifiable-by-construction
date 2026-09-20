from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

from .models import RetrievedDoc, SectionContent, SectionSummary

logger = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────
# The corpus and the library that reads it both live under data/corpus/.
REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "data" / "corpus"
CORPUS_ROOT = CORPUS_DIR / "guidelines"

SUBTOPIC_LEVEL = 3


def _ensure_pipeline_core_on_path() -> None:
    """Make pipeline_core.* importable without packaging it."""
    p = str(CORPUS_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)


# ── in-memory model ──────────────────────────────────────────────────────────
@dataclass
class SectionInfo:
    guideline_id: str
    section_number: int  # 1-indexed ordinal within the guideline (display only)
    section_id: str  # tree node id, e.g. "ch01" — unique within a guideline
    title: str  # display title incl. number prefix, e.g. "5. BP Management"
    summary: str
    subsections: list[str]  # immediate rag-eligible child titles
    url: str | None


@dataclass
class GuidelineInfo:
    guideline_id: str
    guideline_name: str
    description: str
    url: str | None
    dir: Path
    # Registry `default` flag: pre-select this guideline in the UI on first load.
    default: bool = True
    sections: list[SectionInfo] = field(default_factory=list)


# ── corpus loading ────────────────────────────────────────────────────────────
@lru_cache(maxsize=4)
def _load_corpus(tree_level: int = SUBTOPIC_LEVEL) -> dict[str, GuidelineInfo]:
    """Load the enabled guidelines, with retrieval units at `tree_level`. Cached
    per level (so chapter=2 and subtopic=3 each build once).
    """
    _ensure_pipeline_core_on_path()
    from pipeline_core.core.tree import TreeDocument  # noqa: E402

    registry_path = CORPUS_ROOT / "registry.yaml"
    if not registry_path.exists():
        raise FileNotFoundError(f"Missing corpus registry at {registry_path}")

    registry = yaml.safe_load(registry_path.read_text())
    out: dict[str, GuidelineInfo] = {}

    for gid, cfg in registry.items():
        if not cfg.get("enabled", False):
            continue

        gdir = CORPUS_ROOT / gid
        ymeta = yaml.safe_load((gdir / "guideline.yaml").read_text())
        tree = TreeDocument.load(gdir)

        out[gid] = GuidelineInfo(
            guideline_id=gid,
            guideline_name=ymeta["guideline_name"],
            description=ymeta.get("description", "") or "",
            url=ymeta.get("url"),
            dir=gdir,
            default=cfg.get("default", True),
            sections=_build_sections(tree, ymeta.get("url"), tree_level),
        )

    if not out:
        raise RuntimeError("No enabled guidelines found in registry.yaml")
    return out


def _chapter_label(tree, node) -> str | None:
    """The parent chapter (level-2 ancestor) as 'N. Title', for locating a
    sub-topic. Returns None if the node has no chapter ancestor."""
    for anc in tree.ancestors(node.section_id):
        if anc.level == 2:
            return f"{anc.number}. {anc.title}" if anc.number else anc.title
    return None


def _chapter_id_of(tree, node) -> str | None:
    """section_id of the node's level-2 chapter ancestor (itself if it is one)."""
    if node.section_type == "chapter":
        return node.section_id
    for anc in tree.ancestors(node.section_id):
        if anc.level == 2 and anc.section_type == "chapter":
            return anc.section_id
    return None


def _build_sections(
    tree, guideline_url: str | None, tree_level: int
) -> list[SectionInfo]:
    """Enumerate retrieval units at `tree_level`, in document order."""
    # tree.nodes is keyed in document order; filtering preserves that order.
    units = [n for n in tree.nodes.values() if n.rag_eligible and n.level == tree_level]

    if tree_level > 2:
        covered = {_chapter_id_of(tree, n) for n in units}
        extra = [
            n
            for n in tree.nodes.values()
            if n.section_type == "chapter"
            and n.rag_eligible
            and n.section_id not in covered
            and (n.text_file or n.children)
        ]
        if extra:
            order = {sid: i for i, sid in enumerate(tree.nodes.keys())}
            units = sorted(units + extra, key=lambda n: order[n.section_id])

    out: list[SectionInfo] = []
    for ordinal, node in enumerate(units, 1):
        sub_titles = [
            tree.nodes[cid].title
            for cid in node.children
            if cid in tree.nodes and tree.nodes[cid].rag_eligible
        ]
        # Title carries the number prefix so citations read "5. BP Management"
        # rather than a bare "BP Management".
        own = f"{node.number}. {node.title}" if node.number else node.title
        # Sub-topics (below chapter) get a "< parent chapter" suffix so they're
        # locatable. ADA numbers no sub-topics ("Statin Treatment" → which ch?);
        # AHA numbers most ("5.1 …"). Only nodes below chapter level get it.
        chapter = _chapter_label(tree, node) if node.level > 2 else None
        display_title = f"{own} < {chapter}" if chapter else own
        out.append(
            SectionInfo(
                guideline_id=tree.guideline_id,
                section_number=ordinal,
                section_id=node.section_id,
                title=display_title,
                summary=node.summary or "",
                subsections=sub_titles,
                url=guideline_url,  # per-section URLs are not resolved
            )
        )
    return out


def get_loaded_guidelines() -> list[GuidelineInfo]:
    # Guideline-level metadata (id/name/description) is granularity-independent;
    # callers here don't touch .sections, so the default level is fine.
    return list(_load_corpus().values())


# ── doc_id ─────────────────────────────────────────────────────────────────
def _doc_id(guideline_id: str, section_id: str) -> str:
    """Unique, granularity-proof citation key: `{guideline_id}:{section_id}`."""
    return f"{guideline_id}:{section_id}"


# ── tool-facing helpers ────────────────────────────────────────────────────
def list_all_sections(
    guideline_ids: list[str] | None = None,
    tree_level: int = SUBTOPIC_LEVEL,
) -> list[SectionSummary]:
    corpus = _load_corpus(tree_level)
    if guideline_ids:
        selected = [corpus[g] for g in guideline_ids if g in corpus]
    else:
        selected = list(corpus.values())

    out: list[SectionSummary] = []
    for g in selected:
        for s in g.sections:
            out.append(
                SectionSummary(
                    doc_id=_doc_id(g.guideline_id, s.section_id),
                    guideline_id=g.guideline_id,
                    guideline_name=g.guideline_name,
                    section_number=s.section_number,
                    section_id=s.section_id,
                    title=s.title,
                    summary=s.summary,
                    subsections=s.subsections,
                    url=s.url,
                )
            )
    return out


def read_section_text(
    guideline_id: str,
    section_id: str | int,
    tree_level: int = SUBTOPIC_LEVEL,
) -> SectionContent:
    """Read a section's full text via pipeline_core's assemble()."""
    corpus = _load_corpus(tree_level)
    if guideline_id not in corpus:
        raise ValueError(
            f"Unknown guideline_id {guideline_id!r}. Available: {sorted(corpus.keys())}"
        )
    g = corpus[guideline_id]

    # Resolve by node id, else by 1-indexed ordinal (section_number).
    section: SectionInfo | None
    if isinstance(section_id, int) or (
        isinstance(section_id, str) and section_id.isdigit()
    ):
        n = int(section_id)
        section = next((s for s in g.sections if s.section_number == n), None)
    else:
        section = next((s for s in g.sections if s.section_id == section_id), None)
    if section is None:
        available = [s.section_id for s in g.sections]
        raise ValueError(
            f"Unknown section {section_id!r} in {guideline_id}. Available: {available}"
        )

    _ensure_pipeline_core_on_path()
    from pipeline_core.core.assemble import assemble  # noqa: E402

    full_text = assemble(
        g.dir,
        section.section_id,
        include_descendants=True,
        resolve_resources=True,
    )

    return SectionContent(
        doc_id=_doc_id(guideline_id, section.section_id),
        guideline_id=guideline_id,
        guideline_name=g.guideline_name,
        section_number=section.section_number,
        section_id=section.section_id,
        title=section.title,
        summary=section.summary,
        text=full_text,
        subsection=None,
        url=section.url,
    )


def section_to_retrieved_doc(content: SectionContent) -> RetrievedDoc:
    """Build the display payload for a read_section result."""
    return RetrievedDoc(
        doc_id=content.doc_id,
        guideline_id=content.guideline_id,
        guideline_name=content.guideline_name,
        section_number=content.section_number,
        title=content.title,
        summary=content.summary,
        subsection=content.subsection,
        url=content.url,
    )


# ── retrieval scope ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class GuidelineScope:
    """What one turn may retrieve: the selected Guidelines (a hard ceiling — empty = all
    loaded) and the tree level reads resolve at. Built once per request from the request's
    config; the retrieval tools consult it instead of each re-deriving the selection. The
    selection is a CEILING — the agent may narrow within it but never widen past it.
    """

    selected: tuple[str, ...]  # () = all loaded
    tree_level: int

    @classmethod
    def of(cls, selected: list[str] | None) -> GuidelineScope:
        return cls(tuple(selected or ()), SUBTOPIC_LEVEL)

    def guidelines(self) -> list[GuidelineInfo]:
        """Loaded guidelines filtered to the selection (empty selection = all)."""
        loaded = get_loaded_guidelines()
        if not self.selected:
            return loaded
        keep = set(self.selected)
        return [g for g in loaded if g.guideline_id in keep]

    def allows(self, guideline_id: str) -> bool:
        """Whether a guideline_id is in scope (no selection = everything allowed)."""
        return not self.selected or guideline_id in self.selected

    def sections(self, requested: list[str] | None = None) -> list[SectionSummary]:
        """list_all_sections at this scope's tree level, clamped to the ceiling:
        a requested filter is intersected with the selection; with no selection
        the request passes through (None = all)."""
        if self.selected:
            ceiling = set(self.selected)
            ids = [g for g in (requested or self.selected) if g in ceiling]
        else:
            ids = requested
        return list_all_sections(ids, self.tree_level)

    def read(self, guideline_id: str, section_id: str) -> SectionContent:
        """read_section_text at this scope's tree level. Caller checks `allows`
        first for the out-of-scope guidance message."""
        return read_section_text(guideline_id, section_id, self.tree_level)


def section_full_text(guideline_id: str, section_id: str) -> str | None:
    """Assemble a section's full text by node id, granularity-independent (any
    tree node works, unlike read_section_text which resolves within a corpus
    level). Used by the verbatim-citation verifier. None if not found."""
    gdir = CORPUS_ROOT / guideline_id
    if not gdir.exists():
        return None
    _ensure_pipeline_core_on_path()
    from pipeline_core.core.assemble import assemble  # noqa: E402

    try:
        return assemble(
            gdir, section_id, include_descendants=True, resolve_resources=True
        )
    except Exception:
        return None
