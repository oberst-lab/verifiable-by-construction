"""Section tree data structures + load/save."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

TREE_FILENAME = "tree.json"


@dataclass
class SectionNode:
    section_id: str
    title: str
    number: str | None  # parsed numeric prefix, e.g. "4.2.1.1"; may be None
    level: int  # 1=root, 2=chapter, 3=subsec, ...
    parent_id: str | None
    path: list[str]  # ancestor chain incl. self
    children: list[str] = field(default_factory=list)
    text_file: str | None = None  # path relative to guideline_dir
    content_length: int = 0
    figures: list[str] = field(default_factory=list)  # resource ids in citation order
    tables: list[str] = field(default_factory=list)
    rag_eligible: bool = True
    section_type: str = "subsection"  # root|chapter|subsection|container|abstract|bibliography|appendix|metadata
    summary: str | None = None  # filled in by summarize stage (later)


@dataclass
class TreeDocument:
    guideline_id: str
    source: str  # source EPUB filename (no path)
    nodes: dict[str, SectionNode]  # keyed by section_id
    resource_index_figures: list[str]
    resource_index_tables: list[str]

    # ── traversal helpers ────────────────────────────────────────────────────

    def get(self, section_id: str) -> SectionNode:
        return self.nodes[section_id]

    def roots(self) -> list[SectionNode]:
        return [n for n in self.nodes.values() if n.parent_id is None]

    def walk_subtree(self, section_id: str, include_self: bool = True) -> Iterator[str]:
        """DFS in document order; children list already in original order."""
        if include_self:
            yield section_id
        for child_id in self.nodes[section_id].children:
            yield from self.walk_subtree(child_id, include_self=True)

    def ancestors(self, section_id: str) -> list[SectionNode]:
        """Return list of ancestors from root to direct parent (excluding self)."""
        node = self.nodes[section_id]
        return [self.nodes[sid] for sid in node.path[:-1]]

    # ── load / save ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, guideline_dir: Path) -> "TreeDocument":
        data = json.loads((guideline_dir / TREE_FILENAME).read_text(encoding="utf-8"))
        nodes = {n["section_id"]: SectionNode(**n) for n in data["nodes"]}
        return cls(
            guideline_id=data["guideline_id"],
            source=data["source"],
            nodes=nodes,
            resource_index_figures=data["resource_index"]["figures"],
            resource_index_tables=data["resource_index"]["tables"],
        )

    def save(self, guideline_dir: Path) -> None:
        data = {
            "guideline_id": self.guideline_id,
            "source": self.source,
            "nodes": [asdict(n) for n in self.nodes.values()],
            "resource_index": {
                "figures": self.resource_index_figures,
                "tables": self.resource_index_tables,
            },
        }
        (guideline_dir / TREE_FILENAME).write_text(
            json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
        )
