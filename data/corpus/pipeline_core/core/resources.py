"""Resource (figure / table) metadata + load/save helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

FIGURES_DIR = "resources/figures"
TABLES_DIR = "resources/tables"


@dataclass
class ResourceMeta:
    resource_id: str
    kind: Literal["figure", "table"]
    primary_section: str | None
    referencing_sections: list[str] = field(default_factory=list)
    caption: str = ""

    # figure-specific
    image_file: str | None = None
    description_file: str | None = None  # filled in by describe_figures stage

    # table-specific
    html_file: str | None = None
    markdown_file: str | None = None


def _meta_path(guideline_dir: Path, kind: str, resource_id: str) -> Path:
    base = FIGURES_DIR if kind == "figure" else TABLES_DIR
    return guideline_dir / base / f"{resource_id}.meta.json"


def load_meta(guideline_dir: Path, kind: str, resource_id: str) -> ResourceMeta:
    data = json.loads(
        _meta_path(guideline_dir, kind, resource_id).read_text(encoding="utf-8")
    )
    return ResourceMeta(**data)


def save_meta(guideline_dir: Path, meta: ResourceMeta) -> None:
    path = _meta_path(guideline_dir, meta.kind, meta.resource_id)
    path.write_text(
        json.dumps(asdict(meta), indent=2, ensure_ascii=False), encoding="utf-8"
    )
