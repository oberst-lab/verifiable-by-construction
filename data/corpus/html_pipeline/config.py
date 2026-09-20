"""Config loading for html_pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CORPUS_DIR = Path(__file__).resolve().parent.parent
GUIDELINES_DIR = CORPUS_DIR / "guidelines"


@dataclass
class GuidelineConfig:
    guideline_id: str
    guideline_name: str
    guideline_dir: Path
    html_dir: Path
    extract: dict = field(default_factory=dict)
    llm_provider: str = "openai"
    llm_model: str | None = None
    raw: dict = field(default_factory=dict)


def load(guideline_id: str) -> GuidelineConfig:
    guideline_dir = GUIDELINES_DIR / guideline_id
    yaml_path = guideline_dir / "guideline.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Missing guideline.yaml: {yaml_path}\n"
            f"Create it before running the pipeline."
        )
    cfg = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}

    source = cfg.get("source", {}) or {}
    src_type = source.get("type")
    if src_type != "html":
        raise ValueError(
            f"html_pipeline only handles source.type=html; got {src_type!r} "
            f"for {guideline_id}"
        )
    src_dir = source.get("dir")
    if not src_dir:
        raise ValueError(f"source.dir missing in {yaml_path}")
    html_dir = (CORPUS_DIR / src_dir).resolve()

    extract = cfg.get("extract") or {}
    llm = cfg.get("llm") or {}

    return GuidelineConfig(
        guideline_id=cfg.get("guideline_id", guideline_id),
        guideline_name=cfg.get("guideline_name", guideline_id),
        guideline_dir=guideline_dir,
        html_dir=html_dir,
        extract=extract,
        llm_provider=llm.get("provider", "openai"),
        llm_model=llm.get("model"),
        raw=cfg,
    )


def list_available() -> list[str]:
    """Return guideline_ids that have a guideline.yaml with source.type==html."""
    out = []
    for yml in sorted(GUIDELINES_DIR.glob("*/guideline.yaml")):
        try:
            data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
            if (data.get("source") or {}).get("type") == "html":
                out.append(yml.parent.name)
        except Exception:
            continue
    return out
