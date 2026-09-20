"""Config loading for epub_pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

CORPUS_DIR = Path(__file__).resolve().parent.parent
GUIDELINES_DIR = CORPUS_DIR / "guidelines"


@dataclass
class GuidelineConfig:
    guideline_id: str
    guideline_dir: Path
    epub_path: Path
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
    if src_type != "epub":
        raise ValueError(
            f"epub_pipeline only handles source.type=epub; got {src_type!r} for {guideline_id}"
        )
    src_file = source.get("file")
    if not src_file:
        raise ValueError(f"source.file missing in {yaml_path}")
    epub_path = (CORPUS_DIR / src_file).resolve()

    extract = cfg.get("extract") or {}
    llm = cfg.get("llm") or {}

    return GuidelineConfig(
        guideline_id=cfg.get("guideline_id", guideline_id),
        guideline_dir=guideline_dir,
        epub_path=epub_path,
        extract=extract,
        llm_provider=llm.get("provider", "openai"),
        llm_model=llm.get("model"),
        raw=cfg,
    )


def list_available() -> list[str]:
    """Return guideline_ids that have a guideline.yaml with source.type==epub."""
    out = []
    for yml in sorted(GUIDELINES_DIR.glob("*/guideline.yaml")):
        try:
            data = yaml.safe_load(yml.read_text(encoding="utf-8")) or {}
            if (data.get("source") or {}).get("type") == "epub":
                out.append(yml.parent.name)
        except Exception:
            continue
    return out
