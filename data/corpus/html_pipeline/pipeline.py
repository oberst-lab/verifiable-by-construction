"""Orchestrator: dispatches to individual stages and tracks completion state."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from pipeline_core.stages import describe_visuals as describe_visuals_stage
from pipeline_core.stages import summarize as summarize_stage
from pipeline_core.utils.logging import setup_logging

from .config import GuidelineConfig
from .stages import extract as extract_stage

logger = setup_logging("html_pipeline")

STAGE_ORDER = ["extract", "describe_visuals", "summarize"]
STATE_FILENAME = ".html_pipeline_state.json"


# ── state ─────────────────────────────────────────────────────────────────────


def _state_path(guideline_dir: Path) -> Path:
    return guideline_dir / STATE_FILENAME


def load_state(guideline_dir: Path) -> dict:
    p = _state_path(guideline_dir)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def mark_done(guideline_dir: Path, stage: str) -> None:
    state = load_state(guideline_dir)
    state[stage] = datetime.now().isoformat(timespec="seconds")
    _state_path(guideline_dir).write_text(json.dumps(state, indent=2), encoding="utf-8")


# ── stage runners ─────────────────────────────────────────────────────────────


def run_extract(
    cfg: GuidelineConfig, *, force: bool, clear_downstream: bool = False
) -> None:
    cfg.guideline_dir.mkdir(parents=True, exist_ok=True)
    extract_stage.run(
        guideline_id=cfg.guideline_id,
        guideline_name=cfg.guideline_name,
        html_dir=cfg.html_dir,
        guideline_dir=cfg.guideline_dir,
        extract_config=cfg.extract,
        force=force,
        clear_downstream=clear_downstream,
    )


def run_describe_visuals(cfg: GuidelineConfig, *, force: bool, dry_run: bool) -> None:
    describe_visuals_stage.run(
        guideline_dir=cfg.guideline_dir,
        provider=cfg.llm_provider,
        model=cfg.llm_model,
        force=force,
        dry_run=dry_run,
    )


def run_summarize(cfg: GuidelineConfig, *, force: bool, dry_run: bool) -> None:
    summarize_config = cfg.raw.get("summarize") or {}
    levels = summarize_config.get("levels") or list(summarize_stage.DEFAULT_LEVELS)
    max_input_tokens = (
        summarize_config.get("max_input_tokens")
        or summarize_stage.DEFAULT_MAX_INPUT_TOKENS
    )
    summarize_stage.run(
        guideline_dir=cfg.guideline_dir,
        provider=cfg.llm_provider,
        model=summarize_config.get("model") or cfg.llm_model,
        levels=levels,
        max_level=summarize_config.get("max_level"),
        max_input_tokens=max_input_tokens,
        force=force,
        dry_run=dry_run,
    )


# ── orchestration ─────────────────────────────────────────────────────────────


def run(
    cfg: GuidelineConfig,
    *,
    stages: list[str] | None = None,
    force_stages: set[str] | None = None,
    dry_run: bool = False,
    clear_downstream: bool = False,
) -> None:
    force_stages = force_stages or set()
    stages = stages or STAGE_ORDER

    state = load_state(cfg.guideline_dir)
    logger.info("🏥 html_pipeline: %s → %s", cfg.guideline_id, cfg.guideline_dir)
    if dry_run:
        logger.info("🔍 DRY RUN — no changes will be made")

    for stage in stages:
        if stage not in STAGE_ORDER:
            raise ValueError(f"Unknown stage: {stage}. Known: {STAGE_ORDER}")

        already_done = state.get(stage) is not None
        forced = stage in force_stages
        if already_done and not forced:
            logger.info("⏭️  [SKIP] %-20s — already done (%s)", stage, state[stage])
            continue

        logger.info("▶️  [START] %s", stage)
        if stage == "extract":
            run_extract(cfg, force=forced, clear_downstream=clear_downstream)
        elif stage == "describe_visuals":
            run_describe_visuals(cfg, force=forced, dry_run=dry_run)
        elif stage == "summarize":
            run_summarize(cfg, force=forced, dry_run=dry_run)
        if not dry_run:
            mark_done(cfg.guideline_dir, stage)
        logger.info("✅ [DONE]  %s", stage)


def status(cfg: GuidelineConfig) -> dict:
    state = load_state(cfg.guideline_dir)
    tree_exists = (cfg.guideline_dir / "tree.json").exists()
    return {
        "guideline_id": cfg.guideline_id,
        "guideline_dir": str(cfg.guideline_dir),
        "tree.json": "✅" if tree_exists else "—",
        "stages": {s: state.get(s, "pending") for s in STAGE_ORDER},
    }
