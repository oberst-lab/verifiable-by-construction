"""CLI: uv run python -m epub_pipeline <subcommand> [...]"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# allow `python -m epub_pipeline` to run from inside data/corpus/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from epub_pipeline import config as cfg_mod  # noqa: E402
from epub_pipeline import pipeline as pl  # noqa: E402

STAGE_CLI_NAMES = {
    "extract": "extract",
    "describe-visuals": "describe_visuals",
    "summarize": "summarize",
}


def cmd_list(args) -> None:
    ids = cfg_mod.list_available()
    if not ids:
        print("No EPUB-source guidelines found in guidelines/")
        return
    print(f"\nEPUB-source guidelines  ({cfg_mod.GUIDELINES_DIR}):\n")
    for gid in ids:
        try:
            cfg = cfg_mod.load(gid)
        except Exception as e:
            print(f"  ⚠️   {gid:<28} (config error: {e})")
            continue
        tree_done = (cfg.guideline_dir / "tree.json").exists()
        state = pl.load_state(cfg.guideline_dir)
        flags = " ".join(f"{s}={state.get(s, '—')}" for s in pl.STAGE_ORDER)
        marker = "✅" if tree_done else "⏳"
        print(f"  {marker}  {gid:<28} {flags}")
    print()


def cmd_status(args) -> None:
    cfg = cfg_mod.load(args.guideline)
    info = pl.status(cfg)
    print(json.dumps(info, indent=2))


def cmd_extract(args) -> None:
    cfg = cfg_mod.load(args.guideline)
    pl.run(
        cfg,
        stages=["extract"],
        force_stages={"extract"} if args.force else set(),
        dry_run=args.dry_run,
        clear_downstream=args.clear_downstream,
    )


def cmd_describe_visuals(args) -> None:
    cfg = cfg_mod.load(args.guideline)
    pl.run(
        cfg,
        stages=["describe_visuals"],
        force_stages={"describe_visuals"} if args.force else set(),
        dry_run=args.dry_run,
    )


def cmd_summarize(args) -> None:
    cfg = cfg_mod.load(args.guideline)
    pl.run(
        cfg,
        stages=["summarize"],
        force_stages={"summarize"} if args.force else set(),
        dry_run=args.dry_run,
    )


def cmd_run(args) -> None:
    cfg = cfg_mod.load(args.guideline)
    force_stages = set()
    if args.force_stage:
        for s in args.force_stage:
            mapped = STAGE_CLI_NAMES.get(s, s)
            force_stages.add(mapped)
    skip = set(STAGE_CLI_NAMES.get(s, s) for s in (args.skip or []))
    stages = [s for s in pl.STAGE_ORDER if s not in skip]
    pl.run(cfg, stages=stages, force_stages=force_stages, dry_run=args.dry_run)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="epub_pipeline",
        description="Subsection-level EPUB extraction pipeline.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="List EPUB-source guidelines")
    p_list.set_defaults(func=cmd_list)

    p_status = sub.add_parser("status", help="Show extraction status for a guideline")
    p_status.add_argument("guideline", help="guideline_id")
    p_status.set_defaults(func=cmd_status)

    p_extract = sub.add_parser("extract", help="Run the extract stage")
    p_extract.add_argument("guideline", help="guideline_id")
    p_extract.add_argument(
        "--force", action="store_true", help="re-run if already done"
    )
    p_extract.add_argument(
        "--clear-downstream",
        action="store_true",
        help=(
            "also delete artifacts owned by later stages (e.g. .description.md). "
            "Default: preserve them across re-extracts."
        ),
    )
    p_extract.add_argument("--dry-run", action="store_true")
    p_extract.set_defaults(func=cmd_extract)

    p_desc = sub.add_parser(
        "describe-visuals",
        help="Run the visual transcription stage (figures + image-only tables)",
    )
    p_desc.add_argument("guideline", help="guideline_id")
    p_desc.add_argument("--force", action="store_true", help="re-run if already done")
    p_desc.add_argument("--dry-run", action="store_true")
    p_desc.set_defaults(func=cmd_describe_visuals)

    p_sum = sub.add_parser("summarize", help="Run the section summary stage")
    p_sum.add_argument("guideline", help="guideline_id")
    p_sum.add_argument("--force", action="store_true", help="re-run if already done")
    p_sum.add_argument("--dry-run", action="store_true")
    p_sum.set_defaults(func=cmd_summarize)

    p_run = sub.add_parser("run", help="Run all stages")
    p_run.add_argument("guideline", help="guideline_id")
    p_run.add_argument(
        "--force-stage",
        action="append",
        default=[],
        choices=list(STAGE_CLI_NAMES.keys()),
        help="re-run a stage even if already done (repeatable)",
    )
    p_run.add_argument(
        "--skip",
        action="append",
        default=[],
        choices=list(STAGE_CLI_NAMES.keys()),
        help="skip a stage (repeatable)",
    )
    p_run.add_argument("--dry-run", action="store_true")
    p_run.set_defaults(func=cmd_run)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
