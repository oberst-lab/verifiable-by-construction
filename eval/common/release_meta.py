"""Shared helpers for blessing a scratch run into a version-controlled release."""

from __future__ import annotations

import subprocess
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]


def dump_manifest(path: Path, header: str, body: dict) -> None:
    """Write a release MANIFEST: a comment header followed by the body as YAML, in
    declaration order (sort_keys=False). One writer so every stage's manifest shares
    the same serialization (no per-stage drift in flags like allow_unicode)."""
    path.write_text(
        header + yaml.safe_dump(body, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def git_sha() -> str | None:
    """HEAD sha, suffixed `+dirty` when the tree carries uncommitted changes. None if
    unavailable (provenance is best-effort, never fatal).
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True
        ).strip()
    except Exception:  # noqa: BLE001 (provenance is best-effort)
        return None
    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=_REPO_ROOT, text=True
            ).strip()
        )
    except Exception:  # noqa: BLE001
        # Cannot tell: say so rather than imply a clean tree by omission.
        return f"{sha}+unknown"
    return f"{sha}+dirty" if dirty else sha


def rel_to_repo(p: Path) -> str:
    """Repo-relative path string, or the absolute path if it lives outside the repo."""
    try:
        return str(p.resolve().relative_to(_REPO_ROOT))
    except ValueError:
        return str(p)


def model_slug(model: str) -> str:
    """`openai:gpt-4.1-mini` -> `gpt-4pt1-mini`: take the last `:`/`/`-delimited segment,
    the model name proper with no provider or org path, then `.`->`pt`. Load-bearing
    as a release dir name
    AND the hit-rate archive slug, so it must be a single clean path component."""
    return model.rsplit(":", 1)[-1].rsplit("/", 1)[-1].replace(".", "pt")


def unpriced_cost_note(usage: dict | None) -> str | None:
    """A caveat string when a run has real tokens but no priced total (null or 0),
    i.e. no pricing is recorded for that vendor.
    Returned so an absent cost in a provenance file is never misread as "free"; None
    when the cost is genuine."""
    if (
        usage
        and not usage.get("total_cost_usd")  # None (unpriced) or 0
        and any(
            m.get("input") or m.get("output")
            for m in usage.get("by_model", {}).values()
        )
    ):
        return (
            "cost is unpriced (null) because litellm has no rate for this model; "
            "tokens are real, compute cost from the vendor's published rates."
        )
    return None
