"""API-key resolution for the extraction pipeline."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# data/corpus/pipeline_core/utils/env.py -> the repository root is five up.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_ENV_PATH = _REPO_ROOT / ".env"
_loaded = False


def load_pipeline_env() -> None:
    """Load the repo-root .env into the environment once (idempotent)."""
    global _loaded
    if not _loaded:
        load_dotenv(_ENV_PATH)
        _loaded = True


def get_openai_api_key() -> str:
    """Return OPENAI_API_KEY from the environment (after loading .env)."""
    load_pipeline_env()
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            f"OPENAI_API_KEY is not set (looked in the environment and {_ENV_PATH})."
        )
    return key
