"""Shared SHA-256 disk cache for the LLM-judge / extractor modules."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


def cache_key(*parts: str) -> str:
    """SHA-256 over the parts (model + version + payload strings), joined with a
    SOH (\\x01) separator so two different part splits can't collide. The separator
    matches the original per-module implementations byte-for-byte, so existing
    cache files stay valid."""
    return hashlib.sha256("\x01".join(parts).encode("utf-8")).hexdigest()


def cache_read(cache_dir: Path, key: str) -> Any | None:
    """Return the cached JSON value (dict or list) for `key`, or None on miss /
    unreadable file."""
    p = cache_dir / f"{key}.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
    return None


def cache_write(cache_dir: Path, key: str, value: Any) -> None:
    """Write `value` (any JSON-serialisable object) under `key`, atomically."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    target = cache_dir / f"{key}.json"
    tmp = cache_dir / f".{key}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    os.replace(tmp, target)
