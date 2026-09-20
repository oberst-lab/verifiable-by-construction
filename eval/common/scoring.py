"""Shared helpers for the citation-metric scorers (vccr / read_efficiency /
coverage) — JSONL loading, percent formatting, and report writing, so the three
scorers don't each re-declare them."""

from __future__ import annotations

import gzip
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file into a list of dicts (blank lines skipped). A `.gz` path is
    decompressed transparently (frozen-context releases store gzipped JSONL to stay
    under the repo's large-file limit)."""
    p = Path(path)
    opener = gzip.open if p.suffix == ".gz" else open
    out: list[dict] = []
    with opener(p, "rt", encoding="utf-8") as fin:
        for line in fin:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def fmt_pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def assert_absent(path: str | Path | None, overwrite: bool = False) -> None:
    """Refuse to write over an existing output unless overwriting was asked for."""
    if path is None:
        return
    p = Path(path)
    if p.exists() and not overwrite:
        raise SystemExit(
            f"{p} already exists — refusing to overwrite.\n"
            f"  Archive it first (git mv into a sibling archive/<what-and-why>-<date>/ "
            f"with a README), write to a new path, or pass --overwrite if the existing "
            f"file is genuinely disposable."
        )


def write_report(path: str | Path, result: dict, overwrite: bool = False) -> None:
    """Write a metric result as pretty JSON, creating parent dirs. Refuses to clobber
    an existing file unless `overwrite`; see `assert_absent`."""
    p = Path(path)
    assert_absent(p, overwrite)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text_guarded(path: str | Path, text: str, overwrite: bool = False) -> None:
    """Write plain text (a summary.md, a README, a .meta.json sidecar) behind the same refusal
    as `write_report`.
    """
    p = Path(path)
    assert_absent(p, overwrite)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def write_jsonl(path: str | Path, rows: list[dict], overwrite: bool = False) -> None:
    """Write rows as one JSON object per line (the inverse of load_jsonl), creating
    parent dirs. Shared here so tracks that emit JSONL (e.g. applicability's frozen
    claim set) do not each re-declare parent-dir creation / ensure_ascii handling.
    Refuses to clobber an existing file unless `overwrite`; see `assert_absent`."""
    p = Path(path)
    assert_absent(p, overwrite)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
