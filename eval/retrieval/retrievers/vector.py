"""Dense vector baseline: embed each unit's full text (chunk → mean-pool, one
vector per unit) + embed the query → cosine ranking.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from openai import OpenAI

from .base import CandidateUnit

_CACHE_DIR = Path(__file__).resolve().parents[2] / "result" / ".embed_cache"
_CHARS_PER_CHUNK = 24_000  # ~6k tokens; safely under the 8k-token per-input cap
_MAX_CHARS_PER_REQUEST = 500_000  # ~125k tokens; well under the 300k/request cap
_MAX_INPUTS_PER_REQUEST = 2048  # OpenAI's per-request input-count cap


def _chunks(text: str) -> list[str]:
    if not text:
        return [""]
    return [
        text[i : i + _CHARS_PER_CHUNK] for i in range(0, len(text), _CHARS_PER_CHUNK)
    ] or [""]


class VectorRetriever:
    name = "vector"

    def __init__(
        self,
        candidates: list[CandidateUnit],
        *,
        model: str = "text-embedding-3-small",
        client: OpenAI | None = None,
        usage=None,
    ):
        self.candidates = candidates
        self.model = model
        self._client = client or OpenAI()
        self.usage = usage
        self._matrix = self._embed_corpus()  # (N, D), L2-normalized

    def _embed(self, texts: list[str]) -> np.ndarray:
        vecs: list[list[float]] = []
        batch: list[str] = []
        batch_chars = 0

        def flush():
            nonlocal batch, batch_chars
            if not batch:
                return
            resp = self._client.embeddings.create(model=self.model, input=batch)
            if self.usage is not None:
                self.usage.record(self.model, resp.usage)
            vecs.extend(d.embedding for d in resp.data)
            batch, batch_chars = [], 0

        for t in texts:
            if batch and (
                batch_chars + len(t) > _MAX_CHARS_PER_REQUEST
                or len(batch) >= _MAX_INPUTS_PER_REQUEST
            ):
                flush()
            batch.append(t)
            batch_chars += len(t)
        flush()

        arr = np.asarray(vecs, dtype=np.float32)
        arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9
        return arr

    def _embed_corpus(self) -> np.ndarray:
        texts = [c.text or c.summary or c.breadcrumb for c in self.candidates]
        sig = hashlib.sha256(
            (self.model + "\x00" + "\x00".join(texts)).encode("utf-8")
        ).hexdigest()[:16]
        cache = _CACHE_DIR / f"{self.model.replace('/', '_')}_{sig}.npy"
        if cache.exists():
            return np.load(cache)
        # Chunk → embed → mean-pool back to one vector per unit.
        flat: list[str] = []
        spans: list[tuple[int, int]] = []
        for t in texts:
            cs = _chunks(t)
            spans.append((len(flat), len(flat) + len(cs)))
            flat.extend(cs)
        chunk_vecs = self._embed(flat)
        unit_vecs = np.stack([chunk_vecs[a:b].mean(axis=0) for (a, b) in spans]).astype(
            np.float32
        )
        unit_vecs /= np.linalg.norm(unit_vecs, axis=1, keepdims=True) + 1e-9
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, unit_vecs)
        return unit_vecs

    def rank(self, question: str) -> list[str]:
        q = self._embed([question])[0]
        sims = self._matrix @ q
        order = np.argsort(-sims)
        return [self.candidates[i].doc_id for i in order]
