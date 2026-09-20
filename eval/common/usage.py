"""Token accounting for the evaluation scripts."""

from __future__ import annotations

import json
import logging
import threading
from collections import defaultdict
from pathlib import Path


logger = logging.getLogger("usage")


def _cost(model: str, prompt_tokens: int, completion_tokens: int) -> float | None:
    """Always None, recorded as JSON null."""
    return None


def billing_id(model: str) -> str:
    """The id the price tables are keyed on: `provider:model` without the provider."""
    return model.split(":", 1)[-1]


def record_run(tracker: "UsageTracker", model: str, result) -> None:
    """Record one pydantic_ai run's usage against `tracker`, best-effort."""
    try:
        u = result.usage
        if callable(u):  # older pydantic_ai exposed usage() as a method
            u = u()
        inp = getattr(u, "input_tokens", None)
        inp = getattr(u, "request_tokens", 0) if inp is None else inp
        out_t = getattr(u, "output_tokens", None)
        out_t = getattr(u, "response_tokens", 0) if out_t is None else out_t
        reqs = getattr(u, "requests", 1) or 1
        details = getattr(u, "details", None) or {}
        rtok = details.get("reasoning_tokens") if isinstance(details, dict) else None
        tracker.record_tokens(
            billing_id(model), inp, out_t, reasoning_tokens=rtok, calls=reqs
        )
    except Exception:  # noqa: BLE001 -- usage is best-effort
        pass


class UsageTracker:
    def __init__(self) -> None:
        # model -> [input_tokens, output_tokens] (always summable ints).
        self._tok: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        self._reasoning: dict[str, int | None] = {}
        self.calls = 0
        self.cache_hits = 0
        self._lock = threading.Lock()

    def _add_reasoning(self, model: str, reasoning_tokens: int | None) -> None:
        """Accumulate reported reasoning tokens. Caller holds the lock. None means
        the provider did not report reasoning for this call: leave the model's
        value untouched (it stays None until some call reports a real number, so
        the sentinel is produced at output time and never poisons the sum)."""
        if reasoning_tokens is None:
            return
        self._reasoning[model] = (self._reasoning.get(model) or 0) + int(
            reasoning_tokens
        )

    def record_cache_hit(self, n: int = 1) -> None:
        """One call the cache answered: no tokens, no cost, but it happened."""
        with self._lock:
            self.cache_hits += n

    def reset(self) -> None:
        """Zero every counter. Module-level trackers are process-global, so a process
        scoring several answer sets must not pool their usage into one report."""
        with self._lock:
            self._tok.clear()
            self._reasoning.clear()
            self.calls = 0
            self.cache_hits = 0

    def record(self, model: str, usage) -> None:
        """Record one call's usage object (OpenAI SDK `resp.usage`). Thread-safe."""
        if usage is None:
            return
        details = getattr(usage, "completion_tokens_details", None)
        reasoning = getattr(details, "reasoning_tokens", None) if details else None
        with self._lock:
            self._tok[model][0] += getattr(usage, "prompt_tokens", 0) or 0
            self._tok[model][1] += getattr(usage, "completion_tokens", 0) or 0
            self._add_reasoning(model, reasoning)
            self.calls += 1

    def record_tokens(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        reasoning_tokens: int | None = None,
        calls: int = 1,
    ) -> None:
        """Record raw token counts (e.g. from a pydantic_ai run usage). Thread-safe.
        reasoning_tokens None = the provider did not report reasoning separately."""
        with self._lock:
            self._tok[model][0] += int(input_tokens or 0)
            self._tok[model][1] += int(output_tokens or 0)
            self._add_reasoning(model, reasoning_tokens)
            self.calls += calls

    def as_dict(self) -> dict:
        by_model: dict[str, dict] = {}
        costs: list[float | None] = []
        for model, (tin, tout) in self._tok.items():
            c = _cost(model, tin, tout)
            costs.append(c)
            by_model[model] = {
                "input": tin,
                "output": tout,
                # int when the provider reported reasoning, else None (JSON null) =
                # "not available", NOT a real zero. See _add_reasoning.
                "reasoning": self._reasoning.get(model),
                # None (JSON null) when the model is not priceable — never a fake 0.
                "cost_usd": None if c is None else round(c, 6),
            }
        # A total is only meaningful if every model was priced; any unpriced model
        # (None) makes the total unknown, so it is null too rather than a partial sum.
        total = None if any(c is None for c in costs) else round(sum(costs), 6)
        return {
            "calls": self.calls,
            "cache_hits": self.cache_hits,
            "by_model": by_model,
            "total_cost_usd": total,
        }

    def log_summary(self) -> dict:
        s = self.as_dict()

        def money(c: float | None) -> str:
            return "unpriced" if c is None else f"${c:.4f}"

        for model, r in s["by_model"].items():
            rea = r["reasoning"]
            logger.info(
                "💰 %s — in=%d out=%d (reasoning=%s) → %s",
                model,
                r["input"],
                r["output"],
                "n/a" if rea is None else rea,
                money(r["cost_usd"]),
            )
        logger.info(
            "💰 TOTAL — %d calls (%d served from cache) → %s",
            s["calls"],
            s["cache_hits"],
            money(s["total_cost_usd"]),
        )
        return s

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")
        logger.info("💾 usage written to %s", path)
