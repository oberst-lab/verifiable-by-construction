"""Shared builder for cheap, no-thinking background helper agents."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.capabilities import Thinking
from pydantic_ai.settings import ModelSettings


@lru_cache(maxsize=16)
def helper_agent(
    provider_model: str,
    *,
    instructions: str,
    max_tokens: int,
    output_type: Any = str,
    effort: Any = False,
) -> Agent:
    """A cached helper Agent for `provider_model`."""
    kwargs: dict[str, Any] = {
        "instructions": instructions,
        "capabilities": [Thinking(effort=effort)],
        "model_settings": ModelSettings(max_tokens=max_tokens),
    }
    # Only pass output_type when structured — leaves str output as the Agent
    # default and keeps the cache key stable for the plain-text helpers.
    if output_type is not str:
        kwargs["output_type"] = output_type
    return Agent(provider_model, **kwargs)
