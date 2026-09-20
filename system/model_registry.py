"""Model registry — the single source of truth for which models are offered,
their display names, output caps, reasoning capability, valid efforts, defaults.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    """One offered model. `key` is the config id; `provider_model` is the
    Pydantic AI "provider:model" (OpenAI reasoning models use the Responses
    API, `openai-responses:`; chat-completions models use `openai:`)."""

    key: str
    label: str
    provider_model: str
    max_tokens: int  # output ceiling, not a fixed cost
    supports_reasoning: bool


_CATALOG: tuple[ModelInfo, ...] = (
    # The models evaluated in the paper, from the two providers this repository
    # supports. max_tokens is the output ceiling each run was given; the run
    # manifests record the rest of each configuration.
    ModelInfo("claude-opus-5", "Claude Opus 5", "anthropic:claude-opus-5", 25000, True),
    ModelInfo(
        "claude-sonnet-5", "Claude Sonnet 5", "anthropic:claude-sonnet-5", 25000, True
    ),
    ModelInfo(
        "claude-haiku-4-5",
        "Claude Haiku 4.5",
        "anthropic:claude-haiku-4-5",
        25000,
        True,
    ),
    ModelInfo("gpt-5.4", "GPT-5.4", "openai-responses:gpt-5.4", 25000, True),
    ModelInfo(
        "gpt-5.4-mini", "GPT-5.4 mini", "openai-responses:gpt-5.4-mini", 25000, True
    ),
)
_BY_KEY: dict[str, ModelInfo] = {m.key: m for m in _CATALOG}
DEFAULT_MODEL = "gpt-5.4"

PROVIDER_ENV_VAR: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openai-responses": "OPENAI_API_KEY",
}


def provider_of(provider_model: str) -> str:
    """The provider prefix of a `provider:model` string (e.g. 'anthropic')."""
    return provider_model.split(":", 1)[0]


DEFAULT_HELPER_MODEL = "gpt-5.4-mini"

HELPER_MODEL_KEYS: frozenset[str] = frozenset({"gpt-5.4-mini", "claude-haiku-4-5"})

_EFFORTS = frozenset({"off", "minimal", "low", "medium", "high"})
DEFAULT_EFFORT = "high"


@dataclass(frozen=True)
class ResolvedModel:
    """A validated model selection, ready to hand to `make_agent`."""

    model: str  # Pydantic AI "provider:model"
    max_tokens: int
    thinking_effort: str | bool  # False = thinking disabled (Fast mode)
    fast: bool  # Fast mode — use the no-thinking system prompt


def model_keys() -> list[str]:
    """Registry keys of all offered models, in display order."""
    return [m.key for m in _CATALOG]


def public_catalog(available_providers: frozenset[str] | None = None) -> list[dict]:
    """Model metadata for a caller's model picker (id + display name +
    reasoning capability), in display order. The UI renders from THIS — it keeps
    no hardcoded model list, so there's one source and nothing to drift.
    """
    return [
        {
            "id": m.key,
            "label": m.label,
            "supports_reasoning": m.supports_reasoning,
            # Whether this model is offered in the curated, cheap helper-model
            # picker (see HELPER_MODEL_KEYS).
            "helper_option": m.key in HELPER_MODEL_KEYS,
        }
        for m in _CATALOG
        if available_providers is None
        or provider_of(m.provider_model) in available_providers
    ]


def resolve_helper_model(model_key: str | None) -> str:
    """The Pydantic AI "provider:model" for the cheap helper tasks, such as the section
    selection behind `search`. Unknown or missing key falls back to DEFAULT_HELPER_MODEL, so
    these internal paths always resolve to a working model even if a caller sends a stale
    id. No thinking is involved, so any vendor in the registry is valid.
    """
    info = _BY_KEY.get(model_key or "", _BY_KEY[DEFAULT_HELPER_MODEL])
    return info.provider_model


def resolve(model_key: str | None, effort: str | None) -> ResolvedModel:
    """Validate client-supplied model + effort, falling back to defaults."""
    info = _BY_KEY.get(model_key or "", _BY_KEY[DEFAULT_MODEL])
    eff = effort if effort in _EFFORTS else DEFAULT_EFFORT
    if not info.supports_reasoning:
        eff = "off"  # no reasoning channel → always Fast mode
    fast = eff == "off"
    thinking: str | bool = False if fast else eff
    return ResolvedModel(
        model=info.provider_model,
        max_tokens=info.max_tokens,
        thinking_effort=thinking,
        fast=fast,
    )
