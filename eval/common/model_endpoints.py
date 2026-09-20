"""One place that knows how to reach a model string."""

from __future__ import annotations

# Providers pydantic_ai routes on its own, given the matching API key.
NATIVE_PREFIXES: tuple[str, ...] = ("openai:", "openai-responses:", "anthropic:")

# Prefixes reached through a host of their own. Empty here; the scorers read it
# to label a route in their reports.
CUSTOM_ENDPOINT_PREFIXES: tuple[str, ...] = ()


def resolve_compat_endpoint(model: str) -> tuple[str, str, dict | None] | None:
    """(base_url, key_env, headers) for a model reached over an OpenAI-compatible
    endpoint of its own, or None when the provider is routed natively.
    """
    return None


def needs_custom_endpoint(model: str) -> bool:
    """Whether `model` needs an endpoint built by hand rather than a native route."""
    return False


def resolve_route(name: str) -> str:
    """Expand a route alias. There are none here, so this passes through."""
    return name


def build_agent_model(model: str):
    """The value to hand pydantic_ai for `model`."""
    if model.startswith(NATIVE_PREFIXES):
        return model
    raise ValueError(
        f"{model!r} is not a provider this build can reach. Supported prefixes: "
        f"{', '.join(NATIVE_PREFIXES)}. Add a case to build_agent_model returning "
        f"a configured pydantic_ai Model to use another host."
    )


def round_seed(model: str, round_index: int) -> int | None:
    """The sampling seed for one voting round, or None when the route needs none."""
    return None


_THINKING_OFF: tuple[tuple[str, dict | None], ...] = ()


class ThinkingOffUnsupported(RuntimeError):
    """The vendor provides no way to disable reasoning for this model."""


def thinking_off_kwargs(model: str) -> dict:
    """Request kwargs that turn reasoning OFF for `model`."""
    for prefix, kw in sorted(_THINKING_OFF, key=lambda p: -len(p[0])):
        if model.lower().startswith(prefix):
            if kw is None:
                raise ThinkingOffUnsupported(
                    f"{model} cannot have reasoning disabled: the vendor documents "
                    f"it as always on. Run it at its default and report the level."
                )
            return dict(kw)
    raise ThinkingOffUnsupported(
        f"no measured thinking-off lever for {model}. Use pydantic_ai's Thinking "
        f"capability, or establish one and add it to _THINKING_OFF: the wrong "
        f"lever returns 200 with reasoning still running, so guessing produces a "
        f"run labelled thinking-off that thought."
    )


def set_pacer(rpm: float | None):
    """No metered host here, so there is nothing to pace."""
    return None


def limiter_for_model(model: str):
    """No metered host here, so no model needs a pacer."""
    return None


def pacing_state() -> dict | None:
    """Pacing provenance for a report. None: nothing was paced."""
    return None


def reasoning_ran(message: object) -> bool | None:
    """Did reasoning actually run, read off the RESPONSE rather than the request."""
    raw = (
        message.model_dump() if hasattr(message, "model_dump") else dict(message or {})
    )
    for field in ("reasoning_content", "reasoning"):
        if field in raw:
            v = raw.get(field)
            # Not necessarily a string: some OpenAI-compatible hosts return
            # `reasoning` as a dict or a list of blocks.
            return bool(v.strip()) if isinstance(v, str) else bool(v)
    return None
