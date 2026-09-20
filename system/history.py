"""Message-history processing run before each model request."""

from __future__ import annotations

from dataclasses import replace

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ThinkingPart,
    ToolReturnPart,
    UserPromptPart,
)

OMITTED_STUB = (
    "[section text omitted from history — call read_section again if you need it]"
)
# Prior-turn tool returns to shrink → the stub that replaces their content. Both
# are bulky and re-fetchable; the model re-calls the tool if it needs them again.
_STUBBED_RETURNS = {
    "read_section": OMITTED_STUB,
    "search": (
        "[search results omitted from history — call search again if you need them]"
    ),
    # A replayed conversation may still carry list_sections returns; stub them
    # too so replayed history stays cheap.
    "list_sections": (
        "[section index omitted from history — call list_sections again if you need it]"
    ),
}


MAX_HISTORY_TOKENS = 40_000
_CHARS_PER_TOKEN = 4


def _turn_starts(messages: list[ModelMessage]) -> list[int]:
    """Indices of requests that carry a user prompt — each marks a turn start."""
    return [
        i
        for i, m in enumerate(messages)
        if isinstance(m, ModelRequest)
        and any(isinstance(p, UserPromptPart) for p in m.parts)
    ]


def _est_tokens(m: ModelMessage) -> int:
    """Rough token estimate for one message (char/4 over its text-ish parts)."""
    chars = 0
    for p in getattr(m, "parts", None) or []:
        content = getattr(p, "content", None)
        if isinstance(content, str):
            chars += len(content)
        elif content is not None:
            chars += len(str(content))
        args = getattr(p, "args", None)
        if args is not None:
            chars += len(str(args))
    return chars // _CHARS_PER_TOKEN


def _window_by_tokens(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Keep the newest whole turns that fit MAX_HISTORY_TOKENS; the current turn
    is always kept even if it alone exceeds the budget. Slicing at a turn start
    keeps every retained turn self-contained (no dangling tool call)."""
    starts = _turn_starts(messages)
    if len(starts) <= 1:
        return messages  # 0 or 1 turn — nothing to drop
    bounds = starts + [len(messages)]
    keep_from = bounds[-2]  # start of the current turn — always kept
    total = sum(_est_tokens(m) for m in messages[bounds[-2] :])
    for i in range(len(starts) - 2, -1, -1):  # older turns, newest first
        seg_tokens = sum(_est_tokens(m) for m in messages[bounds[i] : bounds[i + 1]])
        if total + seg_tokens > MAX_HISTORY_TOKENS:
            break
        total += seg_tokens
        keep_from = bounds[i]
    return messages[keep_from:]


def _current_turn_start(messages: list[ModelMessage]) -> int:
    """Index of the last request carrying a user prompt — i.e. where the current
    turn begins. Everything before it is a prior turn whose section reads can be
    stubbed."""
    starts = _turn_starts(messages)
    return starts[-1] if starts else 0


def _stub_tool_returns(req: ModelRequest) -> ModelRequest:
    new_parts = [
        replace(p, content=_STUBBED_RETURNS[p.tool_name])
        if isinstance(p, ToolReturnPart) and p.tool_name in _STUBBED_RETURNS
        else p
        for p in req.parts
    ]
    return replace(req, parts=new_parts)


def _strip_thinking(resp: ModelResponse) -> ModelResponse:
    """Drop ThinkingPart from a prior-turn response (see TRIM THINKING above). A
    completed prior turn always keeps a text answer and/or tool call, so it never
    empties out."""
    parts = [p for p in resp.parts if not isinstance(p, ThinkingPart)]
    return replace(resp, parts=parts) if len(parts) != len(resp.parts) else resp


def apply(
    messages: list[ModelMessage], *, trim_thinking: bool = True
) -> list[ModelMessage]:
    """Stub prior-turn read_section / list_sections returns, optionally trim
    prior-turn reasoning, then window the result to a token budget (current turn
    always kept full).
    """
    # 1. In prior (pre-current) turns: stub bulky re-fetchable tool returns, and
    #    (when enabled) trim reasoning to save input tokens.
    cutoff = _current_turn_start(messages)
    pruned = [
        _prune_prior(m, trim_thinking) if i < cutoff else m
        for i, m in enumerate(messages)
    ]
    # 2. Token-budget recency window over the (now small) prior turns.
    return _window_by_tokens(pruned)


def _prune_prior(m: ModelMessage, trim_thinking: bool) -> ModelMessage:
    """Prune one prior-turn message: stub bulky tool returns (requests) and, when
    trim_thinking, strip reasoning (responses). Others pass through unchanged."""
    if isinstance(m, ModelRequest):
        return _stub_tool_returns(m)
    if trim_thinking and isinstance(m, ModelResponse):
        return _strip_thinking(m)
    return m
