"""LLM selection: one call picks the relevant units from their summaries.

This measures the system's own `search` retriever rather than a copy of it. The
selector prompt is loaded from the one source the live tool reads,
`system/prompts/retrieval_select.md`, and the user message is assembled the same
way, as `QUESTION:` / `SECTIONS:`.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import threading

from openai import OpenAI

from .base import (
    CandidateUnit,
    anthropic_rejects_temperature,
    sampling_kwargs,
    token_kwargs,
)

# The endpoint map lives in eval/common, shared with the answer harness so both
# reach models the same way. Add that dir to the path: the retrieval driver
# already does, but this keeps the import working standalone.
_COMMON = Path(__file__).resolve().parents[2] / "common"
if str(_COMMON) not in sys.path:
    sys.path.insert(0, str(_COMMON))

from rate_limit import SyncRateLimiter, is_rate_limit_error  # noqa: E402
from model_endpoints import (  # noqa: E402
    reasoning_ran,
    resolve_compat_endpoint,
    thinking_off_kwargs,
)

# The selector prompt the system itself uses, so the benchmark and the system
# cannot diverge on it.
_SYSTEM = (
    Path(__file__).resolve().parents[3] / "system" / "prompts" / "retrieval_select.md"
).read_text(encoding="utf-8")

_TOOL_NAME = "select_guideline_sections"
_TOOL_DESC = "Select which guideline sections are most relevant to the question."


_PACED_TRANSIENT_RETRIES = 2


class _PacedOpenAI(OpenAI):
    """An OpenAI client that retries transient failures but NEVER a rate limit."""

    def _should_retry(self, response) -> bool:  # noqa: ANN001
        if response.status_code == 429:
            return False
        return super()._should_retry(response)


def _paced_anthropic(**kw):
    """See _PacedOpenAI. Built lazily because `anthropic` is imported on demand, so the
    subclass cannot exist at module scope."""
    from anthropic import Anthropic

    class _PacedAnthropic(Anthropic):
        def _should_retry(self, response) -> bool:  # noqa: ANN001
            if response.status_code == 429:
                return False
            return super()._should_retry(response)

    return _PacedAnthropic(**kw)


def _retry_kwargs(limiter) -> dict:
    """Retry policy for a client the caller is pacing."""
    if limiter is None:
        return {}
    return {"max_retries": _PACED_TRANSIENT_RETRIES}


class SemanticRetriever:
    name = "semantic"

    def __init__(
        self,
        candidates: list[CandidateUnit],
        *,
        model: str = "gpt-4.1-mini",
        client=None,
        max_tokens: int = 1200,
        temperature: float | None = 0.0,
        reasoning_effort: str | None = None,
        limiter: SyncRateLimiter | None = None,
        seed: int | None = None,
        usage=None,
    ):
        self.candidates = candidates
        self.model = model
        self.max_tokens = max_tokens
        # 0.0 = the pinned/greedy temp-0 regime; None = omit `temperature` and run
        # provider-native (the temp-default regime, production-faithful). Reasoning /
        # Claude-5 models reject the param and always run provider-native regardless.
        self.temperature = temperature
        # None → a reasoning model runs at its provider-native default effort
        # (the eval's answer-driver policy); a string pins it (e.g. "minimal").
        # OpenAI path only — the Anthropic branch runs thinking-off regardless.
        self.reasoning_effort = reasoning_effort
        self.seed = seed
        self.usage = usage
        self._is_anthropic = model.lower().startswith("claude")
        # A model reached over an OpenAI-compatible endpoint of its own gives
        # (base_url, key_env, headers), already env-resolved; None when routed
        # natively.
        self._compat = resolve_compat_endpoint(model)
        self.tool_choice_mode = "auto" if self._compat is not None else "forced"
        if self._is_anthropic:
            _sends_temp = temperature is not None and not anthropic_rejects_temperature(
                model
            )
            self.effective_temperature = temperature if _sends_temp else None
            # The Anthropic selector never requests a thinking channel here.
            self.effective_reasoning_effort = "off"
        else:
            _sk = sampling_kwargs(
                model, temperature=temperature, reasoning_effort=reasoning_effort
            )
            self.effective_temperature = _sk.get("temperature")
            # sampling_kwargs emits reasoning_effort only for a reasoning model with an
            # explicit effort; otherwise the provider-native default applies.
            self.effective_reasoning_effort = _sk.get("reasoning_effort", "default")

        self._thinking_off: dict = {}
        if (
            self._compat is not None
            and reasoning_effort is not None
            and str(reasoning_effort).lower() in ("none", "off")
        ):
            self._thinking_off = thinking_off_kwargs(model)
            # The field is carried by the lever now, not by sampling_kwargs.
            _sk.pop("reasoning_effort", None)
            self.effective_reasoning_effort = "none"
        # Observed on the RESPONSE, filled in by the first call. `None` until then, and
        # None afterwards on a route that does not report it at all.
        self.observed_reasoning: bool | None = None
        self.served_model: str | None = None
        self._obs_lock = threading.Lock()
        self._limiter = limiter
        self.system_fingerprint: str | None = None
        self._by_doc = {c.doc_id: c for c in candidates}
        valid_ids = list(self._by_doc)
        # The candidate listing (id + breadcrumb + summary) is identical for every
        # question, so build it once.
        self._listing = "\n".join(
            f"[{c.doc_id}] {c.breadcrumb} — {c.summary or '(no summary)'}"
            for c in candidates
        )
        # enum-constrained schema — the model can only return ids that exist. The
        # JSON Schema body is shared; each provider wraps it in its own tool shape.
        self._schema = {
            "type": "object",
            "properties": {
                "reasoning": {
                    "type": "string",
                    "description": "Brief reasoning for the selection.",
                },
                "section_ids": {
                    "type": "array",
                    "description": "Relevant section ids, most relevant first.",
                    "items": {"type": "string", "enum": valid_ids},
                },
            },
            "required": ["reasoning", "section_ids"],
            "additionalProperties": False,
        }
        if self._is_anthropic:
            from anthropic import Anthropic

            self._client = client or (
                _paced_anthropic(**_retry_kwargs(limiter))
                if limiter is not None
                else Anthropic()
            )
            self._tool = {
                "name": _TOOL_NAME,
                "description": _TOOL_DESC,
                "input_schema": self._schema,
            }
        else:
            # OpenAI, or an OpenAI-compatible vendor via base_url override.
            if client is not None:
                self._client = client
            elif self._compat is not None:
                base_url, key_env, extra_headers = self._compat
                self._client = OpenAI(
                    base_url=base_url,
                    api_key=os.environ[key_env],
                    default_headers=extra_headers,
                    **_retry_kwargs(limiter),
                )
            else:
                self._client = (_PacedOpenAI if limiter is not None else OpenAI)(
                    **_retry_kwargs(limiter)
                )
            self._tool = {
                "type": "function",
                "function": {
                    "name": _TOOL_NAME,
                    "description": _TOOL_DESC,
                    "parameters": self._schema,
                },
            }

    def _note_served(self, served: str | None) -> None:
        """Caller holds `_obs_lock`. First non-empty wins, and a later disagreement is
        recorded rather than
        overwritten: a mid-run snapshot change is exactly the thing this field exists to
        make visible, so it must not be smoothed away by last-write-wins."""
        if not served:
            return
        if self.served_model is None:
            self.served_model = served
        elif served not in self.served_model.split(" | "):
            self.served_model = f"{self.served_model} | {served}"

    def _select_anthropic(self, prompt: str) -> list[str]:
        # effective_temperature already accounts for Claude-5 rejection and the
        # temp-default (None) regime, so send it verbatim when present.
        temp_kwargs = (
            {"temperature": self.effective_temperature}
            if self.effective_temperature is not None
            else {}
        )
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            tools=[self._tool],
            tool_choice={"type": "tool", "name": _TOOL_NAME},
            **temp_kwargs,
        )
        if self.usage is not None:
            u = resp.usage
            self.usage.record_tokens(
                self.model,
                getattr(u, "input_tokens", 0),
                getattr(u, "output_tokens", 0),
            )
        with self._obs_lock:
            self._note_served(getattr(resp, "model", None))
        for block in resp.content:
            if getattr(block, "type", None) == "tool_use" and block.name == _TOOL_NAME:
                return block.input.get("section_ids", []) or []
        return []

    def _select_openai(self, prompt: str) -> list[str]:
        tool_choice = (
            "auto"
            if self._compat is not None
            else {"type": "function", "function": {"name": _TOOL_NAME}}
        )
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            tools=[self._tool],
            tool_choice=tool_choice,
            **({} if self.seed is None else {"seed": self.seed}),
            **token_kwargs(self.model, self.max_tokens),
            **{
                k: v
                for k, v in sampling_kwargs(
                    self.model,
                    temperature=self.temperature,
                    reasoning_effort=self.reasoning_effort,
                ).items()
                if not (self._thinking_off and k == "reasoning_effort")
            },
            **self._thinking_off,
        )
        if self.usage is not None:
            self.usage.record(self.model, resp.usage)
        msg = resp.choices[0].message
        # Did reasoning ACTUALLY run? Read off the response, because a lever the vendor
        # accepts and ignores is invisible in the request. Sticky-true: one reasoning
        # response is enough to say the run reasoned.
        ran = reasoning_ran(msg)
        fp = getattr(resp, "system_fingerprint", None)
        with self._obs_lock:
            if ran is not None:
                # Sticky true: one reasoning response is enough to say the run reasoned.
                self.observed_reasoning = bool(self.observed_reasoning) or ran
            if fp and self.system_fingerprint is None:
                self.system_fingerprint = fp
            self._note_served(getattr(resp, "model", None))
        if not msg.tool_calls:
            return []
        try:
            args = json.loads(msg.tool_calls[0].function.arguments)
            return args.get("section_ids", [])
        except Exception:
            return []

    def rank(self, question: str) -> list[str]:
        # Assemble the user message exactly as the production `search` tool does.
        prompt = f"QUESTION:\n{question}\n\nSECTIONS:\n{self._listing}"
        # Paced here, the one point every route passes through, and BEFORE the call so a
        # slot is spent on a request rather than on a result. A 429 pushes every waiting
        # caller back, not only this one, because the limit is on the key.
        if self._limiter is not None:
            self._limiter.acquire()
        try:
            ids = (
                self._select_anthropic(prompt)
                if self._is_anthropic
                else self._select_openai(prompt)
            )
        except Exception as e:
            if self._limiter is not None and is_rate_limit_error(e):
                self._limiter.penalise()
            raise
        seen: set[str] = set()
        out: list[str] = []
        for i in ids:
            if i in self._by_doc and i not in seen:
                seen.add(i)
                out.append(i)
        return out
