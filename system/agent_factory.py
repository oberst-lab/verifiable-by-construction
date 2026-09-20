from __future__ import annotations

from pathlib import Path

from pydantic_ai import Agent, RunContext, ToolReturn
from pydantic_ai.capabilities import ProcessHistory, Thinking
from pydantic_ai.models import Model
from pydantic_ai.settings import ModelSettings

from . import clinical_tools
from . import history as history_mod
from .emitter import (
    UIEvents,
    emit_calculator,
    record_document,
    record_search,
)
from .guidelines import GuidelineScope, section_to_retrieved_doc
from .models import AgentDeps, Retrieval, SectionContent
from .search import SearchBackend, SemanticSearchBackend

PROMPT_TEMPLATE = Path(__file__).parent / "prompts" / "system_prompt.md"


def build_instructions() -> str:
    """The static protocol prose. The available guidelines are fetched live via
    `list_guidelines`, so no guideline data is injected here. One prompt at every reasoning
    effort: citations are inline ({{cite:doc#N|quote}}), so the protocol does not depend on
    a thinking channel and the effort setting changes only whether thinking is enabled.
    """
    return PROMPT_TEMPLATE.read_text(encoding="utf-8")


def instructions_with_patient(protocol: str, patient: str | None) -> str:
    """System prompt = static protocol + a persistent patient-context block
    (Letta-style "core memory": always at the top, every turn). Patient context
    is per-conversation-stable, so within a conversation this string is constant
    → it sits in the cached prefix (anthropic_cache) without invalidating it."""
    if patient and patient.strip():
        return f"{protocol}\n\n<patient_context>\n{patient.strip()}\n</patient_context>"
    return protocol


def retrieval_budget() -> str:
    """Read-count guidance, appended to the system prompt each turn. It governs
    how many of `search`'s candidates the agent OPENS with `read_section`,
    which is distinct from how many the backend SURFACES: the backend picks the
    candidate set, this nudges how many to actually read.
    """
    return (
        "\n\nRetrieval budget — granularity: SUB-TOPIC (each unit is a small "
        '"4.1"-style sub-topic).\n'
        "Read as many sub-topics as the question needs — typically 3–5, more "
        "when it spans multiple chapters or guidelines. Each sub-topic is small "
        "and narrow, so a single one often won't hold the whole answer."
    )


_ALLOWED_EXTRA_SETTINGS = frozenset(
    {
        "anthropic_thinking",  # {'type': 'enabled'|'disabled'|'adaptive', ...}
        "anthropic_effort",  # -> output_config.effort
        "openai_reasoning_effort",  # -> reasoning.effort / reasoning_effort
        "extra_body",  # raw request-body passthrough; CONTENTS whitelisted below
    }
)

_ALLOWED_EXTRA_BODY_KEYS = frozenset({"chat_template_kwargs"})


def _checked_extras(extra: dict | None) -> dict:
    """Reject anything outside `_ALLOWED_EXTRA_SETTINGS` before it can shadow a setting."""
    if not extra:
        return {}
    bad = sorted(set(extra) - _ALLOWED_EXTRA_SETTINGS)
    if bad:
        raise ValueError(
            f"extra_model_settings may only carry reasoning controls "
            f"{sorted(_ALLOWED_EXTRA_SETTINGS)}; got {bad}. max_tokens has its own "
            "parameter, and temperature must never be set (reasoning models reject it "
            "or ignore it silently, so setting it records a config that never applied)."
        )
    body = extra.get("extra_body")
    if body is not None:
        if not isinstance(body, dict):
            raise ValueError(
                f"extra_body must be a dict of request-body keys; got {type(body).__name__}."
            )
        bad_body = sorted(set(body) - _ALLOWED_EXTRA_BODY_KEYS)
        if bad_body:
            raise ValueError(
                f"extra_body may only carry {sorted(_ALLOWED_EXTRA_BODY_KEYS)}; got "
                f"{bad_body}. It is a raw passthrough into the request JSON, so an "
                "unchecked key here would defeat the whole point of this whitelist -- "
                "temperature included."
            )
    return dict(extra)


def make_agent(
    *,
    events: UIEvents,
    model: str | Model = "openai-responses:gpt-5.4",
    max_tokens: int = 32000,
    thinking_effort: str | bool | None = "medium",
    extra_model_settings: dict | None = None,
    search_backend: SearchBackend | None = None,
    instructions_override: str | None = None,
    retrieval_tools: bool = True,
) -> Agent[AgentDeps, str]:
    """Build the clinical agent. `events` receives the agent's tool side effects;
    the agent itself never names a UI protocol.
    """
    backend = search_backend or SemanticSearchBackend()

    def _process_history(ctx: RunContext[AgentDeps], messages: list):
        # Trim prior-turn section reads + (optionally) reasoning. See history.py.
        return history_mod.apply(messages, trim_thinking=ctx.deps.trim_prior_thinking)

    _protocol = instructions_override or build_instructions()

    # thinking_effort None → no Thinking capability, so no reasoning/thinking
    # parameter is sent and the provider uses its native default.
    thinking = [Thinking(effort=thinking_effort)] if thinking_effort is not None else []
    capabilities = thinking + [ProcessHistory(processor=_process_history)]

    agent = Agent(
        model,
        deps_type=AgentDeps,
        capabilities=capabilities,
        model_settings=ModelSettings(
            max_tokens=max_tokens,
            anthropic_cache=True,
        )
        | _checked_extras(extra_model_settings),
    )

    @agent.instructions
    def instructions(ctx: RunContext[AgentDeps]) -> str:
        protocol = _protocol
        if retrieval_tools:
            protocol = protocol + retrieval_budget()
        return instructions_with_patient(protocol, ctx.deps.patient_context)

    def _scope(ctx: RunContext[AgentDeps]) -> GuidelineScope:
        """This turn's retrieval scope: the selected guidelines, as a ceiling."""
        return GuidelineScope.of(ctx.deps.selected_guidelines)

    async def list_guidelines(ctx: RunContext[AgentDeps]) -> list[dict[str, str]]:
        """List the clinical guidelines currently available to consult (id + name)."""
        return [
            {"guideline_id": g.guideline_id, "guideline_name": g.guideline_name}
            for g in _scope(ctx).guidelines()
        ]

    async def search(
        ctx: RunContext[AgentDeps],
        query: str,
    ) -> ToolReturn:
        """Find the guideline sections most relevant to a clinical query."""
        hits = await backend.search(query, _scope(ctx))
        # Record the retrieve stage (query + surfaced ids) as a first-class signal
        # — observable alongside the read documents, and consumed by the eval.
        retrieval = Retrieval(query=query, doc_ids=[h.doc_id for h in hits])
        return ToolReturn(
            return_value=[h.model_dump() for h in hits],
            metadata=record_search(ctx.deps.state, retrieval, events),
        )

    async def read_section(
        ctx: RunContext[AgentDeps],
        guideline_id: str,
        section_id: str,
    ) -> ToolReturn:
        """Read the full text of one section."""
        scope = _scope(ctx)
        if not scope.allows(guideline_id):
            return ToolReturn(
                return_value=(
                    f"{guideline_id!r} is not among the guidelines the user "
                    f"selected ({list(scope.selected)}). Use only those."
                ),
            )
        try:
            content: SectionContent = scope.read(guideline_id, section_id)
        except ValueError as e:
            return ToolReturn(
                return_value=(
                    f"Could not read that section: {e} Use the exact `section_id` "
                    f"from a `search` result (e.g. 'ch05-s4'), not a number "
                    f"like '5.4'. Call `search` and pick a valid candidate."
                ),
            )
        doc = section_to_retrieved_doc(content)
        return ToolReturn(
            return_value=content.model_dump(),
            metadata=record_document(ctx.deps.state, doc, events),
        )

    if retrieval_tools:
        agent.tool(list_guidelines)
        agent.tool(search)
        agent.tool(read_section)

    @agent.tool
    async def cardiovascular_risk_calculator(
        ctx: RunContext[AgentDeps],
    ) -> ToolReturn:
        """Surface the AHA PREVENT risk calculator as a button in the answer."""
        calc = clinical_tools.PREVENT
        return ToolReturn(
            return_value={"name": calc.name, "url": calc.url},
            metadata=emit_calculator(calc, events),
        )

    return agent
