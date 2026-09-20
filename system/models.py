from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class RetrievedDoc(BaseModel):
    """A section that the agent has actually read, shown as a card in the side panel."""

    doc_id: str  # "{guideline_id}:{section_number}"
    guideline_id: str
    guideline_name: str
    section_number: int
    title: str
    summary: str
    subsection: str | None = None
    url: str | None = None


class Citation(BaseModel):
    doc_id: str
    quote: str


class Retrieval(BaseModel):
    """One `search` call's result: the query and the doc_ids it surfaced (in
    rank order). Exposed as a first-class signal, distinct from what the agent
    then READS (documents) or CITES — so the retrieve→read→cite stages are each
    observable (UI, evaluation) instead of only the read-set."""

    query: str
    doc_ids: list[str] = Field(default_factory=list)


class ClinicalCalculator(BaseModel):
    """An external clinical calculator the agent points the user to. Rendered as
    an inline call-to-action button in the answer. The URL comes from code (a
    vetted catalog), never from the model — so it can't be hallucinated."""

    name: str
    url: str
    description: str


class SectionSummary(BaseModel):
    """An entry in the `search` tool output — a candidate section the agent may read."""

    doc_id: str
    guideline_id: str
    guideline_name: str
    section_number: int
    section_id: str
    title: str
    summary: str
    subsections: list[str] = Field(default_factory=list)
    url: str | None = None


class SectionContent(BaseModel):
    """The full payload returned by read_section."""

    doc_id: str
    guideline_id: str
    guideline_name: str
    section_number: int
    section_id: str
    title: str
    summary: str
    text: str
    subsection: str | None = None
    url: str | None = None


class UIState(BaseModel):
    """Output state accumulated during a run, for the caller to read."""

    documents: list[RetrievedDoc] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    retrievals: list[Retrieval] = Field(default_factory=list)


@dataclass
class AgentDeps:
    """Run dependencies. A dataclass with a `state` field (satisfies the
    `StateHandler` protocol) so a caller's adapter can manage `state` while
    leaving `selected_guidelines` alone. That is why the guideline selection
    lives here and not on UIState, which such an adapter rebuilds each run.
    """

    state: UIState
    selected_guidelines: list[str] | None = None
    patient_context: str | None = None
    # trim_prior_thinking: from config; when True the history processor drops
    # prior-turn reasoning to save input tokens (see history.py TRIM THINKING).
    trim_prior_thinking: bool = True
