"""UI emission — the one place the agent's tool side effects live."""

from __future__ import annotations

from typing import Any, Protocol

from .models import Citation, ClinicalCalculator, Retrieval, RetrievedDoc, UIState


class UIEvents(Protocol):
    """Builds the protocol-specific UI event for each thing the agent surfaces."""

    def document(self, doc: RetrievedDoc) -> Any: ...

    def citation(self, citation: Citation) -> Any: ...

    def calculator(self, calc: ClinicalCalculator) -> Any: ...

    def search(self, retrieval: Retrieval) -> Any: ...


def record_search(state: UIState, retrieval: Retrieval, events: UIEvents) -> Any:
    """Record one `search` call's result (query + surfaced doc_ids) and return
    its UI event. Kept in UIState so the retrieve stage is observable alongside
    the read documents — not deduped (each query's result is its own record)."""
    state.retrievals.append(retrieval)
    return events.search(retrieval)


def record_document(state: UIState, doc: RetrievedDoc, events: UIEvents) -> Any:
    """Add a read Source to the side panel and return its UI event."""
    if not any(d.doc_id == doc.doc_id for d in state.documents):
        state.documents.append(doc)
    return events.document(doc)


def record_citation(state: UIState, citation: Citation, events: UIEvents) -> Any:
    """Attach a Citation (verbatim quote) to its Source and return its UI event."""
    state.citations.append(citation)
    return events.citation(citation)


def emit_calculator(calc: ClinicalCalculator, events: UIEvents) -> Any:
    """Surface an external calculator as an inline call-to-action. Stateless —
    a caller re-derives it from the message parts, so it isn't kept in
    UIState."""
    return events.calculator(calc)
