"""Run the agent locally from a terminal.

uv run python -m system ask "What is the BP treatment goal for most adults?"
uv run python -m system guidelines
uv run python -m system models
"""

from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

# Before the package imports, not after: search.py reads SEARCH_DEBUG at module
# import time, so a value set in .env would otherwise always arrive too late.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

from .agent_factory import make_agent  # noqa: E402
from .guidelines import get_loaded_guidelines  # noqa: E402
from .model_registry import (  # noqa: E402
    DEFAULT_EFFORT,
    DEFAULT_HELPER_MODEL,
    PROVIDER_ENV_VAR,
    model_keys,
    provider_of,
    public_catalog,
    resolve,
    resolve_helper_model,
)

# _EFFORTS: same package, and duplicating the set here is how two copies drift.
from .model_registry import _EFFORTS  # noqa: E402
from .cite_parser import parse_citations  # noqa: E402
from .models import AgentDeps, UIState  # noqa: E402


class _SilentEvents:
    """The `UIEvents` seam, wired to nothing."""

    def document(self, doc: Any) -> None: ...
    def citation(self, citation: Any) -> None: ...
    def calculator(self, calc: Any) -> None: ...
    def search(self, retrieval: Any) -> None: ...


def _require_keys(chosen: Any) -> None:
    """Fail before the run, naming each missing variable."""
    needed = {
        provider_of(chosen.model): f"the answer model, {chosen.model}",
        provider_of(resolve_helper_model(None)): (
            f"the helper model {DEFAULT_HELPER_MODEL}, which every search uses"
        ),
    }
    missing = [
        (PROVIDER_ENV_VAR[p], who)
        for p, who in needed.items()
        if p in PROVIDER_ENV_VAR and not os.environ.get(PROVIDER_ENV_VAR[p])
    ]
    if missing:
        lines = [f"  {var} is needed for {who}." for var, who in missing]
        raise SystemExit(
            "\n".join(lines)
            + "\nCopy .env.example to .env and fill it in, or export the variable."
        )


def _require_corpus() -> list:
    """Turn a missing corpus into a setup message, not a traceback mid-run."""
    try:
        return list(get_loaded_guidelines())
    except (FileNotFoundError, RuntimeError) as e:
        raise SystemExit(
            f"No guideline corpus: {e}\nBuild it first; see data/README.md."
        ) from None


def cmd_guidelines(args: argparse.Namespace) -> None:
    for g in _require_corpus():
        print(f"{g.guideline_id:22} {len(g.sections):>4} units  {g.guideline_name}")


def cmd_models(args: argparse.Namespace) -> None:
    configured = {p for p, var in PROVIDER_ENV_VAR.items() if os.environ.get(var)}
    for m in public_catalog():
        provider = provider_of(resolve(m["id"], None).model)
        tags = []
        if m.get("helper_option"):
            tags.append("helper")
        if provider not in configured:
            tags.append(f"no {PROVIDER_ENV_VAR[provider]}")
        suffix = f"  [{', '.join(tags)}]" if tags else ""
        print(f"{m['id']:20} {m['label']}{suffix}")


def cmd_ask(args: argparse.Namespace) -> None:
    # resolve() applies the registry's own defaults and forces a non-reasoning
    # model to effort "off", so the CLI never has to reimplement that policy.
    chosen = resolve(args.model, args.effort)
    _require_keys(chosen)
    known = {g.guideline_id for g in _require_corpus()}
    unknown = [g for g in (args.guidelines or []) if g not in known]
    if unknown:
        raise SystemExit(
            f"Unknown guideline id(s): {', '.join(unknown)}\n"
            f"Loaded: {', '.join(sorted(known))}"
        )
    agent = make_agent(
        events=_SilentEvents(),
        model=chosen.model,
        max_tokens=chosen.max_tokens,
        thinking_effort=chosen.thinking_effort,
    )
    deps = AgentDeps(
        state=UIState(),
        selected_guidelines=args.guidelines or None,
        patient_context=args.patient,
    )
    result = asyncio.run(agent.run(args.question, deps=deps))

    # Citations are inline markers in the answer, not tool calls, so they are
    # parsed out of the text rather than read off UIState. Numbering them keeps
    # the printed answer readable while showing every quote in full below.
    cites = parse_citations(result.output)
    answer = result.output
    for i, c in enumerate(cites, 1):
        answer = answer.replace(f"{{{{cite:{c.doc_id}|{c.quote}}}}}", f"[{i}]", 1)
    print(answer.strip())

    state = deps.state
    if state.retrievals:
        print("\n--- searched")
        for r in state.retrievals:
            print(f"  {r.query}")
    if state.documents:
        print("\n--- read")
        for d in state.documents:
            print(f"  {getattr(d, 'doc_id', '?')}")
    if cites:
        print("\n--- quoted")
        for i, c in enumerate(cites, 1):
            print(f"  [{i}] {c.doc_id}")
            print(f"      {c.quote.strip()}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="system", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    p_ask = sub.add_parser("ask", help="Ask one question and print the answer")
    p_ask.add_argument("question")
    p_ask.add_argument(
        "--model",
        default=None,
        choices=model_keys(),
        help="Registry key of the model to answer with.",
    )
    p_ask.add_argument(
        "--effort",
        default=DEFAULT_EFFORT,
        choices=sorted(_EFFORTS),
        help="Reasoning effort. A non-reasoning model is forced to 'off'.",
    )
    p_ask.add_argument(
        "--guidelines",
        nargs="+",
        help="Restrict retrieval to these guideline ids (default: all).",
    )
    p_ask.add_argument(
        "--patient", help="Patient context, pinned into the system prompt."
    )
    p_ask.set_defaults(func=cmd_ask)

    p_g = sub.add_parser("guidelines", help="List the loaded guidelines")
    p_g.set_defaults(func=cmd_guidelines)

    p_m = sub.add_parser("models", help="List the models in the registry")
    p_m.set_defaults(func=cmd_models)
    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
