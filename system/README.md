# System

The clinical QA agent that the evaluation measures. It answers a question from
the guideline corpus and attaches a verbatim quote to each claim it makes.

```
system/
├── __main__.py        the local CLI
├── agent_factory.py   builds the agent and registers its tools
├── search.py          the retrieval backend behind the `search` tool
├── guidelines.py      loads the corpus, resolves section text on demand
├── model_registry.py  which models are offered, and how effort maps to each
├── helper_agent.py    a cheap, no-reasoning agent for background calls
├── models.py          the shared dataclasses and Pydantic models
├── emitter.py         the seam a caller implements to observe tool calls
├── history.py         message-history shaping before each request
├── clinical_tools.py  vetted external calculators the agent can surface
└── prompts/
    ├── system_prompt.md       the answer and citation protocol
    └── retrieval_select.md    the section-selection prompt
```

## Running it locally

```bash
uv run python -m system guidelines        # what the corpus offers
uv run python -m system models            # what you can answer with
uv run python -m system ask "What is the blood pressure treatment goal for most adults?"
```

`ask` runs one question through the full agent loop. It prints the answer, then
what the agent searched for, what it read and what it quoted. `--model` picks
from the registry, `--effort` sets reasoning effort, `--guidelines` restricts
retrieval, and `--patient` pins a patient context into the system prompt.

## Retrieval

There is no vector index. The agent calls `search(query)`, and the backend
spends one cheap model call picking relevant sections from their summaries
alone, without reading any full text. `read_section` then returns a chosen
unit's whole subtree in a single read.

A retrieval unit is a sub-topic, tree level 3, the "5.1 / 4.3" layer of the
tree that `data/corpus` builds. The published results use that granularity and
no other. A chapter is the parent a sub-topic hangs off, not a unit you can
retrieve. The one exception is structural. A chapter with no sub-topics beneath
it is surfaced as a unit, so its content stays reachable.

`SearchBackend` is a protocol. A vector, BM25 or hybrid backend can replace the
default without touching the agent or the tool signature.

## Models

`model_registry.py` is the single source of truth for which models are offered,
their output ceilings, whether they support reasoning, and how one unified
effort level maps onto each vendor's API. It reads no environment variables. A
caller decides what to do about a provider whose key is unset, and the CLI's
choice is to list the whole roster and mark the models it cannot currently
reach.

The registry lists OpenAI and Anthropic models. The evaluation harness also
accepts a raw `provider:model` string, which is how the paper ran models that
are not in the registry.

## Corpus

`guidelines.py` reads `data/corpus/guidelines/`, which is generated and not
committed. Build it first; see [../data/README.md](../data/README.md). With no
corpus present, loading raises.

## Web interface

The agent also has a browser front end, which this repository does not carry.
It is a React and TypeScript single page app built with Vite, talking to a
Starlette server that streams the run through Pydantic AI's Vercel AI adapter.
Nothing in the evaluation depends on it; the harness drives the agent directly.
Contact the authors if you would like that source.
