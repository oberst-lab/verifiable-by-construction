# Verifiable by Construction

A clinical question-answering agent that attaches a verbatim quote to each claim
it makes, and the evaluation that measures whether those quotes hold up. Code for
*Verifiable by Construction: A Verbatim-Citation Harness for Clinical Guideline
Question Answering* (<https://arxiv.org/abs/2609.15964>).

| | |
|---|---|
| [`data/`](data/README.md) | four clinical guidelines into a retrievable section tree, and the question set built from it |
| [`system/`](system/README.md) | the agent: retrieval, generation, and the citation protocol it answers under |
| [`eval/`](eval/README.md) | the harness that runs a model over the questions, and the metrics that score what came back |

They run in that order. To reproduce a number from the paper, follow
[eval/README.md](eval/README.md); it starts from an empty checkout.

## Quick start

Python 3.11 or newer, and [uv](https://docs.astral.sh/uv/).

```bash
cp .env.example .env          # OPENAI_API_KEY, ANTHROPIC_API_KEY
uv sync
```

Build the corpus next, following [data/README.md](data/README.md). Once a tree
is in place the agent will answer a question:

```bash
uv run python -m system ask "What is the blood pressure treatment goal for most adults?"
```

## Not in this repository

- **The guideline text**, copyrighted by the AHA and the ADA. `data/README.md`
  says where to obtain each of the four and where to put it.
- **The question set**, published as a dataset:
  <https://huggingface.co/datasets/oberst-lab/verifiable-by-construction>.

## Providers

OpenAI and Anthropic. The paper evaluated twelve models from five vendors,
seven of them from these two; the other five ran through a host that is not
part of this release. `build_agent_model` in `eval/common/model_endpoints.py`
decides how a model id is reached, and is where another route would go.

## License

MIT. See [LICENSE](LICENSE).
