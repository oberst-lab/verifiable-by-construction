# Evaluation

The harness that runs a model over the question set, and the metrics that score
what came back. Every number in the paper comes from one of these.

```
eval/
├── common/
│   ├── harness.py          runs a model over the questions -> answers.jsonl
│   ├── segmenter.py        an answer -> claim sentences, and the validity gate
│   ├── claim_extractor.py  the LLM claim filter
│   ├── citation_locator.py pairs each citation with the claim it supports
│   ├── freeze_context.py   turns one run's read sets into a frozen context
│   ├── scoring.py  cache.py  usage.py  rate_limit.py  release_meta.py
│   ├── model_endpoints.py  how a model id is reached
│   └── prompts/            the system prompt each arm overrides with
├── citation/
│   ├── vccr.py                    verbatim compliance
│   ├── coverage.py                citation coverage
│   ├── quote_length.py            how long the quotes are (no model calls)
│   ├── claim_support.py           the support judge
│   ├── claim_support_sentence.py  claim support, sentence-based
│   ├── verbatim_criteria.json     the match tiers and the pass set
│   └── prompts/                   the claim filter and the support judge
├── claim_funnel/funnel.py  every metric on one unit, the claim
├── recovery/               is the evidence elsewhere in what the model read?
└── retrieval/              hit rate: BM25 and dense baselines, and LLM selection
```

## From zero to the first number

Four steps. The first two happen outside this directory.

**1. Build the corpus.** Every scorer reads guideline text to check a quote
against the section it cites, so the tree has to exist first. See
[../data/README.md](../data/README.md). Building it needs `OPENAI_API_KEY` in a
`.env` at the repository root.

**2. Get the question set.** The 222 questions used in the paper are published
as a dataset: <https://huggingface.co/datasets/oberst-lab/verifiable-by-construction>.
It arrives as one `qa.jsonl` per guideline, and the harness takes a single path,
so concatenate them:

```bash
mkdir -p eval/result/datasets/balanced222
cat <downloaded>/*/qa.jsonl > eval/result/datasets/balanced222/qa.jsonl
```

Nothing needs converting. The harness reads `question_id`, `question` and
`guideline_id` from each record. The retrieval track additionally reads
`question_style` and the `source` block, whose `node_id` is the sub-topic the
question was written from and is the ground truth a hit is scored against.

**3. Answer them.**

```bash
uv run python eval/common/harness.py \
    --questions eval/result/datasets/balanced222/qa.jsonl \
    --output    eval/result/answers/gpt-5.4-mini.jsonl \
    --model gpt-5.4-mini --limit 5
```

Use `--limit` first. A full pass is 222 questions against a reasoning model, and
everything downstream is scored from that run, so look at five answers before
paying for all of them.

**4. Score them.** `vccr.py` makes no model calls, so it is the cheapest thing
to confirm the chain works:

```bash
uv run python eval/citation/vccr.py \
    --answers eval/result/answers/gpt-5.4-mini.jsonl \
    --report  eval/result/verbatim_compliance/gpt-5.4-mini/report.json
```

## The order things run in

```
harness.py            -> answers.jsonl
vccr.py               -> verbatim compliance     (no model calls)
coverage.py           -> citation coverage
claim_support_*.py    -> claim support
recovery/readset_probe.py -> where the evidence actually is
claim_funnel/funnel.py    -> the funnel          (no model calls)
```

`funnel.py` computes nothing of its own. It reads what the scorers wrote and
switches the unit to the claim, so run them first or it has nothing to fold.
`recovery/funnel.py` stands in the same relation to the recovery probe.

Answers, reports and caches all land under `eval/result/`, which is gitignored.
The caches hold guideline prose, so that directory has to stay out of the
repository.

## Arms

`harness.py` runs one of two, and stamps which one in the record:

| Arm | Flag | What the model does |
|---|---|---|
| end-to-end | (default) | retrieves, reads, answers with citations |
| frozen context | `--frozen-context` | answers from sections a fixed retriever already opened |

Frozen context swaps in `common/prompts/system_prompt_frozen_context.md`. It is
how a generation comparison avoids being confounded by which model retrieved
better: build one context with `freeze_context.py`, then point every model at
it. Every published generation run used this arm.

## Reproducing a number

The published reports are not in this repository, so you reproduce a number by
re-running the stage that makes it, as above. `vccr.py` and both funnels call no
model at all. Given the answers and the corpus they are deterministic: free, and
exact every time. The rest call a judge, and cost what a judge costs.

## Providers

The harness reaches OpenAI and Anthropic. The paper evaluated twelve models from
five vendors, seven of them from these two; the other five ran through a host
that is not part of this release, so those rows cannot be reproduced from here.
`build_agent_model` in `common/model_endpoints.py` decides how a model id is
reached, and is where another route would go.

## Judges

A judge is a model like any other. Its decoding is pinned in one place,
`claim_support.judge_settings()`, and every report records that block, so two
reports compare only when the blocks match. The paper's support judge is a model
this release does not reach. Re-running claim support here therefore means
picking a judge and disclosing it; the number will not match a published one.
