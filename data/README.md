# Data

Everything upstream of the evaluation: turning four clinical practice
guidelines into a retrievable section tree, and generating the question set
from that tree.

```
data/
├── corpus/       guideline text -> section tree  (code only; see below)
│   ├── epub_pipeline/     EPUB sources (AHA / Circulation)
│   ├── html_pipeline/     saved HTML chapters (ADA Diabetes Care)
│   ├── pipeline_core/     shared stages: visual description, section summary
│   └── guidelines/<id>/guideline.yaml
└── questions/    section tree -> question set
    ├── generate_questions.py
    └── prompts/
```

## The corpus is not distributed

The four guidelines are copyrighted by the AHA and the ADA. This repository
carries the code that processes them and the per-guideline configuration, never
their text. You obtain each source and put it where the configuration expects
it.

| `guideline_id` | Source | Place at |
|---|---|---|
| `cvd-prevention-2019` | [2019 AHA/ACC Primary Prevention of Cardiovascular Disease](https://www.ahajournals.org/doi/10.1161/cir.0000000000000678) | `data/corpus/epub/cvd-prevention-2019.epub` |
| `blood-pressure-2025` | [2025 AHA High Blood Pressure in Adults](https://www.ahajournals.org/doi/10.1161/CIR.0000000000001356) | `data/corpus/epub/blood-pressure-2025.epub` |
| `dyslipidemia-2026` | [2026 AHA Management of Blood Cholesterol](https://www.ahajournals.org/doi/10.1161/CIR.0000000000001423) | `data/corpus/epub/dyslipidemia-2026.epub` |
| `diabetes-care-2026` | [ADA Standards of Care in Diabetes 2026](https://diabetesjournals.org/care/issue/49/Supplement_1) | `data/corpus/html/diabetes-care-2026/` |

The three AHA guidelines download as EPUB from the article page. The ADA
Standards is not one file but a set of journal chapters, so save each chapter
page from the browser ("Save Page As", complete page). That gives you a
`<N>. <title>.html` file beside a `<N>. <title>_files/` directory of images.
Keep both. Extraction orders chapters by the leading number and pulls figures
out of `_files/`.

All four are free to read at the time of writing. What you may then do with the
files is the publishers' call.

## Building the tree

```bash
cd data/corpus
uv run python -m epub_pipeline run blood-pressure-2025
uv run python -m html_pipeline run diabetes-care-2026
uv run python -m epub_pipeline list          # status of every EPUB guideline
```

Each run writes into `data/corpus/guidelines/<id>/`, beside the
`guideline.yaml` that is already there:

```
tree.json                       hierarchical SectionNode list + resource index
sections/<section_id>.txt       plain text per node, with [[figure:fid]] / [[table:tid]] markers
resources/figures/<fid>.*       image, caption, metadata, model transcription
resources/tables/<tid>.*        HTML fragment, markdown, caption, metadata
```

None of it is committed. `.gitignore` keeps the guideline text out.

Stages run in order and record their own completion, so a re-run resumes where
it stopped. `extract` parses the source and calls no model. `describe_visuals`
sends figures and image-only tables to a vision model. `summarize` writes a
one-line summary per section, down to tree level 3, into `tree.json`. The last
two need `OPENAI_API_KEY`, read from the environment or from a `.env` at the
repository root; copy `.env.example` and fill it in. The tree itself builds
without a key.

## Generating questions

```bash
uv run python data/questions/generate_questions.py blood-pressure-2025 \
    --styles lookup scenario sdm \
    --questions-per-node 2 \
    --output data/questions/out/blood-pressure-2025/questions.jsonl
```

Every question is anchored to a **retrieval unit**, a level-3 sub-topic node,
because that is the granularity the system retrieves at. Anchoring deeper, at
every leaf, yields near-duplicate questions from sibling leaves. Each record
keeps its `chapter_id` and `subtopic_id`, so hit rate can be scored at either
granularity with a field lookup.

Chapters are enumerated for two reasons. A sub-topic is a chapter's child, and
non-clinical front matter is filtered out at chapter level. No question is
anchored at a chapter, with one structural exception: a chapter with no level-3
children holds its content directly, and would be invisible at sub-topic
granularity, so it becomes the unit itself and is assembled whole. Two chapters
here are like that, between them 6 of the 222 released questions.

Three styles, one prompt each in `prompts/`. `lookup` is a direct factual
question, `scenario` a short clinical vignette, `sdm` a shared decision-making
question with a patient preference in it.

The records carry section identifiers, titles and character counts. They never
carry guideline text.

## The released question set

The 222 questions used in the paper are published as a dataset, not rebuilt
from this code. Sampling from the per-guideline output involves a balancing
pass, and a reproduction should start from the same questions.

https://huggingface.co/datasets/oberst-lab/verifiable-by-construction
