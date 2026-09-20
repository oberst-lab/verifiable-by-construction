"""Lexical baseline: Okapi BM25 over each unit's full text."""

from __future__ import annotations

import re

import snowballstemmer
from rank_bm25 import BM25Okapi

from .base import CandidateUnit

_TOKEN = re.compile(r"[a-z0-9]+")

_STOPWORDS = frozenset(
    """
i me my myself we our ours ourselves you you're you've you'll you'd your yours
yourself yourselves he him his himself she she's her hers herself it it's its
itself they them their theirs themselves what which who whom this that that'll
these those am is are was were be been being have has had having do does did
doing a an the and but if or because as until while of at by for with about
against between into through during before after above below to from up down in
out on off over under again further then once here there when where why how all
any both each few more most other some such no nor not only own same so than too
very s t can will just don don't should should've now d ll m o re ve y ain aren
aren't couldn couldn't didn didn't doesn doesn't hadn hadn't hasn hasn't haven
haven't isn isn't ma mightn mightn't mustn mustn't needn needn't shan shan't
shouldn shouldn't wasn wasn't weren weren't won won't wouldn wouldn't
""".split()
)

_STEMMER = snowballstemmer.stemmer("english")  # Porter2


def _tok(s: str) -> list[str]:
    words = [w for w in _TOKEN.findall(s.lower()) if w not in _STOPWORDS]
    return _STEMMER.stemWords(words)


class BM25Retriever:
    name = "bm25"

    def __init__(self, candidates: list[CandidateUnit]):
        self.candidates = candidates
        corpus = [_tok(c.text or c.summary or c.breadcrumb) for c in candidates]
        self._bm25 = BM25Okapi(corpus)

    @property
    def config(self) -> dict:
        """Everything that changes the numbers, recorded into the report `meta`."""
        return {
            "implementation": "rank_bm25.BM25Okapi",
            "tokenizer": "lower-cased [a-z0-9]+ runs",
            "stopwords": f"NLTK English, inlined ({len(_STOPWORDS)} words)",
            "stemmer": "Porter2 (snowballstemmer, English)",
            "k1": self._bm25.k1,
            "b": self._bm25.b,
            "document": "retrieval unit's full text (summary, then breadcrumb, as fallback)",
        }

    def rank(self, question: str) -> list[str]:
        scores = self._bm25.get_scores(_tok(question))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [self.candidates[i].doc_id for i in order]
