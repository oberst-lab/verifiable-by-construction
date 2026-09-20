"""Retrieval methods for the benchmark — one common interface."""

from .base import CandidateUnit, build_candidates, hits, gt_doc_id, gt_chapter_doc_id

__all__ = [
    "CandidateUnit",
    "build_candidates",
    "hits",
    "gt_doc_id",
    "gt_chapter_doc_id",
]
