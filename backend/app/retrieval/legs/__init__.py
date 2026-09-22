"""Retrieval legs: one ranked list each, fused by RRF."""

from __future__ import annotations

from app.retrieval.legs.base import MsearchItem, RetrievalLeg, candidate_from_hit, parse_hits, response_error
from app.retrieval.legs.dense import DenseKnnLeg
from app.retrieval.legs.lexical import Bm25Leg, ExactTokenLeg
from app.retrieval.legs.parent import MAX_CHILDREN_PER_PARENT, ParentBm25Leg, project_to_children

__all__ = [
    "MAX_CHILDREN_PER_PARENT",
    "Bm25Leg",
    "DenseKnnLeg",
    "ExactTokenLeg",
    "MsearchItem",
    "ParentBm25Leg",
    "RetrievalLeg",
    "candidate_from_hit",
    "parse_hits",
    "project_to_children",
    "response_error",
]
