"""Reranking: the second-stage scorer, and the entailment scorer it doubles as."""

from __future__ import annotations

from app.retrieval.rerank.base import IdentityReranker, Reranker, apply_scores
from app.retrieval.rerank.factory import build_entailment_scorer, build_reranker
from app.retrieval.rerank.lexical import LexicalEntailmentScorer, LexicalReranker
from app.retrieval.rerank.onnx_client import CrossEncoderReranker, HostedReranker

__all__ = [
    "CrossEncoderReranker",
    "HostedReranker",
    "IdentityReranker",
    "LexicalEntailmentScorer",
    "LexicalReranker",
    "Reranker",
    "apply_scores",
    "build_entailment_scorer",
    "build_reranker",
]
