"""Selecting a reranker from configuration."""

from __future__ import annotations

from app.core.config import Settings
from app.retrieval.rerank.base import IdentityReranker, Reranker
from app.retrieval.rerank.lexical import LexicalEntailmentScorer, LexicalReranker
from app.retrieval.rerank.onnx_client import CrossEncoderReranker


def build_reranker(settings: Settings) -> Reranker:
    """Construct the configured reranker.

    ``onnx`` without a ``MODELS_SERVICE_URL`` falls back to the lexical reranker rather than
    failing to start. A deployment that forgot to enable the models profile should serve slightly
    worse results with a loud log line, not refuse to boot -- reranking is a quality stage, not a
    dependency.
    """
    provider = settings.reranker_provider
    if provider == "identity":
        return IdentityReranker()
    if provider == "lexical":
        return LexicalReranker()
    if provider == "onnx":
        if not settings.models_service_url:
            return LexicalReranker()
        return CrossEncoderReranker(
            settings.models_service_url,
            max_pairs=settings.rerank_top_n,
            max_length=settings.rerank_max_tokens,
        )
    # Hosted providers are constructed per tenant, since the key and the consent flag are
    # tenant-scoped rather than deployment-scoped.
    return LexicalReranker()


def build_entailment_scorer(settings: Settings) -> object | None:
    """The scorer citation verification uses, if any.

    Returns the cross-encoder when one is deployed -- the same model, the same container -- and
    the lexical proxy otherwise. ``None`` disables the entailment layer entirely, leaving the
    deterministic value checks, which is the correct behaviour when nothing trustworthy is
    available: verification degrades rather than guessing.
    """
    if settings.reranker_provider == "onnx" and settings.models_service_url:
        return CrossEncoderReranker(
            settings.models_service_url,
            max_pairs=settings.rerank_top_n,
            max_length=settings.rerank_max_tokens,
        )
    if settings.reranker_provider == "lexical":
        return LexicalEntailmentScorer()
    return None
