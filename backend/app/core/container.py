"""Composition root: the one place a concrete implementation is chosen.

Every swappable part of this product is a Protocol with two or more implementations -- embedder,
reranker, contextualizer, loader, blob store, rate limiter. That is what makes the offline triple
(`LLM_PROVIDER=none`, `EMBEDDING_PROVIDER=hashing`, `RERANKER_PROVIDER=identity`) possible, and
the offline triple is what keeps the ablation table runnable on every pull request.

It only works if the choosing happens *here*. A module that imports its own dependency by name
cannot be run offline, cannot be run in a BYOC deployment that lacks that dependency, and cannot
be tested without it -- and the failure is discovered late, as an ImportError at startup in the
one environment nobody develops in.

Everything is cached per settings object rather than per process. Two settings objects mean two
containers, which is what lets a test build an app with injected fakes beside a real one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from app.core.config import Settings
from app.embeddings.base import Embedder
from app.ingestion.chunker import StructureAwareChunker
from app.ingestion.contextualize import Contextualizer, TemplateContextualizer
from app.ingestion.loaders.base import LoaderRegistry
from app.ingestion.loaders.registry import build_registry
from app.ingestion.pipeline import IngestionPipeline
from app.storage.blobs import BlobStore, build_blob_store

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class Container:
    """Everything an ingestion or retrieval path needs, already chosen."""

    settings: Settings
    registry: LoaderRegistry
    chunker: StructureAwareChunker
    contextualizer: Contextualizer
    embedder: Embedder
    blobs: BlobStore

    def pipeline(self) -> IngestionPipeline:
        return IngestionPipeline(
            registry=self.registry,
            chunker=self.chunker,
            contextualizer=self.contextualizer,
            embedder=self.embedder,
        )


def build_embedder(settings: Settings) -> Embedder:
    """Currently always the hashing embedder.

    **A real embedder is not yet implemented.** The ``models`` container serves ``/rerank`` and
    nothing else, so there is no ``/embed`` endpoint and no client for one. Setting
    ``EMBEDDING_PROVIDER=onnx`` is accepted and logged, and you still get hashing.

    That is stated loudly rather than failing, because the rest of the system is complete and
    testable without it, and because the alternative -- a deployment that will not boot -- helps
    nobody. It is also stated loudly rather than silently, because a deployment unknowingly
    running hashing embeddings is the worst kind of production problem: retrieval works, BM25
    carries it, relevance is quietly poor, and nothing looks broken.

    The generation fingerprint includes the embedder id, so hashing-embedded and
    properly-embedded chunks can never blend in one index -- whichever is running, the index is
    internally consistent.
    """
    provider = str(getattr(settings, "embedding_provider", "hashing")).lower()
    if provider != "hashing":
        logger.warning(
            "embedder.not_implemented",
            extra={"requested": provider, "using": "hashing", "detail": "the models service exposes no /embed route"},
        )

    from app.embeddings.hashing import HashingEmbedder

    return HashingEmbedder(dimension=int(getattr(settings, "embedding_dimension", 384)))


def build_contextualizer(settings: Settings) -> Contextualizer:
    """Template unless an LLM is configured.

    The template already recovers most of the benefit, because the dominant failure contextual
    retrieval fixes is "this chunk does not say which policy it is about" -- which a deterministic
    header answers. The LLM version is better and is not free.
    """
    if str(getattr(settings, "contextualizer", "template")).lower() == "llm":
        from app.ingestion.llm_contextualize import LLMContextualizer
        from app.llm.factory import build_llm

        provider = build_llm(settings)
        if provider.available:
            return LLMContextualizer(provider)
        logger.warning("contextualizer.llm_unavailable", extra={"detail": "falling back to the template"})

    return TemplateContextualizer()


@lru_cache(maxsize=4)
def build_container(settings: Settings) -> Container:
    return Container(
        settings=settings,
        registry=build_registry(settings),
        chunker=StructureAwareChunker(),
        contextualizer=build_contextualizer(settings),
        embedder=build_embedder(settings),
        blobs=build_blob_store(settings),
    )


def build_ingestion_pipeline(settings: Settings) -> IngestionPipeline:
    return build_container(settings).pipeline()


def build_blob_storage(settings: Settings) -> BlobStore:
    return build_container(settings).blobs


def describe(settings: Settings) -> dict[str, Any]:
    """What was actually chosen, for the health endpoint and the startup log.

    Worth surfacing because the commonest confusing production state is a deployment silently
    running the hashing embedder -- retrieval works, relevance is poor, and nothing looks broken.
    Naming the choices makes that a five-second check rather than an afternoon.
    """
    container = build_container(settings)
    return {
        "embedder": container.embedder.id,
        "contextualizer": container.contextualizer.version,
        "loaders": [loader.name for loader in container.registry.loaders],
        "blob_store": type(container.blobs).__name__,
    }
