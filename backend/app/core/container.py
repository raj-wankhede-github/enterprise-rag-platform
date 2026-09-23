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
from dataclasses import dataclass, replace
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
    """The synchronous path: always the hashing embedder.

    A real embedder must be *probed* before it can be used -- its id and dimension are properties
    of the deployed artefact, not of our configuration, and both enter the generation
    fingerprint. That probe is a network call, so it cannot happen here.

    ``resolve_embedder`` is the async counterpart, and is what the API and worker call at
    startup. This one exists for the offline path, for CI, and for anything that needs a
    container without a network.
    """
    from app.embeddings.hashing import HashingEmbedder

    return HashingEmbedder(dimension=int(getattr(settings, "embedding_dimension", 384)))


async def resolve_embedder(settings: Settings) -> Embedder:
    """The real embedder, discovered from the models service.

    **Raises rather than falling back.** Every other remote dependency here degrades -- a
    reranker timeout drops to fusion order, an LLM outage drops to the template. Falling back
    here would write vectors from a different vector space into the same index, and nothing
    downstream could tell: similarity between a hashed vector and a bge vector is meaningless,
    the dense leg returns plausible nonsense for those documents, and no error appears anywhere.

    A slow ingest is recoverable. A poisoned index is a rebuild.

    A deployment that genuinely wants hashing asks for it by name, and gets it without a warning
    because it was a choice.
    """
    provider = str(getattr(settings, "embedding_provider", "hashing")).lower()
    if provider == "hashing":
        return build_embedder(settings)

    url = getattr(settings, "models_service_url", None)
    if not url:
        raise ValueError(
            f"EMBEDDING_PROVIDER={provider} needs MODELS_SERVICE_URL. Set it, or set "
            "EMBEDDING_PROVIDER=hashing to run without a model service."
        )

    from app.embeddings.onnx_client import build_onnx_embedder

    return await build_onnx_embedder(str(url))


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


async def build_container_async(settings: Settings) -> Container:
    """The container an API or worker process builds at startup.

    Differs from the sync one in exactly one way: the embedder is resolved against the running
    models service rather than assumed. Everything else is identical, so a test that uses the
    sync container is testing the same wiring.
    """
    container = build_container(settings)
    embedder = await resolve_embedder(settings)
    if embedder.id == container.embedder.id:
        return container

    resolved = replace(container, embedder=embedder)
    logger.info("container.embedder_resolved", extra={"id": embedder.id, "dimension": embedder.dimension})
    return resolved


def build_ingestion_pipeline(settings: Settings) -> IngestionPipeline:
    """The offline pipeline. Workers use ``build_container_async`` and pass the result in."""
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
