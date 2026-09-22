"""The ingestion pipeline: bytes in, indexable documents out.

Stages, each a checkpoint so a crash at document 900 of 1000 resumes at 900 rather than at zero:

    probe -> load -> chunk -> contextualize -> embed -> build index actions

The pipeline is deliberately **pure with respect to I/O**: it takes a blob and returns an
``IngestResult`` holding everything needed to write to Postgres and OpenSearch, but performs
neither. That is what lets the whole thing -- including the dedup diff and the reuse accounting --
be tested without a database, a cluster, or a network.

The economics live here. ``reuse`` decides what is *not* recomputed, and the counters it produces
are what an operator sees when they ask why a 400-chunk re-upload cost twelve embeddings.
"""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.embeddings.base import Embedder
from app.ingestion.chunker import StructureAwareChunker
from app.ingestion.contextualize import Contextualizer, DocumentContext, prefixed
from app.ingestion.loaders.base import LoaderRegistry
from app.ingestion.loaders.registry import probe as build_probe
from app.ingestion.types import ChildChunk, ChunkedDocument, ExtractedDocument, ParentChunk
from app.ingestion.versioning import ChunkDiff, diff_chunks, sha256_bytes, sha256_text
from app.search.generations import GenerationSpec
from app.utils.text import simhash64


class UnsupportedDocumentError(Exception):
    """No loader claimed the probe.

    Raised rather than silently skipped: a document that cannot be parsed must appear as a failed
    ingest in the UI, not as a document that exists and never returns from a search.
    """


@dataclass(frozen=True, slots=True)
class PreparedChunk:
    """One child chunk, ready to index."""

    chunk: ChildChunk
    content_sha256: bytes
    context_line: str
    #: The text that was embedded and that BM25 indexes: context line plus chunk text.
    indexed_text: str
    simhash: int
    embedding: list[float] | None
    #: True when the vector came from the cache rather than the embedder.
    embedding_reused: bool


@dataclass(slots=True)
class IngestResult:
    document: ExtractedDocument
    chunked: ChunkedDocument
    prepared: list[PreparedChunk] = field(default_factory=list)
    blob_sha256: bytes = b""
    doc_text_sha256: bytes = b""
    diff: ChunkDiff | None = None
    embedder_id: str = ""
    chunker_version: str = ""
    contextualizer_version: str = ""
    generation_fingerprint: str = ""

    @property
    def parents(self) -> list[ParentChunk]:
        return self.chunked.parents

    @property
    def chunks_embedded(self) -> int:
        return sum(1 for item in self.prepared if not item.embedding_reused)

    @property
    def chunks_reused(self) -> int:
        return sum(1 for item in self.prepared if item.embedding_reused)

    @property
    def reuse_ratio(self) -> float:
        return len(self.prepared) and self.chunks_reused / len(self.prepared)


class IngestionPipeline:
    def __init__(
        self,
        *,
        registry: LoaderRegistry,
        chunker: StructureAwareChunker,
        contextualizer: Contextualizer,
        embedder: Embedder,
    ) -> None:
        self.registry = registry
        self.chunker = chunker
        self.contextualizer = contextualizer
        self.embedder = embedder

    @property
    def generation(self) -> GenerationSpec:
        return GenerationSpec(
            embedder_id=self.embedder.id,
            chunker_version=self.chunker.chunk(ExtractedDocument(blocks=())).chunker_version,
            contextualizer_version=self.contextualizer.version,
            dimension=self.embedder.dimension,
        )

    async def run(
        self,
        blob: bytes,
        filename: str,
        *,
        context: DocumentContext | None = None,
        previous_chunk_hashes: Sequence[bytes] = (),
        vector_cache: Mapping[bytes, list[float]] | None = None,
        context_cache: Mapping[bytes, str] | None = None,
    ) -> IngestResult:
        """Parse, chunk, contextualize and embed.

        ``previous_chunk_hashes`` drives the incremental diff; ``vector_cache`` and
        ``context_cache`` are keyed on chunk content hash and supply whatever has already been
        computed for that exact text -- from an earlier version, from another document, or from
        another tenant. Everything they hit is work that is not paid for twice.
        """
        probe = build_probe(blob, filename)
        loader = self.registry.select(probe)
        if loader is None:
            raise UnsupportedDocumentError(
                f"No loader handles {filename!r} ({probe.mime_type}). "
                "Layout-heavy formats need the parser service; set PARSER_SERVICE_URL."
            )

        document = await loader.load(blob, probe)
        chunked = self.chunker.chunk(document)
        doc_context = context or DocumentContext(title=document.title or probe.filename)

        prepared = await self._prepare(
            chunked.children,
            doc_context,
            vector_cache=vector_cache or {},
            context_cache=context_cache or {},
        )

        generation = GenerationSpec(
            embedder_id=self.embedder.id,
            chunker_version=chunked.chunker_version,
            contextualizer_version=self.contextualizer.version,
            dimension=self.embedder.dimension,
        )

        return IngestResult(
            document=document,
            chunked=chunked,
            prepared=prepared,
            blob_sha256=sha256_bytes(blob),
            doc_text_sha256=sha256_text(document.text),
            diff=diff_chunks(list(previous_chunk_hashes), [item.content_sha256 for item in prepared]),
            embedder_id=self.embedder.id,
            chunker_version=chunked.chunker_version,
            contextualizer_version=self.contextualizer.version,
            generation_fingerprint=generation.fingerprint,
        )

    async def _prepare(
        self,
        children: Sequence[ChildChunk],
        context: DocumentContext,
        *,
        vector_cache: Mapping[bytes, list[float]],
        context_cache: Mapping[bytes, str],
    ) -> list[PreparedChunk]:
        if not children:
            return []

        hashes = [sha256_text(child.text) for child in children]

        # Context lines: reuse cached ones, generate only the rest. Cached by content hash, so
        # boilerplate repeated across a corpus is contextualized once.
        missing = [index for index, digest in enumerate(hashes) if digest not in context_cache]
        generated: dict[int, str] = {}
        if missing:
            lines = await self.contextualizer.contextualize(context, [children[index] for index in missing])
            generated = dict(zip(missing, lines, strict=True))
        context_lines = [context_cache.get(digest) or generated.get(index, "") for index, digest in enumerate(hashes)]

        indexed_texts = [prefixed(line, child.text) for line, child in zip(context_lines, children, strict=True)]

        # Embeddings: the cache is keyed on the *chunk* hash, but what gets embedded is the
        # context-prefixed text. That is safe only because the context line is itself a pure
        # function of (chunk text, document metadata, contextualizer version), all of which are
        # in the fingerprint -- so a cache hit implies the prefixed text would be identical too.
        to_embed = [index for index, digest in enumerate(hashes) if digest not in vector_cache]
        fresh: dict[int, list[float]] = {}
        if to_embed:
            vectors = await self.embedder.embed_passages([indexed_texts[index] for index in to_embed])
            fresh = dict(zip(to_embed, vectors, strict=True))

        prepared: list[PreparedChunk] = []
        for index, child in enumerate(children):
            digest = hashes[index]
            cached = vector_cache.get(digest)
            prepared.append(
                PreparedChunk(
                    chunk=child,
                    content_sha256=digest,
                    context_line=context_lines[index],
                    indexed_text=indexed_texts[index],
                    simhash=simhash64(child.text),
                    embedding=cached if cached is not None else fresh.get(index),
                    embedding_reused=cached is not None,
                )
            )
        return prepared


def index_action(
    item: PreparedChunk,
    *,
    tenant_id: uuid.UUID,
    doc_id: uuid.UUID,
    doc_version_id: uuid.UUID,
    parent_id: str,
    chunk_id: str,
    generation_fingerprint: str,
    title: str,
    visibility_rank: int,
    access_groups: Sequence[str],
    is_active: bool = True,
) -> dict[str, object]:
    """Build the OpenSearch document body for one chunk.

    ``content`` holds the verbatim chunk text and ``context_line`` the generated prefix, stored
    separately so a citation can quote the source without the synthesized sentence. Both are
    analyzed, so both contribute to BM25 -- which is the whole point of contextual retrieval on
    the lexical side.
    """
    chunk = item.chunk
    return {
        "tenant_id": str(tenant_id),
        "doc_id": str(doc_id),
        "doc_version_id": str(doc_version_id),
        "parent_id": parent_id,
        "chunk_id": chunk_id,
        "ordinal": chunk.ordinal,
        "generation_fingerprint": generation_fingerprint,
        "content": chunk.text,
        "context_line": item.context_line,
        "title": title,
        "heading_path": chunk.heading_path,
        "embedding": item.embedding,
        "visibility_rank": visibility_rank,
        "access_groups": list(access_groups) or ["*"],
        "is_active": is_active,
        "is_superseded": False,
        "block_kinds": [kind.value for kind in chunk.block_kinds],
        "page_from": chunk.page_from,
        "page_to": chunk.page_to,
        "token_count": chunk.token_count,
        "content_sha256": item.content_sha256.hex(),
        "simhash64": str(item.simhash),
    }
