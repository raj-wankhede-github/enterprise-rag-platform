"""Reading the corpus back out of Postgres, for a generation rebuild.

Postgres is truth and OpenSearch is a derived projection, so a rebuild is a read of the former
into the latter. Three things make that read non-obvious:

**It re-embeds nothing.** ``chunk_vectors`` holds an fp16 vector per (chunk, embedder), so a
rebuild forced by a mapping or analyzer change -- which is most of them -- costs no embedding
quota at all. Only a rebuild that *changes the embedder* has to embed, and then only chunks whose
text the new model has not already seen. This is the difference between "we can rebuild your
index tonight" and "we need three days and your embedding budget".

**Paging is keyset, on the natural key.** ``(tenant_id, document_version_id, ordinal)`` is
ordered, unique and already indexed, so each page is an index range scan. The obvious alternative
-- paging on the OpenSearch ``_id`` -- is a trap: that id is a SHA-256, so filtering on it means
computing the hash for every candidate row on every page, and its ordering is random with respect
to every index the table has. ``OFFSET`` is the other trap; it re-scans everything already
emitted, which turns a long backfill quadratic exactly when it is already too slow.

**The cursor is opaque to the orchestrator.** It is a string the source emits and the source
parses. That is what lets the same rebuild machinery resume a Postgres backfill, a blob-store
replay or a connector re-sync without knowing anything about their key spaces.

A long backfill *will* be interrupted -- a deploy, an OOM, a node restart -- and restarting from
zero on a ten-hour rebuild means the rebuild never finishes. Documents written after the cursor
passes are covered by the dual-write instead; one written exactly at the boundary is written
twice under the same deterministic ``_id``, which is an overwrite rather than a duplicate. That
determinism is what makes the two mechanisms compose.
"""

from __future__ import annotations

import logging
import struct
import uuid
from collections.abc import AsyncIterator, Sequence
from typing import Any

from sqlalchemy import Select, func, select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.chunk import Chunk, ChunkContext, ChunkVector, Parent
from app.models.document import Document, DocumentVersion
from app.models.tenant import TenantIndexBinding
from app.search.bulk import BulkDoc
from app.search.generations import chunk_document_id, chunk_index_name, parent_index_name
from app.search.rebuild import BackfillBatch

logger = logging.getLogger(__name__)

#: Rows per database round trip. Independent of the bulk batch size: this bounds memory on our
#: side, the bulk batcher bounds request size on the cluster's.
DEFAULT_PAGE = 1000

_CHUNK_PHASE = "c"
_PARENT_PHASE = "p"


class MissingVectorError(Exception):
    """A chunk has no persisted vector for the target embedder.

    Raised rather than indexed with a null vector. A chunk with no embedding is invisible to the
    dense leg while looking perfectly healthy in every count -- the one partial failure the
    reconciliation gate cannot see, because the document *is* there.
    """


class PostgresChunkSource:
    """Streams a pool's chunks and then its parents as indexable documents."""

    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        *,
        generation: int,
        fingerprint: str,
        embedder_id: str,
        contextualizer_version: str,
        page: int = DEFAULT_PAGE,
        require_vectors: bool = True,
    ) -> None:
        self.sessionmaker = sessionmaker
        self.generation = generation
        self.fingerprint = fingerprint
        self.embedder_id = embedder_id
        self.contextualizer_version = contextualizer_version
        self.page = page
        #: False only when the caller produces vectors on the fly, which is the case for a
        #: rebuild that changes the embedder. Never false for a mapping or analyzer rebuild.
        self.require_vectors = require_vectors

    async def count(self, *, pool: int) -> int:
        """Chunks plus parents this pool should hold. Compared against OpenSearch at BACKFILLED."""
        async with self.sessionmaker() as session:
            tenants = await self._tenants_in_pool(session, pool)
            if not tenants:
                return 0
            chunks = await session.scalar(self._active_scope(select(func.count()).select_from(Chunk), Chunk, tenants))
            parents = await session.scalar(
                self._active_scope(select(func.count()).select_from(Parent), Parent, tenants)
            )
            return int(chunks or 0) + int(parents or 0)

    async def stream(self, *, pool: int, after: str | None) -> AsyncIterator[BackfillBatch]:
        """Yield batches of chunks, then of parents, resumable from ``after``.

        Chunks precede parents deliberately. The parent leg projects a section down onto its
        children, so a parent present before its children is briefly a section pointing at
        nothing. Harmless here -- no alias points at this index yet -- but the ordering keeps the
        invariant true at every moment and costs nothing to hold.
        """
        phase, key = _parse_cursor(after)

        async with self.sessionmaker() as session:
            tenants = await self._tenants_in_pool(session, pool)
            if not tenants:
                return

            if phase == _CHUNK_PHASE:
                async for batch in self._stream_chunks(session, pool=pool, tenants=tenants, after=key):
                    yield batch
                key = None

            async for batch in self._stream_parents(session, pool=pool, tenants=tenants, after=key):
                yield batch

    # ------------------------------------------------------------------------------------

    async def _tenants_in_pool(self, session: AsyncSession, pool: int) -> list[uuid.UUID]:
        """Which tenants live in this pool.

        Read from ``tenant_index_bindings`` rather than recomputed from the consistent hash:
        changing ``pool_count`` would otherwise re-point existing tenants at a pool that does not
        hold their chunks, which reads to a customer as "all my documents disappeared".
        """
        rows = await session.scalars(select(TenantIndexBinding.tenant_id).where(TenantIndexBinding.pool == pool))
        return list(rows)

    def _active_scope(
        self, query: Select[Any], model: type[Chunk] | type[Parent], tenants: Sequence[uuid.UUID]
    ) -> Select[Any]:
        """Restrict to the active version of each document.

        Superseded versions stay in Postgres for audit, diffing and one-click rollback. Indexing
        them would roughly double the corpus and let two versions of the same policy answer the
        same question.
        """
        return (
            query.join(DocumentVersion, DocumentVersion.id == model.document_version_id)
            .join(Document, Document.id == DocumentVersion.document_id)
            .where(model.tenant_id.in_(tenants))
            .where(Document.active_version_id == DocumentVersion.id)
            .where(Document.deleted_at.is_(None))
        )

    async def _stream_chunks(
        self, session: AsyncSession, *, pool: int, tenants: Sequence[uuid.UUID], after: tuple[Any, ...] | None
    ) -> AsyncIterator[BackfillBatch]:
        index = chunk_index_name(generation=self.generation, pool=pool)
        key = after

        while True:
            rows = (await session.execute(self._chunk_query(tenants, key))).all()
            if not rows:
                return

            docs: list[BulkDoc] = []
            for chunk, parent, version, document, vector, context in rows:
                chunk_id = chunk_document_id(
                    tenant_id=chunk.tenant_id, doc_version_id=chunk.document_version_id, ordinal=chunk.ordinal
                )
                embedding = decode_vector(vector)
                if embedding is None and self.require_vectors:
                    raise MissingVectorError(
                        f"chunk {chunk.id} has no {self.embedder_id} vector; indexing it would make it "
                        "invisible to the dense leg while still counting as present"
                    )
                docs.append(
                    BulkDoc(
                        index=index,
                        doc_id=chunk_id,
                        routing=str(chunk.tenant_id),
                        body=chunk_body(
                            chunk=chunk,
                            parent=parent,
                            version=version,
                            document=document,
                            chunk_id=chunk_id,
                            embedding=embedding,
                            context_line=context or "",
                            fingerprint=self.fingerprint,
                        ),
                    )
                )

            last = rows[-1][0]
            key = (last.tenant_id, last.document_version_id, last.ordinal)
            yield BackfillBatch(docs=docs, cursor=_format_cursor(_CHUNK_PHASE, key))
            if len(rows) < self.page:
                return

    async def _stream_parents(
        self, session: AsyncSession, *, pool: int, tenants: Sequence[uuid.UUID], after: tuple[Any, ...] | None
    ) -> AsyncIterator[BackfillBatch]:
        index = parent_index_name(generation=self.generation, pool=pool)
        key = after

        while True:
            query = self._active_scope(select(Parent, Document), Parent, tenants).order_by(
                Parent.tenant_id, Parent.document_version_id, Parent.ordinal
            )
            if key is not None:
                query = query.where(tuple_(Parent.tenant_id, Parent.document_version_id, Parent.ordinal) > key)
            rows = (await session.execute(query.limit(self.page))).all()
            if not rows:
                return

            docs = [
                BulkDoc(
                    index=index,
                    doc_id=chunk_document_id(
                        tenant_id=parent.tenant_id,
                        doc_version_id=parent.document_version_id,
                        ordinal=parent.ordinal,
                    ),
                    routing=str(parent.tenant_id),
                    body=parent_body(parent=parent, document=document, fingerprint=self.fingerprint),
                )
                for parent, document in rows
            ]

            last = rows[-1][0]
            key = (last.tenant_id, last.document_version_id, last.ordinal)
            yield BackfillBatch(docs=docs, cursor=_format_cursor(_PARENT_PHASE, key))
            if len(rows) < self.page:
                return

    def _chunk_query(self, tenants: Sequence[uuid.UUID], after: tuple[Any, ...] | None) -> Select[Any]:
        query = self._active_scope(
            select(Chunk, Parent, DocumentVersion, Document, ChunkVector.vector, ChunkContext.text), Chunk, tenants
        )
        query = (
            query.join(Parent, Parent.id == Chunk.parent_id)
            .outerjoin(
                ChunkVector,
                (ChunkVector.chunk_id == Chunk.id) & (ChunkVector.embedder_id == self.embedder_id),
            )
            .outerjoin(
                ChunkContext,
                (ChunkContext.content_sha256 == Chunk.content_sha256)
                & (ChunkContext.contextualizer_version == self.contextualizer_version)
                & (ChunkContext.tenant_id == Chunk.tenant_id),
            )
            .order_by(Chunk.tenant_id, Chunk.document_version_id, Chunk.ordinal)
            .limit(self.page)
        )
        if after is not None:
            # Keyset, not OFFSET: an index range scan whose cost is independent of how far into
            # the corpus the backfill has already got.
            query = query.where(tuple_(Chunk.tenant_id, Chunk.document_version_id, Chunk.ordinal) > after)
        return query


# --------------------------------------------------------------------------------------------
# Cursors
# --------------------------------------------------------------------------------------------


def _format_cursor(phase: str, key: tuple[Any, ...]) -> str:
    tenant, version, ordinal = key
    return f"{phase}:{tenant}:{version}:{ordinal}"


def _parse_cursor(cursor: str | None) -> tuple[str, tuple[Any, ...] | None]:
    """Split a cursor into its phase and key.

    A cursor from a previous run may name the parent phase, in which case the chunk phase is
    already complete and is skipped entirely rather than re-walked.
    """
    if not cursor:
        return _CHUNK_PHASE, None
    phase, tenant, version, ordinal = cursor.split(":", 3)
    return phase, (uuid.UUID(tenant), uuid.UUID(version), int(ordinal))


def decode_vector(raw: bytes | None) -> list[float] | None:
    """fp16 bytes back to a list of floats.

    Stored as fp16 because 1024-d fp32 is 4 KB per chunk against 2 KB, and the precision lost sits
    far below what HNSW's own approximation already costs.
    """
    if not raw:
        return None
    count = len(raw) // 2
    return [float(value) for value in struct.unpack(f"<{count}e", raw[: count * 2])]


def encode_vector(values: Sequence[float]) -> bytes:
    """The inverse, for the ingestion path that writes ``chunk_vectors``."""
    return struct.pack(f"<{len(values)}e", *values)


# --------------------------------------------------------------------------------------------
# Document bodies. These must match ``ingestion.pipeline.index_action`` field for field.
# --------------------------------------------------------------------------------------------


def chunk_body(
    *,
    chunk: Chunk,
    parent: Parent,
    version: DocumentVersion,
    document: Document,
    chunk_id: str,
    embedding: list[float] | None,
    context_line: str,
    fingerprint: str,
) -> dict[str, Any]:
    """Under ``dynamic: strict`` a divergence from the ingest path is a bulk rejection rather
    than silent drift -- which is what that setting is worth the occasional failed rebuild for."""
    return {
        "tenant_id": str(chunk.tenant_id),
        "doc_id": str(document.id),
        "doc_version_id": str(version.id),
        "parent_id": chunk_document_id(
            tenant_id=parent.tenant_id, doc_version_id=parent.document_version_id, ordinal=parent.ordinal
        ),
        "chunk_id": chunk_id,
        "ordinal": chunk.ordinal,
        "generation_fingerprint": fingerprint,
        "content": chunk.text,
        "context_line": context_line,
        "title": document.title,
        "heading_path": parent.heading_path or "",
        "embedding": embedding,
        "visibility_rank": document.visibility_rank,
        "access_groups": list(document.allowed_groups) or ["*"],
        "is_active": True,
        "is_superseded": document.is_superseded,
        "block_kinds": list(chunk.block_kinds),
        "page_from": chunk.page_from,
        "page_to": chunk.page_to,
        "token_count": chunk.token_count,
        "content_sha256": chunk.content_sha256.hex(),
        "simhash64": str(chunk.simhash64),
    }


def parent_body(*, parent: Parent, document: Document, fingerprint: str, child_count: int = 0) -> dict[str, Any]:
    return {
        "tenant_id": str(parent.tenant_id),
        "doc_id": str(document.id),
        "doc_version_id": str(parent.document_version_id),
        # The key the parent leg projects through. Dropping it makes the leg silently link
        # nothing, which has happened here once already.
        "parent_id": chunk_document_id(
            tenant_id=parent.tenant_id, doc_version_id=parent.document_version_id, ordinal=parent.ordinal
        ),
        "generation_fingerprint": fingerprint,
        "content": parent.text,
        "title": document.title,
        "heading_path": parent.heading_path or "",
        # parent_ordinal, not ordinal: the parent mapping renames it, and dynamic: strict turns
        # the wrong name into a rejected bulk item rather than a silently unmapped field.
        "parent_ordinal": parent.ordinal,
        "child_count": child_count,
        "visibility_rank": document.visibility_rank,
        "access_groups": list(document.allowed_groups) or ["*"],
        "is_active": True,
        "is_superseded": document.is_superseded,
        "token_count": parent.token_count,
        "page_from": parent.page_from,
        "page_to": parent.page_to,
    }
