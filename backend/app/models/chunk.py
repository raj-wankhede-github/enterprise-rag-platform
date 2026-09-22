"""Parents, children, and the caches that make a re-upload nearly free.

``ChunkContext`` and ``EmbeddingCache`` are both keyed on ``content_sha256`` rather than on a
chunk id. That is the whole trick: identical text is contextualized once and embedded once, no
matter how many versions, documents or (optionally) tenants contain it.
"""

from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import (
    BigInteger,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    PrimaryKeyConstraint,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TenantScoped, Timestamps, UUIDPrimaryKey


class Parent(UUIDPrimaryKey, Timestamps, TenantScoped, Base):
    """A heading section. Searched as its own retrieval leg, and sent to the LLM on expansion."""

    __tablename__ = "parents"
    __table_args__ = (UniqueConstraint("document_version_id", "ordinal", name="uq_parents_version_ordinal"),)

    document_version_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    heading_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_from: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_to: Mapped[int | None] = mapped_column(Integer, nullable=True)


class Chunk(UUIDPrimaryKey, Timestamps, TenantScoped, Base):
    """A child chunk: what gets embedded and searched.

    ``content_sha256`` is the identity that drives incremental updates. Two chunks with the same
    hash have the same text, so they share a context line and an embedding even across documents.
    """

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_version_id", "ordinal", name="uq_chunks_version_ordinal"),
        Index("ix_chunks_content_sha", "content_sha256"),
        Index("ix_chunks_tenant_version", "tenant_id", "document_version_id"),
    )

    document_version_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=False
    )
    parent_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("parents.id", ondelete="CASCADE"), nullable=False
    )
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    content_sha256: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    #: 64-bit SimHash for near-duplicate suppression at assembly time. Stored signed because
    #: Postgres has no unsigned bigint.
    simhash64: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page_from: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_to: Mapped[int | None] = mapped_column(Integer, nullable=True)
    block_kinds: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list, server_default="{}")
    chunk_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)


class ChunkContext(Timestamps, TenantScoped, Base):
    """The one-line context an LLM wrote for a chunk, cached by content hash.

    This is what stops contextual retrieval being paid for twice. A re-ingest, a mapping change
    or a full embedding-model rebuild all reuse these rows, so the LLM cost is paid once per
    distinct piece of text per prompt version -- and identical boilerplate across a corpus
    collapses to a single entry.
    """

    __tablename__ = "chunk_contexts"
    __table_args__ = (
        PrimaryKeyConstraint("tenant_id", "content_sha256", "contextualizer_version", name="pk_chunk_contexts"),
    )

    content_sha256: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    contextualizer_version: Mapped[str] = mapped_column(String(32), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)


class EmbeddingCache(Timestamps, Base):
    """Vector cache keyed on (model, content hash).

    Global by default: the key is a hash of the content and the value is a deterministic function
    of it, so a shared cache reveals nothing an attacker did not already have. BYOC deployments
    and tenants who ask set ``EMBEDDING_CACHE_SCOPE=tenant``, which makes ``tenant_id`` part of
    the key instead of the all-zero sentinel.
    """

    __tablename__ = "embedding_cache"
    __table_args__ = (PrimaryKeyConstraint("model_id", "content_sha256", "tenant_id", name="pk_embedding_cache"),)

    model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    content_sha256: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    #: All-zero UUID when the cache is global, so the primary key stays non-nullable.
    tenant_id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    #: fp16 little-endian, ``dim`` values. Raw bytes rather than ARRAY(REAL): 4x smaller on the
    #: wire and we never query on the contents.
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    hits: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)


class ChunkVector(Timestamps, TenantScoped, Base):
    """Persisted fp16 vectors, so a generation rebuild never re-embeds.

    1024-d fp16 is ~2 KB per chunk, so 10M chunks is ~20 GB of cheap storage. That is the
    difference between "we can rebuild your index tonight" and "we need three days and your
    embedding quota" when a mapping change forces a rebuild.
    """

    __tablename__ = "chunk_vectors"
    __table_args__ = (PrimaryKeyConstraint("chunk_id", "embedder_id", name="pk_chunk_vectors"),)

    chunk_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("chunks.id", ondelete="CASCADE"), nullable=False
    )
    embedder_id: Mapped[str] = mapped_column(String(128), nullable=False)
    dim: Mapped[int] = mapped_column(Integer, nullable=False)
    vector: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
