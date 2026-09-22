"""Document identity, content-addressed blobs and versions.

The design question this file answers: *what happens when the same document is uploaded twice?*

Identity is ``(tenant_id, source_system, external_id)`` -- deliberately NOT the content hash,
because the same bytes legitimately exist as two documents with different owners, ACLs and
metadata. Content identity is tracked separately, by hash, at three levels:

* ``Blob.blob_sha256`` - raw bytes. Identical bytes skip parsing, chunking and embedding.
* ``DocumentVersion.doc_text_sha256`` - extracted text. A metadata-only change reuses the parse.
* ``Chunk.content_sha256`` - per chunk. Drives the incremental re-embedding diff.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TenantScoped, Timestamps, UUIDPrimaryKey
from app.models.enums import AclMode, DocumentStatus, VersionStatus


class Blob(UUIDPrimaryKey, Timestamps, Base):
    """Content-addressed storage. One row per distinct byte sequence.

    Refcounted rather than owned: the same PDF filed in two collections stores its bytes once,
    and deleting one document must not delete the other's content. Only the storage key is
    shared -- nothing about a blob row reveals which tenants reference it.
    """

    __tablename__ = "blobs"
    __table_args__ = (UniqueConstraint("blob_sha256", name="uq_blobs_sha"),)

    blob_sha256: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    storage_key: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str] = mapped_column(String(128), nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    refcount: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class Document(UUIDPrimaryKey, Timestamps, TenantScoped, Base):
    """A logical document. Content lives in versions; this row holds identity and ACLs."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("tenant_id", "source_system", "external_id", name="uq_documents_external"),
        CheckConstraint("visibility_rank IN (10,20,30,40)", name="visibility_rank_valid"),
        CheckConstraint("acl_mode IN ('tenant_public','restricted')", name="acl_mode_valid"),
        CheckConstraint("status IN ('ACTIVE','INACTIVE','DELETED')", name="status_valid"),
        Index("ix_documents_tenant_status", "tenant_id", "status"),
        Index("ix_documents_tenant_rank", "tenant_id", "visibility_rank"),
    )

    collection_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("collections.id", ondelete="SET NULL"), nullable=True
    )
    title: Mapped[str] = mapped_column(Text, nullable=False)
    source_system: Mapped[str] = mapped_column(String(64), nullable=False, default="upload")
    #: Stable key within the source system. For uploads this is a client-supplied key, falling
    #: back to a hash of the normalized filename scoped to the target collection.
    external_id: Mapped[str] = mapped_column(Text, nullable=False)
    external_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_type: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")

    doc_type: Mapped[str | None] = mapped_column(String(64), nullable=True)
    author: Mapped[str | None] = mapped_column(Text, nullable=True)
    language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    publication_date: Mapped[date | None] = mapped_column(Date, nullable=True)
    effective_from: Mapped[date | None] = mapped_column(Date, nullable=True)
    effective_to: Mapped[date | None] = mapped_column(Date, nullable=True)
    is_superseded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    authority_rank: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    visibility_rank: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=10)
    acl_mode: Mapped[str] = mapped_column(String(16), nullable=False, default=AclMode.TENANT_PUBLIC)
    allowed_groups: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list, server_default="{}")
    denied_groups: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list, server_default="{}")
    allowed_user_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(PgUUID(as_uuid=True)), nullable=False, default=list, server_default="{}"
    )
    denied_user_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(PgUUID(as_uuid=True)), nullable=False, default=list, server_default="{}"
    )

    active_version_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    latest_version_no: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=DocumentStatus.ACTIVE)

    owner_user_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    injection_flags: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list, server_default="{}")
    last_permission_sync_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    doc_metadata: Mapped[dict[str, Any]] = mapped_column("metadata", JSONB, nullable=False, default=dict)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DocumentVersion(UUIDPrimaryKey, Timestamps, TenantScoped, Base):
    """One ingest of one document's bytes.

    A re-upload with *changed* bytes creates a new row here rather than mutating the old one, so
    the previous version stays available for audit, diffing and one-click rollback. Readers only
    ever see ``Document.active_version_id``, which is flipped in a single statement -- that is
    what makes the cutover atomic even though the OpenSearch writes are not.
    """

    __tablename__ = "document_versions"
    __table_args__ = (
        UniqueConstraint("document_id", "version_no", name="uq_doc_versions"),
        CheckConstraint("status IN ('UPLOADED','PROCESSING','READY','FAILED','SUPERSEDED')", name="status_valid"),
        Index("ix_doc_versions_doc", "document_id", "version_no"),
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    version_no: Mapped[int] = mapped_column(Integer, nullable=False)

    blob_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("blobs.id", ondelete="RESTRICT"), nullable=False
    )
    #: Duplicated from the blob so the dedup lookup is one index hit on this table.
    blob_sha256: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False, index=True)
    #: Hash of the *extracted text*. Equal hashes mean the parse can be reused even when the raw
    #: bytes differ -- a re-export that only changed PDF metadata, for instance.
    doc_text_sha256: Mapped[bytes | None] = mapped_column(LargeBinary(32), nullable=True)

    status: Mapped[str] = mapped_column(String(16), nullable=False, default=VersionStatus.UPLOADED)
    parser: Mapped[str | None] = mapped_column(String(64), nullable=True)
    parser_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    chunker_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    embedder_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    contextualizer_version: Mapped[str | None] = mapped_column(String(32), nullable=True)
    generation_fingerprint: Mapped[str | None] = mapped_column(String(32), nullable=True)

    page_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    #: Populated by the incremental diff, so an operator can see what a re-upload actually cost.
    chunks_reused: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    chunks_embedded: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    processed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class DocumentUploadEvent(UUIDPrimaryKey, Timestamps, TenantScoped, Base):
    """Every upload attempt, including the ones that created nothing.

    A deduplicated re-upload writes no version, so without this row there would be no record that
    a user uploaded at all -- which is exactly the question an auditor asks.
    """

    __tablename__ = "document_upload_events"
    __table_args__ = (Index("ix_upload_events_tenant_time", "tenant_id", "created_at"),)

    document_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("documents.id", ondelete="SET NULL"), nullable=True
    )
    document_version_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    blob_sha256: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    filename: Mapped[str] = mapped_column(Text, nullable=False)
    size_bytes: Mapped[int] = mapped_column(BigInteger, nullable=False)
    outcome: Mapped[str] = mapped_column(String(40), nullable=False)
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    #: Recorded because the behaviour is role-independent, and we want evidence of that.
    actor_role: Mapped[str | None] = mapped_column(String(16), nullable=True)


class DocumentRelation(Timestamps, TenantScoped, Base):
    """Supersession and reference graph, flattened onto chunks at index time.

    Kept as a graph here and denormalized into the index because traversal at query time is a
    latency cost we do not need: supersession changes rarely, and a change triggers a cheap
    update-by-query over one document's chunks.
    """

    __tablename__ = "document_relations"
    __table_args__ = (
        CheckConstraint("relation IN ('supersedes','references','translates','attaches')", name="relation_valid"),
    )

    from_document_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
    relation: Mapped[str] = mapped_column(String(16), primary_key=True)
    to_document_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True
    )
