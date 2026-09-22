"""The ingestion queue and the index-generation state machine.

The queue is Postgres with ``FOR UPDATE SKIP LOCKED`` rather than Redis or Celery. The decisive
property is not throughput -- it is that **enqueue commits in the same transaction as the
business row**, so there is no window in which a document is saved but its job is lost. A Redis
queue needs an outbox table to achieve the same thing, which is strictly more machinery.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TenantScoped, Timestamps, UUIDPrimaryKey
from app.models.enums import JobStatus


class IngestJob(UUIDPrimaryKey, Timestamps, TenantScoped, Base):
    __tablename__ = "ingest_jobs"
    __table_args__ = (
        CheckConstraint("status IN ('QUEUED','RUNNING','DONE','FAILED','DEAD')", name="status_valid"),
        # The claim query orders by (priority, run_after). An interactive re-index must beat a
        # 50k-document connector backfill, and per-tenant in-flight caps stop one tenant's
        # backfill starving everyone else -- the most common multi-tenant ingestion failure.
        Index("ix_jobs_claim", "priority", "run_after"),
        Index("ix_jobs_tenant_inflight", "tenant_id", "status"),
    )

    job_type: Mapped[str] = mapped_column(String(32), nullable=False)
    document_version_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("document_versions.id", ondelete="CASCADE"), nullable=True
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    #: Lower runs sooner.
    priority: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=100)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=JobStatus.QUEUED)
    #: Checkpoint, so a crash at document 900 of 1000 resumes at 900 rather than at zero.
    stage: Mapped[str | None] = mapped_column(String(24), nullable=True)
    cursor: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=5)
    run_after: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    #: Reclaimed by the maintenance worker when a process dies mid-job.
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    worker_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class JobEvent(UUIDPrimaryKey, Timestamps, TenantScoped, Base):
    """Per-stage telemetry. Where the per-1k-chunk ingestion cost figure comes from."""

    __tablename__ = "job_events"
    __table_args__ = (Index("ix_job_events_job", "job_id", "created_at"),)

    job_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("ingest_jobs.id", ondelete="CASCADE"), nullable=False
    )
    stage: Mapped[str] = mapped_column(String(24), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cost: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    detail: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)


class IndexGeneration(UUIDPrimaryKey, Timestamps, Base):
    """One generation of one pool, and where it is in the rebuild state machine.

        PLANNED -> BUILDING -> BACKFILLED -> VERIFYING -> LIVE -> DRAINING -> RETIRED

    ``LIVE`` is unreachable without a passing shadow evaluation against the previous
    generation's recorded metrics. That is the gate which stops a stalled backfill from quietly
    degrading relevance in production: wrong-generation documents are invisible to queries, so a
    half-built index shows up as missing results in a shadow eval rather than as worse answers.
    """

    __tablename__ = "index_generations"
    __table_args__ = (
        UniqueConstraint("generation", "pool", name="uq_generation_pool"),
        CheckConstraint(
            "state IN ('PLANNED','BUILDING','BACKFILLED','VERIFYING','LIVE','DRAINING','RETIRED')",
            name="state_valid",
        ),
    )

    generation: Mapped[int] = mapped_column(Integer, nullable=False)
    pool: Mapped[int] = mapped_column(Integer, nullable=False)
    index_name: Mapped[str] = mapped_column(String(255), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="PLANNED")

    #: The digest over embedder id, chunker version, contextualizer version, mapping hash and
    #: analyzer hash. Stamped on every chunk and asserted as a term filter in every query.
    fingerprint: Mapped[str] = mapped_column(String(32), nullable=False)
    embedder_id: Mapped[str] = mapped_column(String(128), nullable=False)
    chunker_version: Mapped[str] = mapped_column(String(32), nullable=False)
    contextualizer_version: Mapped[str] = mapped_column(String(32), nullable=False)
    mapping_sha: Mapped[str] = mapped_column(String(32), nullable=False)
    analysis_sha: Mapped[str] = mapped_column(String(32), nullable=False)

    backfill_cursor: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    verified_metrics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    live_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    retired_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
