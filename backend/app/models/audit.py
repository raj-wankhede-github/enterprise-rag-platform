"""The tenant-facing audit log, and the answer trace behind every response.

Two tables with very different jobs, kept apart on purpose.

``audit_logs`` is what a customer's compliance team reads. It is append-only -- the migration
REVOKEs UPDATE and DELETE from the application role, so "the vendor could have edited it" is
answerable with a grant table rather than with a promise -- and it carries an optional hash chain
so tampering by anyone, including someone with database access, is detectable rather than merely
prohibited.

``answer_traces`` is what a *user* reads when they ask "why did it say that". It is the product's
own record, deliberately not dependent on an observability vendor: a customer-facing explanation
cannot be gated behind whether Langfuse is up, or behind an ops tool's RBAC. Content capture is
off by default, which is what lets "does the vendor store our documents in a third-party tool" be
answered with "no".
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TenantScoped, Timestamps, UUIDPrimaryKey


class AuditAction(StrEnum):
    """Actions worth a permanent record. Reads are not among them, with two exceptions.

    Logging every read would bury the events that matter under search traffic and would itself
    become a privacy problem -- a complete record of what every employee looked at. The
    exceptions are bulk export and anything a platform operator did, because those are the two a
    customer actually audits.
    """

    LOGIN_SUCCEEDED = "login.succeeded"
    LOGIN_FAILED = "login.failed"
    LOGOUT = "logout"
    SESSION_REVOKED = "session.revoked"
    SESSION_FAMILY_REVOKED = "session.family_revoked"

    USER_CREATED = "user.created"
    USER_ROLE_CHANGED = "user.role_changed"
    USER_DEACTIVATED = "user.deactivated"
    USER_ELEVATED_TO_ADMIN = "user.elevated_to_admin"

    IDP_CREATED = "idp.created"
    IDP_TESTED = "idp.tested"
    IDP_ACTIVATED = "idp.activated"
    IDP_DISABLED = "idp.disabled"
    EMERGENCY_LOGIN_OPENED = "emergency_login.opened"

    DOCUMENT_UPLOADED = "document.uploaded"
    DOCUMENT_VERSION_CREATED = "document.version_created"
    DOCUMENT_DELETED = "document.deleted"
    DOCUMENT_VISIBILITY_CHANGED = "document.visibility_changed"
    DOCUMENTS_EXPORTED = "documents.exported"

    APIKEY_CREATED = "apikey.created"
    APIKEY_REVOKED = "apikey.revoked"

    CONNECTOR_CONNECTED = "connector.connected"
    CONNECTOR_SYNC_COMPLETED = "connector.sync_completed"
    CONNECTOR_DISCONNECTED = "connector.disconnected"

    SUPPORT_GRANT_APPROVED = "support_grant.approved"
    SUPPORT_GRANT_REVOKED = "support_grant.revoked"
    SUPPORT_BREAK_GLASS_USED = "support.break_glass_used"

    INDEX_GENERATION_PROMOTED = "index.generation_promoted"
    RETRIEVAL_PROFILE_CHANGED = "retrieval.profile_changed"


class AuditLog(UUIDPrimaryKey, TenantScoped, Base):
    """One recorded action. Append-only, optionally hash-chained.

    Partitioned by month in the migration: an audit table is the fastest-growing thing in the
    schema and the only one whose retention is contractual rather than chosen, so dropping a
    partition has to be how old rows leave.
    """

    __tablename__ = "audit_logs"
    __table_args__ = (
        Index("ix_audit_tenant_time", "tenant_id", "created_at"),
        Index("ix_audit_actor", "tenant_id", "actor_user_id", "created_at"),
        Index("ix_audit_action", "tenant_id", "action", "created_at"),
        Index("ix_audit_resource", "tenant_id", "resource_type", "resource_id"),
    )

    action: Mapped[str] = mapped_column(String(64), nullable=False)

    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    #: Copied rather than joined. A user who is later deleted must not erase the record of what
    #: they did, and a join to a missing row would render the entry as "unknown".
    actor_email: Mapped[str] = mapped_column(String(320), nullable=False, default="")
    actor_role: Mapped[str | None] = mapped_column(String(16), nullable=True)
    #: Set when a platform operator acted in this tenant. The field a customer filters on first.
    actor_operator_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    support_grant_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    #: True when the actor was operating with temporarily raised rights.
    elevated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    resource_type: Mapped[str | None] = mapped_column(String(48), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    resource_name: Mapped[str | None] = mapped_column(Text, nullable=True)

    #: Before/after for a change. Never document content -- a visibility change records the two
    #: ranks, not the document it was applied to.
    changes: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)

    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)

    #: Tamper-evidence, optional per tenant.
    #:
    #: Each entry hashes the previous entry's hash together with its own content, so altering or
    #: removing any row breaks every hash after it. Append-only permissions stop the application
    #: rewriting history; this detects someone with direct database access doing so, which is the
    #: threat a compliance team is actually asking about.
    prev_hash: Mapped[bytes | None] = mapped_column(LargeBinary(32), nullable=True)
    entry_hash: Mapped[bytes | None] = mapped_column(LargeBinary(32), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())


def chain_hash(
    *,
    prev_hash: bytes | None,
    tenant_id: uuid.UUID,
    action: str,
    actor_email: str,
    resource_id: str | None,
    created_at: datetime,
    details: dict[str, Any],
) -> bytes:
    """One link in the chain.

    Field-delimited with a separator that cannot appear in any input, so that two different
    entries cannot serialise to the same string -- otherwise ``("ab", "c")`` and ``("a", "bc")``
    would collide and two distinct histories would share a hash.
    """
    import json

    payload = "\x1f".join(
        [
            (prev_hash or b"").hex(),
            str(tenant_id),
            action,
            actor_email,
            resource_id or "",
            created_at.isoformat(),
            json.dumps(details, sort_keys=True, separators=(",", ":"), default=str),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).digest()


class AnswerTrace(UUIDPrimaryKey, Timestamps, TenantScoped, Base):
    """Why one answer came out the way it did.

    The product's own record, not an observability vendor's. A customer-facing "why did it say
    that" cannot depend on whether Langfuse is reachable, or on an ops tool's access control --
    and it must survive a BYOC deployment that runs no observability stack at all.

    Content capture is off by default. That is what lets "does the vendor store our documents in
    a third-party tool" be answered with "no", and it is also why the per-stage timings and
    counts are stored separately from the text: the numbers are always safe to keep.
    """

    __tablename__ = "answer_traces"
    __table_args__ = (
        Index("ix_traces_tenant_time", "tenant_id", "created_at"),
        Index("ix_traces_user", "tenant_id", "user_id", "created_at"),
        Index("ix_traces_abstained", "tenant_id", "answerable", "created_at"),
    )

    user_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    #: Links this row to the OTel span, so an engineer can move between the two without
    #: correlating on timestamps.
    trace_id: Mapped[str | None] = mapped_column(String(32), nullable=True)

    #: Only when the tenant enabled content capture. Off, this is null and the trace still
    #: carries every number.
    question: Mapped[str | None] = mapped_column(Text, nullable=True)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)

    answerable: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    abstention_reason: Mapped[str | None] = mapped_column(String(48), nullable=True)
    used_fast_path: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    generation_fingerprint: Mapped[str] = mapped_column(String(32), nullable=False, default="")

    #: Per-leg candidate counts, fusion size, rerank size, evidence size. Always recorded.
    stage_counts: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    #: Per-stage milliseconds. This is what makes a latency regression attributable in one query
    #: rather than in an afternoon.
    stage_latency_ms: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    #: Chunk ids only -- never their text, whatever the content-capture setting. A citation is
    #: resolved from the index at read time, so storing the text here would be a second copy
    #: with its own retention problem.
    cited_chunk_ids: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)

    total_latency_ms: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    cached_input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    estimated_cost_usd: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)

    #: Degradations that did not fail the request: a reranker timeout, a dense leg that errored.
    #: Present here because a user complaining about a bad answer and a trace showing
    #: "rerank_status: timeout" is a one-minute diagnosis.
    diagnostics: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
