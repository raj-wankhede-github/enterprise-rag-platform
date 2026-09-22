"""API keys and their usage record.

The shape here follows from one requirement: verification must be a single indexed lookup. A
table keyed only by a hash would force hashing the presented secret against every row in turn,
and at argon2's cost that is a denial-of-service vector against our own database. ``key_id`` is
therefore stored in plaintext and indexed; only the secret is hashed.

``last_used_at`` is deliberately coarse. Writing it on every request turns a read-only API call
into a write and makes the row a contention point for a busy client; it is updated at most once a
minute, which is precise enough for "is this key still in use" -- the only question anyone asks
of it.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TenantScoped, Timestamps, UUIDPrimaryKey


class ApiKey(UUIDPrimaryKey, Timestamps, TenantScoped, Base):
    __tablename__ = "api_keys"
    __table_args__ = (
        # The lookup key. Unique globally rather than per tenant, because verification happens
        # before a tenant is known -- the key is what establishes it.
        UniqueConstraint("key_id", name="uq_api_keys_key_id"),
        Index("ix_api_keys_tenant", "tenant_id"),
        Index("ix_api_keys_expires", "expires_at"),
        CheckConstraint("role IN ('ADMIN','DEV','TEST','PROD')", name="api_key_role_valid"),
        CheckConstraint("environment IN ('live','test')", name="api_key_environment_valid"),
    )

    #: Plaintext and indexed. Not a secret: it identifies the row, the secret authenticates it.
    key_id: Mapped[str] = mapped_column(String(32), nullable=False)
    #: argon2id. The key itself is shown once at creation and is not recoverable from here.
    secret_hash: Mapped[str] = mapped_column(Text, nullable=False)
    environment: Mapped[str] = mapped_column(String(8), nullable=False, default="live")

    name: Mapped[str] = mapped_column(Text, nullable=False, default="")
    #: Enough of the key to recognise it in a list, and not enough to use it.
    prefix_hint: Mapped[str] = mapped_column(String(48), nullable=False, default="")

    #: The role the key acts as. Capped at its creator's role, and never ADMIN-equivalent in
    #: practice: `apikeys.FORBIDDEN_FOR_KEYS` removes the capabilities that would make it so.
    role: Mapped[str] = mapped_column(String(16), nullable=False, default="PROD")
    #: A chosen subset of that role's capabilities. Empty means all of them, minus the forbidden
    #: set. Subtractive only -- it can never widen what the role grants.
    granted_capabilities: Mapped[list[str]] = mapped_column(
        ARRAY(Text), nullable=False, default=list, server_default="{}"
    )
    #: Narrows the key to named collections, as `user_collection_scopes` does for a person.
    collection_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(PgUUID(as_uuid=True)), nullable=False, default=list, server_default="{}"
    )

    created_by_user_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="SET NULL"), nullable=True
    )
    #: Null means it never expires, which is allowed and is also the thing worth nagging about --
    #: a key with no expiry is one that outlives the person who created it.
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_by_user_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    #: Updated at most once a minute. Writing on every request turns a read into a write and
    #: makes this row a contention point for a busy client.
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_used_ip: Mapped[str | None] = mapped_column(String(45), nullable=True)
    use_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    #: Per-key overrides of the default rate limits, for a customer whose integration genuinely
    #: needs more. A plan limit still caps it.
    rate_limits: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)


class ApiKeyEvent(UUIDPrimaryKey, TenantScoped, Base):
    """Creation, revocation and expiry of keys.

    Separate from `audit_logs` because the questions asked of it are different: "who created the
    key that is making these calls" is asked during an incident, under time pressure, and it
    should not require filtering a general audit stream. Usage itself is not recorded here --
    that would be a row per request.
    """

    __tablename__ = "api_key_events"
    __table_args__ = (
        Index("ix_api_key_events_key", "api_key_id", "created_at"),
        Index("ix_api_key_events_tenant", "tenant_id", "created_at"),
        CheckConstraint("action IN ('created','revoked','expired','rotated')", name="api_key_action_valid"),
    )

    api_key_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("api_keys.id", ondelete="CASCADE"), nullable=False
    )
    action: Mapped[str] = mapped_column(String(16), nullable=False)
    actor_user_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    actor_role: Mapped[str | None] = mapped_column(String(16), nullable=True)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
