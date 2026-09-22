"""The vendor's own identity, kept structurally separate from every tenant's.

This is the part of the data model that procurement and a SOC 2 auditor ask about first, and the
answer has to be structural rather than procedural. "Our staff are instructed not to read
customer data" is not an answer. "Our staff *cannot* read customer data without a time-boxed
grant the customer approved, and here is the table" is.

Four separations, and each closes a door the others leave open:

**A separate table.** ``users.role`` has a CHECK listing exactly ADMIN, DEV, TEST and PROD, so a
platform role is literally unstorable inside a tenant. No JIT-provisioning path, no migration and
no seeding script can mint one by accident, because the column would reject the value.

**A separate token audience.** A platform token carries ``aud=platform`` and the tenant API
accepts only ``aud=tenant``. A stolen or misrouted operator token authenticates nowhere useful.

**A separate host and a separate ASGI app.** The operator plane is not mounted on the customer-
facing application at all, so a routing mistake cannot expose an operator endpoint to a tenant,
and a customer's WAF rules never see operator traffic.

**Reading tenant content needs a grant the tenant approved.** Not a role, not a flag -- a row,
with an expiry, a reason and an audit trail. Break-glass exists for genuine emergencies, notifies
every tenant ADMIN the moment it is used, and is its own capability.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, Timestamps, UUIDPrimaryKey


class OperatorRole(StrEnum):
    """Vendor-side roles. Deliberately *not* the tenant roles, and not stored in ``users``."""

    SUPPORT = "SUPPORT"
    """Can see tenant *metadata* -- names, counts, job states, error rates -- and no content."""

    ENGINEER = "ENGINEER"
    """Support, plus index and job operations. Still no content without a grant."""

    OWNER = "OWNER"
    """Everything, including provisioning tenants and using break-glass."""


class GrantScope(StrEnum):
    METADATA = "metadata"
    """Job states, counts, configuration. The default, and enough for most support work."""

    CONTENT = "content"
    """Document text and search results. The one that needs explicit approval."""

    IMPERSONATE = "impersonate"
    """Act as a named user, to reproduce what they see. Every request logged as elevated."""


class PlatformOperator(UUIDPrimaryKey, Timestamps, Base):
    """A member of vendor staff.

    Global by definition -- it exists above tenants rather than inside one. MFA is required
    rather than optional: this account can, with approval, read customer data, and an operator
    account without a second factor is the single highest-value phishing target in the product.
    """

    __tablename__ = "platform_operators"
    __table_args__ = (
        UniqueConstraint("email", name="uq_platform_operators_email"),
        CheckConstraint("role IN ('SUPPORT','ENGINEER','OWNER')", name="platform_role_valid"),
    )

    email: Mapped[str] = mapped_column(String(320), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False, default="")
    role: Mapped[str] = mapped_column(String(16), nullable=False, default=OperatorRole.SUPPORT)
    password_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Not nullable in effect: ``mfa_enrolled`` gates sign-in, and the settings validator refuses
    #: a production deployment that turns that requirement off.
    mfa_secret_enc: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    mfa_enrolled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PlatformSession(UUIDPrimaryKey, Timestamps, Base):
    """An operator's session. Shorter-lived than a tenant user's, on purpose.

    A tenant user is doing their job all day and being signed out is friction. An operator is
    doing something exceptional, and a session that outlives the task is a credential lying
    around. Two hours, not two weeks.
    """

    __tablename__ = "platform_sessions"
    __table_args__ = (
        Index("ix_platform_sessions_operator", "operator_id"),
        Index("ix_platform_sessions_expires", "expires_at"),
    )

    operator_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("platform_operators.id", ondelete="CASCADE"), nullable=False
    )
    refresh_token_hash: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)


class SupportGrant(UUIDPrimaryKey, Timestamps, Base):
    """A tenant's time-boxed permission for one operator to look at something.

    The three properties that make this worth having rather than theatre:

    **It expires by itself.** Not "should be revoked when the ticket closes" -- a grant that
    someone must remember to revoke is a grant that stays open. ``expires_at`` is not nullable.

    **It names a scope, and ``content`` is separate from ``metadata``.** Most support work needs
    to know that a job failed, not what the document said. Making the cheap case cheap is what
    stops people asking for content access habitually.

    **It is approved by a named tenant ADMIN**, except under break-glass -- which sets
    ``is_break_glass``, notifies every tenant ADMIN immediately, and is its own capability that
    only OWNER holds.
    """

    __tablename__ = "support_grants"
    __table_args__ = (
        Index("ix_support_grants_tenant", "tenant_id", "expires_at"),
        Index("ix_support_grants_operator", "operator_id", "expires_at"),
        CheckConstraint("scope IN ('metadata','content','impersonate')", name="grant_scope_valid"),
        # A grant with no expiry is a permanent back door. The database refuses one.
        CheckConstraint("expires_at > created_at", name="grant_must_expire_after_creation"),
        CheckConstraint(
            "(is_break_glass AND approved_by_user_id IS NULL) OR approved_by_user_id IS NOT NULL",
            name="grant_needs_approval_unless_break_glass",
        ),
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    operator_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("platform_operators.id", ondelete="CASCADE"), nullable=False
    )
    scope: Mapped[str] = mapped_column(String(16), nullable=False, default=GrantScope.METADATA)

    #: The tenant ADMIN who approved it. Null only for break-glass, which the CHECK enforces.
    approved_by_user_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    #: Free text shown to the tenant, usually a ticket reference. Required, because a grant
    #: nobody can explain later is one nobody can review.
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    ticket_ref: Mapped[str | None] = mapped_column(String(64), nullable=True)

    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    revoked_by_user_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)

    #: Used without prior approval. Legitimate -- a tenant whose only ADMIN is locked out cannot
    #: approve anything -- and it notifies every tenant ADMIN the moment it happens.
    is_break_glass: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    tenant_notified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    #: Narrows an impersonation grant to one user, so "reproduce what Alice sees" does not become
    #: "act as anyone in the tenant".
    impersonate_user_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    #: Further narrowing, same subtractive principle as `user_collection_scopes`.
    collection_ids: Mapped[list[uuid.UUID]] = mapped_column(
        ARRAY(PgUUID(as_uuid=True)), nullable=False, default=list, server_default="{}"
    )

    #: Every request made under this grant, counted. A grant used four hundred times to answer
    #: one ticket is worth asking about.
    use_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0, server_default="0")
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class PlatformAuditLog(UUIDPrimaryKey, Base):
    """Everything an operator did, in its own table.

    Separate from tenant `audit_logs` deliberately. A tenant must be able to read the record of
    what the vendor did in *their* tenant, and the vendor must not be able to edit it. Keeping
    the two apart means the tenant-facing view can be a filtered, read-only projection of this
    without exposing other customers' rows.

    Append-only in effect: the migration REVOKEs UPDATE and DELETE from the application role.
    """

    __tablename__ = "platform_audit_logs"
    __table_args__ = (
        Index("ix_platform_audit_operator", "operator_id", "created_at"),
        Index("ix_platform_audit_tenant", "tenant_id", "created_at"),
        Index("ix_platform_audit_action", "action", "created_at"),
    )

    operator_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    operator_email: Mapped[str] = mapped_column(String(320), nullable=False, default="")
    #: Null for actions not about a specific tenant, such as signing in.
    tenant_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    grant_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)

    action: Mapped[str] = mapped_column(String(64), nullable=False)
    resource_type: Mapped[str | None] = mapped_column(String(48), nullable=True)
    resource_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: Whether this action read customer *content* as opposed to metadata. The single field a
    #: tenant's security review actually wants to filter on.
    touched_content: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    details: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
