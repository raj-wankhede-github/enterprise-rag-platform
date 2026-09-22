"""Tenant users and collections.

Two decisions here are expensive to change later, so they are made on day one:

* ``users`` is unique on ``(tenant_id, email)``, **not** globally. Consultants legitimately hold
  accounts in several tenants, and a globally unique email cannot be relaxed afterwards without
  a migration that has no correct answer for the rows that already collided.
* ``role`` carries a CHECK constraint listing exactly the four tenant roles. The platform
  operator is a different table entirely, so "PLATFORM" is literally unstorable in a tenant and
  no JIT-provisioning path can be tricked into minting one.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from sqlalchemy import (
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    LargeBinary,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TenantScoped, Timestamps, UUIDPrimaryKey
from app.security.capabilities import Role

_ROLE_CHECK = "role IN ('ADMIN','DEV','TEST','PROD')"


class User(UUIDPrimaryKey, Timestamps, TenantScoped, Base):
    __tablename__ = "users"
    __table_args__ = (
        UniqueConstraint("tenant_id", "email", name="uq_users_tenant_email"),
        CheckConstraint(_ROLE_CHECK, name="role_valid"),
        Index("ix_users_tenant_role", "tenant_id", "role"),
    )

    email: Mapped[str] = mapped_column(String(320), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False, default="")
    role: Mapped[str] = mapped_column(String(16), nullable=False, default=Role.PROD)
    #: A manual override set by an ADMIN. When true, IdP role mapping never downgrades this user.
    role_locked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    password_hash: Mapped[str | None] = mapped_column(Text, nullable=True)
    mfa_secret_enc: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    #: IdP group object ids, cached at login and refreshed on SCIM events. Mirrored into the
    #: ``erp-user-acl`` index so search filters resolve them as a server-side terms lookup.
    group_ids: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list, server_default="{}")
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    deactivated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Collection(UUIDPrimaryKey, Timestamps, TenantScoped, Base):
    """A named grouping of documents. Also the unit of the subtractive user scope."""

    __tablename__ = "collections"
    __table_args__ = (
        UniqueConstraint("tenant_id", "slug", name="uq_collections_tenant_slug"),
        CheckConstraint("default_visibility_rank IN (10,20,30,40)", name="default_visibility_rank_valid"),
    )

    name: Mapped[str] = mapped_column(Text, nullable=False)
    slug: Mapped[str] = mapped_column(String(63), nullable=False)
    default_visibility_rank: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=10)


class UserCollectionScope(Timestamps, TenantScoped, Base):
    """Subtractive scope: narrows a role to named collections.

    Absence of any row means tenant-wide, which is the default. Because this can only ever
    *reduce* what a role grants, the sentence "one role per user, ADMIN > DEV > TEST > PROD"
    stays literally true while still expressing "Alice curates HR only".
    """

    __tablename__ = "user_collection_scopes"

    user_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), primary_key=True
    )
    collection_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("collections.id", ondelete="CASCADE"), primary_key=True
    )


class RoleElevationGrant(UUIDPrimaryKey, Timestamps, TenantScoped, Base):
    """Time-boxed elevation, so "Bob needs DEV for the migration week" is not a role change.

    The principal resolver takes the higher of the stored role and any active grant, and every
    request made under one is logged with ``elevated=true``. At rest the user still has exactly
    one role.
    """

    __tablename__ = "role_elevation_grants"
    __table_args__ = (
        CheckConstraint("granted_role IN ('ADMIN','DEV','TEST','PROD')", name="granted_role_valid"),
        Index("ix_elevation_active", "user_id", "expires_at"),
    )

    user_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    granted_role: Mapped[str] = mapped_column(String(16), nullable=False)
    granted_by: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    reason: Mapped[str] = mapped_column(Text, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
