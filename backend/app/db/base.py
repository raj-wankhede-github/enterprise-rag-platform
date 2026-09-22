"""Declarative base, naming conventions and the tenant-scoping mixin."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import DateTime, ForeignKey, MetaData, func
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Explicit names so Alembic autogenerate produces stable, reviewable migrations rather than
# database-assigned identifiers that differ between environments.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)

    def __repr__(self) -> str:
        pk = getattr(self, "id", None)
        return f"<{type(self).__name__} id={pk}>"


def utcnow() -> datetime:
    return datetime.now(UTC)


class UUIDPrimaryKey:
    id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), primary_key=True, default=uuid.uuid4)


class Timestamps:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=utcnow
    )


class TenantScoped:
    """Marks a table as belonging to exactly one tenant.

    Inheriting this is what arms the guards in ``db/guards.py``: SELECTs are auto-filtered, INSERTs
    are stamped, and any attempt to change ``tenant_id`` raises. ``tests/unit/test_models.py``
    asserts every non-global table carries it, so a new model is scoped by default rather than by
    the author remembering.
    """

    __tenant_scoped__ = True

    @property
    def _tenant_marker(self) -> Any:  # pragma: no cover - documentation shim
        return None

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )


#: Tables that are deliberately not tenant-scoped. Every entry needs a reason, because the
#: architecture test treats this list as the only way to opt out of tenancy.
GLOBAL_TABLES: frozenset[str] = frozenset(
    {
        # --- control plane: exists above tenants, or defines them ---
        "tenants",
        "tenant_domains",
        "platform_operators",
        "platform_sessions",
        "support_grants",
        "platform_audit_logs",
        # Resolved *before* a principal exists (the request must know which pool to query), so
        # it cannot depend on the guard that a principal arms. It carries a tenant_id column but
        # is read by the routing layer, not by tenant-facing queries.
        "tenant_index_bindings",
        # Cluster-wide index state. Generations span every tenant in a pool.
        "index_generations",
        # --- read before a principal exists, which is the whole point of them ---
        # Email-first discovery resolves the tenant FROM these, so they cannot be filtered by a
        # tenant that has not been established yet. Both carry tenant_id and every read asserts
        # it explicitly; neither is reachable from a tenant-facing query path.
        "idp_configs",
        "user_identities",
        # Keyed on the submitted email rather than on a user, because the attempts worth
        # detecting are those against accounts that do not exist -- and therefore against no
        # tenant. Contains no password, token or hash of either.
        "login_attempts",
        # Resolved from a cookie before the principal it identifies has been loaded.
        "sessions",
        # --- deliberately shared caches ---
        # Content-addressed: one row per distinct byte sequence, refcounted across tenants. A
        # blob row reveals nothing about who references it.
        "blobs",
        # Keyed on (model, content hash). The value is a deterministic function of text the
        # holder already has, so sharing leaks nothing. Tenants who object -- and every BYOC
        # deployment -- set EMBEDDING_CACHE_SCOPE=tenant, which puts tenant_id in the key.
        "embedding_cache",
        # --- machinery ---
        "alembic_version",
    }
)
