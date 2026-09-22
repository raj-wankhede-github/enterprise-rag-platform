from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import Boolean, CheckConstraint, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, Timestamps, UUIDPrimaryKey
from app.models.enums import TenantStatus


class Tenant(UUIDPrimaryKey, Timestamps, Base):
    """A customer. Deliberately has no ``tenant_id`` -- it is the control plane."""

    __tablename__ = "tenants"
    __table_args__ = (
        CheckConstraint(
            "status IN ('PROVISIONING','ACTIVE','SUSPENDED','DELETING')",
            name="status_valid",
        ),
        CheckConstraint("default_visibility_rank IN (10,20,30,40)", name="default_visibility_rank_valid"),
    )

    slug: Mapped[str] = mapped_column(String(63), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=TenantStatus.ACTIVE)
    region: Mapped[str] = mapped_column(String(32), nullable=False, default="eu-central-1")

    password_login_enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    require_mfa_for_password: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    # Default false so an ADMIN does not silently read HR files that no group grants them. Some
    # customers demand the opposite; flipping it is an audited tenant setting, not a code change.
    admin_bypasses_group_acl: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    default_visibility_rank: Mapped[int] = mapped_column(Integer, nullable=False, default=10)
    require_approval_for_visibility_increase: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    trace_content_capture: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    settings: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    plan_limits: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)


class TenantIndexBinding(UUIDPrimaryKey, Timestamps, Base):
    """Which OpenSearch pool and generation serve a tenant.

    Recorded rather than recomputed: rehashing a tenant to a different pool would orphan its
    chunks, so the binding is written once at provisioning and only changed by a migration job.
    """

    __tablename__ = "tenant_index_bindings"

    tenant_id: Mapped[uuid.UUID] = mapped_column(nullable=False, unique=True)
    pool: Mapped[int] = mapped_column(Integer, nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    target_generation: Mapped[int | None] = mapped_column(Integer, nullable=True)
    dedicated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
