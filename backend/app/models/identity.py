"""Identity providers, federated identities, sessions and login attempts.

Four decisions here are load-bearing and expensive to change later.

**A federated identity is matched on the IdP's immutable subject, never on email.** Entra's
``oid``, Google's ``sub``. Email is mutable, reassignable and in many directories shared by a
role account -- matching on it means a departing employee's replacement inherits their identity,
and a display-name change can orphan an account. ``user_identities`` is unique on
``(idp_config_id, subject)``, and the email on the row is a convenience copy only.

**``users`` is unique on ``(tenant_id, lower(email))`` and not globally.** Consultants belong to
several customers, and a global uniqueness constraint is unmigratable once real users exist.

**A tenant's ``default_role`` cannot be ADMIN**, enforced by a database CHECK rather than by
validation. JIT provisioning writes that role, so a misconfigured group claim must not be able to
hand ADMIN to an entire directory. A CHECK is the only layer no future code path can bypass.

**An IdP config is draft -> test -> active, and only one may be active per method.** Activating
an untested config whose tenant has password login disabled locks out the administrator who
broke it, which is why ``tenants.emergency_password_login_until`` exists as the break-glass.
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
from sqlalchemy import text as sa_text
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, Timestamps, UUIDPrimaryKey


class IdpProtocol(StrEnum):
    OIDC = "oidc"
    SAML = "saml"


class IdpKind(StrEnum):
    """What the product knows about the provider beyond the protocol.

    The protocol is OIDC for all of these -- SAML customers reach us through the Keycloak broker,
    which speaks SAML outward and OIDC to us. The kind drives discovery defaults, the button
    label, and the two places the providers genuinely differ: Entra puts group object ids in the
    token, and Google does not put groups in the token at all.
    """

    ENTRA = "entra"
    GOOGLE = "google"
    GENERIC = "generic"
    KEYCLOAK_BROKER = "keycloak_broker"


class IdpState(StrEnum):
    DRAFT = "draft"
    TESTED = "tested"
    ACTIVE = "active"
    DISABLED = "disabled"


class TenantDomain(UUIDPrimaryKey, Timestamps, Base):
    """An email domain that resolves to a tenant, for email-first login discovery.

    Global rather than tenant-scoped by necessity: the lookup happens before any principal
    exists, which is the whole point of discovery. Verification matters -- an unverified domain
    would let anyone claim ``@microsoft.com`` and harvest which of its employees have accounts.
    """

    __tablename__ = "tenant_domains"
    __table_args__ = (
        UniqueConstraint("domain", name="uq_tenant_domains_domain"),
        Index("ix_tenant_domains_tenant", "tenant_id"),
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    domain: Mapped[str] = mapped_column(String(253), nullable=False)
    verified_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    verification_token: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: When true this domain alone decides the tenant; otherwise discovery may return several and
    #: the user picks. Needed for shared domains like gmail.com, which must never auto-resolve.
    is_primary: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class IdpConfig(UUIDPrimaryKey, Timestamps, Base):
    """One configured identity provider for one tenant.

    Global rather than TenantScoped for the same reason as ``tenant_domains``: discovery reads it
    before a principal exists. Every read is explicitly scoped by ``tenant_id`` instead, and the
    architecture test lists it accordingly.
    """

    __tablename__ = "idp_configs"
    __table_args__ = (
        # At most one active config per (tenant, kind). A second active Entra config would make
        # "which one did this token come from" undecidable at callback time.
        Index(
            "uq_idp_configs_active",
            "tenant_id",
            "kind",
            unique=True,
            postgresql_where=sa_text("state = 'active'"),
        ),
        Index("ix_idp_configs_tenant", "tenant_id"),
        CheckConstraint(
            "default_role IN ('DEV','TEST','PROD')",
            name="idp_default_role_not_admin",
        ),
        CheckConstraint("state IN ('draft','tested','active','disabled')", name="idp_state_valid"),
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    display_name: Mapped[str] = mapped_column(Text, nullable=False, default="Sign in")
    protocol: Mapped[str] = mapped_column(String(8), nullable=False, default=IdpProtocol.OIDC)
    kind: Mapped[str] = mapped_column(String(24), nullable=False, default=IdpKind.GENERIC)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default=IdpState.DRAFT)

    #: Discovery document URL. Everything else (authorize, token, jwks) is read from it rather
    #: than configured, because hand-entered endpoint URLs are the single most common cause of a
    #: broken SSO setup and the document is authoritative.
    issuer: Mapped[str] = mapped_column(Text, nullable=False)
    discovery_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    client_id: Mapped[str] = mapped_column(Text, nullable=False)
    #: Envelope-encrypted through the SecretStore. Never the plaintext, never reversible by a
    #: database dump alone.
    client_secret_enc: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    scopes: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False, default=list, server_default="{}")

    #: Which claim carries group membership. Entra: "groups". Google: absent -- groups need an
    #: Admin SDK call with domain-wide delegation, which is why that is its own flag.
    groups_claim: Mapped[str | None] = mapped_column(String(64), nullable=True, default="groups")
    fetch_groups_from_directory: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    #: IdP group value -> our role. Applied by ``provisioning.resolve_role`` in declared order,
    #: highest rank wins, and a mapping to ADMIN is audited and notified separately.
    role_mappings: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    default_role: Mapped[str] = mapped_column(String(16), nullable=False, default="PROD")

    #: JIT provisioning off means only users an administrator created may sign in. The right
    #: default for a tenant that wants the directory to authenticate but not to authorize.
    jit_provisioning: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    #: Re-apply role mappings on every login. On by default so a directory demotion takes effect;
    #: off for tenants who manage roles in the product and treat the IdP as authentication only.
    sync_role_on_login: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    tested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: The claim set from the most recent successful test sign-in, so the admin wizard can show a
    #: real claim-to-role preview instead of asking someone to guess what their IdP sends.
    last_test_claims: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
    activated_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class UserIdentity(UUIDPrimaryKey, Timestamps, Base):
    """A link between one of our users and one subject at one identity provider.

    Not TenantScoped: the callback resolves this row *before* a principal exists, from the state
    parameter and the token's subject. ``tenant_id`` is carried and asserted rather than filtered.
    """

    __tablename__ = "user_identities"
    __table_args__ = (
        # The identity matching rule, expressed as a constraint. Matching on email instead would
        # let a reassigned address inherit an account.
        UniqueConstraint("idp_config_id", "subject", name="uq_user_identities_idp_subject"),
        Index("ix_user_identities_user", "user_id"),
        Index("ix_user_identities_tenant", "tenant_id"),
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    idp_config_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("idp_configs.id", ondelete="CASCADE"), nullable=False
    )
    #: The IdP's immutable identifier: ``oid`` for Entra, ``sub`` elsewhere.
    subject: Mapped[str] = mapped_column(String(255), nullable=False)
    #: A convenience copy for display and support. Never a matching key.
    email_at_idp: Mapped[str | None] = mapped_column(String(320), nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_claims: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)


class Session(UUIDPrimaryKey, Timestamps, Base):
    """A browser session and its refresh-token family.

    The family is what makes reuse detection possible. Each refresh rotates the token and
    increments ``generation``; presenting a *previously rotated* token means either a race or a
    stolen token, and since the two are indistinguishable the whole family is revoked. A legitimate
    user re-authenticates; an attacker with a stolen token gets nothing, and the tenant's admins
    get an alert. Accepting the old token instead is how refresh rotation becomes theatre.

    Only the hash of the refresh token is stored, so a database leak does not yield live sessions.
    """

    __tablename__ = "sessions"
    __table_args__ = (
        Index("ix_sessions_user", "user_id"),
        Index("ix_sessions_family", "family_id"),
        Index("ix_sessions_tenant", "tenant_id"),
        Index("ix_sessions_expires", "expires_at"),
    )

    tenant_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    #: Shared by every rotation of one login. Revocation is per family, never per token.
    family_id: Mapped[uuid.UUID] = mapped_column(PgUUID(as_uuid=True), nullable=False)
    generation: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    refresh_token_hash: Mapped[bytes] = mapped_column(LargeBinary(32), nullable=False)

    issued_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    #: Set when the family was revoked because a rotated token was presented again. Distinct from
    #: an ordinary logout, because it is the signal worth alerting on.
    revoked_reason: Mapped[str | None] = mapped_column(String(32), nullable=True)

    #: Recorded for the session list in account settings and for incident response. Not used for
    #: validation: IP-pinning a session breaks every mobile user who changes network.
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    #: Which IdP authenticated this session, or null for password login. Lets a tenant revoke
    #: every session from a provider it has just disconnected.
    idp_config_id: Mapped[uuid.UUID | None] = mapped_column(
        PgUUID(as_uuid=True), ForeignKey("idp_configs.id", ondelete="SET NULL"), nullable=True
    )
    amr: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)


class LoginAttempt(UUIDPrimaryKey, Base):
    """Every authentication attempt, successful or not.

    Deliberately keyed on the *submitted* email rather than on a user id, because the attempts
    worth detecting are the ones against accounts that do not exist. No password, no token, no
    hash of either -- a failed-login table is a frequent exfiltration target and it must be worth
    nothing to an attacker who reads it.
    """

    __tablename__ = "login_attempts"
    __table_args__ = (
        Index("ix_login_attempts_email_time", "email", "created_at"),
        Index("ix_login_attempts_ip_time", "ip_address", "created_at"),
        Index("ix_login_attempts_tenant_time", "tenant_id", "created_at"),
    )

    tenant_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    user_id: Mapped[uuid.UUID | None] = mapped_column(PgUUID(as_uuid=True), nullable=True)
    email: Mapped[str] = mapped_column(String(320), nullable=False)
    method: Mapped[str] = mapped_column(String(16), nullable=False, default="password")
    succeeded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    #: A short machine-readable reason. Never surfaced to the client, which always sees the same
    #: generic message: distinguishing "no such user" from "wrong password" is an enumeration
    #: oracle.
    failure_reason: Mapped[str | None] = mapped_column(String(48), nullable=True)
    ip_address: Mapped[str | None] = mapped_column(String(45), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=func.now())
