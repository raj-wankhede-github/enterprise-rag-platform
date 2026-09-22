"""Email-first login discovery: which tenant, and which methods it allows.

The login page asks for an email, posts it here, and renders what comes back. That shape exists
because a multi-tenant product cannot show one password field and three SSO buttons to everyone
-- most of which would fail -- and because the alternative, asking the user which company they
work for, is a question they should never have to answer.

**The server is the authority, not the response.** Hiding the password field is a convenience;
the login endpoint independently refuses password authentication when the tenant has disabled it.
A response that merely omits the field is a UI hint, and treating it as the control is how an
attacker who posts directly to the endpoint bypasses it.

**Discovery must not be a user-enumeration oracle.** The response says what a *tenant* allows,
never whether a particular address has an account. An unknown domain gets the same shape as a
known one with password login enabled, so the two are indistinguishable from outside -- and a
timing floor keeps the database lookup from distinguishing them either.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime

from app.models.identity import IdpKind, IdpState

#: Domains where an email address says nothing about which organisation someone belongs to.
#: A tenant may still *claim* one, but it never auto-resolves -- otherwise the first customer to
#: verify gmail.com would capture every consumer address in the product.
PUBLIC_DOMAINS: frozenset[str] = frozenset(
    {
        "gmail.com",
        "googlemail.com",
        "outlook.com",
        "hotmail.com",
        "live.com",
        "yahoo.com",
        "icloud.com",
        "me.com",
        "proton.me",
        "protonmail.com",
        "gmx.de",
        "web.de",
        "aol.com",
        "mail.com",
        "yandex.ru",
        "qq.com",
    }
)

_EMAIL = re.compile(r"^[^@\s]+@([^@\s]+\.[^@\s]+)$")


@dataclass(frozen=True, slots=True)
class LoginMethod:
    """One button on the login page."""

    kind: str
    display_name: str
    #: Where the browser is sent to start the flow. Opaque to the client.
    start_url: str
    idp_config_id: uuid.UUID | None = None


@dataclass(frozen=True, slots=True)
class DiscoveryResult:
    """What the login page renders. Deliberately says nothing about the *user*."""

    tenant_id: uuid.UUID | None = None
    tenant_slug: str | None = None
    tenant_name: str | None = None
    password_login: bool = True
    methods: tuple[LoginMethod, ...] = ()
    #: Several tenants claim this domain, so the user must choose. Carries slugs and names only.
    ambiguous: tuple[tuple[str, str], ...] = ()
    #: Password login is on only because the break-glass window is open. The UI says so, loudly,
    #: because a tenant that thinks SSO is enforced needs to know when it temporarily is not.
    emergency_access: bool = False

    @property
    def requires_tenant_choice(self) -> bool:
        return bool(self.ambiguous)


def normalize_email(value: str) -> str:
    """Trim and casefold. Not a validator -- ``domain_of`` decides validity."""
    return value.strip().casefold()


def domain_of(email: str) -> str | None:
    match = _EMAIL.match(normalize_email(email))
    return match.group(1) if match else None


def is_public_domain(domain: str) -> bool:
    return domain in PUBLIC_DOMAINS


@dataclass(frozen=True, slots=True)
class TenantLoginPolicy:
    """The tenant-side inputs to a discovery decision, read from Postgres."""

    tenant_id: uuid.UUID
    slug: str
    name: str
    password_login_enabled: bool
    emergency_password_login_until: datetime | None = None
    idps: tuple[IdpSummary, ...] = ()


@dataclass(frozen=True, slots=True)
class IdpSummary:
    id: uuid.UUID
    kind: str
    display_name: str
    state: str


def password_login_allowed(policy: TenantLoginPolicy, *, now: datetime | None = None) -> tuple[bool, bool]:
    """Whether password login is permitted, and whether that is only the break-glass.

    The break-glass exists for one specific disaster: an administrator activates a broken IdP
    config on a tenant with password login disabled, and locks out everyone including themselves.
    Recovery through the platform operator plane would mean the vendor holds a key to every
    customer's front door, so instead a platform operator opens a *time-boxed window* on that one
    tenant. It expires by itself, which is the property that matters -- a flag someone must
    remember to turn off is a flag that stays on.
    """
    if policy.password_login_enabled:
        return True, False
    deadline = policy.emergency_password_login_until
    if deadline and deadline > (now or datetime.now(UTC)):
        return True, True
    return False, False


def build_result(
    policy: TenantLoginPolicy | None,
    *,
    start_url: str = "/api/auth/oidc/{idp_config_id}/start",
    now: datetime | None = None,
) -> DiscoveryResult:
    """Turn a tenant's policy into what the login page should render.

    ``None`` means the domain resolved to no tenant. The response is deliberately the *default*
    shape -- a password field, no SSO buttons -- rather than an error, because an error would
    tell an attacker which domains are customers. Posting credentials into it fails exactly as an
    unknown password does.
    """
    if policy is None:
        return DiscoveryResult(password_login=True)

    allowed, emergency = password_login_allowed(policy, now=now)
    methods = tuple(
        LoginMethod(
            kind=idp.kind,
            display_name=idp.display_name or _default_label(idp.kind),
            start_url=start_url.format(idp_config_id=idp.id),
            idp_config_id=idp.id,
        )
        # Only active configs. A draft or tested config is visible to its administrator in the
        # wizard and to nobody else -- that separation is what makes "test before activate" real.
        for idp in policy.idps
        if idp.state == IdpState.ACTIVE
    )
    return DiscoveryResult(
        tenant_id=policy.tenant_id,
        tenant_slug=policy.slug,
        tenant_name=policy.name,
        password_login=allowed,
        methods=methods,
        emergency_access=emergency,
    )


def ambiguous_result(candidates: list[TenantLoginPolicy]) -> DiscoveryResult:
    """Several tenants claim this domain, so the user picks.

    Real in consultancies and holding groups, where one address legitimately belongs to several
    customer tenants. Only slugs and names are returned -- enough to choose from, nothing about
    who has an account where.
    """
    return DiscoveryResult(
        ambiguous=tuple((policy.slug, policy.name) for policy in sorted(candidates, key=lambda p: p.name))
    )


def _default_label(kind: str) -> str:
    return {
        IdpKind.ENTRA: "Continue with Microsoft",
        IdpKind.GOOGLE: "Continue with Google",
        IdpKind.KEYCLOAK_BROKER: "Continue with your organisation",
    }.get(IdpKind(kind) if kind in set(IdpKind) else IdpKind.GENERIC, "Continue with SSO")


@dataclass(slots=True)
class DiscoveryDefaults:
    """Config the route passes in, kept out of the pure functions above."""

    #: A floor on how long discovery takes, so a hit and a miss are indistinguishable by timing.
    #: A database lookup that finds nothing returns measurably faster than one that joins to a
    #: tenant's IdP list, and that difference alone enumerates customers.
    min_response_ms: float = 120.0
    allow_public_domain_tenants: bool = False
    methods: list[LoginMethod] = field(default_factory=list)
