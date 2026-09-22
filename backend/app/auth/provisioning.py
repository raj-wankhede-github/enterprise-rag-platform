"""Turning an IdP's claims into one of our users, and into one of our four roles.

This is the highest-risk code in the authentication path, and the risk is not subtle: a wrong
group mapping grants ADMIN to an entire directory, silently, at the next login. The defences are
layered so that no single mistake is sufficient.

1. **Roles are resolved from an explicit allow-map**, never from a claim's value directly. A
   provider that starts sending ``"role": "ADMIN"`` changes nothing here.
2. **The highest matching rank wins**, and that is deliberate rather than incidental: a user in
   both ``engineering`` and ``eng-leads`` gets the higher of the two, which is what an
   administrator who wrote both mappings meant. Taking the first match instead would make
   behaviour depend on dictionary ordering.
3. **A mapping to ADMIN is reported separately** so the caller can audit and notify. Elevation to
   ADMIN via a directory group is legitimate and also exactly what an attacker who can edit
   groups would do; it must never be silent.
4. **JIT provisioning can never create an ADMIN.** A new user is capped at the config's
   ``default_role``, which a database CHECK already forbids from being ADMIN. Group mappings
   apply to *existing* users; the first ADMIN of a tenant is created deliberately, by a human.
5. **``role_locked`` wins over everything.** An administrator who pins a user's role in the
   product is overruling the directory on purpose, and a nightly group sync must not undo it.

Identity matching is on the IdP's immutable subject and never on email. See ``models/identity``.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from app.security.capabilities import RANK, Role


class ProvisioningOutcome(StrEnum):
    LINKED_EXISTING = "linked_existing"
    """The subject was already known. The ordinary case."""

    CREATED = "created"
    """JIT created a new user at the configured default role."""

    LINKED_BY_EMAIL = "linked_by_email"
    """A user existed with this email but no identity for this provider.

    The one place email is consulted, and only to *link* an account an administrator already
    created -- never to match an identity on subsequent logins. Without it, a tenant that
    pre-provisions its users would have every one of them fail their first SSO login.
    """

    REJECTED_NO_JIT = "rejected_no_jit"
    """Unknown subject and JIT is off, so the directory authenticates but does not enroll."""

    REJECTED_INACTIVE = "rejected_inactive"
    """The user exists and is deactivated. Deactivation must survive a fresh SSO login."""


@dataclass(frozen=True, slots=True)
class IdentityClaims:
    """The normalised subset of an ID token we act on."""

    subject: str
    email: str | None = None
    name: str = ""
    groups: tuple[str, ...] = ()
    #: Verified-email claim. Some providers send unverified addresses, and linking an existing
    #: account on an unverified address is an account-takeover primitive.
    email_verified: bool = False
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ExistingUser:
    id: uuid.UUID
    email: str
    role: str
    is_active: bool
    role_locked: bool = False
    group_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ProvisioningDecision:
    outcome: ProvisioningOutcome
    role: Role | None = None
    user_id: uuid.UUID | None = None
    groups: tuple[str, ...] = ()
    #: True when this login moves the user to ADMIN through a group mapping. The caller audits
    #: and notifies every existing tenant ADMIN; a silent promotion is the failure mode.
    elevates_to_admin: bool = False
    #: True when the role changed at all, so a demotion is auditable too.
    role_changed: bool = False
    reason: str = ""

    @property
    def permitted(self) -> bool:
        return self.outcome in {
            ProvisioningOutcome.LINKED_EXISTING,
            ProvisioningOutcome.CREATED,
            ProvisioningOutcome.LINKED_BY_EMAIL,
        }


@dataclass(frozen=True, slots=True)
class ProvisioningPolicy:
    """The parts of an ``IdpConfig`` this module needs."""

    default_role: str = Role.PROD
    role_mappings: dict[str, str] = field(default_factory=dict)
    jit_provisioning: bool = True
    sync_role_on_login: bool = True


def extract_claims(token: dict[str, Any], *, groups_claim: str | None = "groups") -> IdentityClaims:
    """Normalise a provider's ID token.

    ``oid`` before ``sub`` because Entra sends both and they mean different things: ``sub`` is
    pairwise per-application, so it changes if the app registration is recreated, while ``oid``
    is the directory object id and is stable for the life of the account. Matching on ``sub``
    there means an app-registration change orphans every identity in the tenant.
    """
    subject = str(token.get("oid") or token.get("sub") or "").strip()
    if not subject:
        raise ValueError("the ID token carries neither 'oid' nor 'sub'; it cannot identify anyone")

    raw_groups = token.get(groups_claim) if groups_claim else None
    groups: tuple[str, ...] = ()
    if isinstance(raw_groups, list):
        groups = tuple(str(value) for value in raw_groups if str(value).strip())
    elif isinstance(raw_groups, str) and raw_groups.strip():
        # Some providers send a single group as a bare string, and a few send them comma- or
        # space-separated. Treating a string as an iterable of characters is the classic bug.
        groups = tuple(part for part in re_split(raw_groups) if part)

    email = token.get("email") or token.get("preferred_username") or token.get("upn")
    return IdentityClaims(
        subject=subject,
        email=str(email).strip().casefold() if email else None,
        name=str(token.get("name") or "").strip(),
        groups=groups,
        email_verified=bool(token.get("email_verified", False)),
        raw=dict(token),
    )


def re_split(value: str) -> list[str]:
    import re

    return [part.strip() for part in re.split(r"[,\s]+", value) if part.strip()]


def resolve_role(
    claims: IdentityClaims, policy: ProvisioningPolicy, *, current: str | None = None
) -> tuple[Role, bool]:
    """Which role this login should carry, and whether it is an elevation to ADMIN.

    Returns the *highest-ranked* mapped role among the user's groups, falling back to the current
    role for an existing user and to ``default_role`` for a new one. Falling back to the default
    for an existing user would silently demote every administrator the moment their groups stop
    matching -- which happens whenever a claim name changes.
    """
    matched: list[Role] = []
    for group in claims.groups:
        mapped = policy.role_mappings.get(group)
        if mapped and mapped in set(Role):
            matched.append(Role(mapped))

    if matched:
        role = max(matched, key=lambda item: RANK[item])
    elif current and current in set(Role):
        role = Role(current)
    else:
        role = Role(policy.default_role)

    elevates = role is Role.ADMIN and (current is None or current != Role.ADMIN)
    return role, elevates


def plan_login(
    claims: IdentityClaims,
    policy: ProvisioningPolicy,
    *,
    linked: ExistingUser | None,
    by_email: ExistingUser | None = None,
) -> ProvisioningDecision:
    """Decide what this login does, given what already exists.

    ``linked`` is the user found by ``(idp_config_id, subject)`` -- the authoritative lookup.
    ``by_email`` is a user with a matching address and no identity for this provider, which is
    the pre-provisioned case and the only one where email is consulted at all.
    """
    if linked is not None:
        return _existing(claims, policy, linked, ProvisioningOutcome.LINKED_EXISTING)

    if by_email is not None:
        if not claims.email_verified:
            # Linking an administrator-created account on an unverified address hands that
            # account to anyone who can make their IdP assert the string.
            return ProvisioningDecision(
                outcome=ProvisioningOutcome.REJECTED_NO_JIT,
                reason="the provider did not assert that this address is verified",
            )
        return _existing(claims, policy, by_email, ProvisioningOutcome.LINKED_BY_EMAIL)

    if not policy.jit_provisioning:
        return ProvisioningDecision(
            outcome=ProvisioningOutcome.REJECTED_NO_JIT,
            reason="automatic provisioning is disabled for this identity provider",
        )

    # A new user never gets a group-mapped role, only the default -- which the database forbids
    # from being ADMIN. A group mapping is an administrator's statement about *their* people; it
    # must not be a path by which an unknown subject arrives as an administrator.
    return ProvisioningDecision(
        outcome=ProvisioningOutcome.CREATED,
        role=Role(policy.default_role),
        groups=claims.groups,
        reason="provisioned at the configured default role",
    )


def _existing(
    claims: IdentityClaims,
    policy: ProvisioningPolicy,
    user: ExistingUser,
    outcome: ProvisioningOutcome,
) -> ProvisioningDecision:
    if not user.is_active:
        # Must outrank everything below. A deactivated user who can still authenticate at the
        # directory would otherwise walk straight back in.
        return ProvisioningDecision(
            outcome=ProvisioningOutcome.REJECTED_INACTIVE,
            user_id=user.id,
            reason="this account has been deactivated",
        )

    if user.role_locked or not policy.sync_role_on_login:
        return ProvisioningDecision(
            outcome=outcome,
            role=Role(user.role),
            user_id=user.id,
            groups=claims.groups,
            reason="role pinned in the product" if user.role_locked else "role sync disabled",
        )

    role, elevates = resolve_role(claims, policy, current=user.role)
    return ProvisioningDecision(
        outcome=outcome,
        role=role,
        user_id=user.id,
        groups=claims.groups,
        elevates_to_admin=elevates,
        role_changed=role != user.role,
        reason="role resolved from directory groups",
    )


def preview_mapping(
    policy: ProvisioningPolicy, sample_claims: dict[str, Any], *, groups_claim: str | None
) -> dict[str, Any]:
    """What the admin wizard shows before anyone activates a config.

    The wizard is draft -> test -> activate, and this is the "test" half made legible: a real
    sign-in produces real claims, and this states which groups matched, which were ignored, and
    what role the person would have received. An administrator who can see that ``All Company``
    maps to ADMIN before activating will not activate it.
    """
    claims = extract_claims(sample_claims, groups_claim=groups_claim)
    role, elevates = resolve_role(claims, policy)
    matched = {group: policy.role_mappings[group] for group in claims.groups if group in policy.role_mappings}
    return {
        "subject": claims.subject,
        "email": claims.email,
        "email_verified": claims.email_verified,
        "groups_seen": list(claims.groups),
        "groups_matched": matched,
        "groups_ignored": [group for group in claims.groups if group not in matched],
        "resolved_role": str(role),
        "would_elevate_to_admin": elevates,
        "default_role_for_new_users": str(policy.default_role),
        "warnings": _warnings(policy, claims, elevates),
    }


def _warnings(policy: ProvisioningPolicy, claims: IdentityClaims, elevates: bool) -> list[str]:
    warnings: list[str] = []
    if elevates:
        warnings.append(
            "This configuration grants ADMIN through a directory group. Everyone in that group "
            "will hold full administrative rights, including user management and SSO settings."
        )
    if not claims.groups:
        warnings.append(
            "The provider sent no groups. Every user will receive the default role. For Google "
            "Workspace this is expected: groups are not in the ID token and require a Directory "
            "API call with domain-wide delegation."
        )
    if not claims.email_verified:
        warnings.append(
            "The provider did not mark this address as verified, so existing accounts cannot be "
            "linked by email. New users will still be provisioned."
        )
    unmatched = [group for group in policy.role_mappings if group not in claims.groups]
    if unmatched and claims.groups:
        warnings.append(
            f"{len(unmatched)} configured mapping(s) did not appear in this sign-in: "
            f"{', '.join(sorted(unmatched)[:5])}. Check the group values are object ids where "
            "the provider sends ids rather than names."
        )
    return warnings
