"""What an operator may do in a tenant, and when.

One function decides it -- ``authorize`` -- and every operator-plane route goes through it. The
same reasoning as ``search/dsl.py``: a rule expressed once can be tested exhaustively, and a rule
expressed in fourteen route handlers has fourteen chances to be wrong.

The default is nothing. An operator with no grant can see that a tenant exists, because they had
to name it to get here, and that is all. Everything past that is a row with an expiry.

**Scopes are ordered, and the order is the point.** ``metadata`` answers most support questions
-- did the job fail, how many documents, which connector -- and needs only the tenant's standing
consent. ``content`` reads what the customer wrote and needs a named approver. ``impersonate``
acts as a person and is narrower still: one named user, not the tenant.

Making the cheap case genuinely cheap is what keeps the expensive one exceptional. If reading a
job's error required the same approval as reading a document, every ticket would request content
access on principle, and the distinction would stop meaning anything.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from app.models.platform import GrantScope, OperatorRole

#: Scope ordering. A content grant implies metadata; impersonation implies both.
#:
#: Ordered rather than a set of independent flags, because the alternative -- an operator holding
#: `content` but not `metadata` -- is not a state anyone means to create, and allowing it doubles
#: the cases every route has to think about.
_RANK: dict[str, int] = {
    GrantScope.METADATA: 10,
    GrantScope.CONTENT: 20,
    GrantScope.IMPERSONATE: 30,
}


class OperatorAction(StrEnum):
    """What the operator is trying to do. Named by intent, not by route."""

    LIST_TENANTS = "list_tenants"
    VIEW_TENANT_METADATA = "view_tenant_metadata"
    VIEW_JOBS = "view_jobs"
    RETRY_JOB = "retry_job"
    REBUILD_INDEX = "rebuild_index"
    VIEW_DOCUMENT_CONTENT = "view_document_content"
    RUN_SEARCH = "run_search"
    VIEW_QUERY_TEXT = "view_query_text"
    IMPERSONATE_USER = "impersonate_user"
    PROVISION_TENANT = "provision_tenant"
    GRANT_BREAK_GLASS = "grant_break_glass"
    OPEN_EMERGENCY_LOGIN = "open_emergency_login"


#: What each action requires. Absence from this map is a refusal, not a pass.
#:
#: Opt-out rather than opt-in, so a new action added without a decision fails closed. The
#: alternative -- defaulting to metadata -- means the one someone forgot is the one that leaks.
_REQUIREMENTS: dict[OperatorAction, tuple[OperatorRole, str | None]] = {
    # Needs a role, no grant: these are about the platform, not about a tenant's data.
    OperatorAction.LIST_TENANTS: (OperatorRole.SUPPORT, None),
    OperatorAction.PROVISION_TENANT: (OperatorRole.OWNER, None),
    # Metadata about one tenant.
    OperatorAction.VIEW_TENANT_METADATA: (OperatorRole.SUPPORT, GrantScope.METADATA),
    OperatorAction.VIEW_JOBS: (OperatorRole.SUPPORT, GrantScope.METADATA),
    OperatorAction.RETRY_JOB: (OperatorRole.ENGINEER, GrantScope.METADATA),
    OperatorAction.REBUILD_INDEX: (OperatorRole.ENGINEER, GrantScope.METADATA),
    # The customer's own words.
    OperatorAction.VIEW_DOCUMENT_CONTENT: (OperatorRole.SUPPORT, GrantScope.CONTENT),
    OperatorAction.RUN_SEARCH: (OperatorRole.SUPPORT, GrantScope.CONTENT),
    # A query is content: it is what someone asked, often about themselves or a colleague.
    OperatorAction.VIEW_QUERY_TEXT: (OperatorRole.SUPPORT, GrantScope.CONTENT),
    OperatorAction.IMPERSONATE_USER: (OperatorRole.SUPPORT, GrantScope.IMPERSONATE),
    # Break-glass and the emergency login window: OWNER only, and each its own audited event.
    OperatorAction.GRANT_BREAK_GLASS: (OperatorRole.OWNER, None),
    OperatorAction.OPEN_EMERGENCY_LOGIN: (OperatorRole.OWNER, None),
}

_ROLE_RANK: dict[str, int] = {OperatorRole.SUPPORT: 10, OperatorRole.ENGINEER: 20, OperatorRole.OWNER: 30}


@dataclass(frozen=True, slots=True)
class Grant:
    """The parts of a ``SupportGrant`` row this module needs."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    operator_id: uuid.UUID
    scope: str
    expires_at: datetime
    revoked_at: datetime | None = None
    impersonate_user_id: uuid.UUID | None = None
    is_break_glass: bool = False

    def active(self, *, now: datetime | None = None) -> bool:
        moment = now or datetime.now(UTC)
        return self.revoked_at is None and self.expires_at > moment

    def covers(self, scope: str) -> bool:
        return _RANK.get(self.scope, 0) >= _RANK.get(scope, 99)


@dataclass(frozen=True, slots=True)
class Operator:
    id: uuid.UUID
    email: str
    role: str
    is_active: bool = True
    mfa_enrolled: bool = False


@dataclass(frozen=True, slots=True)
class AuthorizationResult:
    allowed: bool
    reason: str = ""
    grant_id: uuid.UUID | None = None
    #: Whether this action reads customer content. Drives the audit row's ``touched_content``,
    #: which is the one field a tenant's security review actually filters on.
    touches_content: bool = False
    impersonated_user_id: uuid.UUID | None = None


def authorize(
    action: OperatorAction,
    *,
    operator: Operator,
    tenant_id: uuid.UUID | None = None,
    grants: list[Grant] | None = None,
    target_user_id: uuid.UUID | None = None,
    now: datetime | None = None,
) -> AuthorizationResult:
    """The single decision. Everything is refused unless this says otherwise."""
    if not operator.is_active:
        return AuthorizationResult(False, "this operator account is deactivated")

    if not operator.mfa_enrolled:
        # Not a soft warning. This account can, with approval, read customer data, which makes it
        # the highest-value phishing target in the product.
        return AuthorizationResult(False, "operator accounts require multi-factor authentication")

    requirement = _REQUIREMENTS.get(action)
    if requirement is None:
        # An action nobody classified. Failing closed means a new endpoint is unreachable until
        # someone decides what it needs, rather than quietly inheriting the weakest rule.
        return AuthorizationResult(False, f"the action {action} has no declared requirement")

    required_role, required_scope = requirement

    if _ROLE_RANK.get(operator.role, 0) < _ROLE_RANK[required_role]:
        return AuthorizationResult(False, f"this action requires the {required_role} role")

    if required_scope is None:
        return AuthorizationResult(True, "permitted by role", touches_content=False)

    if tenant_id is None:
        return AuthorizationResult(False, "this action is about a specific tenant and none was named")

    grant = _find(grants or [], operator=operator, tenant_id=tenant_id, scope=required_scope, now=now)
    if grant is None:
        return AuthorizationResult(
            False,
            f"no active {required_scope} grant from this tenant. "
            "Ask the customer to approve access, or use break-glass if they cannot.",
        )

    if action is OperatorAction.IMPERSONATE_USER:
        # An impersonation grant names one person. "Reproduce what Alice sees" must not become
        # "act as anyone in the tenant" -- which is what an unscoped impersonation grant is.
        if grant.impersonate_user_id is None:
            return AuthorizationResult(False, "this grant does not name a user to impersonate")
        if target_user_id is not None and grant.impersonate_user_id != target_user_id:
            return AuthorizationResult(False, "this grant is for a different user")
        return AuthorizationResult(
            True,
            "permitted by impersonation grant",
            grant_id=grant.id,
            touches_content=True,
            impersonated_user_id=grant.impersonate_user_id,
        )

    return AuthorizationResult(
        True,
        "permitted by grant",
        grant_id=grant.id,
        touches_content=required_scope in {GrantScope.CONTENT, GrantScope.IMPERSONATE},
    )


def _find(
    grants: list[Grant],
    *,
    operator: Operator,
    tenant_id: uuid.UUID,
    scope: str,
    now: datetime | None,
) -> Grant | None:
    """The narrowest active grant that covers what is being asked for.

    Narrowest rather than first: an operator may hold both a metadata grant and an impersonation
    grant for the same tenant, and using the broader one to read a job's status would attribute
    the action to the wrong grant in the audit log -- which is the record a customer reads to
    answer "what did the vendor actually do".
    """
    candidates = [
        grant
        for grant in grants
        if grant.operator_id == operator.id
        and grant.tenant_id == tenant_id
        and grant.active(now=now)
        and grant.covers(scope)
    ]
    if not candidates:
        return None
    return min(candidates, key=lambda grant: _RANK.get(grant.scope, 99))


def requires_content_grant(action: OperatorAction) -> bool:
    """Whether this action reads the customer's own words. Used to label the UI, honestly."""
    requirement = _REQUIREMENTS.get(action)
    return requirement is not None and requirement[1] in {GrantScope.CONTENT, GrantScope.IMPERSONATE}


def declared_actions() -> frozenset[OperatorAction]:
    """Every action with a declared requirement.

    A test asserts this equals the enum, so an action added without a decision fails the build
    rather than being unreachable at runtime and discovered in production.
    """
    return frozenset(_REQUIREMENTS)


def describe(action: OperatorAction) -> str:
    """What the operator UI shows before an action, so the cost is visible up front."""
    requirement = _REQUIREMENTS.get(action)
    if requirement is None:
        return "This action has not been classified and cannot be performed."
    role, scope = requirement
    if scope is None:
        return f"Requires the {role} role."
    if scope == GrantScope.METADATA:
        return f"Requires the {role} role and an active support grant. Reads metadata only."
    if scope == GrantScope.IMPERSONATE:
        return (
            f"Requires the {role} role and an impersonation grant naming one user. "
            "Every request is recorded as elevated and is visible to the customer."
        )
    return (
        f"Requires the {role} role and a content grant approved by a tenant administrator. "
        "This reads the customer's own documents and is recorded as such."
    )
