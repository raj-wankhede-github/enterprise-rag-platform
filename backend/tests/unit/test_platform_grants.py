"""What an operator may do in a tenant, and when.

The acceptance criterion for this step is one sentence: *an operator cannot read tenant content
without a grant.* Most of this file is that sentence, tested from every direction a real
implementation tends to leak from — a role that seems senior enough, a grant for the wrong
tenant, an expired one, one that was revoked, and the OWNER who assumes seniority is sufficient.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.models.platform import GrantScope, OperatorRole
from app.platform.grants import (
    Grant,
    Operator,
    OperatorAction,
    authorize,
    declared_actions,
    describe,
    requires_content_grant,
)

OPERATOR_ID = uuid.UUID("11110000-0000-0000-0000-000000000001")
TENANT = uuid.UUID("22220000-0000-0000-0000-000000000002")
OTHER_TENANT = uuid.UUID("33330000-0000-0000-0000-000000000003")
USER = uuid.UUID("44440000-0000-0000-0000-000000000004")


def an_operator(role: str = OperatorRole.SUPPORT, **kwargs: object) -> Operator:
    base: dict[str, object] = {
        "id": OPERATOR_ID,
        "email": "support@vendor.example",
        "role": role,
        "is_active": True,
        "mfa_enrolled": True,
    }
    base.update(kwargs)
    return Operator(**base)  # type: ignore[arg-type]


def a_grant(scope: str = GrantScope.METADATA, **kwargs: object) -> Grant:
    base: dict[str, object] = {
        "id": uuid.uuid4(),
        "tenant_id": TENANT,
        "operator_id": OPERATOR_ID,
        "scope": scope,
        "expires_at": datetime.now(UTC) + timedelta(hours=4),
    }
    base.update(kwargs)
    return Grant(**base)  # type: ignore[arg-type]


CONTENT_ACTIONS = [
    OperatorAction.VIEW_DOCUMENT_CONTENT,
    OperatorAction.RUN_SEARCH,
    OperatorAction.VIEW_QUERY_TEXT,
]


# ------------------------------------------------------------------------------------------
# The acceptance criterion
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("action", CONTENT_ACTIONS)
@pytest.mark.parametrize("role", list(OperatorRole))
def test_no_operator_role_reads_tenant_content_without_a_grant(action: OperatorAction, role: str) -> None:
    """The sentence this whole plane exists to make true, tested across every role.

    Seniority is not consent. An OWNER can provision tenants and open break-glass windows, and
    still cannot read a customer's documents because they felt like it.
    """
    result = authorize(action, operator=an_operator(role), tenant_id=TENANT, grants=[])
    assert not result.allowed
    assert "grant" in result.reason


@pytest.mark.parametrize("action", CONTENT_ACTIONS)
def test_a_metadata_grant_is_not_enough_for_content(action: OperatorAction) -> None:
    """Most support work needs metadata. Letting it escalate silently is how the distinction
    stops meaning anything."""
    result = authorize(action, operator=an_operator(), tenant_id=TENANT, grants=[a_grant(GrantScope.METADATA)])
    assert not result.allowed


@pytest.mark.parametrize("action", CONTENT_ACTIONS)
def test_a_content_grant_permits_content(action: OperatorAction) -> None:
    result = authorize(action, operator=an_operator(), tenant_id=TENANT, grants=[a_grant(GrantScope.CONTENT)])
    assert result.allowed
    assert result.touches_content, "the audit row must record that customer content was read"


def test_a_grant_for_another_tenant_does_not_transfer() -> None:
    """A support engineer with a grant from customer A must not thereby read customer B."""
    grant = a_grant(GrantScope.CONTENT, tenant_id=OTHER_TENANT)
    result = authorize(
        OperatorAction.VIEW_DOCUMENT_CONTENT, operator=an_operator(), tenant_id=TENANT, grants=[grant]
    )
    assert not result.allowed


def test_another_operators_grant_does_not_transfer() -> None:
    grant = a_grant(GrantScope.CONTENT, operator_id=uuid.uuid4())
    result = authorize(
        OperatorAction.VIEW_DOCUMENT_CONTENT, operator=an_operator(), tenant_id=TENANT, grants=[grant]
    )
    assert not result.allowed


def test_an_expired_grant_stops_working_by_itself() -> None:
    """The property that matters. A grant someone must remember to revoke is one that stays open."""
    expired = a_grant(GrantScope.CONTENT, expires_at=datetime.now(UTC) - timedelta(minutes=1))
    result = authorize(
        OperatorAction.VIEW_DOCUMENT_CONTENT, operator=an_operator(), tenant_id=TENANT, grants=[expired]
    )
    assert not result.allowed


def test_a_revoked_grant_stops_working_immediately() -> None:
    """A customer who revokes access mid-incident must not have to wait for an expiry."""
    revoked = a_grant(GrantScope.CONTENT, revoked_at=datetime.now(UTC))
    result = authorize(
        OperatorAction.VIEW_DOCUMENT_CONTENT, operator=an_operator(), tenant_id=TENANT, grants=[revoked]
    )
    assert not result.allowed


def test_a_content_action_without_a_named_tenant_is_refused() -> None:
    """Otherwise an action with a missing tenant id would fall through the grant check entirely."""
    result = authorize(OperatorAction.VIEW_DOCUMENT_CONTENT, operator=an_operator(), tenant_id=None, grants=[])
    assert not result.allowed


# ------------------------------------------------------------------------------------------
# Metadata, kept genuinely cheap
# ------------------------------------------------------------------------------------------


def test_listing_tenants_needs_no_grant() -> None:
    """An operator had to name the tenant to get here. Requiring a grant to learn it exists
    would make routing a ticket impossible."""
    assert authorize(OperatorAction.LIST_TENANTS, operator=an_operator(), grants=[]).allowed


def test_viewing_job_states_needs_only_a_metadata_grant() -> None:
    """The commonest support question. Making it cheap is what stops every ticket asking for
    content access on principle."""
    result = authorize(OperatorAction.VIEW_JOBS, operator=an_operator(), tenant_id=TENANT, grants=[a_grant()])
    assert result.allowed
    assert not result.touches_content


def test_a_content_grant_also_covers_metadata() -> None:
    """Scopes are ordered rather than independent flags: an operator holding `content` but not
    `metadata` is not a state anyone means to create."""
    result = authorize(
        OperatorAction.VIEW_JOBS, operator=an_operator(), tenant_id=TENANT, grants=[a_grant(GrantScope.CONTENT)]
    )
    assert result.allowed


def test_a_query_string_counts_as_content() -> None:
    """What someone asked is often about themselves or a colleague. It is not metadata."""
    assert requires_content_grant(OperatorAction.VIEW_QUERY_TEXT)


def test_job_metadata_does_not_count_as_content() -> None:
    assert not requires_content_grant(OperatorAction.VIEW_JOBS)


# ------------------------------------------------------------------------------------------
# Roles
# ------------------------------------------------------------------------------------------


def test_support_cannot_retry_a_job() -> None:
    """Reading is support; changing state is engineering."""
    result = authorize(
        OperatorAction.RETRY_JOB, operator=an_operator(OperatorRole.SUPPORT), tenant_id=TENANT, grants=[a_grant()]
    )
    assert not result.allowed
    assert "ENGINEER" in result.reason


def test_an_engineer_can_retry_a_job_with_a_grant() -> None:
    result = authorize(
        OperatorAction.RETRY_JOB, operator=an_operator(OperatorRole.ENGINEER), tenant_id=TENANT, grants=[a_grant()]
    )
    assert result.allowed


def test_an_engineer_still_needs_a_grant_to_retry() -> None:
    """A role is permission to act, not permission to enter."""
    result = authorize(
        OperatorAction.RETRY_JOB, operator=an_operator(OperatorRole.ENGINEER), tenant_id=TENANT, grants=[]
    )
    assert not result.allowed


@pytest.mark.parametrize("role", [OperatorRole.SUPPORT, OperatorRole.ENGINEER])
def test_only_an_owner_can_provision_a_tenant(role: str) -> None:
    assert not authorize(OperatorAction.PROVISION_TENANT, operator=an_operator(role), grants=[]).allowed


@pytest.mark.parametrize("role", [OperatorRole.SUPPORT, OperatorRole.ENGINEER])
def test_only_an_owner_can_use_break_glass(role: str) -> None:
    assert not authorize(OperatorAction.GRANT_BREAK_GLASS, operator=an_operator(role), grants=[]).allowed


@pytest.mark.parametrize("role", [OperatorRole.SUPPORT, OperatorRole.ENGINEER])
def test_only_an_owner_can_open_the_emergency_login_window(role: str) -> None:
    """The break-glass for an administrator who activated a broken IdP and locked everyone out.
    It re-enables password login on a customer's tenant, which is not a support action."""
    assert not authorize(OperatorAction.OPEN_EMERGENCY_LOGIN, operator=an_operator(role), grants=[]).allowed


def test_an_owner_can_do_the_owner_only_things() -> None:
    owner = an_operator(OperatorRole.OWNER)
    for action in (
        OperatorAction.PROVISION_TENANT,
        OperatorAction.GRANT_BREAK_GLASS,
        OperatorAction.OPEN_EMERGENCY_LOGIN,
    ):
        assert authorize(action, operator=owner, grants=[]).allowed


def test_a_customer_cannot_consent_their_way_past_an_owner_only_action() -> None:
    """Break-glass is not about one customer's data, so one customer's approval cannot unlock it."""
    result = authorize(
        OperatorAction.GRANT_BREAK_GLASS,
        operator=an_operator(OperatorRole.SUPPORT),
        tenant_id=TENANT,
        grants=[a_grant(GrantScope.IMPERSONATE, impersonate_user_id=USER)],
    )
    assert not result.allowed


# ------------------------------------------------------------------------------------------
# Impersonation
# ------------------------------------------------------------------------------------------


def test_impersonation_requires_a_grant_naming_one_user() -> None:
    """"Reproduce what Alice sees" must not become "act as anyone in the tenant"."""
    unscoped = a_grant(GrantScope.IMPERSONATE, impersonate_user_id=None)
    result = authorize(
        OperatorAction.IMPERSONATE_USER, operator=an_operator(), tenant_id=TENANT, grants=[unscoped]
    )
    assert not result.allowed


def test_impersonating_the_named_user_is_permitted() -> None:
    grant = a_grant(GrantScope.IMPERSONATE, impersonate_user_id=USER)
    result = authorize(
        OperatorAction.IMPERSONATE_USER,
        operator=an_operator(),
        tenant_id=TENANT,
        grants=[grant],
        target_user_id=USER,
    )
    assert result.allowed
    assert result.impersonated_user_id == USER
    assert result.touches_content


def test_impersonating_a_different_user_is_refused() -> None:
    grant = a_grant(GrantScope.IMPERSONATE, impersonate_user_id=USER)
    result = authorize(
        OperatorAction.IMPERSONATE_USER,
        operator=an_operator(),
        tenant_id=TENANT,
        grants=[grant],
        target_user_id=uuid.uuid4(),
    )
    assert not result.allowed


def test_a_content_grant_does_not_permit_impersonation() -> None:
    """Reading a document and acting as a person are different things, and the second is worse:
    an impersonated action is attributable to the customer's own employee."""
    result = authorize(
        OperatorAction.IMPERSONATE_USER,
        operator=an_operator(),
        tenant_id=TENANT,
        grants=[a_grant(GrantScope.CONTENT)],
    )
    assert not result.allowed


# ------------------------------------------------------------------------------------------
# The account itself
# ------------------------------------------------------------------------------------------


def test_a_deactivated_operator_can_do_nothing_even_with_a_live_grant() -> None:
    """Deactivation must outrank every grant: a departing employee's grants do not depart with
    them, so this is the control that actually ends their access."""
    result = authorize(
        OperatorAction.VIEW_DOCUMENT_CONTENT,
        operator=an_operator(OperatorRole.OWNER, is_active=False),
        tenant_id=TENANT,
        grants=[a_grant(GrantScope.CONTENT)],
    )
    assert not result.allowed
    assert "deactivated" in result.reason


def test_an_operator_without_mfa_can_do_nothing() -> None:
    """This account can, with approval, read customer data. That makes it the highest-value
    phishing target in the product, so MFA is a gate rather than a recommendation."""
    result = authorize(
        OperatorAction.LIST_TENANTS, operator=an_operator(mfa_enrolled=False), grants=[]
    )
    assert not result.allowed
    assert "multi-factor" in result.reason


# ------------------------------------------------------------------------------------------
# Fail-closed
# ------------------------------------------------------------------------------------------


def test_every_action_has_a_declared_requirement() -> None:
    """An action added without a decision must fail the build, not be discovered in production.

    Opt-out rather than opt-in: defaulting a new action to `metadata` means the one someone
    forgot to classify is the one that leaks.
    """
    undeclared = set(OperatorAction) - declared_actions()
    assert not undeclared, f"these actions have no declared requirement: {sorted(undeclared)}"


def test_an_unclassified_action_is_refused_rather_than_defaulted() -> None:
    from app.platform import grants as module

    saved = module._REQUIREMENTS.pop(OperatorAction.VIEW_JOBS)
    try:
        result = authorize(
            OperatorAction.VIEW_JOBS,
            operator=an_operator(OperatorRole.OWNER),
            tenant_id=TENANT,
            grants=[a_grant(GrantScope.IMPERSONATE, impersonate_user_id=USER)],
        )
        assert not result.allowed
        assert "no declared requirement" in result.reason
    finally:
        module._REQUIREMENTS[OperatorAction.VIEW_JOBS] = saved


# ------------------------------------------------------------------------------------------
# Attribution
# ------------------------------------------------------------------------------------------


def test_the_narrowest_grant_is_the_one_recorded() -> None:
    """An operator may hold both a metadata and an impersonation grant. Attributing a job-status
    read to the impersonation grant would misdescribe it in the record the customer reads to
    answer "what did the vendor actually do"."""
    metadata = a_grant(GrantScope.METADATA)
    impersonate = a_grant(GrantScope.IMPERSONATE, impersonate_user_id=USER)

    result = authorize(
        OperatorAction.VIEW_JOBS, operator=an_operator(), tenant_id=TENANT, grants=[impersonate, metadata]
    )
    assert result.grant_id == metadata.id


def test_a_permitted_action_names_the_grant_that_permitted_it() -> None:
    grant = a_grant(GrantScope.CONTENT)
    result = authorize(
        OperatorAction.VIEW_DOCUMENT_CONTENT, operator=an_operator(), tenant_id=TENANT, grants=[grant]
    )
    assert result.grant_id == grant.id


# ------------------------------------------------------------------------------------------
# What the operator is told beforehand
# ------------------------------------------------------------------------------------------


def test_a_content_action_says_up_front_that_it_reads_customer_documents() -> None:
    """Making the cost visible before the click is most of what stops casual access."""
    text = describe(OperatorAction.VIEW_DOCUMENT_CONTENT)
    assert "content grant" in text
    assert "own documents" in text


def test_an_impersonation_action_says_the_customer_will_see_it() -> None:
    text = describe(OperatorAction.IMPERSONATE_USER)
    assert "visible to the customer" in text


def test_every_action_has_a_description() -> None:
    for action in OperatorAction:
        assert describe(action) and "cannot be performed" not in describe(action)
