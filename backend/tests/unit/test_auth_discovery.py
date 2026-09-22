"""Email-first login discovery.

Two properties are being defended: that discovery never becomes a user- or customer-enumeration
oracle, and that hiding the password field is understood as a UI hint rather than as the control.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.auth.discovery import (
    DiscoveryResult,
    IdpSummary,
    TenantLoginPolicy,
    ambiguous_result,
    build_result,
    domain_of,
    is_public_domain,
    normalize_email,
    password_login_allowed,
)
from app.models.identity import IdpKind, IdpState

TENANT = uuid.UUID("11110000-0000-0000-0000-000000000001")


def a_policy(**kwargs: object) -> TenantLoginPolicy:
    base: dict[str, object] = {
        "tenant_id": TENANT,
        "slug": "acme",
        "name": "Acme GmbH",
        "password_login_enabled": True,
    }
    base.update(kwargs)
    return TenantLoginPolicy(**base)  # type: ignore[arg-type]


def an_idp(**kwargs: object) -> IdpSummary:
    base: dict[str, object] = {
        "id": uuid.uuid4(),
        "kind": IdpKind.ENTRA,
        "display_name": "",
        "state": IdpState.ACTIVE,
    }
    base.update(kwargs)
    return IdpSummary(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------
# Parsing
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("  A@Acme.COM ", "acme.com"),
        ("first.last+tag@sub.acme.co.uk", "sub.acme.co.uk"),
    ],
)
def test_the_domain_is_extracted_case_insensitively(value: str, expected: str) -> None:
    assert domain_of(value) == expected


@pytest.mark.parametrize("value", ["", "not-an-email", "@acme.com", "a@", "a@b", "a b@acme.com"])
def test_something_that_is_not_an_address_has_no_domain(value: str) -> None:
    assert domain_of(value) is None


def test_normalisation_is_trim_and_casefold() -> None:
    assert normalize_email("  A@Acme.COM  ") == "a@acme.com"


def test_consumer_domains_never_auto_resolve_to_a_tenant() -> None:
    """Otherwise the first customer to verify gmail.com captures every consumer address in the
    product."""
    assert is_public_domain("gmail.com")
    assert is_public_domain("outlook.com")
    assert not is_public_domain("acme.com")


# ------------------------------------------------------------------------------------------
# Enumeration resistance
# ------------------------------------------------------------------------------------------


def test_an_unknown_domain_looks_exactly_like_an_ordinary_tenant() -> None:
    """An error, or an empty response, would tell an attacker which domains are customers --
    which in a B2B product is a competitor's prospect list."""
    unknown = build_result(None)
    known = build_result(a_policy())

    assert unknown.password_login == known.password_login is True
    assert unknown == DiscoveryResult(password_login=True)


def test_discovery_says_nothing_about_whether_a_user_exists() -> None:
    """The response describes a tenant's policy. It is computed without looking at users at
    all, which is why no field here could carry that information."""
    result = build_result(a_policy(idps=(an_idp(),)))
    assert not hasattr(result, "user_exists")
    assert set(vars(DiscoveryResult()) if hasattr(DiscoveryResult(), "__dict__") else DiscoveryResult.__slots__) == {
        "tenant_id",
        "tenant_slug",
        "tenant_name",
        "password_login",
        "methods",
        "ambiguous",
        "emergency_access",
    }


# ------------------------------------------------------------------------------------------
# Which methods are offered
# ------------------------------------------------------------------------------------------


def test_only_active_configs_appear_on_the_login_page() -> None:
    """A draft or tested config is visible to its administrator in the wizard and to nobody
    else. That separation is what makes draft -> test -> activate real."""
    policy = a_policy(
        idps=(
            an_idp(state=IdpState.ACTIVE, display_name="Live"),
            an_idp(state=IdpState.DRAFT, display_name="Draft"),
            an_idp(state=IdpState.TESTED, display_name="Tested"),
            an_idp(state=IdpState.DISABLED, display_name="Off"),
        )
    )
    assert [method.display_name for method in build_result(policy).methods] == ["Live"]


def test_a_provider_without_a_label_gets_a_sensible_one() -> None:
    methods = build_result(a_policy(idps=(an_idp(kind=IdpKind.ENTRA), an_idp(kind=IdpKind.GOOGLE)))).methods
    assert {method.display_name for method in methods} == {"Continue with Microsoft", "Continue with Google"}


def test_a_configured_label_wins_over_the_default() -> None:
    policy = a_policy(idps=(an_idp(display_name="Sign in with Acme SSO"),))
    assert build_result(policy).methods[0].display_name == "Sign in with Acme SSO"


def test_the_start_url_carries_the_config_id() -> None:
    idp = an_idp()
    result = build_result(a_policy(idps=(idp,)))
    assert str(idp.id) in result.methods[0].start_url


# ------------------------------------------------------------------------------------------
# Password login and the break-glass
# ------------------------------------------------------------------------------------------


def test_a_tenant_that_disabled_passwords_offers_none() -> None:
    assert not build_result(a_policy(password_login_enabled=False)).password_login


def test_the_break_glass_window_re_enables_passwords_and_says_so() -> None:
    """For one specific disaster: an administrator activates a broken IdP config on a tenant
    with password login disabled and locks out everyone including themselves. Recovery through
    the operator plane would mean the vendor holds a key to every customer's front door."""
    policy = a_policy(
        password_login_enabled=False,
        emergency_password_login_until=datetime.now(UTC) + timedelta(hours=1),
    )
    result = build_result(policy)
    assert result.password_login
    assert result.emergency_access, "a tenant that believes SSO is enforced must be told when it temporarily is not"


def test_the_break_glass_window_closes_by_itself() -> None:
    """The property that matters. A flag someone must remember to turn off is a flag that stays
    on."""
    policy = a_policy(
        password_login_enabled=False,
        emergency_password_login_until=datetime.now(UTC) - timedelta(minutes=1),
    )
    assert not build_result(policy).password_login


def test_an_ordinary_password_tenant_is_not_flagged_as_emergency_access() -> None:
    allowed, emergency = password_login_allowed(a_policy(password_login_enabled=True))
    assert allowed and not emergency


def test_the_window_is_evaluated_against_a_supplied_clock() -> None:
    deadline = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    policy = a_policy(password_login_enabled=False, emergency_password_login_until=deadline)
    assert password_login_allowed(policy, now=deadline - timedelta(minutes=1))[0]
    assert not password_login_allowed(policy, now=deadline + timedelta(minutes=1))[0]


# ------------------------------------------------------------------------------------------
# Several tenants on one domain
# ------------------------------------------------------------------------------------------


def test_a_shared_domain_asks_the_user_to_choose() -> None:
    """Real in consultancies and holding groups, where one address legitimately belongs to
    several customer tenants."""
    result = ambiguous_result([a_policy(slug="beta", name="Beta AG"), a_policy(slug="acme", name="Acme GmbH")])
    assert result.requires_tenant_choice
    assert result.ambiguous == (("acme", "Acme GmbH"), ("beta", "Beta AG"))


def test_the_choice_carries_no_identifiers_and_no_methods() -> None:
    """Enough to choose from, nothing about who has an account where."""
    result = ambiguous_result([a_policy()])
    assert result.tenant_id is None
    assert result.methods == ()
