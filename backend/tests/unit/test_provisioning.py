"""Turning IdP claims into one of our users and one of our four roles.

The highest-risk code in the authentication path, and the risk is blunt: a wrong group mapping
grants ADMIN to an entire directory, silently, at the next login. Most of this file is about the
layers that make that require more than one mistake.
"""

from __future__ import annotations

import uuid

import pytest

from app.auth.provisioning import (
    ExistingUser,
    IdentityClaims,
    ProvisioningOutcome,
    ProvisioningPolicy,
    extract_claims,
    plan_login,
    preview_mapping,
    resolve_role,
)
from app.security.capabilities import Role

USER_ID = uuid.UUID("11110000-0000-0000-0000-000000000001")


def claims(**kwargs: object) -> IdentityClaims:
    base: dict[str, object] = {"subject": "subject-1", "email": "a@acme.com", "email_verified": True}
    base.update(kwargs)
    return IdentityClaims(**base)  # type: ignore[arg-type]


def a_user(**kwargs: object) -> ExistingUser:
    base: dict[str, object] = {
        "id": USER_ID,
        "email": "a@acme.com",
        "role": Role.PROD,
        "is_active": True,
    }
    base.update(kwargs)
    return ExistingUser(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------
# Claim extraction
# ------------------------------------------------------------------------------------------


def test_entras_oid_wins_over_sub() -> None:
    """``sub`` is pairwise per application and changes if the app registration is recreated;
    ``oid`` is the directory object id and is stable for the life of the account. Matching on
    ``sub`` orphans every identity in the tenant the day someone recreates the registration."""
    extracted = extract_claims({"oid": "object-id", "sub": "pairwise-sub"})
    assert extracted.subject == "object-id"


def test_sub_is_used_when_there_is_no_oid() -> None:
    assert extract_claims({"sub": "generic-sub"}).subject == "generic-sub"


def test_a_token_identifying_nobody_is_rejected() -> None:
    with pytest.raises(ValueError, match="neither 'oid' nor 'sub'"):
        extract_claims({"email": "a@acme.com"})


def test_a_single_group_sent_as_a_bare_string_is_not_split_into_characters() -> None:
    """The classic bug: a provider sends one group as a string, and treating a str as an
    iterable turns "admins" into six single-character groups that match nothing."""
    assert extract_claims({"sub": "s", "groups": "admins"}).groups == ("admins",)


def test_groups_sent_comma_or_space_separated_are_split() -> None:
    assert extract_claims({"sub": "s", "groups": "admins, engineers"}).groups == ("admins", "engineers")


def test_an_absent_groups_claim_yields_no_groups_rather_than_failing() -> None:
    """Google Workspace does not put groups in the ID token at all."""
    assert extract_claims({"sub": "s"}).groups == ()


def test_the_groups_claim_name_is_configurable() -> None:
    extracted = extract_claims({"sub": "s", "roles": ["a"]}, groups_claim="roles")
    assert extracted.groups == ("a",)


def test_email_falls_back_through_the_providers_that_do_not_send_it_plainly() -> None:
    assert extract_claims({"sub": "s", "upn": "A@Acme.com"}).email == "a@acme.com"
    assert extract_claims({"sub": "s", "preferred_username": "B@acme.com"}).email == "b@acme.com"


# ------------------------------------------------------------------------------------------
# Role resolution
# ------------------------------------------------------------------------------------------


def test_the_highest_matching_role_wins_not_the_first() -> None:
    """A user in both engineering and eng-leads gets the higher of the two, which is what an
    administrator who wrote both mappings meant. First-match would make the outcome depend on
    dictionary ordering."""
    policy = ProvisioningPolicy(role_mappings={"engineering": Role.TEST, "eng-leads": Role.DEV})
    role, _ = resolve_role(claims(groups=("engineering", "eng-leads")), policy)
    assert role is Role.DEV


def test_the_order_groups_arrive_in_does_not_change_the_outcome() -> None:
    policy = ProvisioningPolicy(role_mappings={"a": Role.PROD, "b": Role.ADMIN, "c": Role.DEV})
    forward, _ = resolve_role(claims(groups=("a", "b", "c")), policy)
    backward, _ = resolve_role(claims(groups=("c", "b", "a")), policy)
    assert forward is backward is Role.ADMIN


def test_a_role_value_we_do_not_recognise_is_ignored() -> None:
    """A provider that starts sending "role": "SUPERADMIN" changes nothing here, because roles
    come from an allow-map rather than from a claim's value."""
    policy = ProvisioningPolicy(role_mappings={"weird": "SUPERADMIN"}, default_role=Role.PROD)
    role, _ = resolve_role(claims(groups=("weird",)), policy)
    assert role is Role.PROD


def test_an_existing_user_whose_groups_stop_matching_keeps_their_role() -> None:
    """Falling back to the default would silently demote every administrator the moment a claim
    name changes -- which is a thing that happens, and would be indistinguishable from a bug."""
    policy = ProvisioningPolicy(role_mappings={"admins": Role.ADMIN}, default_role=Role.PROD)
    role, _ = resolve_role(claims(groups=()), policy, current=Role.ADMIN)
    assert role is Role.ADMIN


def test_a_new_user_with_no_matching_groups_gets_the_default() -> None:
    policy = ProvisioningPolicy(default_role=Role.TEST)
    role, _ = resolve_role(claims(groups=("unmapped",)), policy)
    assert role is Role.TEST


def test_promotion_to_admin_is_reported_so_it_can_be_audited_and_notified() -> None:
    """Elevation through a directory group is legitimate, and is also exactly what an attacker
    who can edit groups would do. It must never be silent."""
    policy = ProvisioningPolicy(role_mappings={"admins": Role.ADMIN})
    _, elevates = resolve_role(claims(groups=("admins",)), policy, current=Role.DEV)
    assert elevates


def test_an_admin_who_stays_an_admin_is_not_reported_as_an_elevation() -> None:
    policy = ProvisioningPolicy(role_mappings={"admins": Role.ADMIN})
    _, elevates = resolve_role(claims(groups=("admins",)), policy, current=Role.ADMIN)
    assert not elevates


# ------------------------------------------------------------------------------------------
# What a login does
# ------------------------------------------------------------------------------------------


def test_a_known_subject_signs_in() -> None:
    decision = plan_login(claims(), ProvisioningPolicy(), linked=a_user())
    assert decision.outcome is ProvisioningOutcome.LINKED_EXISTING
    assert decision.permitted


def test_jit_never_creates_an_admin_however_the_groups_are_mapped() -> None:
    """The layer that matters most. A group mapping is an administrator's statement about their
    own people; it must not be a path by which an unknown subject arrives as an administrator.
    The database CHECK on default_role forbids ADMIN independently."""
    policy = ProvisioningPolicy(role_mappings={"admins": Role.ADMIN}, default_role=Role.PROD)
    decision = plan_login(claims(groups=("admins",)), policy, linked=None)

    assert decision.outcome is ProvisioningOutcome.CREATED
    assert decision.role is Role.PROD
    assert not decision.elevates_to_admin


def test_an_unknown_subject_is_refused_when_jit_is_off() -> None:
    """The tenant wants the directory to authenticate but not to enroll."""
    decision = plan_login(claims(), ProvisioningPolicy(jit_provisioning=False), linked=None)
    assert decision.outcome is ProvisioningOutcome.REJECTED_NO_JIT
    assert not decision.permitted


def test_a_deactivated_user_cannot_sign_back_in_through_sso() -> None:
    """Deactivation must outrank everything, or a user removed from the product walks straight
    back in the moment their directory account still authenticates."""
    decision = plan_login(claims(), ProvisioningPolicy(), linked=a_user(is_active=False))
    assert decision.outcome is ProvisioningOutcome.REJECTED_INACTIVE
    assert not decision.permitted


def test_a_deactivated_user_is_refused_even_with_an_admin_group() -> None:
    policy = ProvisioningPolicy(role_mappings={"admins": Role.ADMIN})
    decision = plan_login(claims(groups=("admins",)), policy, linked=a_user(is_active=False))
    assert not decision.permitted
    assert decision.role is None


def test_a_preprovisioned_account_is_linked_by_verified_email_once() -> None:
    """Without this, a tenant that creates its users in advance has every one of them fail their
    first SSO login."""
    decision = plan_login(claims(), ProvisioningPolicy(), linked=None, by_email=a_user())
    assert decision.outcome is ProvisioningOutcome.LINKED_BY_EMAIL
    assert decision.user_id == USER_ID


def test_an_unverified_address_cannot_link_an_existing_account() -> None:
    """Otherwise anyone who can make their IdP assert a string takes over that account."""
    decision = plan_login(claims(email_verified=False), ProvisioningPolicy(), linked=None, by_email=a_user())
    assert not decision.permitted


def test_a_pinned_role_survives_a_directory_that_says_otherwise() -> None:
    """An administrator who pins a role in the product is overruling the directory on purpose."""
    policy = ProvisioningPolicy(role_mappings={"interns": Role.PROD})
    decision = plan_login(claims(groups=("interns",)), policy, linked=a_user(role=Role.DEV, role_locked=True))
    assert decision.role is Role.DEV
    assert not decision.role_changed


def test_role_sync_can_be_turned_off_for_a_tenant_that_manages_roles_itself() -> None:
    policy = ProvisioningPolicy(role_mappings={"admins": Role.ADMIN}, sync_role_on_login=False)
    decision = plan_login(claims(groups=("admins",)), policy, linked=a_user(role=Role.PROD))
    assert decision.role is Role.PROD
    assert not decision.elevates_to_admin


def test_a_directory_demotion_takes_effect_and_is_reported() -> None:
    policy = ProvisioningPolicy(role_mappings={"contractors": Role.PROD})
    decision = plan_login(claims(groups=("contractors",)), policy, linked=a_user(role=Role.DEV))
    assert decision.role is Role.PROD
    assert decision.role_changed


# ------------------------------------------------------------------------------------------
# The wizard's preview
# ------------------------------------------------------------------------------------------


def test_the_preview_warns_before_a_group_can_hand_out_admin() -> None:
    """An administrator who sees that "All Company" maps to ADMIN before activating will not
    activate it. That is the entire value of draft -> test -> activate."""
    policy = ProvisioningPolicy(role_mappings={"All Company": Role.ADMIN})
    preview = preview_mapping(policy, {"sub": "s", "groups": ["All Company"]}, groups_claim="groups")

    assert preview["resolved_role"] == Role.ADMIN
    assert preview["would_elevate_to_admin"]
    assert any("ADMIN" in warning for warning in preview["warnings"])


def test_the_preview_explains_googles_missing_groups_rather_than_showing_nothing() -> None:
    preview = preview_mapping(ProvisioningPolicy(), {"sub": "s"}, groups_claim="groups")
    assert any("Google Workspace" in warning for warning in preview["warnings"])


def test_the_preview_separates_matched_from_ignored_groups() -> None:
    """A user in 40 directory groups needs to see which two mattered."""
    policy = ProvisioningPolicy(role_mappings={"eng": Role.DEV})
    preview = preview_mapping(policy, {"sub": "s", "groups": ["eng", "everyone", "building-3"]}, groups_claim="groups")

    assert preview["groups_matched"] == {"eng": Role.DEV}
    assert set(preview["groups_ignored"]) == {"everyone", "building-3"}


def test_the_preview_flags_mappings_that_did_not_appear() -> None:
    """Almost always the object-id-versus-display-name mistake, which otherwise looks like the
    IdP being broken."""
    policy = ProvisioningPolicy(role_mappings={"Engineering": Role.DEV, "Finance": Role.TEST})
    preview = preview_mapping(policy, {"sub": "s", "groups": ["a1b2-guid"]}, groups_claim="groups")
    assert any("did not appear" in warning for warning in preview["warnings"])


def test_the_preview_never_reports_a_role_outside_the_four() -> None:
    policy = ProvisioningPolicy(role_mappings={"g": "PLATFORM"})
    preview = preview_mapping(policy, {"sub": "s", "groups": ["g"]}, groups_claim="groups")
    assert preview["resolved_role"] in set(Role)
