"""SCIM: deprovisioning, and the directory quirks that make it fail silently.

``test_entra_sends_active_as_the_string_false`` is the one to read. ``bool("False")`` is ``True``
in Python, so the naive implementation *activates* the account the directory just told it to
disable — and reports success while doing it. That is a departed employee keeping their access,
with a green tick in the customer's console.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest

from app.auth.scim import (
    FORBIDDEN,
    PATCH_SCHEMA,
    USER_SCHEMA,
    ScimError,
    ScimUser,
    list_response,
    parse_filter,
    parse_patch,
    parse_user,
    plan_deprovision,
    service_provider_config,
)
from app.security.capabilities import Role

USER_ID = uuid.UUID("11110000-0000-0000-0000-000000000001")


def a_user(**kwargs: Any) -> ScimUser:
    base: dict[str, Any] = {
        "id": USER_ID,
        "user_name": "a@acme.com",
        "active": True,
        "display_name": "A Person",
        "created_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    base.update(kwargs)
    return ScimUser(**base)


# ------------------------------------------------------------------------------------------
# Deactivation: the thing SCIM is actually for
# ------------------------------------------------------------------------------------------


def test_entra_sends_active_as_the_string_false() -> None:
    """The single highest-consequence line in the module.

    `bool("False")` is True. A naive implementation activates the account it was told to disable,
    and reports success -- a departed employee keeping their access, with a green tick in the
    customer's console.
    """
    write = parse_patch(
        {
            "schemas": [PATCH_SCHEMA],
            "Operations": [{"op": "replace", "path": "active", "value": "False"}],
        }
    )
    assert write.active is False
    assert write.deactivates


def test_a_json_boolean_false_also_deactivates() -> None:
    write = parse_patch(
        {"schemas": [PATCH_SCHEMA], "Operations": [{"op": "Replace", "path": "active", "value": False}]}
    )
    assert write.deactivates


def test_okta_sends_a_patch_with_no_path_at_all() -> None:
    """{"op": "replace", "value": {"active": false}}. A path-only implementation ignores it, and
    the account stays live."""
    write = parse_patch({"schemas": [PATCH_SCHEMA], "Operations": [{"op": "replace", "value": {"active": False}}]})
    assert write.deactivates


def test_reactivation_works_too() -> None:
    """Someone returning from leave must not need an administrator to intervene."""
    write = parse_patch(
        {"schemas": [PATCH_SCHEMA], "Operations": [{"op": "replace", "path": "active", "value": "True"}]}
    )
    assert write.active is True
    assert not write.deactivates


def test_a_patch_that_does_not_mention_active_leaves_it_alone() -> None:
    """A name change must not be able to disable an account."""
    write = parse_patch(
        {"schemas": [PATCH_SCHEMA], "Operations": [{"op": "replace", "path": "displayName", "value": "New Name"}]}
    )
    assert write.active is None
    assert write.display_name == "New Name"


def test_a_remove_of_active_is_ignored_rather_than_treated_as_deactivation() -> None:
    """Nobody deactivates this way, and treating it as deactivation would let a partial sync
    disable accounts."""
    write = parse_patch({"schemas": [PATCH_SCHEMA], "Operations": [{"op": "remove", "path": "active"}]})
    assert write.active is None


def test_a_patch_without_the_patch_schema_is_refused() -> None:
    with pytest.raises(ScimError) as raised:
        parse_patch({"Operations": [{"op": "replace", "path": "active", "value": False}]})
    assert raised.value.status == 400


@pytest.mark.parametrize("value", [None, "maybe", {}, []])
def test_an_unreadable_active_value_changes_nothing(value: Any) -> None:
    """Guessing is how an account is disabled, or left live, by accident."""
    write = parse_patch(
        {"schemas": [PATCH_SCHEMA], "Operations": [{"op": "replace", "path": "active", "value": value}]}
    )
    assert write.active is None


# ------------------------------------------------------------------------------------------
# What deactivation must actually do
# ------------------------------------------------------------------------------------------


def test_deprovisioning_revokes_sessions_and_keys_not_just_the_row() -> None:
    """Marking the row inactive is the least interesting part. Without the rest, a departed
    employee's access token works for its remaining lifetime and their API key works
    indefinitely -- so "we deprovisioned them" is true of the database and false of the system."""
    effect = plan_deprovision(USER_ID)
    assert effect.deactivate_user
    assert effect.revoke_sessions
    assert effect.revoke_api_keys
    assert effect.revoke_support_impersonation


def test_deprovisioning_is_audited() -> None:
    assert plan_deprovision(USER_ID).audit_action == "user.deactivated"


def test_deprovisioning_deactivates_rather_than_deletes() -> None:
    """The audit trail must survive the person leaving; a deleted row orphans every entry that
    names them."""
    effect = plan_deprovision(USER_ID)
    assert not hasattr(effect, "delete_user")


# ------------------------------------------------------------------------------------------
# What a directory may not write
# ------------------------------------------------------------------------------------------


def test_a_directory_cannot_set_a_role() -> None:
    """Allowing it would make the customer's IdP administrator an administrator here, through a
    path with none of the SSO wizard's warnings or previews."""
    write = parse_user({"schemas": [USER_SCHEMA], "userName": "a@acme.com", "roles": [{"value": "ADMIN"}]})
    assert "roles" in write.refused
    assert not hasattr(write, "role")


def test_a_directory_cannot_set_a_role_by_patch_either() -> None:
    write = parse_patch(
        {"schemas": [PATCH_SCHEMA], "Operations": [{"op": "replace", "path": "roles", "value": [{"value": "ADMIN"}]}]}
    )
    assert "roles" in write.refused


@pytest.mark.parametrize("attribute", sorted(FORBIDDEN))
def test_no_forbidden_attribute_is_ever_writable(attribute: str) -> None:
    write = parse_user({"schemas": [USER_SCHEMA], "userName": "a@acme.com", attribute: "anything"})
    assert attribute in write.refused


def test_an_unknown_attribute_is_ignored_rather_than_rejected() -> None:
    """Entra and Okta both send fields we do not model. Failing a provisioning run over one
    produces a red error in a customer's console that neither side can act on."""
    write = parse_user(
        {
            "schemas": [USER_SCHEMA],
            "userName": "a@acme.com",
            "urn:ietf:params:scim:schemas:extension:enterprise:2.0:User": {"department": "Legal"},
            "preferredLanguage": "en-GB",
            "timezone": "Europe/Berlin",
        }
    )
    assert write.user_name == "a@acme.com"
    assert "preferredLanguage" in write.ignored


# ------------------------------------------------------------------------------------------
# Reading a user
# ------------------------------------------------------------------------------------------


def test_a_username_is_normalised() -> None:
    assert parse_user({"schemas": [USER_SCHEMA], "userName": "  A@Acme.COM "}).user_name == "a@acme.com"


def test_the_primary_email_is_used_when_there_is_no_username() -> None:
    write = parse_user(
        {
            "schemas": [USER_SCHEMA],
            "emails": [{"value": "secondary@acme.com"}, {"value": "Primary@Acme.com", "primary": True}],
        }
    )
    assert write.user_name == "primary@acme.com"


def test_a_name_object_is_split_out() -> None:
    write = parse_user({"schemas": [USER_SCHEMA], "name": {"givenName": "Ada", "familyName": "Lovelace"}})
    assert (write.given_name, write.family_name) == ("Ada", "Lovelace")


def test_a_formatted_name_becomes_the_display_name_when_there_is_none() -> None:
    write = parse_user({"schemas": [USER_SCHEMA], "name": {"formatted": "Ada Lovelace"}})
    assert write.display_name == "Ada Lovelace"


# ------------------------------------------------------------------------------------------
# The resource we return
# ------------------------------------------------------------------------------------------


def test_a_resource_carries_the_schema_and_a_location() -> None:
    resource = a_user().to_resource()
    assert resource["schemas"] == [USER_SCHEMA]
    assert resource["meta"]["location"].endswith(str(USER_ID))


def test_the_role_is_returned_read_only_so_an_administrator_can_see_it() -> None:
    """Visible in the IdP console, not settable from there."""
    assert a_user(role=Role.DEV).to_resource()["roles"] == [{"value": Role.DEV, "primary": True}]


def test_a_deactivated_user_reports_active_false() -> None:
    assert a_user(active=False).to_resource()["active"] is False


def test_a_list_response_reports_the_true_total_not_the_page_size() -> None:
    """A directory pages on it. Reporting the page size stops the sync after one page, and the
    users it never saw keep their access."""
    response = list_response([a_user()], total=137)
    assert response["totalResults"] == 137
    assert response["itemsPerPage"] == 1


# ------------------------------------------------------------------------------------------
# Filters
# ------------------------------------------------------------------------------------------


def test_the_one_filter_every_directory_sends() -> None:
    assert parse_filter('userName eq "A@Acme.com"') == ("userName", "a@acme.com")


def test_no_filter_means_no_filter() -> None:
    assert parse_filter(None) is None
    assert parse_filter("") is None


@pytest.mark.parametrize(
    "expression",
    ['userName co "acme"', "active pr", 'userName eq "a" and active eq true', "nonsense"],
)
def test_an_unsupported_filter_is_refused_rather_than_matching_everything(expression: str) -> None:
    """An over-broad match would hand a directory the whole user list in response to a query it
    thought was for one person."""
    with pytest.raises(ScimError) as raised:
        parse_filter(expression)
    assert raised.value.status == 400


# ------------------------------------------------------------------------------------------
# What we tell a directory we support
# ------------------------------------------------------------------------------------------


def test_we_claim_patch_because_that_is_how_entra_deactivates() -> None:
    assert service_provider_config()["patch"]["supported"] is True


def test_we_do_not_claim_bulk_or_sort_we_have_not_implemented() -> None:
    """Claiming support we do not have produces a provisioning run that reports success and
    changes nothing -- worse than declining."""
    config = service_provider_config()
    assert config["bulk"]["supported"] is False
    assert config["sort"]["supported"] is False


def test_we_do_not_accept_passwords_over_scim() -> None:
    """Passwords live at the identity provider. There is nothing here for a directory to set."""
    assert service_provider_config()["changePassword"]["supported"] is False


def test_the_error_shape_is_the_one_scim_clients_parse() -> None:
    body = ScimError(404, "No such user.", scim_type="invalidValue").to_dict()
    assert body["schemas"] == ["urn:ietf:params:scim:api:messages:2.0:Error"]
    assert body["status"] == "404"
    assert body["scimType"] == "invalidValue"
