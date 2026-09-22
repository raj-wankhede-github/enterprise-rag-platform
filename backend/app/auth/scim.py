"""SCIM 2.0, for the one thing that actually matters: deprovisioning.

Provisioning is already solved -- JIT creates a user on their first successful sign-in, which is
both simpler and more accurate than a directory push. **Deprovisioning is not.** When someone
leaves, their directory account is disabled and they can no longer authenticate, but their rows
here remain: their sessions stay valid until they expire, their API keys keep working, and their
name stays in the user list looking current.

That gap is what a security questionnaire asks about, and the honest answer has a number in it:

    "How quickly is access revoked when an employee leaves?"
    "Within 15 minutes with SCIM configured. Within 24 hours without it, by nightly
     reconciliation against the directory."

Three decisions:

**``active: false`` deactivates; it does not delete.** SCIM's own semantics, and the right ones:
the audit trail must survive the person leaving, and a deleted row would orphan every entry that
names them. A DELETE is treated as a deactivation for the same reason.

**Deactivation revokes every session and every API key immediately.** Marking the row inactive
alone would leave a valid access token working for its remaining lifetime, and a personal API key
working indefinitely. The row is the least interesting part of the change.

**Roles are not writable over SCIM.** A directory push that could set a role would make the
customer's IdP administrator an administrator of this product too, through a path with none of
the wizard's warnings. Group membership arrives here and is *mapped* by the same rules as a
login, which keeps one place responsible for deciding what a group means.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from app.security.capabilities import Role

logger = logging.getLogger(__name__)

USER_SCHEMA = "urn:ietf:params:scim:schemas:core:2.0:User"
LIST_SCHEMA = "urn:ietf:params:scim:api:messages:2.0:ListResponse"
PATCH_SCHEMA = "urn:ietf:params:scim:api:messages:2.0:PatchOp"
ERROR_SCHEMA = "urn:ietf:params:scim:api:messages:2.0:Error"
ENTERPRISE_SCHEMA = "urn:ietf:params:scim:schemas:extension:enterprise:2.0:User"

#: Attributes a directory may write. Everything else is accepted and ignored.
#:
#: Ignored rather than rejected, deliberately: Entra and Okta both send attributes we do not
#: model, and returning 400 for an attribute we simply do not care about makes a provisioning
#: run fail for no reason a customer can act on.
WRITABLE = frozenset({"userName", "name", "displayName", "active", "emails", "externalId"})

#: Attributes that are never writable however they arrive.
#:
#: ``roles`` is the one that matters. Allowing it would make the customer's IdP administrator an
#: administrator here through a path with none of the SSO wizard's warnings or previews.
FORBIDDEN = frozenset({"roles", "entitlements", "groups"})


class ScimError(Exception):
    """A SCIM-shaped failure. ``status`` is the HTTP code the response carries."""

    def __init__(self, status: int, detail: str, *, scim_type: str | None = None) -> None:
        super().__init__(detail)
        self.status = status
        self.detail = detail
        self.scim_type = scim_type

    def to_dict(self) -> dict[str, Any]:
        body: dict[str, Any] = {"schemas": [ERROR_SCHEMA], "status": str(self.status), "detail": self.detail}
        if self.scim_type:
            body["scimType"] = self.scim_type
        return body


@dataclass(frozen=True, slots=True)
class ScimUser:
    """Our user, in SCIM's vocabulary."""

    id: uuid.UUID
    user_name: str
    active: bool
    display_name: str = ""
    given_name: str = ""
    family_name: str = ""
    external_id: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    #: Returned read-only, so an administrator can see it in their IdP's console without being
    #: able to set it from there.
    role: str = Role.PROD

    def to_resource(self, *, base_url: str = "/scim/v2") -> dict[str, Any]:
        return {
            "schemas": [USER_SCHEMA],
            "id": str(self.id),
            "externalId": self.external_id,
            "userName": self.user_name,
            "name": {
                "givenName": self.given_name,
                "familyName": self.family_name,
                "formatted": self.display_name or f"{self.given_name} {self.family_name}".strip(),
            },
            "displayName": self.display_name,
            "emails": [{"value": self.user_name, "primary": True, "type": "work"}],
            "active": self.active,
            # Read-only, and marked as such in the schema response so a directory does not try.
            "roles": [{"value": self.role, "primary": True}],
            "meta": {
                "resourceType": "User",
                "created": self.created_at.isoformat() if self.created_at else None,
                "lastModified": self.updated_at.isoformat() if self.updated_at else None,
                "location": f"{base_url}/Users/{self.id}",
            },
        }


@dataclass(slots=True)
class UserWrite:
    """What a directory asked us to change, after filtering."""

    user_name: str | None = None
    display_name: str | None = None
    given_name: str | None = None
    family_name: str | None = None
    external_id: str | None = None
    active: bool | None = None
    ignored: list[str] = field(default_factory=list)
    refused: list[str] = field(default_factory=list)

    @property
    def deactivates(self) -> bool:
        return self.active is False


def parse_user(payload: dict[str, Any]) -> UserWrite:
    """Read a SCIM user resource into what we will actually change.

    Unknown attributes are collected rather than rejected. Entra and Okta both send fields we do
    not model, and failing a provisioning run over one of them produces a red error in a
    customer's console that neither side can act on.
    """
    write = UserWrite()

    for key, value in payload.items():
        if key in {"schemas", "id", "meta", ENTERPRISE_SCHEMA}:
            continue
        if key in FORBIDDEN:
            write.refused.append(key)
            continue
        if key not in WRITABLE:
            write.ignored.append(key)
            continue

        if key == "userName":
            write.user_name = str(value).strip().casefold()
        elif key == "displayName":
            write.display_name = str(value)
        elif key == "externalId":
            write.external_id = str(value)
        elif key == "active":
            write.active = _as_bool(value)
        elif key == "name" and isinstance(value, dict):
            write.given_name = str(value.get("givenName", ""))
            write.family_name = str(value.get("familyName", ""))
            if not write.display_name and value.get("formatted"):
                write.display_name = str(value["formatted"])
        elif key == "emails" and isinstance(value, list):
            primary = next((entry for entry in value if entry.get("primary")), value[0] if value else None)
            if primary and primary.get("value") and not write.user_name:
                write.user_name = str(primary["value"]).strip().casefold()

    if write.refused:
        logger.info("scim.attributes_refused", extra={"attributes": write.refused})
    return write


def parse_patch(payload: dict[str, Any]) -> UserWrite:
    """Read a PATCH operation list.

    Entra deactivates with PATCH rather than PUT, and this is the single most important request
    this module handles -- it is how an employee's access actually ends. Getting the several
    shapes right matters more than anything else here, because a PATCH we fail to understand is
    an account that stays live.
    """
    if PATCH_SCHEMA not in payload.get("schemas", []):
        raise ScimError(400, "Expected a PatchOp document.", scim_type="invalidSyntax")

    write = UserWrite()
    for operation in payload.get("Operations", []):
        op = str(operation.get("op", "")).lower()
        if op == "remove":
            # A remove of `active` is not how anyone deactivates, and treating it as one would
            # let a partial sync disable accounts. Recorded and ignored.
            write.ignored.append(f"remove:{operation.get('path')}")
            continue

        path = str(operation.get("path", "")).strip()
        value = operation.get("value")

        if not path and isinstance(value, dict):
            # Okta sends {"op": "replace", "value": {"active": false}} with no path at all.
            merged = parse_user(value)
            write.active = merged.active if merged.active is not None else write.active
            write.user_name = merged.user_name or write.user_name
            write.display_name = merged.display_name or write.display_name
            write.ignored.extend(merged.ignored)
            write.refused.extend(merged.refused)
            continue

        # Entra sends a bare path with a scalar, and sometimes the string "False" rather than a
        # JSON boolean -- which is why _as_bool exists rather than bool().
        attribute = path.split(".")[0].split("[")[0]
        if attribute in FORBIDDEN:
            write.refused.append(attribute)
        elif attribute == "active":
            write.active = _as_bool(value)
        elif attribute == "userName":
            write.user_name = str(value).strip().casefold()
        elif attribute == "displayName":
            write.display_name = str(value)
        elif attribute == "name" and isinstance(value, dict):
            write.given_name = str(value.get("givenName", write.given_name or ""))
            write.family_name = str(value.get("familyName", write.family_name or ""))
        elif attribute == "name" and path.endswith("givenName"):
            write.given_name = str(value)
        elif attribute == "name" and path.endswith("familyName"):
            write.family_name = str(value)
        else:
            write.ignored.append(path)

    return write


def _as_bool(value: Any) -> bool | None:
    """SCIM's ``active``, from every shape a directory sends it in.

    Entra has historically sent the *string* ``"False"``, which ``bool()`` evaluates as true --
    so a naive implementation activates the account it was told to disable. This is the single
    highest-consequence line in the module.
    """
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    if isinstance(value, int):
        return bool(value)
    return None


def list_response(
    users: list[ScimUser], *, total: int, start_index: int = 1, base_url: str = "/scim/v2"
) -> dict[str, Any]:
    return {
        "schemas": [LIST_SCHEMA],
        "totalResults": total,
        "startIndex": start_index,
        "itemsPerPage": len(users),
        "Resources": [user.to_resource(base_url=base_url) for user in users],
    }


#: ``userName eq "a@b.com"``, and nothing else.
#:
#: Anchored at both ends deliberately. A permissive split would read
#: ``userName eq "a" and active eq true`` as a simple filter with a nonsense value, return no
#: results, and lead the directory to conclude the user does not exist -- so it creates a
#: duplicate. Refusing is the only outcome that tells the truth.
_FILTER = re.compile(r'^\s*(\w+)\s+eq\s+"([^"]*)"\s*$', re.IGNORECASE)


def parse_filter(expression: str | None) -> tuple[str, str] | None:
    """The one filter shape every directory sends: ``userName eq "a@b.com"``.

    A full SCIM filter grammar is a parser and a security surface, and no directory in practice
    sends anything else for a Users lookup. Anything else is refused rather than approximated:
    an over-broad match would hand a directory the whole user list in response to a query it
    thought was for one person, and a silently-truncated one produces duplicate accounts.
    """
    if not expression:
        return None

    match = _FILTER.match(expression)
    if match is None:
        logger.info("scim.unsupported_filter", extra={"filter": expression[:120]})
        raise ScimError(400, "Only 'attribute eq \"value\"' filters are supported.", scim_type="invalidFilter")

    return match.group(1), match.group(2).strip().casefold()


@dataclass(frozen=True, slots=True)
class DeprovisionEffect:
    """Everything that must happen when a directory deactivates someone.

    Marking the row inactive is the least interesting part. Without the rest, a departed
    employee's access token keeps working for its remaining lifetime and their personal API key
    works indefinitely -- so "we deprovisioned them" would be true of the database and false of
    the system.
    """

    user_id: uuid.UUID
    deactivate_user: bool = True
    revoke_sessions: bool = True
    revoke_api_keys: bool = True
    revoke_support_impersonation: bool = True
    audit_action: str = "user.deactivated"
    at: datetime = field(default_factory=lambda: datetime.now(UTC))


def plan_deprovision(user_id: uuid.UUID) -> DeprovisionEffect:
    return DeprovisionEffect(user_id=user_id)


def service_provider_config(*, base_url: str = "/scim/v2") -> dict[str, Any]:
    """What we support, stated honestly.

    Directories read this and adjust. Claiming PATCH support we do not have produces a
    provisioning run that reports success and changes nothing, which is worse than declining.
    """
    return {
        "schemas": ["urn:ietf:params:scim:schemas:core:2.0:ServiceProviderConfig"],
        "documentationUri": f"{base_url}/docs",
        "patch": {"supported": True},
        "bulk": {"supported": False, "maxOperations": 0, "maxPayloadSize": 0},
        "filter": {"supported": True, "maxResults": 200},
        # Passwords live at the identity provider. There is nothing here for a directory to set.
        "changePassword": {"supported": False},
        "sort": {"supported": False},
        "etag": {"supported": False},
        "authenticationSchemes": [
            {
                "type": "oauthbearertoken",
                "name": "OAuth Bearer Token",
                "description": "A SCIM token issued from Settings > Sign-in. Scoped to this tenant.",
                "primary": True,
            }
        ],
    }
