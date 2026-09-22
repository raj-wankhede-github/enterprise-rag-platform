"""SharePoint and OneDrive through Microsoft Graph.

**Why this connector first, and not Confluence or Drive.** Entra is already a required identity
provider here, which means the group object ids in ``driveItem.permissions`` are the *same* ids
arriving in the login token's ``groups`` claim. Permission mirroring is therefore nearly free and
exactly correct -- the ACL we index is the one the user is measured against at query time. No
other connector has that property: with Confluence or Drive we would be mapping one directory's
notion of a group onto another's, and every mapping error is a document visible to the wrong
person.

**``Sites.Selected``, never ``Sites.Read.All``.** The broad scope grants the application every
site in the tenant, forever, and no customer security review should accept it. ``Sites.Selected``
grants nothing until the customer names specific sites, which turns "what can this vendor read"
from a promise into a list they control.

**Permissions are cached by folder, and item-level calls are the exception.** A library of
100,000 files answered item by item exhausts the Graph throttling budget long before the sync
finishes. SharePoint inheritance means almost every file has its parent folder's permissions, and
``driveItem.shared`` marks the ones that do not -- so the item-level call happens only where an
inheritance break actually exists, which is typically a few percent of a library.

**Mirrored ACLs are eventually consistent, and we say so.** Between a permission change in
SharePoint and the next delta sync there is a window where our index is wrong. Pretending
otherwise is what turns a documented limitation into an incident: the UI shows when permissions
were last verified, ``strict_acl_recheck`` re-validates at query time for tenants who need it,
and the security questionnaire gets a number rather than a reassurance.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GRAPH = "https://graph.microsoft.com/v1.0"

#: Item-level permission reads per sync before we stop and report degradation.
#:
#: Graph throttles per app per tenant, and exhausting the budget does not merely slow this sync
#: down -- it returns 429s to every other call the application makes for that tenant, including
#: ones a user is waiting on. Stopping with a partial, honestly-labelled result is better.
MAX_ITEM_PERMISSION_CALLS = 2_000

#: Roles in a Graph permission that grant read access. ``owner`` and ``write`` imply read.
_READ_ROLES = frozenset({"read", "write", "owner", "sp.read", "sp.write", "sp.full control"})


class GraphError(Exception):
    """Graph could not be reached, or refused."""


class ThrottledError(GraphError):
    """Graph returned 429. Carries the server's own Retry-After."""

    def __init__(self, retry_after_s: float) -> None:
        super().__init__(f"Microsoft Graph is throttling this application; retry in {retry_after_s:.0f}s")
        self.retry_after_s = retry_after_s


@dataclass(frozen=True, slots=True)
class ItemAcl:
    """Who may read one item, in *our* vocabulary.

    ``group_ids`` are Entra object ids, which is the point: they are the same strings that arrive
    in a login token's ``groups`` claim, so ``dsl.build_filter`` compares like with like.
    """

    group_ids: tuple[str, ...] = ()
    user_ids: tuple[str, ...] = ()
    #: True when the item is readable by anyone in the tenant -- SharePoint's "Everyone except
    #: external users", or an organisation-scoped sharing link.
    tenant_wide: bool = False
    #: True when a link exists that anyone with the URL can use. Indexed, but flagged: a customer
    #: usually wants to know these exist.
    anonymous_link: bool = False
    #: Where this came from, so the UI can explain an inherited permission.
    inherited_from: str | None = None

    @property
    def access_groups(self) -> tuple[str, ...]:
        """What goes into the index. ``*`` means tenant-wide, matching our own convention."""
        return ("*",) if self.tenant_wide else self.group_ids


@dataclass(slots=True)
class SyncStats:
    items_seen: int = 0
    items_indexed: int = 0
    items_deleted: int = 0
    folder_permission_reads: int = 0
    item_permission_reads: int = 0
    throttled_waits: int = 0
    #: Items whose permissions we could not read. Indexed at the *narrowest* ACL we know rather
    #: than the widest -- a document nobody can find is a support ticket, one visible to the
    #: wrong person is a breach.
    permission_failures: list[str] = field(default_factory=list)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    #: The connector's *configured* budget, not the module default.
    #:
    #: Reading the module constant here would mean a deployment that tuned the budget down never
    #: reported a partial sync -- the one configuration where partial syncs are most likely.
    budget: int = MAX_ITEM_PERMISSION_CALLS

    @property
    def degraded(self) -> bool:
        return bool(self.permission_failures) or self.item_permission_reads >= self.budget


class SharePointConnector:
    """Walks selected sites, mirrors permissions, and yields indexable items."""

    name = "sharepoint"
    version = "1"

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        strict_acl_recheck: bool = False,
        max_item_permission_calls: int = MAX_ITEM_PERMISSION_CALLS,
    ) -> None:
        self.client = client
        self.strict_acl_recheck = strict_acl_recheck
        self.max_item_permission_calls = max_item_permission_calls
        self.stats = SyncStats(budget=max_item_permission_calls)
        #: folder id -> ACL. The whole economic argument for this connector.
        self._folder_acls: dict[str, ItemAcl] = {}

    # ------------------------------------------------------------------------------------

    async def sync(self, *, drive_id: str, delta_token: str | None = None) -> AsyncIterator[dict[str, Any]]:
        """Yield changed items since ``delta_token``, each carrying a resolved ACL.

        A delta sync, not a full crawl. A 100,000-item library re-read hourly is both a
        throttling problem and a cost problem, and delta is the only reason a connector can run
        at a useful frequency.
        """
        url = (
            f"{GRAPH}/drives/{drive_id}/root/delta"
            if delta_token is None
            else f"{GRAPH}/drives/{drive_id}/root/delta?token={delta_token}"
        )

        while url:
            payload = await self._get(url)
            for item in payload.get("value", []):
                self.stats.items_seen += 1
                if item.get("deleted"):
                    self.stats.items_deleted += 1
                    yield {"external_id": item["id"], "deleted": True}
                    continue
                if "folder" in item:
                    # Folders are not indexed, but their ACLs are: they are what every file
                    # inside them inherits.
                    await self._cache_folder_acl(drive_id, item)
                    continue
                prepared = await self._prepare(drive_id, item)
                if prepared is not None:
                    self.stats.items_indexed += 1
                    yield prepared

            url = payload.get("@odata.nextLink") or ""
            if not url and payload.get("@odata.deltaLink"):
                # The delta link is the cursor for next time. Losing it means the next sync is a
                # full crawl, so it is yielded rather than merely logged.
                yield {"delta_token": _token_of(str(payload["@odata.deltaLink"]))}

    async def _prepare(self, drive_id: str, item: dict[str, Any]) -> dict[str, Any] | None:
        acl = await self._resolve_acl(drive_id, item)
        if acl is None:
            # Never index an item whose permissions we could not establish. A document nobody
            # can find is a support ticket; one visible to the wrong person is a breach.
            self.stats.permission_failures.append(str(item.get("id")))
            logger.warning("sharepoint.acl_unresolved", extra={"item": item.get("id")})
            return None

        return {
            "external_id": item["id"],
            "filename": item.get("name", ""),
            "mime_type": (item.get("file") or {}).get("mimeType", "application/octet-stream"),
            "size_bytes": int(item.get("size", 0)),
            "download_url": item.get("@microsoft.graph.downloadUrl"),
            "web_url": item.get("webUrl"),
            "modified_at": item.get("lastModifiedDateTime"),
            "etag": item.get("eTag"),
            "acl_mode": "tenant_public" if acl.tenant_wide else "restricted",
            "allowed_groups": list(acl.access_groups),
            "allowed_user_ids": list(acl.user_ids),
            "anonymous_link": acl.anonymous_link,
            "acl_inherited_from": acl.inherited_from,
            "acl_verified_at": datetime.now(UTC).isoformat(),
        }

    # ------------------------------------------------------------------------------------
    # Permissions
    # ------------------------------------------------------------------------------------

    async def _resolve_acl(self, drive_id: str, item: dict[str, Any]) -> ItemAcl | None:
        """The folder's ACL, unless this item breaks inheritance.

        ``driveItem.shared`` is the signal. Its presence means the item has been shared
        *differently* from its parent, and those are the only items worth an item-level call.
        Without this test the connector makes one Graph call per file and exhausts the
        application's throttling budget for the whole tenant.
        """
        parent_id = str((item.get("parentReference") or {}).get("id") or "")

        if not item.get("shared") and parent_id in self._folder_acls:
            inherited = self._folder_acls[parent_id]
            return ItemAcl(
                group_ids=inherited.group_ids,
                user_ids=inherited.user_ids,
                tenant_wide=inherited.tenant_wide,
                anonymous_link=inherited.anonymous_link,
                inherited_from=parent_id,
            )

        if self.stats.item_permission_reads >= self.max_item_permission_calls:
            # Budget spent. Fall back to the parent's ACL if we have one -- which is the correct
            # value for the overwhelming majority of items -- and mark the sync degraded.
            logger.warning("sharepoint.permission_budget_exhausted", extra={"drive": drive_id})
            return self._folder_acls.get(parent_id)

        try:
            self.stats.item_permission_reads += 1
            payload = await self._get(f"{GRAPH}/drives/{drive_id}/items/{item['id']}/permissions")
        except GraphError as exc:
            logger.warning("sharepoint.permissions_failed", extra={"item": item.get("id"), "error": str(exc)})
            return None

        return parse_permissions(payload.get("value", []))

    async def _cache_folder_acl(self, drive_id: str, folder: dict[str, Any]) -> None:
        try:
            self.stats.folder_permission_reads += 1
            payload = await self._get(f"{GRAPH}/drives/{drive_id}/items/{folder['id']}/permissions")
        except GraphError as exc:
            logger.warning(
                "sharepoint.folder_permissions_failed", extra={"folder": folder.get("id"), "error": str(exc)}
            )
            return
        self._folder_acls[str(folder["id"])] = parse_permissions(payload.get("value", []))

    # ------------------------------------------------------------------------------------

    async def _get(self, url: str) -> dict[str, Any]:
        try:
            response = await self.client.get(url, timeout=30.0)
        except httpx.HTTPError as exc:
            raise GraphError(f"could not reach Microsoft Graph: {exc}") from exc

        if response.status_code == 429:
            # Graph's own Retry-After, not a guess. Guessing here is how an application gets its
            # throttling window extended rather than cleared.
            retry = float(response.headers.get("retry-after", "30"))
            self.stats.throttled_waits += 1
            raise ThrottledError(retry)

        if response.status_code >= 400:
            raise GraphError(f"Microsoft Graph returned {response.status_code}")

        result: dict[str, Any] = response.json()
        return result


def parse_permissions(permissions: Sequence[dict[str, Any]]) -> ItemAcl:
    """Map Graph permission entries onto our ACL vocabulary.

    The group ids come out as Entra object ids because that is what Graph stores and what a login
    token carries. Resolving them to display names here would be actively harmful: names are
    mutable and non-unique, so a renamed group would silently change who can read a document.
    """
    groups: list[str] = []
    users: list[str] = []
    tenant_wide = False
    anonymous = False

    for permission in permissions:
        roles = {str(role).lower() for role in permission.get("roles", [])}
        if roles and not (roles & _READ_ROLES):
            continue

        link = permission.get("link") or {}
        scope = str(link.get("scope", "")).lower()
        if scope == "anonymous":
            anonymous = True
            tenant_wide = True
            continue
        if scope == "organization":
            tenant_wide = True
            continue

        identities = permission.get("grantedToIdentitiesV2") or []
        if permission.get("grantedToV2"):
            identities = [*identities, permission["grantedToV2"]]
        # The legacy singular field is still returned by some endpoints, and an item whose only
        # permission arrives there would otherwise resolve to "nobody can read this".
        if permission.get("grantedTo"):
            identities = [*identities, permission["grantedTo"]]

        for identity in identities:
            group = identity.get("group") or identity.get("siteGroup")
            if group and group.get("id"):
                groups.append(str(group["id"]))
                continue
            user = identity.get("user") or identity.get("siteUser")
            if user and user.get("id"):
                users.append(str(user["id"]))

    return ItemAcl(
        group_ids=tuple(dict.fromkeys(groups)),
        user_ids=tuple(dict.fromkeys(users)),
        tenant_wide=tenant_wide,
        anonymous_link=anonymous,
    )


def _token_of(delta_link: str) -> str:
    _, _, tail = delta_link.partition("token=")
    return tail.split("&")[0] if tail else ""


def describe_freshness(verified_at: datetime | None, *, now: datetime | None = None) -> str:
    """What the UI says about how current a mirrored permission is.

    Shown on every result from a connector-sourced document, because the honest answer to "can my
    colleague see this" is "as of four minutes ago, no" -- and a product that implies otherwise
    is making a promise the sync interval cannot keep.
    """
    if verified_at is None:
        return "Permissions have not been verified"
    delta = (now or datetime.now(UTC)) - verified_at
    minutes = int(delta.total_seconds() // 60)
    if minutes < 1:
        return "Permissions verified just now"
    if minutes < 60:
        return f"Permissions last verified {minutes} minute{'s' if minutes != 1 else ''} ago"
    hours = minutes // 60
    if hours < 24:
        return f"Permissions last verified {hours} hour{'s' if hours != 1 else ''} ago"
    return f"Permissions last verified {hours // 24} day{'s' if hours // 24 != 1 else ''} ago"
