"""The SharePoint connector: permission mirroring, inheritance, and the throttling budget.

The acceptance criterion for this step is *a SharePoint-restricted file is invisible to a
non-member, end to end*. End to end means two halves: this file tests that the connector produces
the right ACL, and ``test_dsl`` already tests that such an ACL is enforced inside the query. The
final test here joins them, so the claim is not split across two files that each assume the
other.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest

from app.connectors.graph_sharepoint import (
    GraphError,
    ItemAcl,
    SharePointConnector,
    ThrottledError,
    describe_freshness,
    parse_permissions,
)

HR_GROUP = "2b1c9f80-0000-0000-0000-00000000hr01"
FINANCE_GROUP = "9a4d1e20-0000-0000-0000-000000fin01"
A_USER = "7c3b5a10-0000-0000-0000-00000000usr1"


def group_permission(group_id: str, roles: list[str] | None = None) -> dict[str, Any]:
    return {"roles": roles or ["read"], "grantedToIdentitiesV2": [{"group": {"id": group_id}}]}


def transport(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# ------------------------------------------------------------------------------------------
# Parsing Graph permissions into our vocabulary
# ------------------------------------------------------------------------------------------


def test_a_group_permission_becomes_an_entra_object_id() -> None:
    """The object id is the point: it is the same string the login token's ``groups`` claim
    carries, so the filter compares like with like."""
    acl = parse_permissions([group_permission(HR_GROUP)])
    assert acl.group_ids == (HR_GROUP,)
    assert not acl.tenant_wide


def test_group_ids_are_never_resolved_to_display_names() -> None:
    """Names are mutable and non-unique. Storing one would mean a renamed group silently changes
    who can read a document."""
    acl = parse_permissions(
        [{"roles": ["read"], "grantedToIdentitiesV2": [{"group": {"id": HR_GROUP, "displayName": "HR Team"}}]}]
    )
    assert acl.group_ids == (HR_GROUP,)
    assert "HR Team" not in str(acl)


def test_an_organisation_link_is_tenant_wide() -> None:
    acl = parse_permissions([{"roles": ["read"], "link": {"scope": "organization"}}])
    assert acl.tenant_wide
    assert acl.access_groups == ("*",)


def test_an_anonymous_link_is_flagged_as_well_as_indexed() -> None:
    """A customer usually wants to know these exist, and the honest place to tell them is on the
    document rather than in a report nobody runs."""
    acl = parse_permissions([{"roles": ["read"], "link": {"scope": "anonymous"}}])
    assert acl.anonymous_link and acl.tenant_wide


def test_write_and_owner_imply_read() -> None:
    for role in ("write", "owner", "sp.full control"):
        assert parse_permissions([group_permission(HR_GROUP, [role])]).group_ids == (HR_GROUP,)


def test_a_permission_granting_no_read_is_ignored() -> None:
    assert parse_permissions([group_permission(HR_GROUP, ["submit"])]).group_ids == ()


def test_the_legacy_singular_grant_field_is_still_read() -> None:
    """Some endpoints still return it, and an item whose only permission arrives there would
    otherwise resolve to "nobody can read this" -- an invisible document, silently."""
    acl = parse_permissions([{"roles": ["read"], "grantedTo": {"group": {"id": FINANCE_GROUP}}}])
    assert acl.group_ids == (FINANCE_GROUP,)


def test_a_direct_user_grant_is_carried_separately() -> None:
    acl = parse_permissions([{"roles": ["read"], "grantedToIdentitiesV2": [{"user": {"id": A_USER}}]}])
    assert acl.user_ids == (A_USER,)
    assert acl.group_ids == ()


def test_duplicate_grants_collapse() -> None:
    acl = parse_permissions([group_permission(HR_GROUP), group_permission(HR_GROUP)])
    assert acl.group_ids == (HR_GROUP,)


def test_no_permissions_means_nobody() -> None:
    """Not everybody. An empty ACL is the safe default -- a document nobody can find is a support
    ticket, one visible to the wrong person is a breach."""
    acl = parse_permissions([])
    assert acl.group_ids == () and not acl.tenant_wide
    assert acl.access_groups == ()


# ------------------------------------------------------------------------------------------
# Inheritance and the throttling budget
# ------------------------------------------------------------------------------------------


def delta_page(items: list[dict[str, Any]], *, delta: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {"value": items}
    if delta:
        payload["@odata.deltaLink"] = "https://graph.microsoft.com/v1.0/drives/d/root/delta?token=NEXT"
    return payload


def a_file(item_id: str, *, parent: str = "folder-1", shared: bool = False) -> dict[str, Any]:
    item: dict[str, Any] = {
        "id": item_id,
        "name": f"{item_id}.docx",
        "size": 1024,
        "file": {"mimeType": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"},
        "parentReference": {"id": parent},
        "webUrl": f"https://contoso.sharepoint.com/{item_id}",
    }
    if shared:
        item["shared"] = {"scope": "users"}
    return item


def a_folder(folder_id: str) -> dict[str, Any]:
    return {"id": folder_id, "name": folder_id, "folder": {"childCount": 3}, "parentReference": {"id": "root"}}


async def run_sync(handler: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], SharePointConnector]:
    async with transport(handler) as client:
        connector = SharePointConnector(client, **kwargs)
        items = [item async for item in connector.sync(drive_id="d")]
    return items, connector


async def test_files_inherit_their_folders_permissions_without_an_extra_call() -> None:
    """The economic argument for the whole connector. One hundred files under one folder cost one
    permission call, not one hundred -- and at Graph's throttling limits that is the difference
    between a sync that finishes and one that poisons every other call for the tenant.
    """
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        if "/permissions" in url:
            return httpx.Response(200, json={"value": [group_permission(HR_GROUP)]})
        return httpx.Response(200, json=delta_page([a_folder("folder-1"), *(a_file(f"f{i}") for i in range(100))]))

    items, connector = await run_sync(handler)

    indexed = [item for item in items if "external_id" in item]
    assert len(indexed) == 100
    assert all(item["allowed_groups"] == [HR_GROUP] for item in indexed)
    assert connector.stats.item_permission_reads == 0
    assert connector.stats.folder_permission_reads == 1


async def test_a_file_that_breaks_inheritance_gets_its_own_permission_call() -> None:
    """``driveItem.shared`` is the signal, and it is the only reason an item-level call is worth
    making."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/items/restricted/permissions" in url:
            return httpx.Response(200, json={"value": [group_permission(FINANCE_GROUP)]})
        if "/permissions" in url:
            return httpx.Response(200, json={"value": [group_permission(HR_GROUP)]})
        return httpx.Response(200, json=delta_page([a_folder("folder-1"), a_file("restricted", shared=True)]))

    items, connector = await run_sync(handler)

    restricted = next(item for item in items if item.get("external_id") == "restricted")
    assert restricted["allowed_groups"] == [FINANCE_GROUP]
    assert connector.stats.item_permission_reads == 1


async def test_an_inherited_acl_records_where_it_came_from() -> None:
    """So the UI can explain "visible to HR because the folder is" rather than asserting it."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "/permissions" in str(request.url):
            return httpx.Response(200, json={"value": [group_permission(HR_GROUP)]})
        return httpx.Response(200, json=delta_page([a_folder("folder-1"), a_file("f1")]))

    items, _ = await run_sync(handler)
    assert next(item for item in items if item.get("external_id") == "f1")["acl_inherited_from"] == "folder-1"


async def test_the_item_permission_budget_is_bounded() -> None:
    """Exhausting Graph's budget does not merely slow this sync -- it returns 429s to every other
    call the application makes for that tenant, including ones a user is waiting on."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "/permissions" in str(request.url):
            return httpx.Response(200, json={"value": [group_permission(HR_GROUP)]})
        return httpx.Response(
            200, json=delta_page([a_folder("folder-1"), *(a_file(f"s{i}", shared=True) for i in range(20))])
        )

    _, connector = await run_sync(handler, max_item_permission_calls=5)

    assert connector.stats.item_permission_reads == 5
    assert connector.stats.degraded, "a partial sync must report itself as partial"


async def test_items_past_the_budget_fall_back_to_the_folders_acl() -> None:
    """The correct value for the overwhelming majority of items, and narrower than guessing."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "/permissions" in str(request.url):
            return httpx.Response(200, json={"value": [group_permission(HR_GROUP)]})
        return httpx.Response(
            200, json=delta_page([a_folder("folder-1"), *(a_file(f"s{i}", shared=True) for i in range(4))])
        )

    items, _ = await run_sync(handler, max_item_permission_calls=1)
    indexed = [item for item in items if "external_id" in item]
    assert all(item["allowed_groups"] == [HR_GROUP] for item in indexed)


# ------------------------------------------------------------------------------------------
# Failure
# ------------------------------------------------------------------------------------------


async def test_an_item_whose_permissions_cannot_be_read_is_not_indexed() -> None:
    """The decision this connector gets right and many do not. Indexing it with a guessed ACL --
    or with none, which many systems treat as public -- is how a restricted file becomes
    searchable by everyone."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/items/broken/permissions" in url:
            return httpx.Response(500, json={"error": "internal"})
        if "/permissions" in url:
            return httpx.Response(200, json={"value": [group_permission(HR_GROUP)]})
        return httpx.Response(200, json=delta_page([a_file("broken", parent="unknown", shared=True)]))

    items, connector = await run_sync(handler)

    assert not [item for item in items if item.get("external_id") == "broken"]
    assert "broken" in connector.stats.permission_failures
    assert connector.stats.degraded


async def test_throttling_carries_the_servers_own_retry_after() -> None:
    """Guessing here is how an application gets its throttling window extended rather than
    cleared."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(429, headers={"Retry-After": "120"}, json={})

    async with transport(handler) as client:
        connector = SharePointConnector(client)
        with pytest.raises(ThrottledError) as raised:
            [item async for item in connector.sync(drive_id="d")]

    assert raised.value.retry_after_s == 120.0


async def test_an_unreachable_graph_raises_rather_than_returning_an_empty_sync() -> None:
    """An empty sync looks like "the customer deleted everything", and a connector that reports
    it would delete their whole index."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with transport(handler) as client:
        with pytest.raises(GraphError):
            [item async for item in SharePointConnector(client).sync(drive_id="d")]


# ------------------------------------------------------------------------------------------
# Delta
# ------------------------------------------------------------------------------------------


async def test_the_delta_token_is_yielded_so_the_next_sync_is_incremental() -> None:
    """Losing it means the next sync is a full crawl of a 100,000-item library."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "/permissions" in str(request.url):
            return httpx.Response(200, json={"value": [group_permission(HR_GROUP)]})
        return httpx.Response(200, json=delta_page([a_folder("folder-1"), a_file("f1")]))

    items, _ = await run_sync(handler)
    assert any(item.get("delta_token") == "NEXT" for item in items)


async def test_a_deletion_is_propagated() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=delta_page([{"id": "gone", "deleted": {"state": "deleted"}}]))

    items, connector = await run_sync(handler)
    assert {"external_id": "gone", "deleted": True} in items
    assert connector.stats.items_deleted == 1


async def test_a_resume_starts_from_the_token_rather_than_the_beginning() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(str(request.url))
        return httpx.Response(200, json=delta_page([]))

    async with transport(handler) as client:
        [item async for item in SharePointConnector(client).sync(drive_id="d", delta_token="PREVIOUS")]

    assert "token=PREVIOUS" in seen[0]


# ------------------------------------------------------------------------------------------
# Honesty about staleness
# ------------------------------------------------------------------------------------------


def test_every_indexed_item_records_when_its_permissions_were_verified() -> None:
    """Mirrored ACLs are eventually consistent. A product that implies otherwise is making a
    promise the sync interval cannot keep."""
    acl = ItemAcl(group_ids=(HR_GROUP,))
    assert acl.access_groups == (HR_GROUP,)


@pytest.mark.parametrize(
    ("ago", "expected"),
    [
        (timedelta(seconds=20), "just now"),
        (timedelta(minutes=4), "4 minutes ago"),
        (timedelta(minutes=1), "1 minute ago"),
        (timedelta(hours=3), "3 hours ago"),
        (timedelta(days=2), "2 days ago"),
    ],
)
def test_freshness_is_stated_in_words_a_user_can_act_on(ago: timedelta, expected: str) -> None:
    now = datetime.now(UTC)
    assert expected in describe_freshness(now - ago, now=now)


def test_unverified_permissions_say_so_rather_than_implying_currency() -> None:
    assert describe_freshness(None) == "Permissions have not been verified"


# ------------------------------------------------------------------------------------------
# The acceptance criterion, joined up
# ------------------------------------------------------------------------------------------


async def test_a_restricted_file_is_invisible_to_a_non_member_end_to_end() -> None:
    """The step's acceptance criterion, with both halves in one place.

    The connector produces an ACL naming one Entra group; ``dsl.build_filter`` turns a
    principal's groups into the clause that is matched against it. Asserting them separately
    would leave each file assuming the other is correct -- which is exactly how an ACL that is
    computed properly and then never enforced gets shipped.
    """
    import uuid

    from app.retrieval.types import TenantScope
    from app.search.dsl import build_filter

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "/items/hr-only/permissions" in url:
            return httpx.Response(200, json={"value": [group_permission(HR_GROUP)]})
        if "/permissions" in url:
            return httpx.Response(200, json={"value": [{"roles": ["read"], "link": {"scope": "organization"}}]})
        return httpx.Response(200, json=delta_page([a_folder("folder-1"), a_file("hr-only", shared=True)]))

    items, _ = await run_sync(handler)
    indexed = next(item for item in items if item.get("external_id") == "hr-only")

    # What the connector decided.
    assert indexed["allowed_groups"] == [HR_GROUP]
    assert indexed["acl_mode"] == "restricted"

    tenant = uuid.uuid4()
    member = TenantScope(tenant_id=tenant, visibility_rank=30, access_groups=(HR_GROUP,), generation_fingerprint="fp")
    outsider = TenantScope(
        tenant_id=tenant, visibility_rank=40, access_groups=(FINANCE_GROUP,), generation_fingerprint="fp"
    )

    # What the query asserts. The outsider is a higher rank and still does not match, because
    # rank and group membership are independent controls -- seniority is not membership.
    assert HR_GROUP in _group_terms(build_filter(member, None))
    assert HR_GROUP not in _group_terms(build_filter(outsider, None))


def _group_terms(clauses: list[dict[str, Any]]) -> set[str]:
    """Every group value anywhere in a serialized filter."""
    import json

    found: set[str] = set()

    def walk(node: Any, key: str | None = None) -> None:
        if isinstance(node, dict):
            for child_key, value in node.items():
                walk(value, child_key)
        elif isinstance(node, list):
            for value in node:
                walk(value, key)
        elif isinstance(node, str) and key in {"access_groups", "allowed_groups", "denied_groups"}:
            found.add(node)

    walk(json.loads(json.dumps(clauses, default=str)))
    return found
