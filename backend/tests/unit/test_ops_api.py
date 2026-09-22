"""The operator plane as an application: audience separation, guarding, and the audit trail.

The two tests worth reading are the audience ones. They are what makes "separate app, separate
host, separate audience" a structural claim rather than a description of intent -- a tenant token
must be useless against the operator API, and an operator token useless against the tenant API,
with neither app containing a code path that would accept the other.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.errors import register_exception_handlers
from app.api.routes import auth as auth_routes
from app.core.config import Settings
from app.models.platform import GrantScope, OperatorRole
from app.platform.grants import Grant, Operator, OperatorAction
from app.platform.ops_api import OWNER_ONLY, actions_for_role, create_ops_app
from app.security.tokens import (
    AUDIENCE_PLATFORM,
    AUDIENCE_TENANT,
    KeySet,
    SigningKey,
    issue_access_token,
)

OPERATOR_ID = uuid.UUID("11110000-0000-0000-0000-000000000001")
TENANT = uuid.UUID("22220000-0000-0000-0000-000000000002")
DOCUMENT = uuid.UUID("33330000-0000-0000-0000-000000000003")

KEYS = KeySet(active=SigningKey(kid="k1", secret="k" * 48))


class FakeOpsService:
    def __init__(self) -> None:
        self.operator_row: Operator | None = Operator(
            id=OPERATOR_ID, email="support@vendor.example", role=OperatorRole.SUPPORT, mfa_enrolled=True
        )
        self.grants: list[Grant] = []
        self.audits: list[dict[str, Any]] = []
        self.notifications: list[dict[str, Any]] = []
        self.created_grants: list[dict[str, Any]] = []

    async def operator(self, operator_id: uuid.UUID) -> Operator | None:
        return self.operator_row

    async def grants_for(self, operator_id: uuid.UUID, tenant_id: uuid.UUID) -> list[Grant]:
        return self.grants

    async def tenants(self) -> list[dict[str, Any]]:
        return [
            {
                "id": TENANT,
                "slug": "acme",
                "name": "Acme GmbH",
                "status": "active",
                "created_at": datetime.now(UTC),
                "document_count": 412,
                "user_count": 37,
            }
        ]

    async def tenant_metadata(self, tenant_id: uuid.UUID) -> dict[str, Any] | None:
        return {"id": str(tenant_id), "slug": "acme", "document_count": 412}

    async def jobs(self, tenant_id: uuid.UUID) -> list[dict[str, Any]]:
        return [{"id": str(uuid.uuid4()), "state": "FAILED", "stage": "PARSED", "error_class": "ParserUnavailable"}]

    async def request_grant(self, **kwargs: Any) -> dict[str, Any]:
        self.created_grants.append(kwargs)
        return {"id": str(uuid.uuid4()), **{k: str(v) for k, v in kwargs.items()}}

    async def audit(self, **kwargs: Any) -> None:
        self.audits.append(kwargs)

    async def notify_tenant_admins(self, tenant_id: uuid.UUID, *, subject: str, body: str) -> None:
        self.notifications.append({"tenant_id": tenant_id, "subject": subject, "body": body})


@pytest.fixture
def service() -> FakeOpsService:
    return FakeOpsService()


@pytest.fixture
def client(service: FakeOpsService) -> TestClient:
    app = create_ops_app(Settings(jwt_secret="x" * 48))
    app.state.ops_service = service
    app.state.signing_keys = KEYS
    return TestClient(app, base_url="https://ops.vendor.example")


def operator_token(role: str = OperatorRole.SUPPORT) -> str:
    return issue_access_token(
        KEYS,
        user_id=OPERATOR_ID,
        tenant_id=uuid.UUID(int=0),
        session_id=uuid.uuid4(),
        role=role,
        audience=AUDIENCE_PLATFORM,
    )


def production_settings() -> Settings:
    """A configuration the production validator accepts.

    Assembling it here rather than passing `environment="production"` alone is deliberate: the
    validator refuses a production deployment without TLS and without Redis, and a test that
    worked around that would be testing a configuration the product will not run in.
    """
    return Settings(
        jwt_secret="x" * 48,
        environment="production",
        cookie_secure=True,
        opensearch_url="https://opensearch.internal:9200",
        redis_url="redis://redis.internal:6379/0",
    )


def auth(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def a_grant(scope: str = GrantScope.METADATA, **kwargs: Any) -> Grant:
    base: dict[str, Any] = {
        "id": uuid.uuid4(),
        "tenant_id": TENANT,
        "operator_id": OPERATOR_ID,
        "scope": scope,
        "expires_at": datetime.now(UTC) + timedelta(hours=4),
    }
    base.update(kwargs)
    return Grant(**base)


# ------------------------------------------------------------------------------------------
# Audience separation
# ------------------------------------------------------------------------------------------


def test_a_tenant_token_is_useless_against_the_operator_api(client: TestClient) -> None:
    """The structural half of "separate plane". A customer's own token -- validly signed, by the
    same key -- authenticates nothing here."""
    tenant_token = issue_access_token(
        KEYS,
        user_id=uuid.uuid4(),
        tenant_id=TENANT,
        session_id=uuid.uuid4(),
        role="ADMIN",
        audience=AUDIENCE_TENANT,
    )
    assert client.get("/ops/tenants", headers=auth(tenant_token)).status_code == 401


def test_an_operator_token_is_useless_against_the_tenant_api() -> None:
    """The other direction, which matters just as much: a leaked operator token must not
    authenticate against a customer's API."""
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_routes.router)
    app.state.settings = Settings(jwt_secret="x" * 48)
    app.state.signing_keys = KEYS

    with TestClient(app, base_url="https://acme.app.example.com") as tenant_api:
        response = tenant_api.get("/api/auth/me", headers=auth(operator_token()))
    assert response.status_code == 401


def test_the_operator_api_is_not_mounted_on_the_tenant_app() -> None:
    """A routing mistake must not be able to put an operator endpoint on a customer's domain."""
    from app.main import create_app

    paths = {getattr(route, "path", "") for route in create_app(Settings(jwt_secret="x" * 48)).routes}
    assert paths, "the tenant app registered no routes, so this assertion would pass vacuously"
    assert not any(path.startswith("/ops") for path in paths)


def test_no_authorization_header_is_refused(client: TestClient) -> None:
    assert client.get("/ops/tenants").status_code == 401


def test_a_deactivated_operator_is_refused_at_the_door(client: TestClient, service: FakeOpsService) -> None:
    """Reloaded from the database, not read from the token -- so revoking access takes effect
    immediately rather than at the next token expiry."""
    service.operator_row = Operator(id=OPERATOR_ID, email="x@y", role=OperatorRole.OWNER, is_active=False)
    assert client.get("/ops/tenants", headers=auth(operator_token())).status_code == 401


# ------------------------------------------------------------------------------------------
# Reading without a grant
# ------------------------------------------------------------------------------------------


def test_listing_tenants_works_without_a_grant(client: TestClient) -> None:
    response = client.get("/ops/tenants", headers=auth(operator_token()))
    assert response.status_code == 200
    assert response.json()[0]["slug"] == "acme"


def test_the_tenant_list_carries_counts_and_no_content(client: TestClient) -> None:
    payload = client.get("/ops/tenants", headers=auth(operator_token())).json()[0]
    assert payload["document_count"] == 412
    assert not any(key in payload for key in ("documents", "content", "queries"))


def test_reading_a_document_without_a_grant_is_refused(client: TestClient, service: FakeOpsService) -> None:
    """The acceptance criterion, at the route level."""
    service.grants = []
    response = client.get(f"/ops/tenants/{TENANT}/documents/{DOCUMENT}", headers=auth(operator_token()))

    assert response.status_code == 403
    assert "grant" in response.json()["message"]


def test_a_metadata_grant_does_not_unlock_a_document(client: TestClient, service: FakeOpsService) -> None:
    service.grants = [a_grant(GrantScope.METADATA)]
    response = client.get(f"/ops/tenants/{TENANT}/documents/{DOCUMENT}", headers=auth(operator_token()))
    assert response.status_code == 403


def test_job_states_are_readable_with_a_metadata_grant(client: TestClient, service: FakeOpsService) -> None:
    service.grants = [a_grant(GrantScope.METADATA)]
    response = client.get(f"/ops/tenants/{TENANT}/jobs", headers=auth(operator_token()))

    assert response.status_code == 200
    assert response.json()[0]["state"] == "FAILED"


def test_job_states_carry_an_error_class_rather_than_the_document(client: TestClient, service: FakeOpsService) -> None:
    """Enough to answer the ticket; nothing the customer wrote."""
    service.grants = [a_grant(GrantScope.METADATA)]
    job = client.get(f"/ops/tenants/{TENANT}/jobs", headers=auth(operator_token())).json()[0]
    assert job["error_class"] == "ParserUnavailable"
    assert "content" not in job and "text" not in job


# ------------------------------------------------------------------------------------------
# Audit
# ------------------------------------------------------------------------------------------


def test_every_action_is_audited(client: TestClient, service: FakeOpsService) -> None:
    client.get("/ops/tenants", headers=auth(operator_token()))
    assert service.audits[-1]["action"] == OperatorAction.LIST_TENANTS
    assert service.audits[-1]["operator_email"] == "support@vendor.example"


def test_a_refusal_is_audited_too(client: TestClient, service: FakeOpsService) -> None:
    """An operator repeatedly attempting content access they do not have is the signal worth
    alerting on, and it is invisible if only successes are recorded."""
    service.grants = []
    client.get(f"/ops/tenants/{TENANT}/documents/{DOCUMENT}", headers=auth(operator_token()))

    entry = service.audits[-1]
    assert entry["details"]["allowed"] is False
    assert entry["action"] == OperatorAction.VIEW_DOCUMENT_CONTENT


def test_a_refused_content_read_is_not_recorded_as_having_touched_content(
    client: TestClient, service: FakeOpsService
) -> None:
    """`touched_content` is the field a customer's security review filters on. A refusal that
    marked it true would make every review start with a false alarm."""
    service.grants = []
    client.get(f"/ops/tenants/{TENANT}/documents/{DOCUMENT}", headers=auth(operator_token()))
    assert service.audits[-1]["touched_content"] is False


def test_metadata_access_is_not_recorded_as_touching_content(client: TestClient, service: FakeOpsService) -> None:
    service.grants = [a_grant(GrantScope.METADATA)]
    client.get(f"/ops/tenants/{TENANT}/jobs", headers=auth(operator_token()))
    assert service.audits[-1]["touched_content"] is False


def test_the_audit_row_names_the_grant_that_permitted_it(client: TestClient, service: FakeOpsService) -> None:
    grant = a_grant(GrantScope.METADATA)
    service.grants = [grant]
    client.get(f"/ops/tenants/{TENANT}/jobs", headers=auth(operator_token()))
    assert service.audits[-1]["grant_id"] == grant.id


# ------------------------------------------------------------------------------------------
# Break-glass
# ------------------------------------------------------------------------------------------


def test_support_cannot_take_break_glass_access(client: TestClient) -> None:
    response = client.post(
        "/ops/grants",
        headers=auth(operator_token(OperatorRole.SUPPORT)),
        json={"tenant_id": str(TENANT), "reason": "customer locked out", "break_glass": True},
    )
    assert response.status_code == 403


def test_an_owner_can_take_break_glass_access(client: TestClient, service: FakeOpsService) -> None:
    """Legitimate: a tenant whose only administrator is locked out cannot approve anything, and
    refusing to help them is not a security posture."""
    service.operator_row = Operator(
        id=OPERATOR_ID, email="owner@vendor.example", role=OperatorRole.OWNER, mfa_enrolled=True
    )
    response = client.post(
        "/ops/grants",
        headers=auth(operator_token(OperatorRole.OWNER)),
        json={"tenant_id": str(TENANT), "reason": "sole admin locked out by SSO misconfiguration", "break_glass": True},
    )
    assert response.status_code == 200
    assert service.created_grants[-1]["is_break_glass"] is True


def test_break_glass_notifies_the_tenant_immediately(client: TestClient, service: FakeOpsService) -> None:
    """A customer finding out tomorrow that the vendor read their data today is the incident,
    regardless of whether the access was justified."""
    service.operator_row = Operator(
        id=OPERATOR_ID, email="owner@vendor.example", role=OperatorRole.OWNER, mfa_enrolled=True
    )
    client.post(
        "/ops/grants",
        headers=auth(operator_token(OperatorRole.OWNER)),
        json={"tenant_id": str(TENANT), "reason": "sole admin locked out", "break_glass": True},
    )

    notice = service.notifications[-1]
    assert notice["tenant_id"] == TENANT
    assert "owner@vendor.example" in notice["body"]
    assert "revoke" in notice["body"].lower()


def test_an_ordinary_grant_request_notifies_nobody(client: TestClient, service: FakeOpsService) -> None:
    """It is a request the customer will approve, not an access already taken."""
    client.post(
        "/ops/grants",
        headers=auth(operator_token()),
        json={"tenant_id": str(TENANT), "reason": "investigating ticket 4412"},
    )
    assert service.notifications == []


def test_a_grant_request_must_explain_itself(client: TestClient) -> None:
    """A grant nobody can explain later is one nobody can review."""
    response = client.post(
        "/ops/grants", headers=auth(operator_token()), json={"tenant_id": str(TENANT), "reason": "x"}
    )
    assert response.status_code == 422


def test_a_grant_cannot_be_requested_for_longer_than_three_days(client: TestClient) -> None:
    response = client.post(
        "/ops/grants",
        headers=auth(operator_token()),
        json={"tenant_id": str(TENANT), "reason": "long investigation", "hours": 500},
    )
    assert response.status_code == 422


# ------------------------------------------------------------------------------------------
# Hygiene
# ------------------------------------------------------------------------------------------


def test_operator_responses_are_never_cached(client: TestClient) -> None:
    """Anything here may contain customer data. An intermediate cache holding it is a copy
    nobody knows about and nobody can revoke a grant from."""
    response = client.get("/ops/tenants", headers=auth(operator_token()))
    assert "no-store" in response.headers["cache-control"]


def test_the_operator_api_publishes_no_schema_in_production() -> None:
    """Publishing its shape tells an attacker exactly what to look for on the operator host."""
    app = create_ops_app(production_settings())
    assert app.docs_url is None
    assert app.openapi_url is None


def test_the_owner_only_set_matches_what_the_rules_enforce() -> None:
    support = actions_for_role(OperatorRole.SUPPORT)
    engineer = actions_for_role(OperatorRole.ENGINEER)
    for action in OWNER_ONLY:
        assert action not in support
        assert action not in engineer
    assert actions_for_role(OperatorRole.OWNER) >= OWNER_ONLY


def test_the_actions_endpoint_states_the_cost_of_each(client: TestClient) -> None:
    rows = client.get("/ops/actions", headers=auth(operator_token())).json()
    by_action = {row["action"]: row["requirement"] for row in rows}
    assert "own documents" in by_action[OperatorAction.VIEW_DOCUMENT_CONTENT]
    assert len(by_action) == len(list(OperatorAction))
