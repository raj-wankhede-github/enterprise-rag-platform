"""The auth routes end to end, against a fake service.

These exist for the properties that only hold at the route level: that every failure looks the
same from outside, that cookies carry the attributes they are supposed to, and that the OIDC
callback never trusts anything it was not handed in a signed state.
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
from app.api.routes import oidc as oidc_routes
from app.auth.discovery import DiscoveryResult, IdpSummary, LoginMethod, TenantLoginPolicy
from app.core.config import Settings
from app.models.identity import IdpKind, IdpState
from app.security.capabilities import Capability, Role
from app.security.tokens import (
    KeySet,
    RefreshTokenReuseError,
    SigningKey,
    TokenError,
    access_cookie_name,
    refresh_cookie_name,
)

TENANT = uuid.UUID("11110000-0000-0000-0000-000000000001")
USER = uuid.UUID("22220000-0000-0000-0000-000000000002")
SESSION = uuid.UUID("33330000-0000-0000-0000-000000000003")
CONFIG = uuid.UUID("44440000-0000-0000-0000-000000000004")


class FakeUser:
    id = USER
    email = "a@acme.com"
    name = "A Person"
    role = Role.DEV
    group_ids = ("eng",)


class FakeSession:
    id = SESSION
    tenant_id = TENANT
    tenant_slug = "acme"
    user = FakeUser()
    refresh_token = "a-refresh-token"
    capabilities = (Capability.SSO_CONFIGURE,)
    access_expires_at = datetime.now(UTC) + timedelta(minutes=10)
    refresh_expires_at = datetime.now(UTC) + timedelta(days=14)


class Outcome:
    def __init__(self, ok: bool, reason: str | None = None) -> None:
        self.ok = ok
        self.reason = reason
        self.user = FakeUser()
        self.tenant_id = TENANT
        self.user_id = USER if ok else None


class FakeAuthService:
    def __init__(self) -> None:
        self.policy: Any = None
        self.outcome = Outcome(True)
        self.attempts: list[dict[str, Any]] = []
        self.rotate_error: Exception | None = None
        self.ended: list[dict[str, Any]] = []

    async def policy_for_domain(self, domain: str, *, tenant_slug: str | None) -> Any:
        return self.policy

    async def authenticate_password(self, *, email: str, password: str, tenant_slug: str | None) -> Any:
        return self.outcome

    async def start_session(self, *, user: Any, request: Any, method: str) -> Any:
        return FakeSession()

    async def rotate_session(self, *, refresh_token: str) -> Any:
        if self.rotate_error:
            raise self.rotate_error
        return FakeSession()

    async def end_session(self, *, refresh_token: str, all_devices: bool = False) -> None:
        self.ended.append({"token": refresh_token, "all": all_devices})

    async def record_attempt(self, **kwargs: Any) -> None:
        self.attempts.append(kwargs)


@pytest.fixture
def service() -> FakeAuthService:
    return FakeAuthService()


@pytest.fixture
def client(service: FakeAuthService) -> TestClient:
    app = FastAPI()
    register_exception_handlers(app)
    app.include_router(auth_routes.router)
    app.include_router(oidc_routes.router)
    app.state.settings = Settings(jwt_secret="x" * 48, public_base_url="https://acme.app.example.com")
    app.state.auth_service = service
    app.state.oidc_service = service
    app.state.signing_keys = KeySet(active=SigningKey(kid="k1", secret="k" * 48))
    # base_url over https so __Host- cookies and the secure path are exercised.
    return TestClient(app, base_url="https://acme.app.example.com")


def a_policy(**kwargs: Any) -> TenantLoginPolicy:
    base: dict[str, Any] = {
        "tenant_id": TENANT,
        "slug": "acme",
        "name": "Acme GmbH",
        "password_login_enabled": True,
    }
    base.update(kwargs)
    return TenantLoginPolicy(**base)


# ------------------------------------------------------------------------------------------
# Discovery
# ------------------------------------------------------------------------------------------


def test_discovery_returns_a_tenants_methods(client: TestClient, service: FakeAuthService) -> None:
    service.policy = a_policy(
        idps=(IdpSummary(id=CONFIG, kind=IdpKind.ENTRA, display_name="", state=IdpState.ACTIVE),)
    )
    body = client.post("/api/auth/discover", json={"email": "a@acme.com"}).json()

    assert body["tenant_slug"] == "acme"
    assert [method["kind"] for method in body["methods"]] == [IdpKind.ENTRA]
    assert str(CONFIG) in body["methods"][0]["start_url"]


def test_an_unknown_domain_is_indistinguishable_from_a_password_tenant(
    client: TestClient, service: FakeAuthService
) -> None:
    """The response an attacker sees must not say which domains are customers."""
    service.policy = None
    unknown = client.post("/api/auth/discover", json={"email": "a@nowhere.example"})

    service.policy = a_policy()
    known = client.post("/api/auth/discover", json={"email": "a@acme.com"})

    assert unknown.status_code == known.status_code == 200
    assert unknown.json()["password_login"] == known.json()["password_login"] is True
    assert unknown.json()["methods"] == known.json()["methods"] == []


def test_discovery_never_reports_whether_a_user_exists(client: TestClient, service: FakeAuthService) -> None:
    service.policy = a_policy()
    body = client.post("/api/auth/discover", json={"email": "nobody@acme.com"}).json()
    assert "user_exists" not in body
    assert not any("user" in key for key in body)


def test_a_malformed_address_is_rejected_by_validation_not_by_a_lookup(client: TestClient) -> None:
    assert client.post("/api/auth/discover", json={"email": "not-an-email"}).status_code == 422


def test_a_tenant_with_sso_only_offers_no_password_field(client: TestClient, service: FakeAuthService) -> None:
    service.policy = a_policy(password_login_enabled=False)
    assert client.post("/api/auth/discover", json={"email": "a@acme.com"}).json()["password_login"] is False


def test_the_break_glass_window_is_surfaced_so_a_tenant_knows_sso_is_not_enforced(
    client: TestClient, service: FakeAuthService
) -> None:
    service.policy = a_policy(
        password_login_enabled=False,
        emergency_password_login_until=datetime.now(UTC) + timedelta(hours=1),
    )
    body = client.post("/api/auth/discover", json={"email": "a@acme.com"}).json()
    assert body["password_login"] and body["emergency_access"]


def test_a_shared_domain_asks_the_user_to_choose(client: TestClient, service: FakeAuthService) -> None:
    service.policy = DiscoveryResult(ambiguous=(("acme", "Acme GmbH"), ("beta", "Beta AG")))
    body = client.post("/api/auth/discover", json={"email": "a@shared.example"}).json()
    assert body["requires_tenant_choice"]
    assert [entry["slug"] for entry in body["tenants"]] == ["acme", "beta"]


# ------------------------------------------------------------------------------------------
# Password login
# ------------------------------------------------------------------------------------------


def test_a_successful_login_sets_all_three_cookies(client: TestClient) -> None:
    response = client.post("/api/auth/login", json={"email": "a@acme.com", "password": "a-password"})

    assert response.status_code == 200
    names = set(response.cookies)
    assert access_cookie_name(secure=True) in names
    assert refresh_cookie_name(secure=True) in names
    assert "erp_csrf" in names


def test_no_token_ever_appears_in_the_response_body(client: TestClient) -> None:
    """A token in a JSON body invites the client to store it where script can read it, which is
    the entire thing HttpOnly cookies avoid."""
    body = client.post("/api/auth/login", json={"email": "a@acme.com", "password": "p"}).json()
    serialized = str(body)
    assert "a-refresh-token" not in serialized
    assert not any("token" in key for key in body)


def test_the_response_carries_the_capabilities_the_frontend_gates_on(client: TestClient) -> None:
    body = client.post("/api/auth/login", json={"email": "a@acme.com", "password": "p"}).json()
    assert body["role"] == Role.DEV
    assert Capability.SSO_CONFIGURE in body["capabilities"]


@pytest.mark.parametrize(
    "reason",
    ["no_such_user", "wrong_password", "deactivated", "sso_only", "password_login_disabled"],
)
def test_every_cause_of_failure_produces_an_identical_response(
    client: TestClient, service: FakeAuthService, reason: str
) -> None:
    """Each distinction is an enumeration oracle. In a multi-tenant product "does this address
    have an account here" leaks the customer list as well as the user list."""
    service.outcome = Outcome(False, reason)
    response = client.post("/api/auth/login", json={"email": "a@acme.com", "password": "p"})

    assert response.status_code == 401
    assert response.json()["message"] == auth_routes.GENERIC_LOGIN_FAILURE
    assert reason not in str(response.json())


def test_the_specific_reason_is_recorded_where_an_administrator_can_see_it(
    client: TestClient, service: FakeAuthService
) -> None:
    service.outcome = Outcome(False, "deactivated")
    client.post("/api/auth/login", json={"email": "a@acme.com", "password": "p"})
    assert service.attempts[-1]["failure_reason"] == "deactivated"
    assert service.attempts[-1]["succeeded"] is False


def test_an_attempt_against_an_account_that_does_not_exist_is_still_recorded(
    client: TestClient, service: FakeAuthService
) -> None:
    """The only record that exists for that case, and exactly the pattern worth detecting."""
    service.outcome = Outcome(False, "no_such_user")
    client.post("/api/auth/login", json={"email": "nobody@nowhere.example", "password": "p"})
    assert service.attempts[-1]["email"] == "nobody@nowhere.example"


def test_a_successful_login_is_recorded_too(client: TestClient, service: FakeAuthService) -> None:
    client.post("/api/auth/login", json={"email": "a@acme.com", "password": "p"})
    assert service.attempts[-1]["succeeded"] is True


def test_the_submitted_address_is_normalised_before_it_is_recorded(
    client: TestClient, service: FakeAuthService
) -> None:
    client.post("/api/auth/login", json={"email": "  A@Acme.COM ", "password": "p"})
    assert service.attempts[-1]["email"] == "a@acme.com"


# ------------------------------------------------------------------------------------------
# Refresh
# ------------------------------------------------------------------------------------------


def test_a_refresh_without_a_cookie_is_an_ended_session_not_a_crash(client: TestClient) -> None:
    assert client.post("/api/auth/refresh").status_code == 401


def test_a_replayed_refresh_token_signs_the_user_out_everywhere(
    client: TestClient, service: FakeAuthService
) -> None:
    """The user-visible half of family revocation. Being told why matters: an unexplained
    sign-out is a support ticket, and this one is worth their attention."""
    client.cookies.set(refresh_cookie_name(secure=True), "stale-token")
    service.rotate_error = RefreshTokenReuseError("replayed")

    response = client.post("/api/auth/refresh")
    assert response.status_code == 401
    assert "all devices" in response.json()["message"]


def test_a_failed_refresh_clears_the_cookies_rather_than_leaving_a_dead_session(
    client: TestClient, service: FakeAuthService
) -> None:
    client.cookies.set(refresh_cookie_name(secure=True), "stale")
    service.rotate_error = TokenError("expired")

    response = client.post("/api/auth/refresh")
    assert response.status_code == 401
    assert 'Max-Age=0' in response.headers.get("set-cookie", "") or "expires=" in response.headers.get(
        "set-cookie", ""
    ).lower()


def test_a_valid_refresh_issues_a_new_pair(client: TestClient) -> None:
    client.cookies.set(refresh_cookie_name(secure=True), "current")
    response = client.post("/api/auth/refresh")
    assert response.status_code == 200
    assert refresh_cookie_name(secure=True) in response.cookies


# ------------------------------------------------------------------------------------------
# Logout
# ------------------------------------------------------------------------------------------


def test_logout_clears_the_cookies_and_ends_the_session(client: TestClient, service: FakeAuthService) -> None:
    client.cookies.set(refresh_cookie_name(secure=True), "a-token")
    response = client.post("/api/auth/logout")

    assert response.status_code == 204
    assert service.ended[-1]["token"] == "a-token"


def test_logging_out_twice_is_not_an_error(client: TestClient) -> None:
    assert client.post("/api/auth/logout").status_code == 204
    assert client.post("/api/auth/logout").status_code == 204


def test_a_failure_to_clean_up_still_clears_the_cookies(client: TestClient, service: FakeAuthService) -> None:
    """A logout that leaves a cookie in place because a database call failed is worse than one
    that leaves a row behind."""

    async def broken(**kwargs: Any) -> None:
        raise RuntimeError("database unavailable")

    service.end_session = broken  # type: ignore[assignment]
    client.cookies.set(refresh_cookie_name(secure=True), "a-token")

    response = client.post("/api/auth/logout")
    assert response.status_code == 204
    assert "erp" in response.headers.get("set-cookie", "")


def test_logout_can_end_every_session_for_the_user(client: TestClient, service: FakeAuthService) -> None:
    client.cookies.set(refresh_cookie_name(secure=True), "a-token")
    client.post("/api/auth/logout?all_devices=true")
    assert service.ended[-1]["all"] is True


# ------------------------------------------------------------------------------------------
# OIDC
# ------------------------------------------------------------------------------------------


def test_a_callback_with_no_state_goes_to_the_login_page_not_to_a_json_error(client: TestClient) -> None:
    """The user arrived by redirect, not by fetch. A JSON body would render as raw text."""
    response = client.get("/api/auth/oidc/callback?code=abc", follow_redirects=False)
    assert response.status_code == 302
    assert response.headers["location"].startswith("/login?error=")


def test_a_provider_that_refused_is_reported_without_a_code_exchange(client: TestClient) -> None:
    response = client.get(
        "/api/auth/oidc/callback?error=access_denied&error_description=consent+declined",
        follow_redirects=False,
    )
    assert response.status_code == 302
    assert "provider_declined" in response.headers["location"]


def test_a_forged_state_is_refused(client: TestClient) -> None:
    response = client.get(
        "/api/auth/oidc/callback?code=abc&state=not-a-signed-state", follow_redirects=False
    )
    assert "expired" in response.headers["location"]


def test_starting_a_flow_for_an_unknown_config_does_not_confirm_which_ids_exist(
    client: TestClient, service: FakeAuthService
) -> None:
    async def no_config(idp_config_id: uuid.UUID) -> Any:
        return None

    service.config = no_config  # type: ignore[assignment]
    assert client.get(f"/api/auth/oidc/{CONFIG}/start", follow_redirects=False).status_code == 404


def test_the_wizards_test_flow_requires_permission(client: TestClient) -> None:
    """A real sign-in against a config that is not yet active, producing a preview that names
    directory groups."""
    assert client.get(f"/api/auth/oidc/{CONFIG}/test", follow_redirects=False).status_code == 401
