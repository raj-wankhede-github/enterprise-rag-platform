"""Authentication routes: discovery, password login, refresh, logout.

The OIDC half lives in ``routes/oidc.py``. This module holds the parts that do not involve an
external provider, plus the session-issuing helper both share.

Three behaviours are enforced here rather than left to the caller:

**The server decides whether password login is allowed**, independently of what discovery
returned. Discovery hides the field; this refuses the request. A client that posts straight to
this endpoint gets the same answer as one that read the hint.

**Every failure looks the same.** Unknown address, wrong password, deactivated account,
SSO-only account: one message, one status, one response time. Each distinction is an oracle, and
in a multi-tenant product "does this address have an account here" leaks the customer list as
well as the user list.

**A login always writes a ``login_attempts`` row**, successful or not. It is the only record
that exists for an attempt against an account that does not exist, which is exactly the pattern
worth detecting.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Request, Response, status
from pydantic import BaseModel, EmailStr, Field

from app.auth.discovery import (
    DiscoveryResult,
    build_result,
    domain_of,
    normalize_email,
)
from app.core.errors import AuthenticationError, ValidationError
from app.security import csrf as csrf_module
from app.security.tokens import (
    AUDIENCE_TENANT,
    KeySet,
    RefreshTokenReuseError,
    TokenError,
    access_cookie_name,
    cookie_settings,
    issue_access_token,
    refresh_cookie_name,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth", tags=["auth"])

#: The one message every failed password login returns. Never "no such user", never "wrong
#: password", never "this account uses SSO" -- each is an enumeration oracle.
GENERIC_LOGIN_FAILURE = "That email address and password combination was not recognised."

#: A floor on how long a discovery or login response takes, so a hit and a miss cannot be told
#: apart by timing. The database work differs measurably between them; this hides that.
MIN_AUTH_RESPONSE_S = 0.12


class DiscoverRequest(BaseModel):
    email: EmailStr
    #: Set when the user has already chosen among several tenants claiming their domain.
    tenant_slug: str | None = Field(default=None, max_length=63)


class LoginMethodOut(BaseModel):
    kind: str
    display_name: str
    start_url: str


class DiscoverResponse(BaseModel):
    """Describes a *tenant's* policy. Nothing here depends on whether the user exists."""

    tenant_slug: str | None = None
    tenant_name: str | None = None
    password_login: bool = True
    methods: list[LoginMethodOut] = Field(default_factory=list)
    requires_tenant_choice: bool = False
    tenants: list[dict[str, str]] = Field(default_factory=list)
    emergency_access: bool = False

    @classmethod
    def of(cls, result: DiscoveryResult) -> DiscoverResponse:
        return cls(
            tenant_slug=result.tenant_slug,
            tenant_name=result.tenant_name,
            password_login=result.password_login,
            methods=[
                LoginMethodOut(kind=m.kind, display_name=m.display_name, start_url=m.start_url) for m in result.methods
            ],
            requires_tenant_choice=result.requires_tenant_choice,
            tenants=[{"slug": slug, "name": name} for slug, name in result.ambiguous],
            emergency_access=result.emergency_access,
        )


class LoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=1, max_length=1024)
    tenant_slug: str | None = Field(default=None, max_length=63)


class SessionResponse(BaseModel):
    """What the client learns after a successful login.

    No tokens in the body: they are set as cookies. A token in a JSON body invites the client to
    store it somewhere script can read, which is the whole thing HttpOnly cookies avoid.
    """

    user_id: uuid.UUID
    tenant_slug: str
    email: str
    display_name: str
    role: str
    capabilities: list[str]
    expires_at: datetime


# ----------------------------------------------------------------------------------------------


class AuthService:
    """Everything these routes need from the database.

    A protocol-shaped seam rather than direct session use, so the route logic -- which is where
    the enumeration and CSRF properties live -- is testable without a database.
    """

    async def policy_for_domain(self, domain: str, *, tenant_slug: str | None) -> Any: ...
    async def authenticate_password(self, *, email: str, password: str, tenant_slug: str | None) -> Any: ...
    async def start_session(self, *, user: Any, request: Request, method: str) -> Any: ...
    async def rotate_session(self, *, refresh_token: str) -> Any: ...
    async def end_session(self, *, refresh_token: str, all_devices: bool = False) -> None: ...
    async def record_attempt(self, **kwargs: Any) -> None: ...


def get_auth_service(request: Request) -> AuthService:
    service = getattr(request.app.state, "auth_service", None)
    if service is None:  # pragma: no cover - wiring error, not a runtime path
        raise RuntimeError("auth_service is not configured on the application")
    return service  # type: ignore[no-any-return]


def get_keys(request: Request) -> KeySet:
    return request.app.state.signing_keys  # type: ignore[no-any-return]


ServiceDep = Annotated[AuthService, Depends(get_auth_service)]
KeysDep = Annotated[KeySet, Depends(get_keys)]


# ----------------------------------------------------------------------------------------------


@router.post("/discover", response_model=DiscoverResponse)
async def discover(body: DiscoverRequest, service: ServiceDep) -> DiscoverResponse:
    """Which tenant this address belongs to, and how that tenant allows signing in.

    Deliberately unauthenticated and deliberately uninformative about users. It is rate-limited
    per IP by the middleware, because it is the cheapest endpoint from which to probe which
    domains are customers.
    """
    started = time.perf_counter()
    domain = domain_of(body.email)

    result = DiscoveryResult(password_login=True)
    if domain:
        policy = await service.policy_for_domain(domain, tenant_slug=body.tenant_slug)
        result = policy if isinstance(policy, DiscoveryResult) else build_result(policy)

    await _pad_to_floor(started)
    return DiscoverResponse.of(result)


@router.post("/login", response_model=SessionResponse)
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    service: ServiceDep,
    keys: KeysDep,
) -> SessionResponse:
    """Password login. The server, not the discovery response, decides whether it is allowed."""
    started = time.perf_counter()
    email = normalize_email(body.email)

    outcome = await service.authenticate_password(email=email, password=body.password, tenant_slug=body.tenant_slug)

    await service.record_attempt(
        email=email,
        tenant_id=getattr(outcome, "tenant_id", None),
        user_id=getattr(outcome, "user_id", None),
        method="password",
        succeeded=bool(getattr(outcome, "ok", False)),
        failure_reason=getattr(outcome, "reason", None),
        ip_address=_client_ip(request),
        user_agent=request.headers.get("user-agent"),
    )

    if not getattr(outcome, "ok", False):
        await _pad_to_floor(started)
        # One message for every cause. The specific reason is on the login_attempts row, where
        # an administrator can see it and an attacker cannot.
        raise AuthenticationError(GENERIC_LOGIN_FAILURE, detail=str(getattr(outcome, "reason", "")))

    session = await service.start_session(user=outcome.user, request=request, method="password")
    _set_session_cookies(response, session=session, keys=keys, request=request)
    await _pad_to_floor(started)

    return _session_response(outcome.user, session)


@router.get("/me", response_model=SessionResponse)
async def me(request: Request) -> SessionResponse:
    """The current principal, for the frontend to render against.

    Deliberately not ``/refresh``. A page load that rotated the refresh token would burn one
    rotation per load, and two tabs opened at once would each present a token the other had
    already rotated -- which the backend correctly reads as replay and answers by revoking the
    session family. The user would be signed out of everything by opening a second tab.

    The principal is the one the middleware loaded from Postgres this request, not the token's
    claims, so a demotion or deactivation is reflected here within the principal cache window.
    """
    principal = getattr(request.state, "principal", None)
    if principal is None:
        raise AuthenticationError("Your session has ended. Please sign in again.")

    return SessionResponse(
        user_id=principal.user_id or uuid.UUID(int=0),
        tenant_slug=getattr(request.state, "tenant_slug", ""),
        email=principal.email,
        display_name=principal.display_name,
        role=str(principal.role),
        capabilities=sorted(str(capability) for capability in principal.capabilities),
        expires_at=principal.expires_at or datetime.now(UTC) + timedelta(minutes=10),
    )


@router.post("/refresh", response_model=SessionResponse)
async def refresh(request: Request, response: Response, service: ServiceDep, keys: KeysDep) -> SessionResponse:
    """Exchange the refresh cookie for a new pair.

    A replayed token revokes the whole family, which signs the user out everywhere. That is the
    intended behaviour: the token was either stolen or raced, and from here the two cannot be
    told apart.
    """
    secure = _is_secure(request)
    stale = _session_cookie_names(secure=secure)
    presented = request.cookies.get(refresh_cookie_name(secure=secure))
    if not presented:
        raise AuthenticationError("Your session has ended. Please sign in again.", clear_cookies=stale)

    try:
        session = await service.rotate_session(refresh_token=presented)
    except RefreshTokenReuseError as exc:
        logger.warning("auth.refresh_reuse_detected")
        raise AuthenticationError(
            "For your security we have signed you out of all devices. Please sign in again.",
            detail=str(exc),
            clear_cookies=stale,
        ) from exc
    except TokenError as exc:
        raise AuthenticationError(
            "Your session has ended. Please sign in again.", detail=str(exc), clear_cookies=stale
        ) from exc

    _set_session_cookies(response, session=session, keys=keys, request=request)
    return _session_response(session.user, session)


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(request: Request, response: Response, service: ServiceDep, all_devices: bool = False) -> Response:
    """End this session. Idempotent: logging out twice is not an error.

    The cookies are cleared whatever happens server-side. A logout that leaves a cookie in place
    because a database call failed is worse than one that leaves a row behind.
    """
    secure = _is_secure(request)
    presented = request.cookies.get(refresh_cookie_name(secure=secure))
    _clear_session_cookies(response, request)

    if presented:
        try:
            # Resolved from the token rather than from a claim, because a logout must work even
            # when the access token has already expired -- which is exactly when people use it.
            await service.end_session(refresh_token=presented, all_devices=all_devices)
        except Exception as exc:
            logger.warning("auth.logout_cleanup_failed", extra={"error": str(exc)})

    response.status_code = status.HTTP_204_NO_CONTENT
    return response


# ----------------------------------------------------------------------------------------------


def _session_response(user: Any, session: Any) -> SessionResponse:
    return SessionResponse(
        user_id=user.id,
        tenant_slug=getattr(session, "tenant_slug", ""),
        email=user.email,
        display_name=getattr(user, "name", ""),
        role=str(user.role),
        capabilities=sorted(str(capability) for capability in getattr(session, "capabilities", ())),
        expires_at=session.access_expires_at,
    )


def _set_session_cookies(response: Response, *, session: Any, keys: KeySet, request: Request) -> None:
    """Set the access, refresh and CSRF cookies.

    The CSRF cookie is reissued on every session change on purpose: a stale token after a
    refresh produces a mismatch on the next write, which reads to the user as a random failure.
    """
    secure = _is_secure(request)
    settings = cookie_settings(secure=secure)

    access = issue_access_token(
        keys,
        user_id=session.user.id,
        tenant_id=session.tenant_id,
        session_id=session.id,
        role=str(session.user.role),
        groups=tuple(getattr(session.user, "group_ids", ()) or ()),
        audience=AUDIENCE_TENANT,
    )
    response.set_cookie(access_cookie_name(secure=secure), access, **settings)
    response.set_cookie(
        refresh_cookie_name(secure=secure),
        session.refresh_token,
        **settings,
        max_age=int((session.refresh_expires_at - datetime.now(UTC)).total_seconds()),
    )
    response.set_cookie(
        csrf_module.CSRF_COOKIE,
        csrf_module.new_token(),
        **csrf_module.csrf_cookie_settings(secure=secure),  # type: ignore[arg-type]
    )


def _session_cookie_names(*, secure: bool) -> tuple[str, ...]:
    return (access_cookie_name(secure=secure), refresh_cookie_name(secure=secure), csrf_module.CSRF_COOKIE)


def _clear_session_cookies(response: Response, request: Request) -> None:
    for name in _session_cookie_names(secure=_is_secure(request)):
        response.delete_cookie(name, path="/")


def _is_secure(request: Request) -> bool:
    """Whether to use the ``__Host-`` prefix, which requires HTTPS.

    Read from the request rather than from configuration so that a single build works both in
    local development over http and in production behind TLS, without a flag someone can set
    wrongly in exactly one environment.
    """
    return request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"


def _client_ip(request: Request) -> str | None:
    """The client's address, trusting ``X-Forwarded-For`` only as far as it deserves.

    Taken as the *first* entry, which is the client as the nearest trusted proxy saw it. This is
    forgeable by the client and is therefore recorded for forensics, never used for an
    authorisation or rate-limiting decision on its own.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()[:45]
    return request.client.host if request.client else None


async def _pad_to_floor(started: float, floor: float = MIN_AUTH_RESPONSE_S) -> None:
    """Hold the response until the floor has elapsed.

    Without this, a lookup that finds no tenant returns measurably faster than one that joins
    through to a tenant's IdP list -- and that difference alone enumerates customers. The cost is
    a fixed latency on an endpoint called once per session.
    """
    remaining = floor - (time.perf_counter() - started)
    if remaining > 0:
        await asyncio.sleep(remaining)


def session_expiry(ttl: timedelta) -> datetime:
    return datetime.now(UTC) + ttl


def require_password_login(policy: Any) -> None:
    """Refuse password login when the tenant has disabled it.

    Called by the service before any hashing, and it is the authority -- discovery merely hides
    the field. A client posting directly to ``/login`` gets the same answer as one that read the
    hint, which is what makes "hiding a field is UX, not security" true rather than aspirational.
    """
    from app.auth.discovery import password_login_allowed

    allowed, _ = password_login_allowed(policy)
    if not allowed:
        raise ValidationError(
            "This organisation requires signing in through its identity provider.",
            detail="password_login_disabled",
        )
