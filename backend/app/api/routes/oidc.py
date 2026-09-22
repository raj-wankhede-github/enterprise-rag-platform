"""OIDC start and callback, plus the administrator's draft -> test -> activate wizard.

The two flows share one callback on purpose. A separate test callback would mean the thing an
administrator validates is not the thing users will use, and the difference between them is
exactly where a configuration bug hides. ``state.test_mode`` decides what happens *after* the
token validates: a real sign-in provisions and issues a session, a test returns a claim preview
and issues nothing.

The callback never trusts a cookie. Everything it needs -- the config id, the nonce, the PKCE
verifier, the redirect it must echo -- rides in the signed ``state``. A cookie-dependent callback
fails intermittently in Safari, in in-app webviews, and anywhere a SameSite rule drops it on the
way back from the provider, and those failures are close to unreproducible.
"""

from __future__ import annotations

import logging
import uuid
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from app.api.routes.auth import KeysDep, ServiceDep, _set_session_cookies
from app.auth.oidc import (
    OidcError,
    build_authorization_request,
    decode_state,
    decode_without_verification,
    safe_next_path,
    summarise_for_diagnostics,
)
from app.auth.provisioning import ProvisioningOutcome
from app.core.errors import AuthenticationError, NotFoundError, ValidationError
from app.security.capabilities import Capability

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/auth/oidc", tags=["auth"])

#: Where the browser lands when sign-in fails. A JSON error body would be shown as raw text --
#: the user arrived here by redirect, not by fetch.
LOGIN_ERROR_PATH = "/login?error={code}"


class OidcService:
    """What these routes need from the database and the provider."""

    async def config(self, idp_config_id: uuid.UUID) -> Any: ...
    async def metadata_for(self, config: Any) -> Any: ...
    async def complete_callback(self, *, config: Any, state: Any, code: str) -> Any: ...
    async def start_session(self, *, user: Any, request: Request, method: str) -> Any: ...
    async def record_test_result(self, *, config: Any, claims: dict[str, Any]) -> None: ...
    async def record_attempt(self, **kwargs: Any) -> None: ...


def get_oidc_service(request: Request) -> OidcService:
    service = getattr(request.app.state, "oidc_service", None)
    if service is None:  # pragma: no cover - wiring error
        raise RuntimeError("oidc_service is not configured on the application")
    return service  # type: ignore[no-any-return]


OidcDep = Annotated[OidcService, Depends(get_oidc_service)]


class ClaimPreview(BaseModel):
    """What the wizard shows after a test sign-in. No session, no provisioning."""

    subject: str
    email: str | None = None
    email_verified: bool = False
    groups_seen: list[str] = Field(default_factory=list)
    groups_matched: dict[str, str] = Field(default_factory=dict)
    groups_ignored: list[str] = Field(default_factory=list)
    resolved_role: str = ""
    would_elevate_to_admin: bool = False
    default_role_for_new_users: str = ""
    warnings: list[str] = Field(default_factory=list)


# ----------------------------------------------------------------------------------------------


@router.get("/{idp_config_id}/start")
async def start(
    idp_config_id: uuid.UUID,
    request: Request,
    service: OidcDep,
    next: Annotated[str, Query(max_length=512)] = "/",
    login_hint: Annotated[str | None, Query(max_length=320)] = None,
) -> RedirectResponse:
    """Begin the authorization-code flow. A redirect, not a JSON response."""
    config = await service.config(idp_config_id)
    if config is None:
        # 404 rather than 403: the id is not secret, but confirming which ids exist across every
        # tenant in the deployment is a map of who the customers are.
        raise NotFoundError("That sign-in method is not available.")

    metadata = await service.metadata_for(config)
    authorization = build_authorization_request(
        metadata,
        client_id=config.client_id,
        redirect_uri=_callback_uri(request),
        idp_config_id=config.id,
        tenant_id=config.tenant_id,
        state_secret=_state_secret(request),
        scopes=tuple(config.scopes or ("openid", "profile", "email")),
        next_path=safe_next_path(next),
        login_hint=login_hint,
    )
    # 302, not 307: the browser must issue a fresh GET to the provider.
    return RedirectResponse(authorization.url, status_code=302)


@router.get("/{idp_config_id}/test")
async def start_test(
    idp_config_id: uuid.UUID,
    request: Request,
    service: OidcDep,
) -> RedirectResponse:
    """The wizard's test sign-in. Same flow, same callback, no session at the end.

    Requires ``sso:configure``, because it is a real sign-in against a config that is not yet
    active -- and the claim preview it produces names directory groups.
    """
    principal = getattr(request.state, "principal", None)
    if principal is None or not principal.can(Capability.SSO_CONFIGURE):
        raise AuthenticationError("You do not have permission to test sign-in configurations.")

    config = await service.config(idp_config_id)
    if config is None or config.tenant_id != principal.tenant_id:
        raise NotFoundError("That sign-in method is not available.")

    metadata = await service.metadata_for(config)
    authorization = build_authorization_request(
        metadata,
        client_id=config.client_id,
        redirect_uri=_callback_uri(request),
        idp_config_id=config.id,
        tenant_id=config.tenant_id,
        state_secret=_state_secret(request),
        scopes=tuple(config.scopes or ("openid", "profile", "email")),
        test_mode=True,
        # Always show the picker on a test: an administrator testing a config needs to choose
        # which account they are testing with, and a silent reuse of their current session is
        # the least informative possible outcome.
        extra={"prompt": "select_account"},
    )
    return RedirectResponse(authorization.url, status_code=302)


@router.get("/callback")
async def callback(
    request: Request,
    response: Response,
    service: OidcDep,
    auth: ServiceDep,
    keys: KeysDep,
    code: Annotated[str | None, Query()] = None,
    state: Annotated[str | None, Query()] = None,
    error: Annotated[str | None, Query()] = None,
    error_description: Annotated[str | None, Query()] = None,
) -> Response:
    """One callback for both the real flow and the wizard's test."""
    if error:
        # The provider refused before we ever saw a code -- consent declined, the app blocked by
        # a conditional-access policy, the user not assigned. Their wording is the useful one.
        logger.info("oidc.provider_error", extra={"error": error, "description": error_description})
        return _redirect_to_login("provider_declined")

    if not code or not state:
        return _redirect_to_login("incomplete")

    try:
        decoded = decode_state(state, _state_secret(request))
    except OidcError:
        return _redirect_to_login("expired")

    config = await service.config(decoded.idp_config_id)
    if config is None or config.tenant_id != decoded.tenant_id:
        return _redirect_to_login("unavailable")

    try:
        result = await service.complete_callback(config=config, state=decoded, code=code)
    except OidcError as exc:
        # Log the token's shape, never its contents: a diagnostic that carries an email or a
        # group list is a data-protection problem nobody notices until an audit.
        logger.warning(
            "oidc.callback_failed",
            extra={"error": str(exc), "token": summarise_for_diagnostics(decode_without_verification(""))},
        )
        return _redirect_to_login("verification_failed")

    if decoded.test_mode:
        await service.record_test_result(config=config, claims=result.claims)
        return _json(ClaimPreview(**result.preview))

    decision = result.decision
    await service.record_attempt(
        email=result.claims.get("email", ""),
        tenant_id=config.tenant_id,
        user_id=decision.user_id,
        method="oidc",
        succeeded=decision.permitted,
        failure_reason=None if decision.permitted else decision.outcome.value,
        ip_address=request.headers.get("x-forwarded-for", "").split(",")[0].strip() or None,
        user_agent=request.headers.get("user-agent"),
    )

    if not decision.permitted:
        return _redirect_to_login(_code_for(decision.outcome))

    session = await auth.start_session(user=result.user, request=request, method="oidc")
    redirect = RedirectResponse(safe_next_path(decoded.next_path), status_code=302)
    _set_session_cookies(redirect, session=session, keys=keys, request=request)
    return redirect


# ----------------------------------------------------------------------------------------------


def _code_for(outcome: ProvisioningOutcome) -> str:
    """A stable, non-disclosing code the login page turns into wording.

    Distinguishing "deactivated" from "not provisioned" is worth it here and is not an
    enumeration risk: the user has already authenticated at the directory, so they have proved
    who they are. Telling them to contact their administrator is far better than a generic
    failure they will retry six times before raising a ticket.
    """
    return {
        ProvisioningOutcome.REJECTED_INACTIVE: "account_deactivated",
        ProvisioningOutcome.REJECTED_NO_JIT: "not_provisioned",
    }.get(outcome, "sign_in_failed")


def _redirect_to_login(code: str) -> RedirectResponse:
    return RedirectResponse(LOGIN_ERROR_PATH.format(code=code), status_code=302)


def _json(model: BaseModel) -> Response:
    from fastapi.responses import JSONResponse

    return JSONResponse(model.model_dump(mode="json"))


def _callback_uri(request: Request) -> str:
    """The redirect URI, derived from our own configuration and never from a request parameter.

    The provider compares it byte for byte against its registration, and a caller-supplied value
    here is an open redirect that hands over the authorization code.
    """
    configured = getattr(request.app.state.settings, "public_base_url", None)
    if configured:
        return f"{str(configured).rstrip('/')}/api/auth/oidc/callback"
    return str(request.url_for("callback"))


def _state_secret(request: Request) -> str:
    secret = request.app.state.settings.jwt_secret
    value = secret.get_secret_value() if hasattr(secret, "get_secret_value") else str(secret)
    if len(value) < 32:
        raise ValidationError(
            "The deployment's signing secret is too short.",
            detail="jwt_secret must be at least 32 bytes for HS256",
        )
    return value
