"""OIDC authorization-code flow with PKCE, for Entra, Google and anything generic.

**One implementation, not three vendor SDKs.** Entra, Google and a generic provider differ in
exactly three places -- the discovery URL, whether groups arrive in the token, and whether the
provider needs a tenant id in its issuer -- and all three are configuration. A vendor SDK per
provider means three code paths to keep secure, of which two are always the less-tested ones.

SAML is deliberately absent. Customers who need it reach us through a self-hosted Keycloak
broker that speaks SAML outward and OIDC to us. That removes ``xmlsec`` -- which has no usable
Windows wheels -- and, more importantly, removes XML signature-wrapping from our attack surface,
which is a classic place to build a critical vulnerability by accident.

Four things this module will not do:

* **No implicit flow, and no tokens in a URL fragment.** Authorization code with PKCE only.
* **No unvalidated ``redirect_uri``.** The callback URL is derived from our own configuration and
  compared exactly. An open redirect here hands the authorization code to whoever asked.
* **No state that lives in a cookie alone.** ``state`` is signed and carries the nonce, the PKCE
  verifier's digest and the config id, so a callback can be validated without trusting that the
  browser kept anything.
* **No ``nonce`` skipped because "the code flow doesn't need it".** It is what binds the ID token
  to this particular authorization request, and without it a token replayed from another session
  validates perfectly.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import secrets
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import httpx
import jwt
from jwt import PyJWKClient
from jwt.exceptions import InvalidTokenError

logger = logging.getLogger(__name__)

#: How long an authorization request stays valid. Long enough for a user to complete an MFA
#: prompt, short enough that a captured state parameter is not useful tomorrow.
STATE_TTL_SECONDS = 600

#: Clock skew allowed when validating the ID token. Providers and our servers do drift, and a
#: 30-second allowance is the usual figure; more than a minute starts to matter for replay.
LEEWAY_SECONDS = 30

DEFAULT_SCOPES = ("openid", "profile", "email")


class OidcError(Exception):
    """The flow could not be completed. The message is safe to show a user."""


@dataclass(frozen=True, slots=True)
class ProviderMetadata:
    """The parts of a discovery document we use."""

    issuer: str
    authorization_endpoint: str
    token_endpoint: str
    jwks_uri: str
    userinfo_endpoint: str | None = None
    end_session_endpoint: str | None = None

    @classmethod
    def from_document(cls, document: dict[str, Any]) -> ProviderMetadata:
        try:
            return cls(
                issuer=str(document["issuer"]),
                authorization_endpoint=str(document["authorization_endpoint"]),
                token_endpoint=str(document["token_endpoint"]),
                jwks_uri=str(document["jwks_uri"]),
                userinfo_endpoint=_optional(document.get("userinfo_endpoint")),
                end_session_endpoint=_optional(document.get("end_session_endpoint")),
            )
        except KeyError as exc:
            raise OidcError(f"The provider's discovery document is missing {exc}.") from exc


def _optional(value: Any) -> str | None:
    return str(value) if value else None


def discovery_url(issuer: str) -> str:
    """The well-known location. Appended to the issuer, never replacing its path.

    Entra issuers carry a tenant segment (``.../{tenant}/v2.0``) and truncating to the host --
    which is what naive URL joining does -- produces a document for the wrong tenant.
    """
    return f"{issuer.rstrip('/')}/.well-known/openid-configuration"


async def fetch_metadata(
    issuer: str, *, client: httpx.AsyncClient | None = None, url: str | None = None
) -> ProviderMetadata:
    target = url or discovery_url(issuer)
    try:
        if client is not None:
            response = await client.get(target, timeout=10.0)
        else:
            async with httpx.AsyncClient(timeout=10.0) as owned:
                response = await owned.get(target)
        response.raise_for_status()
        document = response.json()
    except httpx.HTTPError as exc:
        raise OidcError(f"Could not reach the identity provider's discovery document at {target}.") from exc
    except ValueError as exc:
        raise OidcError("The identity provider's discovery document was not valid JSON.") from exc

    metadata = ProviderMetadata.from_document(document)
    if metadata.issuer.rstrip("/") != issuer.rstrip("/"):
        # The document declares who it speaks for. A mismatch means the configured issuer is
        # wrong, and accepting it would validate tokens against the wrong authority.
        raise OidcError(
            f"The provider's document declares issuer {metadata.issuer!r}, "
            f"which does not match the configured {issuer!r}."
        )
    return metadata


# ----------------------------------------------------------------------------------------------
# PKCE and state
# ----------------------------------------------------------------------------------------------


def new_verifier() -> str:
    """A PKCE code verifier: 43-128 unreserved characters."""
    return secrets.token_urlsafe(64)[:128]


def challenge_for(verifier: str) -> str:
    """S256, never ``plain``.

    ``plain`` puts the verifier itself in the authorization request, which defeats the point --
    anyone who intercepts the request can complete the exchange.
    """
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


@dataclass(frozen=True, slots=True)
class AuthState:
    """What the callback needs, carried in a signed ``state`` parameter.

    Signed and self-contained rather than stored server-side or in a cookie. A cookie is lost
    whenever the provider's redirect crosses a browser that drops it -- Safari's partitioning,
    a SameSite mismatch, an in-app webview -- and the resulting failure is intermittent and
    unreproducible. Signing it means the callback can validate itself from what it was handed.
    """

    idp_config_id: uuid.UUID
    tenant_id: uuid.UUID
    nonce: str
    verifier: str
    redirect_uri: str
    issued_at: int
    #: Where to send the user once they are signed in. Validated as a same-origin path by the
    #: caller before it is ever used -- an open redirect is the classic bug in this field.
    next_path: str = "/"
    #: Set for the admin wizard's test sign-in, which validates the config and shows a claim
    #: preview without provisioning anyone or issuing a session.
    test_mode: bool = False


def encode_state(state: AuthState, secret: str) -> str:
    payload = {
        "cid": str(state.idp_config_id),
        "tid": str(state.tenant_id),
        "nonce": state.nonce,
        "ver": state.verifier,
        "ruri": state.redirect_uri,
        "iat": state.issued_at,
        "next": state.next_path,
        "test": state.test_mode,
    }
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_state(token: str, secret: str, *, now: int | None = None) -> AuthState:
    try:
        payload = jwt.decode(token, secret, algorithms=["HS256"], options={"require": ["cid", "tid", "iat"]})
    except InvalidTokenError as exc:
        raise OidcError("The sign-in request could not be verified. Please try again.") from exc

    issued = int(payload["iat"])
    if (now or int(time.time())) - issued > STATE_TTL_SECONDS:
        raise OidcError("This sign-in request has expired. Please try again.")

    return AuthState(
        idp_config_id=uuid.UUID(str(payload["cid"])),
        tenant_id=uuid.UUID(str(payload["tid"])),
        nonce=str(payload.get("nonce", "")),
        verifier=str(payload.get("ver", "")),
        redirect_uri=str(payload.get("ruri", "")),
        issued_at=issued,
        next_path=str(payload.get("next", "/")),
        test_mode=bool(payload.get("test", False)),
    )


def safe_next_path(candidate: str | None, *, default: str = "/") -> str:
    """Only a same-origin absolute path is allowed after login.

    Anything with a scheme, a host, or a protocol-relative ``//`` prefix is discarded. This field
    comes straight from a query parameter, and it is the standard way an open redirect turns a
    login page into a phishing relay -- the user really did sign in to us, then landed on the
    attacker's page still trusting the address bar they saw a moment ago.
    """
    if not candidate or not candidate.startswith("/") or candidate.startswith("//"):
        return default
    if "\\" in candidate or "\n" in candidate or "\r" in candidate:
        return default
    return candidate


# ----------------------------------------------------------------------------------------------
# The flow
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuthorizationRequest:
    url: str
    state: str
    nonce: str
    verifier: str


def build_authorization_request(
    metadata: ProviderMetadata,
    *,
    client_id: str,
    redirect_uri: str,
    idp_config_id: uuid.UUID,
    tenant_id: uuid.UUID,
    state_secret: str,
    scopes: tuple[str, ...] = DEFAULT_SCOPES,
    next_path: str = "/",
    test_mode: bool = False,
    login_hint: str | None = None,
    extra: dict[str, str] | None = None,
) -> AuthorizationRequest:
    from urllib.parse import urlencode

    verifier = new_verifier()
    nonce = secrets.token_urlsafe(24)
    state = AuthState(
        idp_config_id=idp_config_id,
        tenant_id=tenant_id,
        nonce=nonce,
        verifier=verifier,
        redirect_uri=redirect_uri,
        issued_at=int(time.time()),
        next_path=safe_next_path(next_path),
        test_mode=test_mode,
    )
    encoded = encode_state(state, state_secret)

    params = {
        "response_type": "code",
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "scope": " ".join(dict.fromkeys(("openid", *scopes))),
        "state": encoded,
        "nonce": nonce,
        "code_challenge": challenge_for(verifier),
        "code_challenge_method": "S256",
    }
    if login_hint:
        # Skips the provider's account picker when we already know the address, which removes
        # the commonest support complaint about SSO: "it signed me in as the wrong account".
        params["login_hint"] = login_hint
    params.update(extra or {})

    return AuthorizationRequest(
        url=f"{metadata.authorization_endpoint}?{urlencode(params)}",
        state=encoded,
        nonce=nonce,
        verifier=verifier,
    )


@dataclass(slots=True)
class TokenResponse:
    id_token: str
    access_token: str | None = None
    refresh_token: str | None = None
    expires_in: int | None = None
    raw: dict[str, Any] = field(default_factory=dict)


async def exchange_code(
    metadata: ProviderMetadata,
    *,
    code: str,
    client_id: str,
    client_secret: str | None,
    redirect_uri: str,
    verifier: str,
    client: httpx.AsyncClient | None = None,
) -> TokenResponse:
    """Exchange the authorization code. Server to server, never from the browser."""
    form = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
        "client_id": client_id,
        "code_verifier": verifier,
    }
    if client_secret:
        form["client_secret"] = client_secret

    try:
        if client is not None:
            response = await client.post(metadata.token_endpoint, data=form, timeout=15.0)
        else:
            async with httpx.AsyncClient(timeout=15.0) as owned:
                response = await owned.post(metadata.token_endpoint, data=form)
        body = response.json()
    except httpx.HTTPError as exc:
        raise OidcError("Could not reach the identity provider to complete sign-in.") from exc
    except ValueError as exc:
        raise OidcError("The identity provider returned an unreadable token response.") from exc

    if response.status_code >= 400:
        # The provider's own error code is genuinely useful here and safe to surface: it is about
        # our configuration, not about the user.
        detail = str(body.get("error_description") or body.get("error") or response.status_code)
        raise OidcError(f"The identity provider rejected the sign-in: {detail}")

    if "id_token" not in body:
        raise OidcError(
            "The identity provider did not return an ID token. Check the requested scopes include 'openid'."
        )

    return TokenResponse(
        id_token=str(body["id_token"]),
        access_token=_optional(body.get("access_token")),
        refresh_token=_optional(body.get("refresh_token")),
        expires_in=int(body["expires_in"]) if body.get("expires_in") else None,
        raw=dict(body),
    )


def validate_id_token(
    id_token: str,
    *,
    jwks_client: PyJWKClient,
    issuer: str,
    audience: str,
    nonce: str,
    leeway: int = LEEWAY_SECONDS,
) -> dict[str, Any]:
    """Verify the ID token's signature and every claim that binds it to this request.

    The nonce check is the one most often skipped, on the reasoning that the code flow already
    proves possession. It does not prove *freshness*: without the nonce, an ID token captured
    from an earlier sign-in validates perfectly against signature, issuer, audience and expiry.
    """
    try:
        key = jwks_client.get_signing_key_from_jwt(id_token)
        claims: dict[str, Any] = jwt.decode(
            id_token,
            key.key,
            algorithms=["RS256", "ES256", "PS256"],
            audience=audience,
            issuer=issuer,
            leeway=leeway,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
    except InvalidTokenError as exc:
        raise OidcError("The identity provider's response could not be verified.") from exc
    except Exception as exc:  # jwks fetch failures surface as assorted urllib errors
        raise OidcError("Could not retrieve the identity provider's signing keys.") from exc

    presented = str(claims.get("nonce", ""))
    if not nonce or not secrets.compare_digest(presented, nonce):
        raise OidcError("The identity provider's response did not match this sign-in request.")

    return claims


async def fetch_google_groups(
    access_token: str, *, email: str, client: httpx.AsyncClient | None = None
) -> tuple[str, ...]:
    """Google Workspace groups, which are **not** in the ID token.

    This surprises everyone once. Entra puts group object ids in the token; Google requires a
    Directory API call, and that call needs domain-wide delegation configured by the customer's
    Workspace administrator. A deployment without it gets no groups and every user lands on the
    default role -- which is why ``preview_mapping`` warns about it explicitly rather than
    leaving an administrator to discover it after activation.

    Failure returns no groups rather than raising: losing group-based roles degrades a login,
    while failing it locks people out of a product they are entitled to use.
    """
    url = "https://admin.googleapis.com/admin/directory/v1/groups"
    try:
        if client is not None:
            response = await client.get(
                url, params={"userKey": email}, headers={"Authorization": f"Bearer {access_token}"}, timeout=10.0
            )
        else:
            async with httpx.AsyncClient(timeout=10.0) as owned:
                response = await owned.get(
                    url, params={"userKey": email}, headers={"Authorization": f"Bearer {access_token}"}
                )
        if response.status_code >= 400:
            logger.warning("google.groups_unavailable", extra={"status": response.status_code})
            return ()
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        logger.warning("google.groups_failed", extra={"error": str(exc)})
        return ()

    return tuple(str(group.get("email") or group.get("id")) for group in payload.get("groups", []) if group)


def decode_without_verification(id_token: str) -> dict[str, Any]:
    """Claims from an unverified token, for error messages and the wizard's preview only.

    Never for an authorisation decision. It exists because "your token's audience was X, we
    expected Y" is the difference between a five-minute fix and a support ticket, and that
    message cannot be produced from a token that failed verification any other way.
    """
    try:
        payload = jwt.decode(id_token, options={"verify_signature": False})
        return dict(payload)
    except InvalidTokenError:
        return {}


def summarise_for_diagnostics(claims: dict[str, Any]) -> str:
    """A one-line description of a token, safe to log. No email, no name, no group values."""
    return json.dumps(
        {
            "iss": claims.get("iss"),
            "aud": claims.get("aud"),
            "has_sub": bool(claims.get("sub")),
            "has_oid": bool(claims.get("oid")),
            "group_count": len(claims.get("groups", []) or []),
            "email_verified": claims.get("email_verified"),
        },
        sort_keys=True,
    )
