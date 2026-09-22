"""The OIDC flow: PKCE, signed state, open-redirect refusal, and token validation.

Most of these guard against a specific, named way of getting OIDC wrong. The nonce test and the
``safe_next_path`` tests are the two that cover bugs which ship most often, because neither
failure is visible in a working sign-in.
"""

from __future__ import annotations

import time
import uuid

import httpx
import pytest

from app.auth.oidc import (
    STATE_TTL_SECONDS,
    AuthState,
    OidcError,
    ProviderMetadata,
    build_authorization_request,
    challenge_for,
    decode_state,
    decode_without_verification,
    discovery_url,
    encode_state,
    exchange_code,
    fetch_google_groups,
    fetch_metadata,
    new_verifier,
    safe_next_path,
    summarise_for_diagnostics,
)

SECRET = "state-signing-secret-long-enough"
CONFIG_ID = uuid.UUID("11110000-0000-0000-0000-000000000001")
TENANT = uuid.UUID("22220000-0000-0000-0000-000000000002")

METADATA = ProviderMetadata(
    issuer="https://login.microsoftonline.com/tid/v2.0",
    authorization_endpoint="https://login.microsoftonline.com/tid/oauth2/v2.0/authorize",
    token_endpoint="https://login.microsoftonline.com/tid/oauth2/v2.0/token",
    jwks_uri="https://login.microsoftonline.com/tid/discovery/v2.0/keys",
)


def transport(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------
# Discovery
# ------------------------------------------------------------------------------------------


def test_the_well_known_path_is_appended_not_substituted() -> None:
    """Entra issuers carry a tenant segment, and naive URL joining truncates to the host --
    which fetches a document for the wrong tenant, or for none."""
    assert discovery_url("https://login.microsoftonline.com/tid/v2.0") == (
        "https://login.microsoftonline.com/tid/v2.0/.well-known/openid-configuration"
    )


def test_a_trailing_slash_does_not_double_up() -> None:
    assert discovery_url("https://idp.example.com/").count("//") == 1


async def test_a_document_declaring_a_different_issuer_is_refused() -> None:
    """The document says who it speaks for. Accepting a mismatch means validating tokens against
    the wrong authority."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "issuer": "https://attacker.example.com",
                "authorization_endpoint": "https://a/authorize",
                "token_endpoint": "https://a/token",
                "jwks_uri": "https://a/keys",
            },
        )

    async with transport(handler) as client:
        with pytest.raises(OidcError, match="does not match the configured"):
            await fetch_metadata("https://idp.example.com", client=client)


async def test_a_document_missing_a_required_endpoint_is_refused_clearly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"issuer": "https://idp.example.com"})

    async with transport(handler) as client:
        with pytest.raises(OidcError, match="missing"):
            await fetch_metadata("https://idp.example.com", client=client)


async def test_an_unreachable_provider_names_the_url_it_could_not_reach() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with transport(handler) as client:
        with pytest.raises(OidcError, match="discovery document at"):
            await fetch_metadata("https://idp.example.com", client=client)


# ------------------------------------------------------------------------------------------
# PKCE
# ------------------------------------------------------------------------------------------


def test_the_challenge_is_s256_and_base64url_without_padding() -> None:
    challenge = challenge_for("a" * 64)
    assert "=" not in challenge
    assert "+" not in challenge and "/" not in challenge


def test_the_verifier_is_never_sent_in_the_authorization_request() -> None:
    """``plain`` puts the verifier in the request, which defeats the entire point: anyone who
    intercepts the request can complete the exchange."""
    request = build_authorization_request(
        METADATA, client_id="c", redirect_uri="https://app/cb", idp_config_id=CONFIG_ID,
        tenant_id=TENANT, state_secret=SECRET,
    )
    assert "code_challenge_method=S256" in request.url
    assert request.verifier not in request.url


def test_verifiers_are_unique_and_of_a_legal_length() -> None:
    verifiers = {new_verifier() for _ in range(100)}
    assert len(verifiers) == 100
    assert all(43 <= len(value) <= 128 for value in verifiers)


# ------------------------------------------------------------------------------------------
# State
# ------------------------------------------------------------------------------------------


def a_state(**kwargs: object) -> AuthState:
    base: dict[str, object] = {
        "idp_config_id": CONFIG_ID,
        "tenant_id": TENANT,
        "nonce": "nonce-value",
        "verifier": "verifier-value",
        "redirect_uri": "https://app.example.com/callback",
        "issued_at": int(time.time()),
    }
    base.update(kwargs)
    return AuthState(**base)  # type: ignore[arg-type]


def test_state_round_trips() -> None:
    decoded = decode_state(encode_state(a_state(), SECRET), SECRET)
    assert decoded.idp_config_id == CONFIG_ID
    assert decoded.nonce == "nonce-value"
    assert decoded.verifier == "verifier-value"


def test_state_signed_with_another_secret_is_refused() -> None:
    with pytest.raises(OidcError):
        decode_state(encode_state(a_state(), "another-secret-entirely-and-long-enough-for-hs256"), SECRET)


def test_a_tampered_state_is_refused() -> None:
    encoded = encode_state(a_state(), SECRET)
    tampered = encoded[:-4] + ("aaaa" if not encoded.endswith("aaaa") else "bbbb")
    with pytest.raises(OidcError):
        decode_state(tampered, SECRET)


def test_state_expires() -> None:
    """A captured state parameter must not be useful tomorrow."""
    old = encode_state(a_state(issued_at=int(time.time()) - STATE_TTL_SECONDS - 60), SECRET)
    with pytest.raises(OidcError, match="expired"):
        decode_state(old, SECRET)


def test_state_is_self_contained_so_a_dropped_cookie_does_not_break_sign_in() -> None:
    """Safari partitioning, a SameSite mismatch, an in-app webview -- a cookie-only state
    produces intermittent, unreproducible login failures."""
    decoded = decode_state(encode_state(a_state(), SECRET), SECRET)
    assert decoded.verifier and decoded.nonce and decoded.redirect_uri


# ------------------------------------------------------------------------------------------
# Open redirect
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "candidate",
    [
        "https://attacker.example.com",
        "//attacker.example.com",
        "http://attacker.example.com/x",
        "javascript:alert(1)",
        "/\\attacker.example.com",
        "/path\nLocation: https://attacker",
        "",
        None,
    ],
)
def test_a_next_path_that_could_leave_the_origin_is_discarded(candidate: str | None) -> None:
    """The standard way a login page becomes a phishing relay: the user really did sign in to
    us, then landed on the attacker's page still trusting the address bar they just saw."""
    assert safe_next_path(candidate) == "/"


@pytest.mark.parametrize("candidate", ["/", "/search", "/documents/123?tab=chunks"])
def test_a_same_origin_path_survives(candidate: str) -> None:
    assert safe_next_path(candidate) == candidate


def test_the_authorization_request_sanitises_the_next_path_before_signing_it() -> None:
    request = build_authorization_request(
        METADATA, client_id="c", redirect_uri="https://app/cb", idp_config_id=CONFIG_ID,
        tenant_id=TENANT, state_secret=SECRET, next_path="https://attacker.example.com",
    )
    assert decode_state(request.state, SECRET).next_path == "/"


# ------------------------------------------------------------------------------------------
# The authorization request
# ------------------------------------------------------------------------------------------


def test_the_request_is_a_code_flow_with_a_nonce() -> None:
    request = build_authorization_request(
        METADATA, client_id="client-id", redirect_uri="https://app/cb", idp_config_id=CONFIG_ID,
        tenant_id=TENANT, state_secret=SECRET,
    )
    assert "response_type=code" in request.url
    assert f"nonce={request.nonce}" in request.url
    assert "response_type=id_token" not in request.url


def test_openid_is_always_requested_even_if_a_tenant_configured_odd_scopes() -> None:
    """Without it the provider returns no ID token at all, and the failure arrives at the
    callback rather than at configuration time."""
    request = build_authorization_request(
        METADATA, client_id="c", redirect_uri="https://app/cb", idp_config_id=CONFIG_ID,
        tenant_id=TENANT, state_secret=SECRET, scopes=("email",),
    )
    assert "scope=openid+email" in request.url


def test_a_login_hint_skips_the_account_picker() -> None:
    """Removes the commonest support complaint about SSO: "it signed me in as the wrong
    account"."""
    request = build_authorization_request(
        METADATA, client_id="c", redirect_uri="https://app/cb", idp_config_id=CONFIG_ID,
        tenant_id=TENANT, state_secret=SECRET, login_hint="a@acme.com",
    )
    assert "login_hint=a%40acme.com" in request.url


def test_each_request_has_a_fresh_nonce_and_verifier() -> None:
    args = dict(
        client_id="c", redirect_uri="https://app/cb", idp_config_id=CONFIG_ID,
        tenant_id=TENANT, state_secret=SECRET,
    )
    first = build_authorization_request(METADATA, **args)  # type: ignore[arg-type]
    second = build_authorization_request(METADATA, **args)  # type: ignore[arg-type]
    assert first.nonce != second.nonce
    assert first.verifier != second.verifier


def test_test_mode_survives_the_round_trip_so_the_callback_provisions_nobody() -> None:
    request = build_authorization_request(
        METADATA, client_id="c", redirect_uri="https://app/cb", idp_config_id=CONFIG_ID,
        tenant_id=TENANT, state_secret=SECRET, test_mode=True,
    )
    assert decode_state(request.state, SECRET).test_mode


# ------------------------------------------------------------------------------------------
# The code exchange
# ------------------------------------------------------------------------------------------


async def test_the_verifier_is_sent_to_the_token_endpoint_not_to_the_browser() -> None:
    sent: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(dict(pair.split("=", 1) for pair in request.content.decode().split("&")))
        return httpx.Response(200, json={"id_token": "header.body.sig", "access_token": "at"})

    async with transport(handler) as client:
        result = await exchange_code(
            METADATA, code="the-code", client_id="c", client_secret="s",
            redirect_uri="https://app/cb", verifier="the-verifier", client=client,
        )
    assert sent["code_verifier"] == "the-verifier"
    assert sent["grant_type"] == "authorization_code"
    assert result.id_token == "header.body.sig"


async def test_a_rejected_exchange_surfaces_the_providers_own_reason() -> None:
    """The provider's error is about *our* configuration, not about the user, so it is both
    safe and genuinely useful to show an administrator."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": "invalid_client", "error_description": "secret expired"})

    async with transport(handler) as client:
        with pytest.raises(OidcError, match="secret expired"):
            await exchange_code(
                METADATA, code="c", client_id="c", client_secret="s",
                redirect_uri="https://app/cb", verifier="v", client=client,
            )


async def test_a_response_without_an_id_token_points_at_the_scope() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"access_token": "at"})

    async with transport(handler) as client:
        with pytest.raises(OidcError, match="openid"):
            await exchange_code(
                METADATA, code="c", client_id="c", client_secret=None,
                redirect_uri="https://app/cb", verifier="v", client=client,
            )


async def test_a_public_client_omits_the_secret_rather_than_sending_an_empty_one() -> None:
    sent: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent.update(dict(pair.split("=", 1) for pair in request.content.decode().split("&")))
        return httpx.Response(200, json={"id_token": "t"})

    async with transport(handler) as client:
        await exchange_code(
            METADATA, code="c", client_id="c", client_secret=None,
            redirect_uri="https://app/cb", verifier="v", client=client,
        )
    assert "client_secret" not in sent


# ------------------------------------------------------------------------------------------
# Google groups
# ------------------------------------------------------------------------------------------


async def test_google_groups_come_from_the_directory_api() -> None:
    """They are not in the ID token. This surprises everyone exactly once."""

    def handler(request: httpx.Request) -> httpx.Response:
        assert "userKey=a%40acme.com" in str(request.url)
        return httpx.Response(200, json={"groups": [{"email": "eng@acme.com"}, {"email": "all@acme.com"}]})

    async with transport(handler) as client:
        groups = await fetch_google_groups("token", email="a@acme.com", client=client)
    assert groups == ("eng@acme.com", "all@acme.com")


async def test_a_directory_call_that_fails_degrades_to_no_groups() -> None:
    """Losing group-based roles degrades a login; failing it locks people out of a product they
    are entitled to use. The customer most likely has not configured domain-wide delegation."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, json={"error": "insufficient permissions"})

    async with transport(handler) as client:
        assert await fetch_google_groups("token", email="a@acme.com", client=client) == ()


async def test_an_unreachable_directory_also_degrades() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with transport(handler) as client:
        assert await fetch_google_groups("t", email="a@acme.com", client=client) == ()


# ------------------------------------------------------------------------------------------
# Diagnostics
# ------------------------------------------------------------------------------------------


def test_the_diagnostic_summary_carries_no_personal_data() -> None:
    """It goes to logs. An email or a group list there is a data-protection problem that nobody
    notices until an audit."""
    summary = summarise_for_diagnostics(
        {"iss": "https://idp", "aud": "client", "sub": "s", "email": "a@acme.com", "groups": ["eng"], "name": "A Person"}
    )
    assert "a@acme.com" not in summary
    assert "A Person" not in summary
    assert "eng" not in summary
    assert '"group_count": 1' in summary


def test_unverified_decoding_exists_for_error_messages_only() -> None:
    """"Your token's audience was X, we expected Y" is the difference between a five-minute fix
    and a support ticket -- and it cannot be produced from a token that failed verification."""
    import jwt

    token = jwt.encode({"aud": "wrong-client", "iss": "https://idp"}, "k" * 32, algorithm="HS256")
    assert decode_without_verification(token)["aud"] == "wrong-client"


def test_unverified_decoding_of_rubbish_returns_nothing_rather_than_raising() -> None:
    assert decode_without_verification("not-a-token") == {}
