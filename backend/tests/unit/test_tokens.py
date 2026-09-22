"""Access tokens, refresh rotation, and the reuse detection that makes rotation mean something.

The rotation tests are the ones worth reading. Refresh rotation without reuse detection is
theatre: it changes the token on every exchange and does nothing at all about a stolen one.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.security.tokens import (
    AUDIENCE_PLATFORM,
    AUDIENCE_TENANT,
    KeySet,
    RefreshTokenReuseError,
    SigningKey,
    StoredSession,
    TokenError,
    access_cookie_name,
    cookie_settings,
    hash_refresh_token,
    issue_access_token,
    new_refresh_token,
    refresh_token_matches,
    rotate,
    verify_access_token,
)

USER = uuid.UUID("11110000-0000-0000-0000-000000000001")
TENANT = uuid.UUID("22220000-0000-0000-0000-000000000002")
SESSION = uuid.UUID("33330000-0000-0000-0000-000000000003")
FAMILY = uuid.UUID("44440000-0000-0000-0000-000000000004")

KEYS = KeySet(active=SigningKey(kid="k1", secret="a" * 48))


def a_token(keys: KeySet = KEYS, **kwargs: object) -> str:
    base: dict[str, object] = {
        "user_id": USER,
        "tenant_id": TENANT,
        "session_id": SESSION,
        "role": "DEV",
    }
    base.update(kwargs)
    return issue_access_token(keys, **base)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------
# Access tokens
# ------------------------------------------------------------------------------------------


def test_a_token_round_trips() -> None:
    verified = verify_access_token(a_token(groups=("eng", "finance")), KEYS)
    assert verified.user_id == USER
    assert verified.tenant_id == TENANT
    assert verified.session_id == SESSION
    assert verified.groups == ("eng", "finance")


def test_a_platform_token_does_not_authenticate_against_the_tenant_api() -> None:
    """The audience split is what stops a vendor employee's token working against a customer's
    API. Without it, one signing key would authenticate both planes."""
    platform = a_token(audience=AUDIENCE_PLATFORM)
    with pytest.raises(TokenError):
        verify_access_token(platform, KEYS, audience=AUDIENCE_TENANT)
    assert verify_access_token(platform, KEYS, audience=AUDIENCE_PLATFORM).audience == AUDIENCE_PLATFORM


def test_an_expired_token_is_refused() -> None:
    expired = a_token(now=datetime.now(UTC) - timedelta(hours=2))
    with pytest.raises(TokenError, match="invalid or expired"):
        verify_access_token(expired, KEYS)


def test_a_token_signed_with_another_key_is_refused() -> None:
    other = KeySet(active=SigningKey(kid="k1", secret="b" * 48))
    with pytest.raises(TokenError):
        verify_access_token(a_token(other), KEYS)


def test_garbage_is_refused_as_malformed_rather_than_raising_something_else() -> None:
    with pytest.raises(TokenError, match="malformed"):
        verify_access_token("not-a-token", KEYS)


# ------------------------------------------------------------------------------------------
# Key rotation
# ------------------------------------------------------------------------------------------


def test_a_token_signed_before_a_rotation_still_verifies_after_it() -> None:
    """The property that makes rotation invisible to users -- and therefore the property that
    makes rotation something a team will actually do."""
    old = KeySet(active=SigningKey(kid="k1", secret="a" * 48))
    token = a_token(old)

    rotated = KeySet(active=SigningKey(kid="k2", secret="c" * 48), accepted=[old.active])
    assert verify_access_token(token, rotated).user_id == USER


def test_a_retired_key_stops_verifying_once_it_is_dropped() -> None:
    token = a_token(KeySet(active=SigningKey(kid="k1", secret="a" * 48)))
    only_new = KeySet(active=SigningKey(kid="k2", secret="c" * 48))
    with pytest.raises(TokenError, match="unknown signing key"):
        verify_access_token(token, only_new)


def test_the_kid_is_not_echoed_in_the_error() -> None:
    """It fingerprints the deployment's rotation state, which a client has no business knowing."""
    token = a_token(KeySet(active=SigningKey(kid="internal-prod-2026-03", secret="a" * 48)))
    with pytest.raises(TokenError) as raised:
        verify_access_token(token, KeySet(active=SigningKey(kid="k9", secret="z" * 48)))
    assert "internal-prod-2026-03" not in str(raised.value)


def test_new_tokens_use_the_active_key() -> None:
    import jwt

    keys = KeySet(active=SigningKey(kid="k2", secret="c" * 48), accepted=[SigningKey(kid="k1", secret="a" * 48)])
    assert jwt.get_unverified_header(a_token(keys))["kid"] == "k2"


# ------------------------------------------------------------------------------------------
# Refresh tokens
# ------------------------------------------------------------------------------------------


def test_a_refresh_token_carries_real_entropy() -> None:
    """256 bits from the OS. A uuid4 would supply 122, and nothing here benefits from looking
    like a UUID."""
    tokens = {new_refresh_token() for _ in range(200)}
    assert len(tokens) == 200
    assert all(len(token) >= 40 for token in tokens)


def test_matching_is_constant_time_against_the_stored_hash() -> None:
    token = new_refresh_token()
    assert refresh_token_matches(token, hash_refresh_token(token))
    assert not refresh_token_matches(new_refresh_token(), hash_refresh_token(token))


def a_session(token: str, *, generation: int = 0, **kwargs: object) -> StoredSession:
    base: dict[str, object] = {
        "id": SESSION,
        "family_id": FAMILY,
        "generation": generation,
        "refresh_token_hash": hash_refresh_token(token),
        "expires_at": datetime.now(UTC) + timedelta(days=7),
    }
    base.update(kwargs)
    return StoredSession(**base)  # type: ignore[arg-type]


def test_a_valid_refresh_rotates_the_token_and_advances_the_generation() -> None:
    token = new_refresh_token()
    result = rotate(token, current=a_session(token), family=[])

    assert result.refresh_token != token
    assert result.generation == 1
    assert refresh_token_matches(result.refresh_token, result.refresh_token_hash)


def test_replaying_an_already_rotated_token_revokes_the_whole_family() -> None:
    """The test this module exists for.

    Either a race or a stolen token, and from here the two are indistinguishable -- so the
    family goes. A legitimate user signs in again; an attacker with a stolen token gets nothing.
    Accepting the old token instead, which is the tempting fix when a user complains about being
    signed out, turns rotation back into decoration.
    """
    first = new_refresh_token()
    second = new_refresh_token()
    old_row = a_session(first, generation=0)
    current = a_session(second, generation=1, id=uuid.uuid4())

    with pytest.raises(RefreshTokenReuseError):
        rotate(first, current=current, family=[old_row, current])


def test_a_token_from_no_generation_at_all_is_simply_invalid() -> None:
    """A guess is not a compromise, and revoking a family on every wrong guess would let anyone
    sign a user out by posting rubbish."""
    current = a_session(new_refresh_token())
    with pytest.raises(TokenError, match="invalid refresh token"):
        rotate(new_refresh_token(), current=current, family=[current])


def test_an_already_revoked_family_refuses_immediately() -> None:
    token = new_refresh_token()
    revoked = a_session(token, revoked_at=datetime.now(UTC) - timedelta(minutes=1))
    with pytest.raises(RefreshTokenReuseError):
        rotate(token, current=revoked, family=[revoked])


def test_an_expired_refresh_token_is_an_expiry_not_a_compromise() -> None:
    token = new_refresh_token()
    stale = a_session(token, expires_at=datetime.now(UTC) - timedelta(days=1))
    with pytest.raises(TokenError, match="expired"):
        rotate(token, current=stale, family=[stale])


def test_active_use_slides_the_expiry_forward() -> None:
    token = new_refresh_token()
    session = a_session(token, expires_at=datetime.now(UTC) + timedelta(hours=1))
    assert rotate(token, current=session, family=[]).expires_at > session.expires_at


# ------------------------------------------------------------------------------------------
# Cookies
# ------------------------------------------------------------------------------------------


def test_the_access_cookie_uses_the_host_prefix_in_production() -> None:
    """``__Host-`` is what stops a sibling subdomain setting or overwriting the cookie. Serving
    tenants at acme.app.example.com makes that a real boundary, not a formality."""
    assert access_cookie_name(secure=True).startswith("__Host-")
    assert not access_cookie_name(secure=False).startswith("__Host-")


def test_cookies_are_httponly_and_secure_and_never_carry_a_domain() -> None:
    settings = cookie_settings(secure=True, domain="app.example.com")
    assert settings["httponly"] and settings["secure"]
    assert "domain" not in settings, "a Domain attribute makes __Host- invalid and re-opens the sibling-subdomain hole"


def test_samesite_is_lax_so_the_oidc_redirect_still_carries_the_session() -> None:
    """Strict would drop the cookie on the top-level cross-site navigation back from the
    provider, which breaks every SSO login. CSRF is covered by the Origin allowlist and the
    double-submit header as well."""
    assert cookie_settings()["samesite"] == "lax"


def test_a_development_cookie_may_carry_a_domain_because_host_cannot_apply_over_http() -> None:
    assert cookie_settings(secure=False, domain="localhost")["domain"] == "localhost"
