"""Access tokens and refresh tokens.

Two different things with two different designs, and conflating them is the usual mistake.

**The access token is a short-lived signed JWT.** It is checked without a database round trip,
which is what makes it cheap enough to put on every request. Ten minutes, because the claims
inside it are a *cache* of the principal and the cache has to expire.

**The refresh token is an opaque random string, stored hashed.** Nothing is encoded in it;
everything about it is a database row. That is the point -- it is long-lived, so it must be
revocable, and a self-contained token cannot be revoked without the database lookup it exists to
avoid.

Three decisions worth stating:

**``kid``-based rotation from day one.** The verifier accepts a *set* of keys and the signer uses
one. Retrofitting that into a live product means a window where every existing session is
invalid, which in practice means it never happens and the signing key is never rotated.

**The role in the token is a hint, never the authority.** ``Principal`` is reloaded from Postgres
on every request behind a short cache keyed on the session id, so a demotion or a deactivation
takes effect in seconds rather than at the next token expiry. The claims are there for logging
and for a fast reject, not for authorisation.

**The audience separates the tenant plane from the platform plane.** A tenant token carries
``aud=tenant`` and a platform-operator token ``aud=platform``; each app accepts only its own.
Without that, a vendor employee's token would authenticate against a customer's API.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from jwt.exceptions import InvalidTokenError

#: Tenant-facing API. A platform-operator token must not authenticate here.
AUDIENCE_TENANT = "tenant"
AUDIENCE_PLATFORM = "platform"

#: Short, because the claims are a cache of the principal. Long enough that a refresh per request
#: is not needed; short enough that a stale role is measured in minutes.
ACCESS_TTL = timedelta(minutes=10)
#: Long, because it is revocable and rotating. This is the "stay signed in" window.
REFRESH_TTL = timedelta(days=14)

#: 256 bits from the OS. Not a UUID: uuid4 carries version and variant bits, so it supplies 122
#: bits, and nothing here benefits from looking like a UUID.
REFRESH_TOKEN_BYTES = 32

ALGORITHM = "HS256"


class TokenError(Exception):
    """A token was absent, malformed, expired, or for the wrong audience."""


@dataclass(frozen=True, slots=True)
class SigningKey:
    kid: str
    secret: str


@dataclass(slots=True)
class KeySet:
    """One signing key and every key still accepted for verification.

    A rotation adds the new key as ``active`` and keeps the old one in ``accepted`` for at least
    one access-token lifetime. Tokens signed a minute before the rotation stay valid, so a
    rotation is invisible to users -- which is the only kind of rotation that actually gets done.
    """

    active: SigningKey
    accepted: list[SigningKey] = field(default_factory=list)

    def all_keys(self) -> list[SigningKey]:
        return [self.active, *self.accepted]

    def by_kid(self, kid: str) -> SigningKey | None:
        return next((key for key in self.all_keys() if key.kid == kid), None)


@dataclass(frozen=True, slots=True)
class AccessClaims:
    """What a verified access token asserts.

    ``role`` and ``groups`` are present for logging and for a cheap early reject. They are not
    the authorisation decision -- that is made against the principal loaded from Postgres.
    """

    user_id: uuid.UUID
    tenant_id: uuid.UUID
    session_id: uuid.UUID
    role: str
    groups: tuple[str, ...]
    audience: str
    expires_at: datetime
    issued_at: datetime


def issue_access_token(
    keys: KeySet,
    *,
    user_id: uuid.UUID,
    tenant_id: uuid.UUID,
    session_id: uuid.UUID,
    role: str,
    groups: tuple[str, ...] = (),
    audience: str = AUDIENCE_TENANT,
    ttl: timedelta = ACCESS_TTL,
    now: datetime | None = None,
) -> str:
    issued = now or datetime.now(UTC)
    payload: dict[str, Any] = {
        "sub": str(user_id),
        "tid": str(tenant_id),
        "sid": str(session_id),
        "role": role,
        "groups": list(groups),
        "aud": audience,
        "iat": int(issued.timestamp()),
        "exp": int((issued + ttl).timestamp()),
        # A distinct id per token, so a specific token can be denylisted in the rare case that
        # matters without revoking the session it belongs to.
        "jti": secrets.token_urlsafe(12),
    }
    return jwt.encode(payload, keys.active.secret, algorithm=ALGORITHM, headers={"kid": keys.active.kid})


def verify_access_token(token: str, keys: KeySet, *, audience: str = AUDIENCE_TENANT) -> AccessClaims:
    """Verify signature, expiry and audience.

    The ``kid`` selects the key rather than trying each in turn. Trying every key makes a
    rotation quietly O(keys) and, worse, hides a misconfiguration in which the wrong key
    validates by accident.
    """
    try:
        header = jwt.get_unverified_header(token)
    except InvalidTokenError as exc:
        raise TokenError("malformed token") from exc

    kid = str(header.get("kid", ""))
    key = keys.by_kid(kid)
    if key is None:
        # Never name the kid in an error that reaches a client; it is a fingerprint of the
        # deployment's rotation state.
        raise TokenError("unknown signing key")

    try:
        payload = jwt.decode(
            token,
            key.secret,
            algorithms=[ALGORITHM],
            audience=audience,
            options={"require": ["exp", "iat", "sub", "tid", "sid", "aud"]},
        )
    except InvalidTokenError as exc:
        raise TokenError("invalid or expired token") from exc

    try:
        return AccessClaims(
            user_id=uuid.UUID(str(payload["sub"])),
            tenant_id=uuid.UUID(str(payload["tid"])),
            session_id=uuid.UUID(str(payload["sid"])),
            role=str(payload.get("role", "")),
            groups=tuple(str(value) for value in payload.get("groups", [])),
            audience=str(payload["aud"]),
            expires_at=datetime.fromtimestamp(int(payload["exp"]), tz=UTC),
            issued_at=datetime.fromtimestamp(int(payload["iat"]), tz=UTC),
        )
    except (KeyError, ValueError) as exc:
        raise TokenError("token claims are not well formed") from exc


# ----------------------------------------------------------------------------------------------
# Refresh tokens
# ----------------------------------------------------------------------------------------------


def new_refresh_token() -> str:
    return secrets.token_urlsafe(REFRESH_TOKEN_BYTES)


def hash_refresh_token(token: str) -> bytes:
    """A plain SHA-256, deliberately, where a password would use argon2id.

    The reasoning is the difference between the two secrets. A password is low-entropy, chosen by
    a human and often reused, so it needs a slow hash to make offline cracking expensive. A
    refresh token is 256 bits from the OS -- there is nothing to crack, and a slow hash would
    only add cost to a request the user makes constantly.
    """
    return hashlib.sha256(token.encode("utf-8")).digest()


def refresh_token_matches(presented: str, stored_hash: bytes) -> bool:
    return hmac.compare_digest(hash_refresh_token(presented), stored_hash)


class RefreshTokenReuseError(Exception):
    """A previously rotated refresh token was presented again.

    Either a race or a stolen token, and the two are indistinguishable from here. The whole
    family is revoked: a legitimate user signs in again, an attacker gets nothing, and the
    tenant's administrators are alerted. Accepting the old token instead -- which is the
    tempting fix when a user complains about being signed out -- makes rotation theatre.
    """


@dataclass(frozen=True, slots=True)
class StoredSession:
    id: uuid.UUID
    family_id: uuid.UUID
    generation: int
    refresh_token_hash: bytes
    expires_at: datetime
    revoked_at: datetime | None = None


@dataclass(frozen=True, slots=True)
class RotationResult:
    session_id: uuid.UUID
    family_id: uuid.UUID
    generation: int
    refresh_token: str
    refresh_token_hash: bytes
    expires_at: datetime


def rotate(
    presented: str,
    *,
    current: StoredSession,
    family: list[StoredSession],
    now: datetime | None = None,
    ttl: timedelta = REFRESH_TTL,
) -> RotationResult:
    """Exchange a refresh token for the next one in its family.

    ``family`` is every session row sharing ``family_id``, which is what makes reuse detection
    possible: a token matching an *older* generation is one that has already been exchanged.
    """
    moment = now or datetime.now(UTC)

    if current.revoked_at is not None:
        raise RefreshTokenReuseError("this session family has already been revoked")
    if current.expires_at <= moment:
        raise TokenError("the refresh token has expired")

    if not refresh_token_matches(presented, current.refresh_token_hash):
        # Not simply "invalid": if it matches any earlier generation, this is a replay of a token
        # that was already exchanged, and the family is compromised.
        if any(
            refresh_token_matches(presented, older.refresh_token_hash) for older in family if older.id != current.id
        ):
            raise RefreshTokenReuseError("a previously rotated refresh token was presented")
        raise TokenError("invalid refresh token")

    token = new_refresh_token()
    return RotationResult(
        session_id=current.id,
        family_id=current.family_id,
        generation=current.generation + 1,
        refresh_token=token,
        refresh_token_hash=hash_refresh_token(token),
        # A sliding expiry: active use extends the session, inactivity ends it. An absolute cap
        # belongs on the family and is enforced by the caller against the family's first issue.
        expires_at=moment + ttl,
    )


def cookie_settings(*, secure: bool = True, domain: str | None = None) -> dict[str, Any]:
    """Cookie attributes for the access token.

    ``__Host-`` is the prefix that makes the guarantee: a browser accepts it only with Secure,
    Path=/ and **no Domain**, which means a sibling subdomain cannot set or overwrite it. In a
    multi-tenant product served at ``acme.app.example.com`` that is not a nicety -- without it,
    one tenant's subdomain could set a cookie the next tenant's page would send.

    SameSite=Lax rather than Strict, because Strict breaks the OIDC redirect back from the
    provider: the browser would not send the cookie on that top-level cross-site navigation.
    CSRF is covered by the Origin allowlist and the double-submit header as well.
    """
    settings: dict[str, Any] = {
        "httponly": True,
        "secure": secure,
        "samesite": "lax",
        "path": "/",
    }
    if domain and not secure:
        # Only for local development over http, where __Host- cannot apply anyway.
        settings["domain"] = domain
    return settings


def access_cookie_name(*, secure: bool = True) -> str:
    return "__Host-erp_access" if secure else "erp_access"


def refresh_cookie_name(*, secure: bool = True) -> str:
    return "__Host-erp_refresh" if secure else "erp_refresh"
