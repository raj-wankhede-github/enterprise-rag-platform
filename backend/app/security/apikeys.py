"""API keys: format, minting and verification.

    erpk_live_7f3a1b2c4d5e6f70_xK9mQ2vR8nL4pT6wY1zA3bC5dE7fG9hJ

    ^^^^ ^^^^ ^^^^^^^^^^^^^^^^ ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    |    |    |                |
    |    |    |                secret -- argon2id-hashed, shown once, never stored
    |    |    key id -- indexed in plaintext
    |    environment
    prefix

**The key id is the reason for the format.** Without it, verifying a key means hashing the
presented secret against every stored hash in turn, and argon2id at 50 ms per row makes that a
denial-of-service vector against our own database. With it, verification is one indexed lookup
and one hash.

**The prefix is the reason for the environment segment.** It makes a leaked key greppable --
GitHub's secret scanning and every internal leak detector work on distinctive prefixes -- and it
makes "is this a production key" answerable without a lookup, which matters when one turns up in
a support ticket or a log.

**Keys can never hold `sso:configure`, `user:manage` or `audit:view`.** Those are the
capabilities that turn a leaked CI credential into a tenant takeover, and they are the ones a
machine client has no legitimate use for. The restriction is structural rather than advisory: it
is applied when the principal is built, so no admin UI can grant them by mistake.
"""

from __future__ import annotations

import hmac
import re
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

from app.security.capabilities import Capability, Role, capabilities_for

PREFIX = "erpk"

#: 16 hex characters. Long enough that guessing one is pointless, short enough to be quoted in a
#: support ticket and an audit row.
KEY_ID_BYTES = 8
#: Secret length in characters, over a 62-character alphabet: ~256 bits.
SECRET_CHARS = 43

#: Base62 -- deliberately no ``_`` or ``-``.
#:
#: ``token_urlsafe`` would be the obvious choice and is wrong here: its alphabet contains the
#: underscore we use as the field separator, so ``token.split("_")`` returns a truncated secret.
#: The regex parser survives that, but every other consumer -- a log scrubber, a support script,
#: a customer's own client -- does the obvious thing and gets it wrong. Base62 also survives
#: double-click selection in a terminal, which is how these actually get copied.
_ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

_SHAPE = rf"{PREFIX}_(live|test)_([0-9a-f]{{{KEY_ID_BYTES * 2}}})_([0-9A-Za-z]{{{SECRET_CHARS}}})"
_PATTERN = re.compile(rf"^{_SHAPE}$")

#: The same shape, unanchored, for finding a key inside a longer string.
#:
#: ``_PATTERN`` cannot do this job: anchored, it matches a whole string and nothing else, so
#: using it to scrub a log line silently does nothing at all -- which is the worst possible
#: outcome for a redaction function, because it reports success.
_EMBEDDED = re.compile(_SHAPE)

#: Cheaper argon2 parameters than a password uses, deliberately.
#:
#: A password is low-entropy and human-chosen, so it needs a slow hash to make offline cracking
#: expensive. An API key secret is 256 bits from the OS -- there is nothing to crack -- and it is
#: verified on *every request* from a machine client, where 50 ms per call is a real cost. This
#: still resists the only attack that applies: someone who steals the database and wants to use
#: the keys directly.
_hasher = PasswordHasher(time_cost=1, memory_cost=8192, parallelism=1, hash_len=32, salt_len=16)

#: Capabilities an API key may never hold, whatever its owner's role.
#:
#: These are what turn a leaked CI credential into a tenant takeover, and a machine client has no
#: legitimate use for any of them. Enforced when the principal is built, so it cannot be granted
#: by an admin UI, a migration or a seeding script.
FORBIDDEN_FOR_KEYS: frozenset[Capability] = frozenset(
    {
        Capability.SSO_CONFIGURE,
        Capability.USER_MANAGE,
        Capability.AUDIT_VIEW,
        Capability.APIKEY_MANAGE_ANY,
        Capability.EXPORT_DOCUMENTS,
    }
)


class KeyEnvironment(StrEnum):
    LIVE = "live"
    TEST = "test"


class InvalidApiKeyError(Exception):
    """The presented string is not a well-formed key, or does not match."""


@dataclass(frozen=True, slots=True)
class ParsedKey:
    environment: str
    key_id: str
    secret: str


@dataclass(frozen=True, slots=True)
class MintedKey:
    """A newly created key. ``token`` is shown once and never recoverable."""

    token: str
    key_id: str
    secret_hash: str
    environment: str
    prefix_hint: str
    """The first characters, stored so the UI can say which key a row refers to."""


@dataclass(frozen=True, slots=True)
class StoredKey:
    """The row a key id resolves to."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    key_id: str
    secret_hash: str
    role: Role
    #: A subset of the role's capabilities, chosen at creation. Subtractive only.
    granted: frozenset[Capability] = frozenset()
    expires_at: datetime | None = None
    revoked_at: datetime | None = None
    is_active: bool = True


def mint(environment: str = KeyEnvironment.LIVE) -> MintedKey:
    key_id = secrets.token_hex(KEY_ID_BYTES)
    secret = "".join(secrets.choice(_ALPHABET) for _ in range(SECRET_CHARS))
    token = f"{PREFIX}_{environment}_{key_id}_{secret}"
    return MintedKey(
        token=token,
        key_id=key_id,
        secret_hash=_hasher.hash(secret),
        environment=str(environment),
        # Enough to recognise a key in a list without being enough to use it.
        prefix_hint=f"{PREFIX}_{environment}_{key_id[:6]}",
    )


def parse(token: str) -> ParsedKey:
    """Split a presented key. Raises rather than returning None, because every caller must stop.

    Validated by shape before any database work, so a malformed string never becomes a lookup.
    """
    match = _PATTERN.match(token.strip())
    if not match:
        raise InvalidApiKeyError("That API key is not valid.")
    return ParsedKey(environment=match.group(1), key_id=match.group(2), secret=match.group(3))


def verify(token: str, stored: StoredKey, *, now: datetime | None = None) -> None:
    """Raise unless the key is well-formed, matches, and is currently usable.

    The order matters. The hash comparison happens before the expiry and revocation checks, so
    that a valid-but-expired key and an invalid one take the same time -- otherwise the response
    time distinguishes "this key existed" from "it never did".
    """
    parsed = parse(token)
    moment = now or datetime.now(UTC)

    if not hmac.compare_digest(parsed.key_id, stored.key_id):
        raise InvalidApiKeyError("That API key is not valid.")

    try:
        _hasher.verify(stored.secret_hash, parsed.secret)
    except (VerifyMismatchError, VerificationError, InvalidHashError) as exc:
        raise InvalidApiKeyError("That API key is not valid.") from exc

    if stored.revoked_at is not None or not stored.is_active:
        raise InvalidApiKeyError("That API key has been revoked.")
    if stored.expires_at is not None and stored.expires_at <= moment:
        raise InvalidApiKeyError("That API key has expired.")


def needs_rehash(stored: StoredKey) -> bool:
    return bool(_hasher.check_needs_rehash(stored.secret_hash))


def capabilities_of(stored: StoredKey) -> frozenset[Capability]:
    """What a key may actually do.

    Three reductions, applied in order, and each can only narrow:

    1. the capabilities of its owner's role;
    2. the subset chosen when the key was created;
    3. minus everything on ``FORBIDDEN_FOR_KEYS``.

    The third is the one that matters. It applies even to a key created by an ADMIN and even if
    an admin UI or a migration tried to grant more, because it is applied here rather than at the
    point of creation -- a key minted before this rule existed is narrowed the next time it is
    used, not left as it was.
    """
    from_role = capabilities_for(stored.role)
    chosen = stored.granted & from_role if stored.granted else from_role
    return frozenset(chosen - FORBIDDEN_FOR_KEYS)


def describe_expiry(stored: StoredKey, *, now: datetime | None = None) -> str:
    """Human wording for a key list. Expiry is the control people actually forget."""
    if stored.revoked_at:
        return "Revoked"
    if stored.expires_at is None:
        return "Never expires"
    remaining = stored.expires_at - (now or datetime.now(UTC))
    if remaining.total_seconds() <= 0:
        return "Expired"
    days = remaining.days
    if days == 0:
        return "Expires today"
    if days < 14:
        return f"Expires in {days} day{'s' if days != 1 else ''}"
    return f"Expires {stored.expires_at.date().isoformat()}"


def looks_like_api_key(value: str) -> bool:
    """Whether a string is one of ours.

    Used to redact keys out of logs and error messages before they are written. A key that
    reaches a log is a key that must be rotated, and the commonest way one gets there is a client
    pasting it into a field that gets echoed back.
    """
    return bool(_PATTERN.match(value.strip()))


def redact(text: str) -> str:
    """Replace every key in a string with its recognisable prefix and nothing else.

    Enough survives to identify which key must be rotated; nothing survives that could be used.
    """
    return _EMBEDDED.sub(lambda m: f"{PREFIX}_{m.group(1)}_{m.group(2)[:6]}_REDACTED", text)
