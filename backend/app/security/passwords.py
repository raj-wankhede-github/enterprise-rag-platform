"""Password hashing and verification.

argon2id, because it is the only widely-available hash that resists GPU *and* ASIC attacks --
bcrypt's 4 KB working set fits in silicon cheaply, and PBKDF2 is almost free on a GPU. The
parameters below follow OWASP's 2024 guidance: 19 MiB, two iterations, one lane, which costs
roughly 40-60 ms on a server core.

Two behaviours matter more than the parameters.

**Verification takes the same time whether or not the user exists.** A login against an unknown
address must still perform a hash, or the response time answers "does this person have an
account here" for anyone who asks -- which in a multi-tenant product is a customer-list leak as
much as a user-list leak. ``verify_or_dummy`` exists for exactly that.

**Hashes are re-hashed on successful login when the parameters change.** Raising the cost is
otherwise a change that only affects accounts created afterwards, and the oldest accounts --
which are the ones most likely to have a reused password -- keep the weakest hash forever.
"""

from __future__ import annotations

import hmac
import secrets
import string
from dataclasses import dataclass

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError, VerifyMismatchError

#: OWASP 2024: m=19456 KiB, t=2, p=1. Tuned for a server core, not for a laptop -- this is a
#: production parameter and the development machine is not a design input.
_hasher = PasswordHasher(
    time_cost=2,
    memory_cost=19_456,
    parallelism=1,
    hash_len=32,
    salt_len=16,
)

#: Hashed once at import, so a login against an unknown address can still spend the same time.
#: The value is irrelevant; only the work is.
_DUMMY_HASH = _hasher.hash("a password that is never a password")

MIN_LENGTH = 12
MAX_LENGTH = 1024


@dataclass(frozen=True, slots=True)
class PasswordCheck:
    ok: bool
    #: True when the stored hash used weaker parameters than the current policy and should be
    #: replaced with a fresh hash of the password the user just proved they know.
    needs_rehash: bool = False


def hash_password(password: str) -> str:
    _guard_length(password)
    return _hasher.hash(password)


def verify_password(password: str, stored_hash: str) -> PasswordCheck:
    try:
        _hasher.verify(stored_hash, password)
    except VerifyMismatchError, VerificationError, InvalidHashError:
        return PasswordCheck(ok=False)
    return PasswordCheck(ok=True, needs_rehash=_hasher.check_needs_rehash(stored_hash))


def verify_or_dummy(password: str, stored_hash: str | None) -> PasswordCheck:
    """Verify, or burn the same work against a dummy when there is no hash to check.

    The case this covers is not only an unknown email. A user who signs in through SSO has
    ``password_hash = NULL``, so an early return here would distinguish "SSO-only account" from
    "wrong password" -- which tells an attacker exactly which accounts to phish instead.
    """
    if stored_hash is None:
        # The work is the point, not the result. It always mismatches.
        verify_password(password, _DUMMY_HASH)
        return PasswordCheck(ok=False)
    return verify_password(password, stored_hash)


def _guard_length(password: str) -> None:
    if len(password) < MIN_LENGTH:
        raise ValueError(f"Password must be at least {MIN_LENGTH} characters.")
    if len(password) > MAX_LENGTH:
        # Not a strength rule. An unbounded input to a memory-hard hash is a denial-of-service
        # vector -- the server does the work, and the attacker pays nothing to submit a megabyte.
        raise ValueError(f"Password must be at most {MAX_LENGTH} characters.")


def generate_password(length: int = 20) -> str:
    """A password for an administrator-created account or a break-glass reset.

    Ambiguous characters are excluded because these get read aloud and retyped. No punctuation:
    it survives being pasted through terminals, spreadsheets and ticketing systems, and the
    entropy is made up with length instead.
    """
    alphabet = "".join(sorted(set(string.ascii_letters + string.digits) - set("O0Il1")))
    return "".join(secrets.choice(alphabet) for _ in range(max(length, MIN_LENGTH)))


def constant_time_equals(left: str, right: str) -> bool:
    return hmac.compare_digest(left.encode("utf-8"), right.encode("utf-8"))


def looks_weak(password: str, *, email: str | None = None, name: str | None = None) -> str | None:
    """A short, deliberately incomplete check for the obviously bad.

    Not a complexity policy. Character-class rules push people toward ``Password1!`` and are
    worse than length alone, which is why the floor is twelve characters and the rest of this
    catches only what is embarrassing: the user's own address, their name, and the handful of
    passwords that appear in every breach corpus.
    """
    lowered = password.casefold()
    if email:
        local = email.split("@")[0].casefold()
        if len(local) >= 4 and local in lowered:
            return "The password must not contain your email address."
    if name and len(name) >= 4 and name.casefold() in lowered:
        return "The password must not contain your name."
    if lowered in _COMMON:
        return "This password appears in lists of commonly used passwords."
    if len(set(lowered)) <= 4:
        return "The password repeats too few distinct characters."
    return None


#: Not a breach corpus -- that belongs behind a service call. These are the ones long enough to
#: pass the length floor and still be among the first an attacker tries.
_COMMON: frozenset[str] = frozenset(
    {
        "password1234",
        "passw0rd1234",
        "qwertyuiop123",
        "123456789012",
        "iloveyou1234",
        "administrator",
        "letmein12345",
        "welcome12345",
        "monkey123456",
        "football1234",
        "trustno1trustno1",
        "changeme1234",
    }
)
