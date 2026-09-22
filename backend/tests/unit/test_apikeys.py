"""API key format, verification and — the one that matters — capability narrowing.

The forbidden-capability tests are the point of this file. A leaked CI key that can reconfigure
SSO or manage users is a tenant takeover; one that can only search is an annoyance. The
restriction is applied when the principal is built rather than when the key is created, so a key
minted before the rule existed is narrowed the next time it is used.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.security.apikeys import (
    FORBIDDEN_FOR_KEYS,
    PREFIX,
    InvalidApiKeyError,
    StoredKey,
    capabilities_of,
    describe_expiry,
    looks_like_api_key,
    mint,
    parse,
    redact,
    verify,
)
from app.security.capabilities import Capability, Role, capabilities_for

TENANT = uuid.UUID("11110000-0000-0000-0000-000000000001")


def a_key(role: Role = Role.DEV, **kwargs: object) -> tuple[str, StoredKey]:
    minted = mint()
    base: dict[str, object] = {
        "id": uuid.uuid4(),
        "tenant_id": TENANT,
        "key_id": minted.key_id,
        "secret_hash": minted.secret_hash,
        "role": role,
    }
    base.update(kwargs)
    return minted.token, StoredKey(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------
# Format
# ------------------------------------------------------------------------------------------


def test_a_minted_key_has_the_documented_shape() -> None:
    minted = mint()
    parts = minted.token.split("_")
    assert parts[0] == PREFIX
    assert parts[1] == "live"
    assert parts[2] == minted.key_id
    assert len(parts[3]) >= 32


def test_the_key_id_is_recoverable_without_the_secret() -> None:
    """The whole reason for the format: verification is one indexed lookup, not a hash per row."""
    minted = mint()
    assert parse(minted.token).key_id == minted.key_id


def test_the_secret_is_never_stored() -> None:
    minted = mint()
    secret = parse(minted.token).secret
    assert secret not in minted.secret_hash
    assert minted.secret_hash.startswith("$argon2")


def test_the_prefix_hint_identifies_a_key_without_being_usable() -> None:
    minted = mint()
    assert minted.prefix_hint in minted.token
    assert len(minted.prefix_hint) < len(minted.token) // 2


def test_keys_are_unique() -> None:
    assert len({mint().token for _ in range(200)}) == 200


@pytest.mark.parametrize(
    "token",
    [
        "",
        "not-a-key",
        "erpk_live_short_secret",
        "erpk_prod_7f3a1b2c4d5e6f70_" + "x" * 40,  # unknown environment
        "erpk_live_ZZZZZZZZZZZZZZZZ_" + "x" * 40,  # non-hex key id
        "sk-openai-style-key-from-another-product",
    ],
)
def test_a_malformed_key_is_rejected_before_any_lookup(token: str) -> None:
    with pytest.raises(InvalidApiKeyError):
        parse(token)


def test_an_environment_is_visible_without_a_lookup() -> None:
    """Answers "is this a production key" when one turns up in a support ticket or a log."""
    assert parse(mint("test").token).environment == "test"
    assert parse(mint("live").token).environment == "live"


# ------------------------------------------------------------------------------------------
# Verification
# ------------------------------------------------------------------------------------------


def test_a_valid_key_verifies() -> None:
    token, stored = a_key()
    verify(token, stored)


def test_a_key_whose_secret_was_altered_is_refused() -> None:
    token, stored = a_key()
    tampered = token[:-4] + ("aaaa" if not token.endswith("aaaa") else "bbbb")
    with pytest.raises(InvalidApiKeyError):
        verify(tampered, stored)


def test_a_key_presented_against_another_rows_hash_is_refused() -> None:
    token, _ = a_key()
    _, other = a_key()
    with pytest.raises(InvalidApiKeyError):
        verify(token, other)


def test_a_revoked_key_is_refused() -> None:
    token, stored = a_key(revoked_at=datetime.now(UTC) - timedelta(minutes=1))
    with pytest.raises(InvalidApiKeyError, match="revoked"):
        verify(token, stored)


def test_a_deactivated_key_is_refused() -> None:
    token, stored = a_key(is_active=False)
    with pytest.raises(InvalidApiKeyError, match="revoked"):
        verify(token, stored)


def test_an_expired_key_is_refused() -> None:
    token, stored = a_key(expires_at=datetime.now(UTC) - timedelta(days=1))
    with pytest.raises(InvalidApiKeyError, match="expired"):
        verify(token, stored)


def test_a_key_expiring_in_the_future_still_works() -> None:
    token, stored = a_key(expires_at=datetime.now(UTC) + timedelta(days=30))
    verify(token, stored)


def test_the_hash_is_checked_before_expiry_so_the_two_cannot_be_told_apart_by_timing() -> None:
    """Otherwise the response time distinguishes "this key existed" from "it never did"."""
    import inspect

    from app.security import apikeys

    source = inspect.getsource(apikeys.verify)
    assert source.index("_hasher.verify") < source.index("revoked_at")


# ------------------------------------------------------------------------------------------
# Capabilities: the part that limits the blast radius
# ------------------------------------------------------------------------------------------


def test_a_key_can_never_reconfigure_sso_even_when_created_by_an_admin() -> None:
    """The capability that turns a leaked CI credential into a tenant takeover."""
    _, stored = a_key(role=Role.ADMIN)
    assert Capability.SSO_CONFIGURE not in capabilities_of(stored)


def test_a_key_can_never_manage_users() -> None:
    _, stored = a_key(role=Role.ADMIN)
    assert Capability.USER_MANAGE not in capabilities_of(stored)


def test_a_key_can_never_read_the_audit_log() -> None:
    _, stored = a_key(role=Role.ADMIN)
    assert Capability.AUDIT_VIEW not in capabilities_of(stored)


def test_a_key_can_never_bulk_export_documents() -> None:
    """Reading in an integration and walking out with the corpus are different risk classes."""
    _, stored = a_key(role=Role.ADMIN)
    assert Capability.EXPORT_DOCUMENTS not in capabilities_of(stored)


def test_a_key_can_never_mint_more_keys() -> None:
    """Otherwise one leaked key is an unbounded supply of them."""
    _, stored = a_key(role=Role.ADMIN)
    assert Capability.APIKEY_MANAGE_ANY not in capabilities_of(stored)


@pytest.mark.parametrize("forbidden", sorted(FORBIDDEN_FOR_KEYS))
def test_no_role_can_give_a_key_a_forbidden_capability(forbidden: Capability) -> None:
    for role in Role:
        _, stored = a_key(role=role)
        assert forbidden not in capabilities_of(stored)


def test_an_explicit_grant_cannot_widen_beyond_the_forbidden_set() -> None:
    """Applied when the principal is built rather than at creation, so a key minted before this
    rule existed is narrowed the next time it is used rather than left as it was."""
    _, stored = a_key(role=Role.ADMIN, granted=frozenset({Capability.SSO_CONFIGURE, Capability.SEARCH}))
    assert capabilities_of(stored) == frozenset({Capability.SEARCH})


def test_an_explicit_grant_cannot_widen_beyond_the_role() -> None:
    _, stored = a_key(role=Role.PROD, granted=frozenset(capabilities_for(Role.ADMIN)))
    assert capabilities_of(stored) <= capabilities_for(Role.PROD)


def test_no_grant_means_the_whole_role_minus_the_forbidden_set() -> None:
    _, stored = a_key(role=Role.DEV)
    assert capabilities_of(stored) == capabilities_for(Role.DEV) - FORBIDDEN_FOR_KEYS


def test_a_key_still_has_enough_to_be_useful() -> None:
    """The restrictions must not leave an integration unable to do its job."""
    _, stored = a_key(role=Role.DEV)
    granted = capabilities_of(stored)
    assert Capability.SEARCH in granted
    assert Capability.ASK in granted
    assert Capability.DOC_UPLOAD in granted


# ------------------------------------------------------------------------------------------
# Leak handling
# ------------------------------------------------------------------------------------------


def test_a_key_in_a_log_line_is_redacted_to_its_prefix() -> None:
    """A key that reaches a log must be rotated, and the commonest way one gets there is a
    client pasting it into a field that is echoed back."""
    token = mint().token
    line = f"request failed with Authorization: Bearer {token} on /api/search"
    redacted = redact(line)

    assert token not in redacted
    assert parse(token).secret not in redacted
    assert PREFIX in redacted, "enough must survive to recognise which key to rotate"


def test_redaction_leaves_ordinary_text_alone() -> None:
    assert redact("no keys here, just words") == "no keys here, just words"


def test_recognising_one_of_ours() -> None:
    assert looks_like_api_key(mint().token)
    assert not looks_like_api_key("sk-not-ours")
    assert not looks_like_api_key("")


# ------------------------------------------------------------------------------------------
# Expiry wording
# ------------------------------------------------------------------------------------------


def test_a_key_with_no_expiry_says_so_plainly() -> None:
    """Allowed, and also the thing worth nagging about: it outlives whoever created it."""
    _, stored = a_key()
    assert describe_expiry(stored) == "Never expires"


def test_an_expiry_within_two_weeks_is_counted_in_days() -> None:
    _, stored = a_key(expires_at=datetime.now(UTC) + timedelta(days=3, hours=1))
    assert describe_expiry(stored) == "Expires in 3 days"


def test_a_distant_expiry_is_given_as_a_date() -> None:
    deadline = datetime.now(UTC) + timedelta(days=200)
    _, stored = a_key(expires_at=deadline)
    assert describe_expiry(stored) == f"Expires {deadline.date().isoformat()}"


def test_revocation_outranks_expiry_in_the_wording() -> None:
    _, stored = a_key(
        revoked_at=datetime.now(UTC) - timedelta(days=1),
        expires_at=datetime.now(UTC) - timedelta(days=2),
    )
    assert describe_expiry(stored) == "Revoked"
