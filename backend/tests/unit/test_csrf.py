"""CSRF: the Origin allowlist, the double-submit token, and what is deliberately exempt."""

from __future__ import annotations

import pytest

from app.security.csrf import (
    CsrfError,
    CsrfPolicy,
    check,
    csrf_cookie_settings,
    new_token,
    normalize_origin,
    origin_of,
    tokens_match,
)

POLICY = CsrfPolicy(allowed_origins=frozenset({"https://acme.app.example.com"}))
TOKEN = "a-csrf-token-value"


def a_check(**kwargs: object) -> None:
    base: dict[str, object] = {
        "method": "POST",
        "origin": "https://acme.app.example.com",
        "referer": None,
        "cookie_token": TOKEN,
        "header_token": TOKEN,
        "policy": POLICY,
    }
    base.update(kwargs)
    check(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------
# Origin normalisation
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("https://app.example.com", "https://app.example.com"),
        ("https://app.example.com:443", "https://app.example.com"),
        ("http://localhost:80", "http://localhost"),
        ("http://localhost:5173", "http://localhost:5173"),
        ("HTTPS://APP.EXAMPLE.COM", "https://app.example.com"),
    ],
)
def test_the_default_port_is_implicit_and_the_host_is_lowercased(value: str, expected: str) -> None:
    """A browser may send either form. Comparing raw strings makes the allowlist wrong in a way
    that only shows up behind certain proxies."""
    assert normalize_origin(value) == expected


def test_a_referer_is_reduced_to_its_origin() -> None:
    assert origin_of("https://app.example.com/search?q=secret") == "https://app.example.com"


def test_no_referer_yields_no_origin() -> None:
    assert origin_of(None) is None
    assert origin_of("") is None


# ------------------------------------------------------------------------------------------
# What passes and what does not
# ------------------------------------------------------------------------------------------


def test_a_well_formed_request_passes() -> None:
    a_check()


@pytest.mark.parametrize("method", ["GET", "HEAD", "OPTIONS", "get", "options"])
def test_safe_methods_are_not_checked(method: str) -> None:
    """A preflight must not require a token it has no way to carry."""
    a_check(method=method, origin="https://attacker.example.com", cookie_token=None, header_token=None)


def test_a_foreign_origin_is_refused_even_with_matching_tokens() -> None:
    """The Origin check is the strongest of the three: page script cannot forge it."""
    with pytest.raises(CsrfError, match="allowed origin"):
        a_check(origin="https://attacker.example.com")


def test_the_rejection_does_not_echo_the_origin_back() -> None:
    """Echoing it confirms to an attacker which of their attempts is being evaluated."""
    with pytest.raises(CsrfError) as raised:
        a_check(origin="https://attacker.example.com")
    assert "attacker.example.com" not in str(raised.value)


def test_a_missing_token_is_refused_even_from_an_allowed_origin() -> None:
    with pytest.raises(CsrfError, match="security token"):
        a_check(header_token=None)


def test_a_mismatched_token_is_refused() -> None:
    with pytest.raises(CsrfError):
        a_check(header_token="a-different-token")


def test_tokens_are_compared_in_constant_time_and_never_match_when_empty() -> None:
    assert tokens_match(TOKEN, TOKEN)
    assert not tokens_match("", "")
    assert not tokens_match(None, TOKEN)
    assert not tokens_match(TOKEN, None)


def test_the_referer_is_used_when_origin_is_absent() -> None:
    """Weaker, but a check that runs beats one that is skipped."""
    a_check(origin=None, referer="https://acme.app.example.com/documents")


def test_a_foreign_referer_is_refused() -> None:
    with pytest.raises(CsrfError):
        a_check(origin=None, referer="https://attacker.example.com/page")


def test_a_request_with_neither_origin_nor_referer_is_refused_in_production() -> None:
    """Every browser sends Origin on a cross-site state-changing request. A missing one means a
    non-browser client, which belongs on the bearer path with an API key."""
    with pytest.raises(CsrfError):
        a_check(origin=None, referer=None)


def test_a_deployment_behind_a_header_stripping_proxy_can_relax_that() -> None:
    relaxed = CsrfPolicy(allowed_origins=POLICY.allowed_origins, require_origin=False)
    a_check(origin=None, referer=None, policy=relaxed)


def test_a_relaxed_policy_still_requires_the_double_submit_token() -> None:
    """Relaxing the Origin requirement leaves the token doing the work alone; dropping both
    would leave no CSRF protection at all."""
    relaxed = CsrfPolicy(allowed_origins=POLICY.allowed_origins, require_origin=False)
    with pytest.raises(CsrfError):
        a_check(origin=None, referer=None, header_token=None, policy=relaxed)


# ------------------------------------------------------------------------------------------
# Bearer exemption
# ------------------------------------------------------------------------------------------


def test_bearer_authenticated_requests_are_exempt() -> None:
    """Correct rather than a shortcut: an Authorization header is never attached automatically
    by a browser, so there is nothing for a cross-site page to forge. Requiring CSRF here would
    break every server-to-server client for no gain."""
    a_check(origin=None, referer=None, cookie_token=None, header_token=None, is_bearer=True)


# ------------------------------------------------------------------------------------------
# The cookie
# ------------------------------------------------------------------------------------------


def test_the_csrf_cookie_is_deliberately_readable_by_script() -> None:
    """Page script has to read it to set the header -- that is the whole mechanism. It is not a
    credential: on its own it authenticates nothing."""
    assert csrf_cookie_settings()["httponly"] is False
    assert csrf_cookie_settings()["secure"] is True


def test_tokens_carry_real_entropy_and_do_not_repeat() -> None:
    assert len({new_token() for _ in range(200)}) == 200
