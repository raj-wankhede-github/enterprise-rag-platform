"""CSRF protection: three overlapping controls, none of them sufficient alone.

Cookie-carried credentials are convenient and safe from XSS exfiltration, and they are also the
reason CSRF exists at all -- the browser attaches them to a cross-site request just as
eagerly as to a legitimate one.

1. **``SameSite=Lax``** stops the simple cases outright. It cannot be ``Strict``, because that
   would drop the cookie on the top-level navigation back from an OIDC provider and break every
   SSO login. Lax leaves top-level GET navigations covered by the others.
2. **Origin allowlist.** ``Origin`` is set by the browser on every state-changing request and
   cannot be forged by page script. It is the strongest of the three and the cheapest to check.
3. **Double-submit token.** A random value in a readable cookie, echoed in a header. Same-origin
   policy stops a cross-site page reading the cookie, so it cannot produce the header.

Three rather than one because the sibling-subdomain case matters here. In BYOC a customer may
serve the product beside their own applications on the same registrable domain, and a cookie
without the ``__Host-`` prefix is writable from a sibling -- which defeats double-submit on its
own. The Origin check does not care, and ``__Host-`` closes it properly where it applies.
"""

from __future__ import annotations

import hmac
import logging
import secrets
from dataclasses import dataclass
from urllib.parse import urlsplit

logger = logging.getLogger(__name__)

CSRF_COOKIE = "erp_csrf"
CSRF_HEADER = "X-CSRF-Token"

#: Methods that cannot change state, so they are not protected. HEAD and OPTIONS are included
#: because a preflight must not require a token it has no way to carry.
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})

TOKEN_BYTES = 32


class CsrfError(Exception):
    """The request failed a cross-site request forgery check."""


def new_token() -> str:
    return secrets.token_urlsafe(TOKEN_BYTES)


def tokens_match(cookie_value: str | None, header_value: str | None) -> bool:
    if not cookie_value or not header_value:
        return False
    return hmac.compare_digest(cookie_value, header_value)


@dataclass(frozen=True, slots=True)
class CsrfPolicy:
    """The allowlist, and how strictly a missing Origin is treated."""

    allowed_origins: frozenset[str] = frozenset()
    #: Whether a request with no Origin *and* no Referer is refused.
    #:
    #: True in production. Every browser sends Origin on a cross-site state-changing request, so
    #: a missing one means a non-browser client -- which should be using an API key against the
    #: bearer path, not a cookie session. False only where a proxy is known to strip the header,
    #: and then the double-submit token is doing the work alone.
    require_origin: bool = True

    def origin_allowed(self, origin: str | None) -> bool:
        if not origin:
            return not self.require_origin
        return normalize_origin(origin) in self.allowed_origins


def normalize_origin(value: str) -> str:
    """Scheme, host and port only -- never a path, and the default port made implicit.

    ``https://app.example.com`` and ``https://app.example.com:443`` are the same origin and a
    browser may send either. Comparing the raw strings makes the allowlist wrong in a way that
    only appears behind certain proxies.
    """
    parts = urlsplit(value if "//" in value else f"//{value}")
    scheme = (parts.scheme or "https").lower()
    host = (parts.hostname or "").lower()
    port = parts.port
    if port is None or (scheme == "https" and port == 443) or (scheme == "http" and port == 80):
        return f"{scheme}://{host}"
    return f"{scheme}://{host}:{port}"


def origin_of(referer: str | None) -> str | None:
    """Fall back to Referer when Origin is absent.

    Weaker -- a referrer policy can strip or truncate it, and it carries a path we must discard
    -- but it is present on some requests where Origin is not, and a check that runs is better
    than one that is skipped.
    """
    if not referer:
        return None
    parts = urlsplit(referer)
    return normalize_origin(f"{parts.scheme}://{parts.netloc}") if parts.netloc else None


def check(
    *,
    method: str,
    origin: str | None,
    referer: str | None,
    cookie_token: str | None,
    header_token: str | None,
    policy: CsrfPolicy,
    is_bearer: bool = False,
) -> None:
    """Raise ``CsrfError`` unless the request passes.

    Bearer-authenticated requests are exempt, and that is correct rather than a shortcut: an
    ``Authorization`` header is never attached by the browser automatically, so there is nothing
    for a cross-site page to forge. Applying CSRF to API keys would break every server-to-server
    client for no gain.
    """
    if method.upper() in SAFE_METHODS or is_bearer:
        return

    effective = origin or origin_of(referer)
    if not policy.origin_allowed(effective):
        # Deliberately does not echo the origin back to the client -- it would confirm to an
        # attacker exactly which of their attempts is being evaluated.
        logger.warning("csrf.origin_rejected", extra={"origin": effective})
        raise CsrfError("This request did not come from an allowed origin.")

    if not tokens_match(cookie_token, header_token):
        logger.warning(
            "csrf.token_mismatch", extra={"has_cookie": bool(cookie_token), "has_header": bool(header_token)}
        )
        raise CsrfError("This request is missing a valid security token. Please reload the page and try again.")


def csrf_cookie_settings(*, secure: bool = True) -> dict[str, object]:
    """The CSRF cookie is the one cookie that is deliberately **not** HttpOnly.

    Page script has to read it to set the header -- that is the entire mechanism. It is not a
    credential: on its own it authenticates nothing, and it is useless to a cross-site page
    because same-origin policy stops that page reading it.
    """
    return {"httponly": False, "secure": secure, "samesite": "lax", "path": "/"}
