"""Application error taxonomy.

Every error carries a *safe* ``message`` (returned to the client) and an optional ``detail``
that is logged but never serialized. The split exists so that an authorization failure can be
debugged from the logs without telling the caller which capability they lacked.
"""

from __future__ import annotations

from typing import Any


class AppError(Exception):
    """Base class. ``status_code`` and ``code`` are what the API handler serializes."""

    status_code: int = 500
    code: str = "internal_error"
    message: str = "Something went wrong."

    def __init__(
        self,
        message: str | None = None,
        *,
        detail: str | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.message = message or type(self).message
        self.detail = detail
        self.extra = extra or {}
        super().__init__(self.message)


class NotFoundError(AppError):
    """Also raised for resources in another tenant.

    Returning 404 rather than 403 is deliberate: a 403 confirms the resource exists, which is
    itself a cross-tenant information leak.
    """

    status_code = 404
    code = "not_found"
    message = "Not found."


class AuthenticationError(AppError):
    status_code = 401
    code = "unauthenticated"
    message = "Authentication required."


class AuthorizationError(AppError):
    status_code = 403
    code = "forbidden"
    message = "You do not have permission to perform this action."


class ValidationError(AppError):
    status_code = 422
    code = "invalid_request"
    message = "The request was not valid."


class ConflictError(AppError):
    status_code = 409
    code = "conflict"
    message = "The request conflicts with the current state."


class RateLimitedError(AppError):
    status_code = 429
    code = "rate_limited"
    message = "Too many requests."

    def __init__(self, *, retry_after_seconds: int, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.retry_after_seconds = retry_after_seconds


class FeatureUnavailableError(AppError):
    """A capability that is not installed in this deployment (e.g. the parser container is off)."""

    status_code = 503
    code = "feature_unavailable"
    message = "That feature is not available in this deployment."


class CrossTenantWriteError(AppError):
    """A programming error, never a user error. Raised by the ORM guard.

    This must never reach a client as a 4xx: if it fires, the application tried to write a row
    into the wrong tenant and the correct response is a 500 plus a very loud log line.
    """

    status_code = 500
    code = "internal_error"
    message = "Something went wrong."


class ProgrammingError(AppError):
    """No principal bound, unscoped query, or similar contract violation."""

    status_code = 500
    code = "internal_error"
    message = "Something went wrong."
