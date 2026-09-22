"""Request-scoped principal binding.

The ORM tenant guard (``db/guards.py``) and the audit writer both need the current tenant without
it being passed explicitly, so it lives in a ``ContextVar``. Workers bind it per job via
``with_principal`` -- there is no ambient default, because a missing principal must be a loud
error rather than a silent query across every tenant.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from app.core.errors import ProgrammingError
from app.security.principal import Principal

_current_principal: ContextVar[Principal | None] = ContextVar("current_principal", default=None)


def get_principal() -> Principal | None:
    return _current_principal.get()


def require_principal() -> Principal:
    principal = _current_principal.get()
    if principal is None:
        raise ProgrammingError(
            detail="No principal bound. API requests bind one in deps.get_principal; "
            "workers must use core.context.with_principal(...) around each job."
        )
    return principal


def require_tenant_id() -> uuid.UUID:
    return require_principal().tenant_id


@contextmanager
def with_principal(principal: Principal) -> Iterator[Principal]:
    token = _current_principal.set(principal)
    try:
        yield principal
    finally:
        _current_principal.reset(token)
