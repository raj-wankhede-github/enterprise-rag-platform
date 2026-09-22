"""Tenant isolation layers 3 and 4: the ORM guards.

Layer 1 is the ``ContextVar`` principal (``core/context.py``); layer 2 is ``SET LOCAL
app.tenant_id`` plus Postgres row-level security (``db/session.py`` and the RLS migration).
This module adds the two that produce good error messages and work without a database:

* ``do_orm_execute`` appends a ``tenant_id`` criterion to every SELECT against a
  ``TenantScoped`` model, so a repository that forgets the filter is still scoped.
* ``before_flush`` stamps ``tenant_id`` on inserts and raises ``CrossTenantWriteError`` on any
  attempt to write or move a row into another tenant.

The guards are armed per ``Session`` in ``db/session.py``. They are deliberately loud: a
violation is a programming error, so it raises rather than silently filtering.
"""

from __future__ import annotations

from typing import Any

from sqlalchemy import event
from sqlalchemy.orm import ORMExecuteState, Session, with_loader_criteria
from sqlalchemy.orm.attributes import get_history

from app.core.context import require_tenant_id
from app.core.errors import CrossTenantWriteError
from app.db.base import TenantScoped

#: Execution option that skips the SELECT guard. Only the platform control plane sets it, and
#: every use is audited. Grep for it in review -- there should be a handful, all in app/platform.
SKIP_TENANT_GUARD = "skip_tenant_guard"


def _apply_tenant_filter(state: ORMExecuteState) -> None:
    if state.is_column_load or state.is_relationship_load:
        # Lazy loads of an already-scoped parent; re-filtering here would break eager loading.
        return
    if not state.is_select:
        return
    if state.execution_options.get(SKIP_TENANT_GUARD):
        return
    tenant_id = require_tenant_id()
    state.statement = state.statement.options(
        with_loader_criteria(
            TenantScoped,
            lambda cls: cls.tenant_id == tenant_id,
            include_aliases=True,
        )
    )


def _stamp_and_verify(session: Session, flush_context: Any, instances: Any) -> None:
    tenant_id = require_tenant_id()

    for obj in session.new:
        if not isinstance(obj, TenantScoped):
            continue
        current = getattr(obj, "tenant_id", None)
        if current is None:
            obj.tenant_id = tenant_id
        elif current != tenant_id:
            raise CrossTenantWriteError(
                detail=f"Refused to insert {type(obj).__name__} into tenant {current} "
                f"while the bound principal is in tenant {tenant_id}."
            )

    for obj in session.dirty:
        if not isinstance(obj, TenantScoped):
            continue
        history = get_history(obj, "tenant_id")
        if history.has_changes():
            raise CrossTenantWriteError(
                detail=f"{type(obj).__name__}.tenant_id is immutable; attempted {history.deleted} -> {history.added}."
            )
        if getattr(obj, "tenant_id", None) != tenant_id:
            raise CrossTenantWriteError(detail=f"Refused to update {type(obj).__name__} belonging to another tenant.")

    for obj in session.deleted:
        if isinstance(obj, TenantScoped) and getattr(obj, "tenant_id", None) != tenant_id:
            raise CrossTenantWriteError(detail=f"Refused to delete {type(obj).__name__} belonging to another tenant.")


def arm(session_factory: Any) -> None:
    """Attach the guards to a sessionmaker (or the Session class in tests)."""
    event.listen(session_factory, "do_orm_execute", _apply_tenant_filter)
    event.listen(session_factory, "before_flush", _stamp_and_verify)


def disarm(session_factory: Any) -> None:
    """Tests only: proves that RLS alone still blocks cross-tenant reads."""
    event.remove(session_factory, "do_orm_execute", _apply_tenant_filter)
    event.remove(session_factory, "before_flush", _stamp_and_verify)
