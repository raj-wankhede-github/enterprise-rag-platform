"""Engine, session factory, and the transaction-scoped tenant binding.

This is enforcement layer 2 of the four (``CLAUDE.md`` -> Tenancy). Layer 1 is the ContextVar
principal, layer 3 the ORM guards, layer 4 ``TenantRepository``. This layer is the one Postgres
itself enforces, and it is the only one that still holds when the application is wrong.

``SET LOCAL app.tenant_id`` is what row-level security reads. Three properties make it the right
mechanism rather than a filtered session:

* ``SET LOCAL`` is scoped to the **transaction**, so it cannot leak to the next checkout of a
  pooled connection. A plain ``SET`` would, and the resulting bug -- one request seeing another
  tenant's rows, only under load, only when the pool recycles -- is close to undebuggable.
* The application connects as a **non-owner** role, because ``FORCE ROW LEVEL SECURITY`` does not
  bind for a table's owner. Running migrations as the owner and the app as a non-owner is what
  makes the policies real rather than decorative.
* It is set from the *same* principal the ORM guards read, so the two agree by construction. Two
  independent notions of "the current tenant" would eventually disagree.

The session factory is deliberately not a global singleton created at import. A CLI command, a
worker and the API each build their own with different pool sizes, and an engine created at
import time binds to whatever event loop happens to import it first -- which is how a worker ends
up sharing connections with a test.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import event, text
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.core.config import Settings

logger = logging.getLogger(__name__)

#: The Postgres GUC row-level-security policies read. Namespaced so it cannot collide with a
#: setting Postgres or an extension owns.
TENANT_SETTING = "app.tenant_id"


def build_engine(settings: Settings, *, pool: bool = True) -> AsyncEngine:
    """One engine per process.

    ``statement_timeout`` is set per connection rather than per query: a runaway query on a
    shared cluster is a multi-tenant availability problem, and relying on every call site to
    remember a timeout is relying on the one that forgot.
    """
    return create_async_engine(
        settings.database_url,
        # NullPool for one-shot processes -- a CLI command or a migration -- where a pool just
        # delays exit while it drains.
        poolclass=None if pool else NullPool,
        pool_size=settings.db_pool_size if pool else 5,
        max_overflow=settings.db_max_overflow if pool else 0,
        pool_pre_ping=True,
        connect_args={
            "server_settings": {
                "statement_timeout": str(settings.db_statement_timeout_ms),
                "application_name": "enterprise-rag-platform",
            }
        },
        echo=False,
    )


def build_sessionmaker(settings: Settings, *, pool: bool = True) -> async_sessionmaker[AsyncSession]:
    """A session factory, with the ORM tenancy guards already installed.

    Installing the guards here rather than at each call site is the point: a session built any
    other way would be unguarded, and layer 3 would be opt-in rather than opt-out.
    """
    from app.db.guards import arm

    factory = async_sessionmaker(
        build_engine(settings, pool=pool),
        expire_on_commit=False,
        autoflush=False,
        class_=AsyncSession,
    )
    arm(factory)
    return factory


async def bind_tenant(session: AsyncSession, tenant_id: uuid.UUID | None) -> None:
    """Bind the transaction to a tenant so RLS policies apply.

    ``set_config(..., true)`` is the parameterised form of ``SET LOCAL``. The ``true`` is the
    part that matters: it makes the setting transaction-local, so it is discarded at commit or
    rollback and cannot survive into the next user of a pooled connection.

    Passing ``None`` clears the binding rather than leaving the previous value in place. A
    cleared binding makes every RLS-protected table return nothing, which is the correct failure
    for a code path that forgot to establish a principal -- an empty result is recoverable,
    another tenant's rows are not.
    """
    await session.execute(
        text(f"SELECT set_config('{TENANT_SETTING}', :tenant, true)"),
        {"tenant": str(tenant_id) if tenant_id else ""},
    )


@asynccontextmanager
async def tenant_session(
    factory: async_sessionmaker[AsyncSession], tenant_id: uuid.UUID
) -> AsyncIterator[AsyncSession]:
    """A session whose transaction is bound to one tenant, committed or rolled back on exit.

    Workers use this per job. There is deliberately no default tenant and no ambient fallback: a
    background job that forgot to bind should fail loudly, not quietly operate as whoever ran
    last.
    """
    async with factory() as session:
        await session.begin()
        await bind_tenant(session, tenant_id)
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()


@asynccontextmanager
async def platform_session(factory: async_sessionmaker[AsyncSession]) -> AsyncIterator[AsyncSession]:
    """A session with no tenant bound, for genuinely global tables.

    Named rather than reached by omission, so that "this query crosses tenants" is a visible,
    greppable decision. ``tenants``, ``platform_operators`` and ``index_generations`` are the
    legitimate users; anything else appearing here is a bug worth arguing about.
    """
    async with factory() as session:
        await session.begin()
        await bind_tenant(session, None)
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()


def log_slow_statements(engine: AsyncEngine, *, threshold_ms: float = 200.0) -> None:
    """Attach a slow-query log. Opt-in, because it costs a clock read per statement."""
    import time

    @event.listens_for(engine.sync_engine, "before_cursor_execute")
    def _before(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        context._query_start = time.perf_counter()

    @event.listens_for(engine.sync_engine, "after_cursor_execute")
    def _after(conn, cursor, statement, parameters, context, executemany):  # type: ignore[no-untyped-def]
        started = getattr(context, "_query_start", None)
        if started is None:
            return
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        if elapsed_ms >= threshold_ms:
            logger.warning("db.slow_statement", extra={"ms": round(elapsed_ms, 1), "sql": statement[:500]})
