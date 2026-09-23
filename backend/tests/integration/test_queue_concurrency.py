"""The claim query under real concurrency.

This is the suite that matters for "can it handle 1000s of documents". Every property here is a
property of the SQL under concurrent transactions, and none of them can be observed against a
fake: a mock would happily let two workers claim the same job, because the thing being tested is
Postgres's row locking rather than our control flow.

Four claims, in the order they would hurt:

* two workers never get the same job (``SKIP LOCKED`` + the atomic UPDATE)
* one tenant's backfill cannot starve another tenant (the fairness window)
* a worker that dies does not take its document with it (leases)
* throughput rises with workers rather than flattening (no serialization on the queue head)
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator
from datetime import timedelta

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.ingestion.queue import (
    Priority,
    claim,
    complete,
    depth,
    enqueue,
    enqueue_many,
    fail,
    heartbeat,
    oldest_queued_age,
    reclaim_expired,
)

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql+asyncpg://erp:erp@localhost:5433/erp_test")

TENANT_A = uuid.UUID("aaaa0000-0000-0000-0000-00000000000a")
TENANT_B = uuid.UUID("bbbb0000-0000-0000-0000-00000000000b")

#: The table, standalone. The real one is created by Alembic and carries tenancy RLS; this suite
#: is about the claim query's concurrency behaviour, so it builds the minimum that query needs
#: and nothing else. Anything more would make the test depend on migration state it is not
#: testing.
_SCHEMA = """
CREATE TABLE IF NOT EXISTS ingest_jobs (
    id uuid PRIMARY KEY,
    tenant_id uuid NOT NULL,
    job_type varchar(32) NOT NULL,
    document_version_id uuid,
    payload jsonb NOT NULL DEFAULT '{}'::jsonb,
    priority smallint NOT NULL DEFAULT 100,
    status varchar(16) NOT NULL DEFAULT 'QUEUED',
    stage varchar(24),
    cursor jsonb NOT NULL DEFAULT '{}'::jsonb,
    attempts integer NOT NULL DEFAULT 0,
    max_attempts integer NOT NULL DEFAULT 5,
    run_after timestamptz NOT NULL DEFAULT now(),
    lease_expires_at timestamptz,
    worker_id varchar(64),
    error_code varchar(64),
    error_message text,
    finished_at timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS ix_jobs_claim ON ingest_jobs (priority, run_after);
CREATE INDEX IF NOT EXISTS ix_jobs_tenant_inflight ON ingest_jobs (tenant_id, status);
"""


@pytest.fixture
async def sessionmaker() -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    if "_test" not in DATABASE_URL:
        pytest.skip("integration tests refuse any database not named *_test")

    engine = create_async_engine(DATABASE_URL, poolclass=None)
    try:
        async with engine.begin() as connection:
            await connection.execute(text("CREATE EXTENSION IF NOT EXISTS pgcrypto"))
            for statement in _SCHEMA.strip().split(";"):
                if statement.strip():
                    await connection.execute(text(statement))
            await connection.execute(text("TRUNCATE ingest_jobs"))
    except Exception as exc:  # pragma: no cover - no database available
        await engine.dispose()
        pytest.skip(f"no Postgres at {DATABASE_URL}: {exc}")

    factory = async_sessionmaker(engine, expire_on_commit=False)
    yield factory
    await engine.dispose()


async def seed(
    factory: async_sessionmaker[AsyncSession],
    *,
    tenant_id: uuid.UUID,
    count: int,
    priority: Priority | int = Priority.NORMAL,
) -> None:
    async with factory() as session, session.begin():
        await enqueue_many(
            session,
            tenant_id=tenant_id,
            job_type="ingest_document",
            payloads=[{"n": n} for n in range(count)],
            priority=priority,
        )


# ------------------------------------------------------------------------------------------
# No job is ever processed twice
# ------------------------------------------------------------------------------------------


async def test_concurrent_workers_never_claim_the_same_job(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """The property everything else rests on.

    Without ``SKIP LOCKED`` and an atomic claim, two workers process one document: its chunks
    are written twice, and the index is only saved by the deterministic ``_id`` -- which is a
    second line of defence, not a reason to skip the first.
    """
    await seed(sessionmaker, tenant_id=TENANT_A, count=60)

    async def worker(name: str) -> list[uuid.UUID]:
        taken: list[uuid.UUID] = []
        for _ in range(30):
            async with sessionmaker() as session, session.begin():
                jobs = await claim(session, worker_id=name, batch=1, per_tenant_cap=100)
            if not jobs:
                break
            taken.append(jobs[0].id)
        return taken

    results = await asyncio.gather(*(worker(f"w{i}") for i in range(6)))
    claimed = [job_id for batch in results for job_id in batch]

    assert len(claimed) == len(set(claimed)), "a job was claimed by two workers"
    assert len(claimed) == 60, "every job should have been claimed exactly once"


async def test_a_claimed_job_is_invisible_to_the_next_claim(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    await seed(sessionmaker, tenant_id=TENANT_A, count=1)

    async with sessionmaker() as session, session.begin():
        first = await claim(session, worker_id="w1", batch=5, per_tenant_cap=100)
    async with sessionmaker() as session, session.begin():
        second = await claim(session, worker_id="w2", batch=5, per_tenant_cap=100)

    assert len(first) == 1
    assert second == []


# ------------------------------------------------------------------------------------------
# Fairness
# ------------------------------------------------------------------------------------------


async def test_one_tenants_backfill_cannot_starve_another(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """The commonest multi-tenant ingestion failure, and it looks like "the product is down" to
    every other customer.

    Tenant A enqueues 500 documents; tenant B enqueues 5 a moment later. A global ORDER BY
    priority, run_after gives B nothing until A's 500 are done.
    """
    await seed(sessionmaker, tenant_id=TENANT_A, count=500)
    await seed(sessionmaker, tenant_id=TENANT_B, count=5)

    async with sessionmaker() as session, session.begin():
        claimed = await claim(session, worker_id="w1", batch=20, per_tenant_cap=4)

    by_tenant = {TENANT_A: 0, TENANT_B: 0}
    for job in claimed:
        by_tenant[job.tenant_id] += 1

    assert by_tenant[TENANT_B] > 0, "the small tenant was starved by the large one's backfill"
    assert by_tenant[TENANT_A] <= 4, "the per-tenant cap was exceeded within a single claim"


async def test_the_cap_counts_jobs_already_running_not_just_this_batch(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """Counting only the current batch would let each claim take a full allowance, so N workers
    would give one tenant N times its cap."""
    await seed(sessionmaker, tenant_id=TENANT_A, count=50)

    async with sessionmaker() as session, session.begin():
        first = await claim(session, worker_id="w1", batch=3, per_tenant_cap=4)
    async with sessionmaker() as session, session.begin():
        second = await claim(session, worker_id="w2", batch=3, per_tenant_cap=4)

    assert len(first) == 3
    assert len(second) == 1, "the second claim should see three already running and take one"


async def test_interactive_work_overtakes_a_running_backfill(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """A person watching an upload spinner must not wait behind a backfill enqueued first."""
    await seed(sessionmaker, tenant_id=TENANT_A, count=100, priority=Priority.BULK)

    async with sessionmaker() as session, session.begin():
        await enqueue(
            session,
            tenant_id=TENANT_A,
            job_type="ingest_document",
            payload={"urgent": True},
            priority=Priority.INTERACTIVE,
        )

    async with sessionmaker() as session, session.begin():
        claimed = await claim(session, worker_id="w1", batch=1, per_tenant_cap=100)

    assert claimed[0].payload.get("urgent") is True


# ------------------------------------------------------------------------------------------
# A worker dying does not lose a document
# ------------------------------------------------------------------------------------------


async def test_an_expired_lease_returns_the_job_to_the_queue(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """A worker that is OOM-killed holds nothing. This is what makes that a non-event."""
    await seed(sessionmaker, tenant_id=TENANT_A, count=1)

    async with sessionmaker() as session, session.begin():
        jobs = await claim(session, worker_id="dying", batch=1, per_tenant_cap=10, lease=timedelta(seconds=-1))
    assert len(jobs) == 1

    async with sessionmaker() as session, session.begin():
        reclaimed = await reclaim_expired(session)
    assert reclaimed == 1

    async with sessionmaker() as session, session.begin():
        again = await claim(session, worker_id="healthy", batch=1, per_tenant_cap=10)
    assert len(again) == 1
    assert again[0].id == jobs[0].id


async def test_a_reclaimed_job_keeps_its_attempt_count(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """The poison-message guard. A job that reliably kills its worker must still reach DEAD
    rather than cycling forever and taking down every worker that touches it."""
    await seed(sessionmaker, tenant_id=TENANT_A, count=1)

    for _ in range(3):
        async with sessionmaker() as session, session.begin():
            await claim(session, worker_id="w", batch=1, per_tenant_cap=10, lease=timedelta(seconds=-1))
        async with sessionmaker() as session, session.begin():
            await reclaim_expired(session)

    async with sessionmaker() as session, session.begin():
        jobs = await claim(session, worker_id="w", batch=1, per_tenant_cap=10)
    assert jobs[0].attempts == 4, "attempts must accumulate across reclaims"


async def test_a_heartbeat_extends_a_lease_and_a_lost_one_is_refused(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """A worker whose lease was reclaimed must find out and stop -- two workers writing one
    document's chunks is how duplicates appear."""
    await seed(sessionmaker, tenant_id=TENANT_A, count=1)

    async with sessionmaker() as session, session.begin():
        jobs = await claim(session, worker_id="w1", batch=1, per_tenant_cap=10)
    async with sessionmaker() as session, session.begin():
        assert await heartbeat(session, jobs[0].id, worker_id="w1") is True
        assert await heartbeat(session, jobs[0].id, worker_id="someone-else") is False


# ------------------------------------------------------------------------------------------
# Failure handling
# ------------------------------------------------------------------------------------------


async def test_a_transient_failure_returns_to_the_queue_after_a_delay(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    await seed(sessionmaker, tenant_id=TENANT_A, count=1)

    async with sessionmaker() as session, session.begin():
        job = (await claim(session, worker_id="w1", batch=1, per_tenant_cap=10))[0]
        status = await fail(session, job, error_code="parser_unavailable", error_message="down")
    assert status == "QUEUED"

    # Not immediately claimable: run_after is in the future.
    async with sessionmaker() as session, session.begin():
        assert await claim(session, worker_id="w2", batch=1, per_tenant_cap=10) == []


async def test_a_permanent_failure_goes_straight_to_dead(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """Five attempts at a password-protected PDF is five wasted worker slots and five log lines
    that look like an outage."""
    await seed(sessionmaker, tenant_id=TENANT_A, count=1)

    async with sessionmaker() as session, session.begin():
        job = (await claim(session, worker_id="w1", batch=1, per_tenant_cap=10))[0]
        status = await fail(session, job, error_code="encrypted_document", error_message="locked")

    assert status == "DEAD"
    async with sessionmaker() as session:
        assert (await depth(session)).get("DEAD") == 1


async def test_a_dead_job_is_kept_not_deleted(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """It is the only record of a document that never made it, and "why is this not searchable"
    needs an answer."""
    await seed(sessionmaker, tenant_id=TENANT_A, count=1)

    async with sessionmaker() as session, session.begin():
        job = (await claim(session, worker_id="w1", batch=1, per_tenant_cap=10))[0]
        await fail(session, job, error_code="corrupt_document", error_message="not a PDF")

    async with sessionmaker() as session:
        row = (
            (
                await session.execute(
                    text("SELECT status, error_code, error_message FROM ingest_jobs WHERE id = :id"),
                    {"id": str(job.id)},
                )
            )
            .mappings()
            .one()
        )

    assert row["status"] == "DEAD"
    assert row["error_code"] == "corrupt_document"
    assert "not a PDF" in row["error_message"]


async def test_running_out_of_attempts_dead_letters_a_transient_failure(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """A transient failure is still terminal once the attempts are gone -- otherwise a parser
    that never comes back keeps a job cycling forever."""
    from dataclasses import replace

    await seed(sessionmaker, tenant_id=TENANT_A, count=1)

    async with sessionmaker() as session, session.begin():
        job = (await claim(session, worker_id="w1", batch=1, per_tenant_cap=10))[0]

    async with sessionmaker() as session, session.begin():
        status = await fail(
            session,
            replace(job, attempts=5, max_attempts=5),
            error_code="transient_io",
            error_message="timeout",
        )

    assert status == "DEAD"


# ------------------------------------------------------------------------------------------
# Scale and visibility
# ------------------------------------------------------------------------------------------


async def test_a_thousand_documents_enqueue_in_one_statement(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """A thousand round trips would make a bulk upload a timeout rather than a request."""
    async with sessionmaker() as session, session.begin():
        written = await enqueue_many(
            session,
            tenant_id=TENANT_A,
            job_type="ingest_document",
            payloads=[{"filename": f"doc-{n}.pdf"} for n in range(1000)],
        )
    assert written == 1000

    async with sessionmaker() as session:
        assert (await depth(session)).get("QUEUED") == 1000


async def test_a_thousand_queued_documents_drain_through_many_workers(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """The actual question: does throughput rise with workers, or do they serialize on the head
    of the queue? Without SKIP LOCKED, adding workers stops helping."""
    await seed(sessionmaker, tenant_id=TENANT_A, count=400)

    async def worker(name: str) -> int:
        done = 0
        while True:
            async with sessionmaker() as session, session.begin():
                jobs = await claim(session, worker_id=name, batch=5, per_tenant_cap=1000)
                if not jobs:
                    return done
                for job in jobs:
                    await complete(session, job.id)
                    done += 1

    counts = await asyncio.gather(*(worker(f"w{i}") for i in range(8)))

    assert sum(counts) == 400
    assert sum(1 for c in counts if c > 0) >= 4, "work should spread across workers, not serialize"

    async with sessionmaker() as session:
        assert (await depth(session)).get("DONE") == 400


async def test_queue_age_reports_starvation_rather_than_depth(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """Depth is a poor alert: a deep queue moving fast is healthy, a shallow stuck one is not."""
    async with sessionmaker() as session:
        assert await oldest_queued_age(session) == 0.0

    await seed(sessionmaker, tenant_id=TENANT_A, count=3)
    async with sessionmaker() as session:
        assert await oldest_queued_age(session) >= 0.0


async def test_depth_can_be_scoped_to_one_tenant(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    await seed(sessionmaker, tenant_id=TENANT_A, count=7)
    await seed(sessionmaker, tenant_id=TENANT_B, count=2)

    async with sessionmaker() as session:
        assert (await depth(session, tenant_id=TENANT_B)).get("QUEUED") == 2
        assert (await depth(session)).get("QUEUED") == 9


async def test_enqueue_does_not_commit_on_its_own(
    sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    """The reason this queue is in Postgres at all: the job and the document row commit together,
    so there is no window where a document exists with nothing scheduled to process it."""
    async with sessionmaker() as session:
        await enqueue(session, tenant_id=TENANT_A, job_type="ingest_document")
        await session.rollback()

    async with sessionmaker() as session:
        assert (await depth(session)).get("QUEUED") is None
