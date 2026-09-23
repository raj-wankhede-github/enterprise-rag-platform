"""The ingestion queue: Postgres, ``FOR UPDATE SKIP LOCKED``, leases and per-tenant fairness.

**Why Postgres and not Redis or SQS.** The decisive property is that *enqueue commits in the same
transaction as the business row*. A document row and its job are written atomically, so there is
no window in which a document exists with no job to process it -- the "saved but never ingested"
bug that a separate broker needs an outbox pattern to avoid. We already have a transactional
database in the request path; adding a broker to get a weaker guarantee is a poor trade at this
scale. At a genuinely large one, the right move is an outbox *into* a broker, and this module is
the thing that would feed it.

Four properties the implementation has to get right, in order of how expensive they are to
discover in production:

**A lease, not a lock.** A worker that is OOM-killed mid-document holds nothing. Its job has a
``lease_expires_at`` in the past, and ``reclaim_expired`` returns it to the queue. A lock held by
a dead process needs an operator; a lease that expires needs nobody.

**Per-tenant fairness, enforced in the claim query.** One tenant's 200,000-document backfill must
not starve everyone else, and a global ``ORDER BY priority`` does exactly that -- it is the
commonest multi-tenant ingestion failure and it looks like "the product is down" to every other
customer. The claim caps in-flight jobs per tenant, counting both what is already running and
what this batch is about to take.

**Stage checkpointing.** A crash at document 900 of 1000 resumes at 900, not at 1. The stages are
ordered and recorded, so re-running a job skips the expensive work it already paid for --
parsing, contextualization and embedding, in that order of cost.

**Permanent failures do not retry.** A password-protected PDF fails identically five times, and
retrying it wastes a worker slot that a fetchable document needs. Retryable and permanent are
different outcomes with different destinations.
"""

from __future__ import annotations

import logging
import random
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import IntEnum, StrEnum
from typing import Any

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


class JobStage(StrEnum):
    """Checkpoints within one document's ingestion, in the order they happen.

    Recorded so a retry resumes rather than restarts. The ordering is what makes
    ``reached`` meaningful, and the cost of re-doing a stage rises steeply down this list.
    """

    FETCHED = "FETCHED"
    PARSED = "PARSED"
    CHUNKED = "CHUNKED"
    CONTEXTUALIZED = "CONTEXTUALIZED"
    EMBEDDED = "EMBEDDED"
    INDEXED = "INDEXED"
    ACTIVATED = "ACTIVATED"


_STAGE_ORDER: dict[str, int] = {stage: index for index, stage in enumerate(JobStage)}


def reached(current: str | None, target: JobStage) -> bool:
    """Whether a job has already got past ``target`` and can skip that work.

    The whole point of checkpointing: a retry that re-parses and re-embeds a document it already
    embedded is a retry that costs as much as the original attempt.
    """
    if current is None:
        return False
    return _STAGE_ORDER.get(current, -1) >= _STAGE_ORDER[target]


class Priority(IntEnum):
    """Lower runs first. Spaced so a band can be inserted without renumbering.

    The separation that matters is interactive from bulk. Someone waiting at an upload dialog and
    a 200,000-document backfill are not the same request, and a single queue that cannot tell
    them apart makes the person wait behind the backfill.
    """

    INTERACTIVE = 10
    """A user uploaded a file and is watching a spinner."""

    NORMAL = 100
    """A connector's incremental sync."""

    BULK = 500
    """A full backfill or re-ingest. Runs when nothing else needs the slot."""


class JobOutcome(StrEnum):
    DONE = "DONE"
    RETRY = "RETRY"
    """Transient: a parser restarting, a 429, a network blip. Worth another attempt."""

    PERMANENT = "PERMANENT"
    """A property of the document, not of the moment. Retrying wastes a worker slot."""


#: Error codes that must never be retried.
#:
#: Each fails identically on every attempt, so a retry costs a worker slot that a fetchable
#: document needs -- and five attempts turn one bad document into five log entries that look
#: like an outage.
PERMANENT_ERRORS: frozenset[str] = frozenset(
    {
        "unsupported_format",
        "encrypted_document",
        "corrupt_document",
        "document_too_large",
        "mapping_rejected",
        "tenant_deleted",
        "blob_missing",
    }
)

#: Ceiling on the retry backoff. Beyond a few minutes the queue stops looking like a queue and
#: starts looking like a scheduler, and a genuinely transient failure has long since cleared.
MAX_BACKOFF = timedelta(minutes=10)

#: How long a claimed job is the worker's before another may take it.
#:
#: Longer than the slowest realistic document -- an OCR-heavy scan through Docling -- because a
#: lease that expires under a worker that is still running produces two workers on one document.
#: The heartbeat extends it, so this is the tolerance for a worker dying, not for it being slow.
DEFAULT_LEASE = timedelta(minutes=10)


@dataclass(frozen=True, slots=True)
class ClaimedJob:
    """A job this worker holds a lease on."""

    id: uuid.UUID
    tenant_id: uuid.UUID
    job_type: str
    document_version_id: uuid.UUID | None
    payload: dict[str, Any]
    stage: str | None
    cursor: dict[str, Any]
    attempts: int
    max_attempts: int
    priority: int
    lease_expires_at: datetime

    @property
    def is_last_attempt(self) -> bool:
        return self.attempts >= self.max_attempts


def backoff_for(attempt: int, *, base_seconds: float = 2.0, jitter: bool = True) -> timedelta:
    """Exponential backoff with full jitter, capped.

    Jitter is not decoration. A parser container restarting fails every in-flight job at the same
    instant; without jitter all of them retry at the same instant too, and the retry storm is
    what keeps the parser down. Full jitter spreads them across the whole window rather than
    synchronising them on its edge.
    """
    raw = min(base_seconds * (2 ** max(0, attempt - 1)), MAX_BACKOFF.total_seconds())
    # Scheduling jitter, not a secret: predicting a retry time buys an attacker nothing, and
    # a CSPRNG here would cost entropy for no gain.
    return timedelta(seconds=random.uniform(0, raw) if jitter else raw)  # noqa: S311


# ----------------------------------------------------------------------------------------------
# Enqueue
# ----------------------------------------------------------------------------------------------

_ENQUEUE = text("""
    INSERT INTO ingest_jobs (
        id, tenant_id, job_type, document_version_id, payload,
        priority, status, attempts, max_attempts, run_after, created_at, updated_at
    )
    VALUES (
        gen_random_uuid(), :tenant_id, :job_type, :document_version_id, CAST(:payload AS jsonb),
        :priority, 'QUEUED', 0, :max_attempts, now(), now(), now()
    )
    RETURNING id
""")


async def enqueue(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    job_type: str,
    document_version_id: uuid.UUID | None = None,
    payload: dict[str, Any] | None = None,
    priority: Priority | int = Priority.NORMAL,
    max_attempts: int = 5,
) -> uuid.UUID:
    """Add one job. **Does not commit** -- that is the entire point.

    The caller commits the document row and the job together, so there is no window in which a
    document exists with nothing scheduled to process it. A queue that commits on its own behalf
    reintroduces exactly the gap this design exists to close.
    """
    import json

    result = await session.execute(
        _ENQUEUE,
        {
            "tenant_id": str(tenant_id),
            "job_type": job_type,
            "document_version_id": str(document_version_id) if document_version_id else None,
            "payload": json.dumps(payload or {}),
            "priority": int(priority),
            "max_attempts": max_attempts,
        },
    )
    return uuid.UUID(str(result.scalar_one()))


async def enqueue_many(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID,
    job_type: str,
    payloads: list[dict[str, Any]],
    priority: Priority | int = Priority.BULK,
    max_attempts: int = 5,
) -> int:
    """Bulk enqueue, one statement.

    A thousand-document upload is one round trip, not a thousand. Defaults to ``BULK`` priority
    because that is what a thousand documents at once almost always is, and a caller who means
    otherwise should have to say so.
    """
    import json

    if not payloads:
        return 0

    await session.execute(
        text("""
            INSERT INTO ingest_jobs (
                id, tenant_id, job_type, document_version_id, payload,
                priority, status, attempts, max_attempts, run_after, created_at, updated_at
            )
            SELECT
                gen_random_uuid(), :tenant_id, :job_type,
                CAST(item->>'document_version_id' AS uuid), item,
                :priority, 'QUEUED', 0, :max_attempts, now(), now(), now()
            FROM jsonb_array_elements(CAST(:items AS jsonb)) AS item
        """),
        {
            "tenant_id": str(tenant_id),
            "job_type": job_type,
            "priority": int(priority),
            "max_attempts": max_attempts,
            "items": json.dumps(payloads),
        },
    )
    return len(payloads)


# ----------------------------------------------------------------------------------------------
# Claim
# ----------------------------------------------------------------------------------------------

#: The claim. One statement, and every property of the queue lives in it.
#:
#: ``FOR UPDATE ... SKIP LOCKED`` is what lets N workers claim concurrently without coordinating:
#: each skips rows another has locked rather than blocking on them. Without SKIP LOCKED, workers
#: serialize on the head of the queue and adding workers stops helping.
#:
#: ``FOR UPDATE OF j`` names the table explicitly because Postgres refuses to lock the nullable
#: side of an outer join, and the in-flight count is a LEFT JOIN.
#:
#: The window function is the fairness. Counting only what is *already* running would let a
#: single claim take a whole batch for one tenant and blow past the cap; ``rn`` is that job's
#: position within its own tenant in this batch, so ``rn + running <= cap`` holds afterwards too.
_CLAIM = text("""
    WITH in_flight AS (
        SELECT tenant_id, count(*) AS running
        FROM ingest_jobs
        WHERE status = 'RUNNING' AND lease_expires_at > now()
        GROUP BY tenant_id
    ),
    ranked AS (
        SELECT
            j.id,
            row_number() OVER (PARTITION BY j.tenant_id ORDER BY j.priority, j.run_after, j.id) AS rn,
            coalesce(f.running, 0) AS running
        FROM ingest_jobs j
        LEFT JOIN in_flight f ON f.tenant_id = j.tenant_id
        WHERE j.status = 'QUEUED' AND j.run_after <= now()
    ),
    claimable AS (
        SELECT r.id
        FROM ranked r
        JOIN ingest_jobs j ON j.id = r.id
        WHERE r.rn + r.running <= :per_tenant_cap
        ORDER BY j.priority, j.run_after, j.id
        LIMIT :batch
        FOR UPDATE OF j SKIP LOCKED
    )
    UPDATE ingest_jobs j
    SET status = 'RUNNING',
        attempts = j.attempts + 1,
        worker_id = :worker_id,
        lease_expires_at = now() + make_interval(secs => :lease_seconds),
        updated_at = now()
    FROM claimable c
    -- The status is re-checked HERE, not only in the CTE, and that is not redundant.
    --
    -- Under READ COMMITTED, Postgres re-reads a concurrently-updated row and re-evaluates the
    -- quals of the *locking* query before proceeding (EvalPlanQual). A predicate inside a CTE is
    -- not part of that query level, so it is NOT re-checked. The window: this transaction's CTE
    -- snapshot sees a job QUEUED, another transaction claims and commits it, the row is no
    -- longer locked so SKIP LOCKED does not skip it -- and without this clause we claim a job
    -- that is already RUNNING. Two workers then process one document.
    --
    -- Found by test_a_thousand_queued_documents_drain_through_many_workers, which completed 420
    -- of 400 jobs before this line existed.
    WHERE j.id = c.id AND j.status = 'QUEUED' AND j.run_after <= now()
    RETURNING j.id, j.tenant_id, j.job_type, j.document_version_id, j.payload,
              j.stage, j.cursor, j.attempts, j.max_attempts, j.priority, j.lease_expires_at
""")


async def claim(
    session: AsyncSession,
    *,
    worker_id: str,
    batch: int = 1,
    per_tenant_cap: int = 4,
    lease: timedelta = DEFAULT_LEASE,
) -> list[ClaimedJob]:
    """Take up to ``batch`` jobs, respecting the per-tenant in-flight cap."""
    result = await session.execute(
        _CLAIM,
        {
            "worker_id": worker_id[:64],
            "batch": batch,
            "per_tenant_cap": per_tenant_cap,
            "lease_seconds": lease.total_seconds(),
        },
    )
    rows = result.mappings().all()
    return [
        ClaimedJob(
            id=row["id"],
            tenant_id=row["tenant_id"],
            job_type=row["job_type"],
            document_version_id=row["document_version_id"],
            payload=dict(row["payload"] or {}),
            stage=row["stage"],
            cursor=dict(row["cursor"] or {}),
            attempts=row["attempts"],
            max_attempts=row["max_attempts"],
            priority=row["priority"],
            lease_expires_at=row["lease_expires_at"],
        )
        for row in rows
    ]


# ----------------------------------------------------------------------------------------------
# Progress and completion
# ----------------------------------------------------------------------------------------------


async def heartbeat(
    session: AsyncSession, job_id: uuid.UUID, *, worker_id: str, lease: timedelta = DEFAULT_LEASE
) -> bool:
    """Extend the lease on a job still being worked.

    Returns False when the lease was already lost -- reclaimed by the sweeper because this worker
    stalled. The worker must then **stop**: another worker owns the job, and two workers writing
    one document's chunks is how duplicates appear in an index that is otherwise idempotent.
    """
    result = await session.execute(
        text("""
            UPDATE ingest_jobs
            SET lease_expires_at = now() + make_interval(secs => :lease_seconds), updated_at = now()
            WHERE id = :id AND worker_id = :worker_id AND status = 'RUNNING'
            RETURNING id
        """),
        {"id": str(job_id), "worker_id": worker_id[:64], "lease_seconds": lease.total_seconds()},
    )
    return result.first() is not None


async def checkpoint(
    session: AsyncSession, job_id: uuid.UUID, *, stage: JobStage, cursor: dict[str, Any] | None = None
) -> None:
    """Record that a stage completed, so a retry resumes after it."""
    import json

    await session.execute(
        text("""
            UPDATE ingest_jobs
            SET stage = :stage,
                cursor = COALESCE(CAST(:cursor AS jsonb), cursor),
                updated_at = now()
            WHERE id = :id
        """),
        {"id": str(job_id), "stage": str(stage), "cursor": json.dumps(cursor) if cursor is not None else None},
    )


async def complete(session: AsyncSession, job_id: uuid.UUID) -> None:
    await session.execute(
        text("""
            UPDATE ingest_jobs
            SET status = 'DONE', stage = 'ACTIVATED', finished_at = now(),
                lease_expires_at = NULL, error_code = NULL, error_message = NULL, updated_at = now()
            WHERE id = :id
        """),
        {"id": str(job_id)},
    )


async def fail(
    session: AsyncSession,
    job: ClaimedJob,
    *,
    error_code: str,
    error_message: str,
    outcome: JobOutcome | None = None,
) -> str:
    """Record a failure, and decide whether it goes back on the queue.

    Returns the resulting status, which is one of three:

    ``QUEUED``  transient, attempts remain -- back on the queue after a jittered backoff.
    ``DEAD``    permanent, or out of attempts -- kept, not deleted, because a dead job is the
                only record of a document that never made it, and a customer asking "why is this
                not searchable" needs an answer.
    ``FAILED``  reserved for a job whose *tenant* went away mid-flight; nothing to retry into.

    The error message is truncated rather than stored whole: a parser traceback can be megabytes,
    and a job table that grows by megabytes per failure becomes its own incident.
    """
    resolved = outcome or (JobOutcome.PERMANENT if error_code in PERMANENT_ERRORS else JobOutcome.RETRY)
    terminal = resolved is JobOutcome.PERMANENT or job.is_last_attempt

    if terminal:
        await session.execute(
            text("""
                UPDATE ingest_jobs
                SET status = 'DEAD', finished_at = now(), lease_expires_at = NULL,
                    error_code = :code, error_message = :message, updated_at = now()
                WHERE id = :id
            """),
            {"id": str(job.id), "code": error_code[:64], "message": error_message[:2000]},
        )
        logger.error(
            "ingest.job_dead",
            extra={"job": str(job.id), "tenant": str(job.tenant_id), "code": error_code, "attempts": job.attempts},
        )
        return "DEAD"

    delay = backoff_for(job.attempts)
    await session.execute(
        text("""
            UPDATE ingest_jobs
            SET status = 'QUEUED', run_after = now() + make_interval(secs => :delay),
                lease_expires_at = NULL, worker_id = NULL,
                error_code = :code, error_message = :message, updated_at = now()
            WHERE id = :id
        """),
        {"id": str(job.id), "delay": delay.total_seconds(), "code": error_code[:64], "message": error_message[:2000]},
    )
    logger.warning(
        "ingest.job_retry",
        extra={"job": str(job.id), "code": error_code, "attempt": job.attempts, "in_s": round(delay.total_seconds())},
    )
    return "QUEUED"


# ----------------------------------------------------------------------------------------------
# Recovery and visibility
# ----------------------------------------------------------------------------------------------


async def reclaim_expired(session: AsyncSession, *, limit: int = 100) -> int:
    """Return jobs whose lease lapsed to the queue.

    This is what makes a worker being OOM-killed a non-event. The attempt is already counted, so
    a job that reliably kills its worker still reaches ``DEAD`` rather than cycling forever --
    the poison-message case, which an unbounded reclaim turns into an infinite loop that also
    takes down every worker that touches it.
    """
    result = await session.execute(
        text("""
            UPDATE ingest_jobs
            SET status = 'QUEUED', worker_id = NULL, lease_expires_at = NULL,
                run_after = now(), updated_at = now()
            WHERE id IN (
                SELECT id FROM ingest_jobs
                WHERE status = 'RUNNING' AND lease_expires_at < now()
                ORDER BY lease_expires_at
                LIMIT :limit
                FOR UPDATE SKIP LOCKED
            )
            RETURNING id
        """),
        {"limit": limit},
    )
    reclaimed = len(result.fetchall())
    if reclaimed:
        logger.warning("ingest.leases_reclaimed", extra={"count": reclaimed})
    return reclaimed


async def depth(session: AsyncSession, *, tenant_id: uuid.UUID | None = None) -> dict[str, int]:
    """Queue depth by status. What the admin UI and the alert both read."""
    result = await session.execute(
        text("""
            SELECT status, count(*) AS n
            FROM ingest_jobs
            WHERE (CAST(:tenant_id AS uuid) IS NULL OR tenant_id = CAST(:tenant_id AS uuid))
            GROUP BY status
        """),
        {"tenant_id": str(tenant_id) if tenant_id else None},
    )
    return {row["status"]: row["n"] for row in result.mappings()}


async def oldest_queued_age(session: AsyncSession) -> float:
    """Seconds the oldest ready job has been waiting.

    The number to alert on. Queue *depth* is a poor signal -- a deep queue moving fast is
    healthy and a shallow one that is stuck is not -- whereas age answers "is anything actually
    being starved", which is the question.
    """
    result = await session.execute(
        text("""
            SELECT EXTRACT(EPOCH FROM (now() - min(run_after))) AS age
            FROM ingest_jobs WHERE status = 'QUEUED' AND run_after <= now()
        """)
    )
    return float(result.scalar() or 0.0)


async def notify_new_work(session: AsyncSession) -> None:
    """Wake an idle worker immediately rather than at its next poll.

    Polling alone is correct but adds up to half the poll interval to every upload's latency,
    which is visible to a user watching a spinner. NOTIFY removes that on the happy path;
    the poll remains as the fallback that makes a missed notification a latency problem rather
    than a lost document.
    """
    await session.execute(text("NOTIFY ingest_jobs_new"))
