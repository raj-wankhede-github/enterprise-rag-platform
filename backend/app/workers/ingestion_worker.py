"""The ingestion worker: claim, process, checkpoint, repeat.

``docker-compose.yml`` has pointed at this module since the skeleton was written; until now it
did not exist, so the ``worker`` service crashed on start and every ingest ran synchronously in
the API request. That is the gap this closes.

The loop is deliberately boring. Everything interesting -- fairness, leases, backoff, dead-
lettering -- lives in ``ingestion/queue.py`` as SQL, because that is where it can be correct
under concurrency. A worker that tried to be clever about ordering would be re-implementing the
claim query in Python, where it has no transaction to be correct within.

Three behaviours worth stating:

**One document per claim, by default.** Batching claims improves throughput and worsens the
thing that matters more: a batch is lost wholesale when a worker dies, and its lease covers the
slowest document in it. Throughput comes from more workers, which the claim query already
supports.

**The heartbeat can revoke the job.** If the lease lapsed -- this worker stalled long enough for
the sweeper to reclaim it -- the worker stops immediately. Another worker owns the document now,
and two workers writing one document's chunks is how duplicates appear in an index whose
idempotency depends on one writer per version.

**SIGTERM finishes the current document.** A deploy that killed workers mid-document would leave
every in-flight job to time out its lease, which is a ten-minute stall for no reason. Draining
costs seconds and the orchestrator's grace period is designed for exactly this.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import socket
import uuid
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.db.session import build_sessionmaker, tenant_session
from app.ingestion.queue import (
    ClaimedJob,
    JobStage,
    checkpoint,
    claim,
    complete,
    fail,
    heartbeat,
    reclaim_expired,
)
from app.observability.otel import StageTiming, new_request_id, stage
from app.search.client import build_client

logger = logging.getLogger(__name__)

#: How long to wait for work before polling again.
#:
#: The upper bound on how late a job starts when its NOTIFY is missed. Short enough that a missed
#: notification is a latency blip rather than a stall; long enough that an idle deployment is not
#: running a query per second per worker forever.
POLL_INTERVAL = timedelta(seconds=5)

#: How often to extend the lease while working. Comfortably under the lease so a slow document
#: does not lose its claim to its own next heartbeat.
HEARTBEAT_INTERVAL = timedelta(seconds=30)

#: How often to return lapsed leases to the queue. Only one worker needs to win this; the others
#: find nothing and move on, which is cheaper than electing a leader.
SWEEP_INTERVAL = timedelta(minutes=1)


class LeaseLostError(Exception):
    """The lease lapsed while working. Another worker owns this job now."""


@dataclass(slots=True)
class WorkerStats:
    claimed: int = 0
    completed: int = 0
    retried: int = 0
    dead: int = 0
    reclaimed: int = 0
    leases_lost: int = 0
    stages: StageTiming = field(default_factory=StageTiming)

    def as_dict(self) -> dict[str, Any]:
        return {
            "claimed": self.claimed,
            "completed": self.completed,
            "retried": self.retried,
            "dead": self.dead,
            "reclaimed": self.reclaimed,
            "leases_lost": self.leases_lost,
        }


class IngestionWorker:
    def __init__(
        self,
        sessionmaker: async_sessionmaker[AsyncSession],
        *,
        settings: Settings,
        search_client: Any,
        worker_id: str | None = None,
        per_tenant_cap: int = 4,
        poll_interval: timedelta = POLL_INTERVAL,
    ) -> None:
        self.sessionmaker = sessionmaker
        self.settings = settings
        self.search_client = search_client
        # Host plus pid: two replicas on one host are distinguishable, and the id survives into
        # the job row so "which worker had this document when it stalled" is answerable.
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"
        self.per_tenant_cap = per_tenant_cap
        self.poll_interval = poll_interval
        self.stats = WorkerStats()
        self._stopping = asyncio.Event()
        self._draining = False

    # ------------------------------------------------------------------------------------

    async def run(self) -> WorkerStats:
        """Claim and process until asked to stop."""
        logger.info("worker.started", extra={"worker": self.worker_id, "cap": self.per_tenant_cap})
        sweeper = asyncio.create_task(self._sweep_forever())

        try:
            while not self._stopping.is_set():
                job = await self._claim_one()
                if job is None:
                    await self._wait_for_work()
                    continue
                await self._run_job(job)
        finally:
            sweeper.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await sweeper
            logger.info("worker.stopped", extra={"worker": self.worker_id, **self.stats.as_dict()})

        return self.stats

    def stop(self) -> None:
        """Ask the loop to finish the current document and exit."""
        if not self._stopping.is_set():
            logger.info("worker.draining", extra={"worker": self.worker_id})
        self._stopping.set()

    # ------------------------------------------------------------------------------------

    async def _claim_one(self) -> ClaimedJob | None:
        async with self.sessionmaker() as session, session.begin():
            jobs = await claim(session, worker_id=self.worker_id, batch=1, per_tenant_cap=self.per_tenant_cap)
        if not jobs:
            return None
        self.stats.claimed += 1
        return jobs[0]

    async def _wait_for_work(self) -> None:
        """Sleep until the poll interval elapses or a stop is requested.

        ``wait_for`` on the stop event rather than a bare sleep, so SIGTERM during an idle period
        exits immediately instead of after a full poll -- the difference between a deploy taking
        seconds and taking the poll interval per worker.
        """
        with contextlib.suppress(TimeoutError):
            await asyncio.wait_for(self._stopping.wait(), timeout=self.poll_interval.total_seconds())

    async def _sweep_forever(self) -> None:
        while not self._stopping.is_set():
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=SWEEP_INTERVAL.total_seconds())
                return
            try:
                async with self.sessionmaker() as session, session.begin():
                    self.stats.reclaimed += await reclaim_expired(session)
            except Exception as exc:
                # A failing sweeper must not take the worker down with it. Its job is recovery,
                # and a worker that stops processing because recovery failed is a worse outcome.
                logger.warning("worker.sweep_failed", extra={"error": str(exc)})

    # ------------------------------------------------------------------------------------

    async def _run_job(self, job: ClaimedJob) -> None:
        request_id = new_request_id()
        log = {"job": str(job.id), "tenant": str(job.tenant_id), "attempt": job.attempts, "rid": request_id}
        logger.info("worker.job_started", extra=log)

        beating = asyncio.create_task(self._heartbeat_forever(job))
        try:
            await self._process(job)
        except LeaseLostError:
            self.stats.leases_lost += 1
            logger.warning("worker.lease_lost", extra=log)
            return
        except Exception as exc:
            await self._record_failure(job, exc, log)
            return
        finally:
            beating.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await beating

        async with self.sessionmaker() as session, session.begin():
            await complete(session, job.id)
        self.stats.completed += 1
        logger.info("worker.job_done", extra=log)

    async def _heartbeat_forever(self, job: ClaimedJob) -> None:
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL.total_seconds())
            async with self.sessionmaker() as session, session.begin():
                alive = await heartbeat(session, job.id, worker_id=self.worker_id)
            if not alive:
                # The sweeper took it. Nothing to do here: the next checkpoint or completion
                # will find the row no longer ours, and _process raises LeaseLostError.
                logger.warning("worker.heartbeat_rejected", extra={"job": str(job.id)})
                return

    async def _record_failure(self, job: ClaimedJob, exc: Exception, log: dict[str, Any]) -> None:
        code = classify(exc)
        async with self.sessionmaker() as session, session.begin():
            status = await fail(session, job, error_code=code, error_message=str(exc))
        if status == "DEAD":
            self.stats.dead += 1
        else:
            self.stats.retried += 1
        logger.warning("worker.job_failed", extra={**log, "code": code, "status": status}, exc_info=exc)

    # ------------------------------------------------------------------------------------

    async def _process(self, job: ClaimedJob) -> None:
        """Fetch, ingest and index one document, skipping stages already passed.

        Every stage is bound to the job's tenant, not to an ambient principal. A worker serves
        every tenant in turn, so there is no request to inherit a tenant from -- and a worker
        that guessed would be the one place in the product where the tenancy guarantee is a
        convention rather than a binding.
        """
        from app.core.container import build_container
        from app.ingestion.contextualize import DocumentContext
        from app.ingestion.pipeline import index_action
        from app.search.bulk import BulkDoc, BulkIndexer
        from app.search.generations import chunk_document_id, chunk_index_name
        from app.storage.blobs import BlobNotFoundError

        container = build_container(self.settings)
        payload = job.payload

        blob_sha = str(payload.get("blob_sha256", ""))
        if not blob_sha:
            raise ValueError("job payload carries no blob_sha256")

        async with tenant_session(self.sessionmaker, job.tenant_id) as session:
            # --- fetch -------------------------------------------------------------------
            with stage("ingest.fetch", timing=self.stats.stages):
                try:
                    data = await container.blobs.get(blob_sha)
                except BlobNotFoundError:
                    # The row and the store have diverged. Retrying will not conjure the bytes,
                    # so this is classified permanent rather than burning five attempts.
                    raise
                await checkpoint(session, job.id, stage=JobStage.FETCHED)

            # --- parse, chunk, contextualize, embed --------------------------------------
            # One call, because the pipeline already checkpoints internally by reusing cached
            # context lines and vectors keyed on content hash. Splitting it here would mean
            # re-parsing to reach the embedding stage.
            await self._guarded(session, job)
            with stage("ingest.prepare", timing=self.stats.stages) as attributes:
                result = await container.pipeline().run(
                    data,
                    str(payload.get("filename", "document")),
                    context=DocumentContext(
                        title=str(payload.get("title") or payload.get("filename") or "Document"),
                        doc_type=payload.get("doc_type"),
                        source_system=payload.get("source_system", "upload"),
                    ),
                    previous_chunk_hashes=[bytes.fromhex(h) for h in payload.get("previous_chunk_hashes", [])],
                )
                attributes["erp.candidates"] = len(result.prepared)
                await checkpoint(
                    session,
                    job.id,
                    stage=JobStage.EMBEDDED,
                    cursor={"chunks": len(result.prepared), "embedded": result.chunks_embedded},
                )

            # --- index --------------------------------------------------------------------
            await self._guarded(session, job)
            with stage("ingest.index", timing=self.stats.stages) as attributes:
                pool = int(payload.get("pool", 0))
                generation = int(payload.get("generation", 1))
                index = chunk_index_name(generation=generation, pool=pool)
                doc_id = uuid.UUID(str(payload["doc_id"]))
                version_id = uuid.UUID(str(payload["doc_version_id"]))

                docs = [
                    BulkDoc(
                        index=index,
                        doc_id=chunk_document_id(
                            tenant_id=job.tenant_id, doc_version_id=version_id, ordinal=item.chunk.ordinal
                        ),
                        routing=str(job.tenant_id),
                        body=index_action(
                            item,
                            tenant_id=job.tenant_id,
                            doc_id=doc_id,
                            doc_version_id=version_id,
                            parent_id=chunk_document_id(
                                tenant_id=job.tenant_id,
                                doc_version_id=version_id,
                                ordinal=item.chunk.parent_ordinal,
                            ),
                            chunk_id=chunk_document_id(
                                tenant_id=job.tenant_id, doc_version_id=version_id, ordinal=item.chunk.ordinal
                            ),
                            generation_fingerprint=result.generation_fingerprint,
                            title=str(payload.get("title") or "Document"),
                            visibility_rank=int(payload.get("visibility_rank", 10)),
                            access_groups=list(payload.get("access_groups", [])),
                        ),
                    )
                    for item in result.prepared
                ]

                written = await BulkIndexer(self.search_client).index(docs)
                # A partial bulk write must fail the job rather than mark it done: half a
                # document in the index is worse than none, because it looks complete.
                written.raise_for_failures(context=f"indexing document {doc_id}")
                attributes["erp.candidates"] = written.indexed
                await checkpoint(session, job.id, stage=JobStage.INDEXED)

    async def _guarded(self, session: AsyncSession, job: ClaimedJob) -> None:
        """Confirm the lease before doing expensive or index-mutating work.

        Checked between stages rather than only at the start, because the stages are minutes
        long. A worker that stalled during parsing and lost its lease must not go on to write
        chunks that a second worker is also writing.
        """
        if not await heartbeat(session, job.id, worker_id=self.worker_id):
            raise LeaseLostError(f"lease on job {job.id} was reclaimed")


def classify(exc: Exception) -> str:
    """Map an exception to a queue error code.

    The distinction that matters is retryable from permanent, and it is made here rather than at
    each raise site so that a new exception type defaults to *retryable* -- the safe direction.
    Defaulting to permanent would silently dead-letter documents on a transient fault nobody had
    classified yet.
    """
    from app.ingestion.loaders.docling_client import ParserUnavailableError
    from app.ingestion.pipeline import UnsupportedDocumentError
    from app.search.bulk import BulkIndexError

    if isinstance(exc, UnsupportedDocumentError):
        return "unsupported_format"
    if isinstance(exc, BulkIndexError):
        # A mapping rejection fails identically forever; a 429 was already retried inside the
        # bulk writer, so reaching here means it was not transient.
        return "mapping_rejected"
    if isinstance(exc, ParserUnavailableError):
        return "parser_unavailable"
    if isinstance(exc, TimeoutError | ConnectionError):
        return "transient_io"
    return "unexpected_error"


# ----------------------------------------------------------------------------------------------


async def main() -> None:
    """Entry point. ``python -m app.workers.ingestion_worker``."""
    logging.basicConfig(
        level=os.environ.get("LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = get_settings()
    sessionmaker = build_sessionmaker(settings)
    search_client = build_client(settings)
    worker = IngestionWorker(
        sessionmaker,
        settings=settings,
        search_client=search_client,
        per_tenant_cap=int(os.environ.get("WORKER_PER_TENANT_CAP", "4")),
    )

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        with contextlib.suppress(NotImplementedError):
            # Windows has no add_signal_handler; KeyboardInterrupt covers the local case.
            loop.add_signal_handler(sig, worker.stop)

    try:
        await worker.run()
    finally:
        await search_client.close()


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(main())
