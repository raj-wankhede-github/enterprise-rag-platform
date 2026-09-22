"""The single bulk-write path into OpenSearch.

Three things every caller needs and nobody writes correctly the first time:

**A 200 from ``_bulk`` does not mean the write succeeded.** OpenSearch returns 200 with
``errors: true`` and a per-item status, so a caller that checks only the HTTP status silently
drops documents. Under ``dynamic: strict`` that is exactly how a mapping drift becomes an index
missing a tenth of its chunks with no error anywhere. ``BulkResult`` forces the per-item view.

**429 is not a failure, it is backpressure.** A rejected item means the write queue is full, and
the correct response is to wait and re-send *that item*, not to fail the batch. During a backfill
this is the common case rather than the exceptional one, because the backfill is deliberately
driving the cluster toward saturation. Retries are per-item and bounded; a mapping rejection is a
bug in our document and fails immediately rather than four times.

**Dual-write is explicit application code.** An alias cannot have two write indices, so during a
rebuild the writer sends every document to both generations itself. Faking it with
``is_write_index`` loses documents. Failure on the *old* generation is fatal; failure on the
*new* one is logged and surfaces at verification -- the live index must never be held hostage by
a rebuild.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: Items per request. ``batched`` also caps by payload size; this is the count ceiling.
DEFAULT_BATCH_SIZE = 500

#: Roughly 8 MB of NDJSON per request. ``http.max_content_length`` defaults to 100 MB, but a
#: batch that large occupies a bulk thread long enough to stall unrelated writes, and the whole
#: batch has to be re-sent on a timeout.
DEFAULT_BATCH_BYTES = 8 * 1024 * 1024

_RETRYABLE_STATUSES = frozenset({429, 502, 503, 504})


@dataclass(frozen=True, slots=True)
class BulkDoc:
    """One document destined for one index."""

    index: str
    doc_id: str
    body: dict[str, Any]
    routing: str | None = None


@dataclass(slots=True)
class BulkResult:
    indexed: int = 0
    #: ``(doc_id, status, reason)`` for items that did not survive their retries.
    failures: list[tuple[str, int, str]] = field(default_factory=list)
    retried: int = 0
    batches: int = 0

    @property
    def ok(self) -> bool:
        return not self.failures

    @property
    def attempted(self) -> int:
        return self.indexed + len(self.failures)

    def merge(self, other: BulkResult) -> None:
        self.indexed += other.indexed
        self.failures.extend(other.failures)
        self.retried += other.retried
        self.batches += other.batches

    def raise_for_failures(self, *, context: str) -> None:
        if not self.failures:
            return
        sample = "; ".join(f"{doc_id}: {status} {reason}" for doc_id, status, reason in self.failures[:3])
        raise BulkIndexError(f"{context}: {len(self.failures)} of {self.attempted} items failed ({sample})")


class BulkIndexError(Exception):
    """Bulk items failed after their retries were exhausted."""


class BulkIndexer:
    def __init__(
        self,
        client: Any,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_bytes: int = DEFAULT_BATCH_BYTES,
        max_retries: int = 4,
        base_backoff_s: float = 0.5,
        refresh: bool = False,
    ) -> None:
        self.client = client
        self.batch_size = batch_size
        self.batch_bytes = batch_bytes
        self.max_retries = max_retries
        self.base_backoff_s = base_backoff_s
        #: Only tests and the eval harness refresh. A backfill that refreshes per batch spends
        #: most of its time building segments it will immediately merge away.
        self.refresh = refresh

    async def index(self, docs: Iterable[BulkDoc]) -> BulkResult:
        total = BulkResult()
        for batch in batched(docs, size=self.batch_size, max_bytes=self.batch_bytes):
            total.merge(await self._one_batch(batch))
        return total

    async def update_metadata(self, docs: Iterable[BulkDoc]) -> BulkResult:
        """Partial updates, for chunks whose text is unchanged but whose position moved.

        This is what makes an incremental re-index cheap: a one-paragraph policy edit turns a
        400-chunk reindex into a dozen embeddings and several hundred metadata updates, and the
        vector -- by far the largest field -- is never re-sent.
        """
        total = BulkResult()
        for batch in batched(docs, size=self.batch_size, max_bytes=self.batch_bytes):
            total.merge(await self._one_batch(batch, op="update"))
        return total

    async def _one_batch(self, batch: Sequence[BulkDoc], *, op: str = "index") -> BulkResult:
        result = BulkResult(batches=1)
        pending = list(batch)

        for attempt in range(self.max_retries + 1):
            if not pending:
                break
            if attempt:
                # Exponential with a ceiling. The cluster is shedding load; hammering it is how
                # a backfill turns a slow rebuild into an outage for live traffic.
                await asyncio.sleep(min(self.base_backoff_s * 2**attempt, 8.0))
                result.retried += len(pending)

            response = await self.client.bulk(body=to_ndjson(pending, op=op), refresh=self.refresh)
            succeeded, retryable, permanent = partition_response(response, pending, op=op)

            result.indexed += succeeded
            result.failures.extend(permanent)
            pending = retryable

        result.failures.extend((doc.doc_id, 429, "retries exhausted") for doc in pending)
        return result


async def dual_write(
    docs: Sequence[BulkDoc],
    *,
    live: BulkIndexer,
    building: BulkIndexer | None,
    building_docs: Sequence[BulkDoc] | None = None,
) -> tuple[BulkResult, BulkResult | None]:
    """Write to the live generation and, during a rebuild, to the one being built.

    The asymmetry is deliberate and is the whole design. The live write is awaited and its
    failure propagates. The building write is best-effort: a rebuild target that is slow, full or
    momentarily missing must never fail a user's upload. Documents it drops are caught by the
    count reconciliation at ``BACKFILLED`` and by the shadow evaluation at ``VERIFYING`` -- which
    is precisely why those two gates exist.

    ``building_docs`` exists because the two generations do not necessarily agree on a document's
    body: a rebuild that changes the embedder produces a different vector, and one that changes
    the chunker produces different chunks entirely.
    """
    live_result = await live.index(docs)
    live_result.raise_for_failures(context="live generation write")

    if building is None:
        return live_result, None

    try:
        building_result = await building.index(building_docs if building_docs is not None else docs)
    except Exception as exc:
        logger.warning("dual_write.building_failed", extra={"error": str(exc), "docs": len(docs)})
        return live_result, BulkResult(failures=[("*", 500, str(exc))])

    if not building_result.ok:
        logger.warning(
            "dual_write.building_incomplete",
            extra={"failed": len(building_result.failures), "docs": len(docs)},
        )
    return live_result, building_result


def to_ndjson(docs: Sequence[BulkDoc], *, op: str = "index") -> str:
    lines: list[str] = []
    for doc in docs:
        action: dict[str, Any] = {"_index": doc.index, "_id": doc.doc_id}
        if doc.routing:
            action["routing"] = doc.routing
        lines.append(json.dumps({op: action}, separators=(",", ":")))
        # An update needs its body wrapped. doc_as_upsert keeps a retried backfill idempotent
        # when a metadata update races the initial index of the same chunk.
        payload: dict[str, Any] = {"doc": doc.body, "doc_as_upsert": True} if op == "update" else doc.body
        lines.append(json.dumps(payload, separators=(",", ":"), default=str))
    return "\n".join(lines) + "\n"


def batched(docs: Iterable[BulkDoc], *, size: int, max_bytes: int) -> Iterator[list[BulkDoc]]:
    """Batch by count *and* by payload size.

    Count alone is not enough: 500 table chunks carrying 1024-d vectors are an order of magnitude
    larger than 500 short paragraphs, and it is the byte size that decides whether the request
    times out.
    """
    batch: list[BulkDoc] = []
    batch_bytes = 0
    for doc in docs:
        doc_bytes = approx_bytes(doc)
        if batch and (len(batch) >= size or batch_bytes + doc_bytes > max_bytes):
            yield batch
            batch, batch_bytes = [], 0
        batch.append(doc)
        batch_bytes += doc_bytes
    if batch:
        yield batch


def approx_bytes(doc: BulkDoc) -> int:
    """Cheap size estimate: the vector dominates and everything else is noise beside it."""
    embedding = doc.body.get("embedding")
    vector_bytes = len(embedding) * 9 if isinstance(embedding, list) else 0
    text_bytes = len(str(doc.body.get("content", ""))) + len(str(doc.body.get("context_line", "")))
    return vector_bytes + text_bytes + 512


def partition_response(
    response: dict[str, Any], batch: Sequence[BulkDoc], *, op: str = "index"
) -> tuple[int, list[BulkDoc], list[tuple[str, int, str]]]:
    """Split a bulk response into succeeded / worth retrying / permanently failed."""
    if not response.get("errors"):
        return len(batch), [], []

    by_id = {doc.doc_id: doc for doc in batch}
    succeeded = 0
    retryable: list[BulkDoc] = []
    permanent: list[tuple[str, int, str]] = []

    for entry in response.get("items", []):
        item: dict[str, Any] = entry.get(op) or next(iter(entry.values()), {})
        status = int(item.get("status", 500))
        doc_id = str(item.get("_id", ""))

        if status < 300:
            succeeded += 1
        elif status in _RETRYABLE_STATUSES and doc_id in by_id:
            retryable.append(by_id[doc_id])
        else:
            error = item.get("error") or {}
            reason = str(error.get("reason") or error.get("type") or "unknown")
            # A mapping rejection is a bug in our document and will fail identically on every
            # retry. Surfacing it immediately is the point of dynamic: strict.
            permanent.append((doc_id, status, reason))

    return succeeded, retryable, permanent
