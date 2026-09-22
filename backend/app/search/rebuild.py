"""Zero-downtime generation rebuilds.

A rebuild is what happens when any of the five fingerprint inputs changes -- a new embedding
model, a chunker fix, a contextualizer prompt edit, a mapping change, an analyzer change. All of
them require every chunk in the corpus to be re-indexed, and none of them may interrupt search.

The shape of the solution:

    PLANNED -> BUILDING -> BACKFILLED -> VERIFYING -> LIVE -> DRAINING -> RETIRED

``BUILDING`` is the only interesting state. Live traffic still reads and writes the old
generation, the backfill walks Postgres writing the new one, and every *new* write goes to both
(``bulk.dual_write``). Without the dual-write, every document uploaded during a backfill that
takes six hours is missing from the new generation -- and the backfill cursor has already passed
it, so nothing ever notices.

Two gates stand between a backfill and production traffic, and they catch different failures:

* **Reconciliation** at ``BACKFILLED`` compares OpenSearch counts against Postgres. It catches a
  stalled or partial backfill -- a crash at document 900 of 1000, a pool that was never started.
* **Shadow evaluation** at ``VERIFYING`` runs the eval harness against the new generation and
  compares to the previous generation's recorded metrics. It catches the backfill that completed
  and is *wrong*: an embedder returning normalized vectors where the old one did not, a chunker
  change that quietly halves recall.

A full count with a poor eval score is the dangerous case, because it looks fine from every
operational dashboard. That is the case the shadow eval exists for, and it is why ``LIVE`` is
reachable only from ``VERIFYING`` in ``generations._TRANSITIONS``.

The promotion itself is one ``_aliases`` call holding every pool's actions. ``DRAINING`` then
holds the old generation intact for a soak window, so a rollback is the same single call with the
generations reversed -- seconds, not a rebuild.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from app.search.admin import IndexAdmin
from app.search.bulk import BulkDoc, BulkIndexer, BulkResult
from app.search.generations import (
    GenerationSpec,
    GenerationState,
    assert_transition,
    chunk_index_name,
    parent_index_name,
)

logger = logging.getLogger(__name__)

#: How far OpenSearch counts may fall short of Postgres and still pass reconciliation.
#: Not zero: a backfill running against live traffic races the dual-write, and a chunk written
#: moments before the count is legitimately in flight. 0.1% over a million chunks is a thousand
#: documents, which is small enough not to move any eval metric and large enough to absorb that
#: race. Anything worse is a stalled backfill, not a race.
DEFAULT_COUNT_TOLERANCE = 0.001

#: How far each metric may regress against the previous generation before promotion is refused.
#: Recall is the gate that matters; nDCG can move on tie ordering alone.
DEFAULT_METRIC_TOLERANCE: dict[str, float] = {
    "recall@10": 0.02,
    "recall@50": 0.02,
    "ndcg@10": 0.03,
    "mrr@10": 0.03,
}


@dataclass(frozen=True, slots=True)
class BackfillBatch:
    """One page of documents, and the cursor that resumes immediately after it.

    The cursor is **opaque to the orchestrator**: a string the source emits and the source
    parses. That is what lets the same rebuild machinery drive a Postgres backfill, a blob-store
    replay and a connector re-sync without knowing anything about their key spaces -- and it
    keeps the orchestrator from assuming the cursor is the last document's id, which is true for
    none of them.
    """

    docs: Sequence[BulkDoc]
    cursor: str


class ChunkSource(Protocol):
    """Where the backfill reads from. Postgres is truth; this is that read."""

    async def count(self, *, pool: int) -> int:
        """Total documents this pool should hold, chunks and parents together."""

    def stream(self, *, pool: int, after: str | None) -> AsyncIterator[BackfillBatch]:
        """Yield batches in a stable order, resumable from ``after``.

        The order must be stable and every batch must carry a cursor, because a backfill *will*
        be interrupted -- by a deploy, an OOM, a node restart -- and resuming from the start on a
        ten-hour rebuild is how a rebuild never finishes.
        """


ShadowEvaluator = Callable[[str], Awaitable[dict[str, float]]]
"""Run the eval harness against one generation fingerprint, returning its metrics."""


@dataclass(slots=True)
class RebuildProgress:
    pool: int
    indexed: int = 0
    failed: int = 0
    cursor: str | None = None
    batches: int = 0

    def record(self, result: BulkResult, cursor: str | None) -> None:
        self.indexed += result.indexed
        self.failed += len(result.failures)
        self.batches += result.batches
        self.cursor = cursor


@dataclass(slots=True)
class VerificationReport:
    """Why a generation was or was not promoted. Persisted to ``verified_metrics``."""

    passed: bool
    counts: dict[str, int] = field(default_factory=dict)
    expected_counts: dict[int, int] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    baseline: dict[str, float] = field(default_factory=dict)
    failures: list[str] = field(default_factory=list)

    def as_json(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "counts": self.counts,
            "expected_counts": {str(pool): value for pool, value in self.expected_counts.items()},
            "metrics": self.metrics,
            "baseline": self.baseline,
            "failures": self.failures,
            "verified_at": datetime.now(UTC).isoformat(),
        }


class RebuildOrchestrator:
    """Drives one generation from PLANNED to LIVE, then the old one to RETIRED."""

    def __init__(
        self,
        *,
        admin: IndexAdmin,
        source: ChunkSource,
        client: Any,
        spec: GenerationSpec,
        generation: int,
        pools: Sequence[int],
        previous_generation: int | None = None,
        count_tolerance: float = DEFAULT_COUNT_TOLERANCE,
        metric_tolerance: dict[str, float] | None = None,
        concurrency: int = 2,
    ) -> None:
        self.admin = admin
        self.source = source
        self.client = client
        self.spec = spec
        self.generation = generation
        self.pools = list(pools)
        self.previous_generation = previous_generation
        self.count_tolerance = count_tolerance
        self.metric_tolerance = dict(metric_tolerance or DEFAULT_METRIC_TOLERANCE)
        #: Pools backfilled at once. Not unbounded: the backfill shares a cluster with live
        #: traffic, and saturating the write queue turns a slow rebuild into a slow *search*.
        self.concurrency = concurrency
        self.state = GenerationState.PLANNED
        self.progress: dict[int, RebuildProgress] = {pool: RebuildProgress(pool=pool) for pool in self.pools}

    # ---------------------------------------------------------------------------------------
    # The state machine
    # ---------------------------------------------------------------------------------------

    def _advance(self, target: GenerationState) -> None:
        assert_transition(self.state, target)
        logger.info("generation.state", extra={"generation": self.generation, "from": self.state, "to": target})
        self.state = target

    async def prepare(self, *, dimension: int | None = None) -> None:
        """Create the new generation's indices. No aliases point at them yet."""
        self._advance(GenerationState.BUILDING)
        for pool in self.pools:
            await self.admin.create_generation(
                generation=self.generation,
                pool=pool,
                dimension=dimension if dimension is not None else self.spec.dimension,
                quantize=self.spec.quantize,
                for_backfill=True,
            )

    def writer_for_new_generation(self) -> BulkIndexer:
        """The indexer the live ingestion path dual-writes through while BUILDING.

        Deliberately not the same object as the live writer: it carries a shorter retry budget,
        because a document the rebuild target drops is recoverable (reconciliation catches it)
        while blocking a user's upload on a struggling rebuild index is not.
        """
        return BulkIndexer(self.client, max_retries=1, base_backoff_s=0.25)

    async def backfill(self, *, resume: dict[int, str] | None = None) -> None:
        """Walk Postgres into the new generation, pool by pool, resumably."""
        if self.state is not GenerationState.BUILDING:
            raise ValueError(f"backfill requires BUILDING, not {self.state}")

        cursors = dict(resume or {})
        semaphore = asyncio.Semaphore(self.concurrency)

        async def one_pool(pool: int) -> None:
            async with semaphore:
                await self._backfill_pool(pool, after=cursors.get(pool))

        await asyncio.gather(*(one_pool(pool) for pool in self.pools))

        failed = sum(progress.failed for progress in self.progress.values())
        if failed:
            # Not a raise: the state machine's own gate decides. Leaving the generation in
            # BUILDING is the correct outcome -- it can be resumed, and nothing was promoted.
            logger.error("backfill.incomplete", extra={"generation": self.generation, "failed": failed})
            return

        await self._finalise()

    async def _backfill_pool(self, pool: int, *, after: str | None) -> None:
        indexer = BulkIndexer(self.client, refresh=False)
        progress = self.progress[pool]

        async for batch in self.source.stream(pool=pool, after=after):
            if not batch.docs:
                continue
            result = await indexer.index(batch.docs)
            progress.record(result, cursor=batch.cursor)
            if result.failures:
                # Permanent failures are mapping rejections. Continuing would produce an index
                # that reconciles short with no record of why, so stop and keep the cursor.
                logger.error(
                    "backfill.failed",
                    extra={"pool": pool, "cursor": progress.cursor, "failures": result.failures[:3]},
                )
                return

    async def _finalise(self) -> None:
        """Restore replicas and refresh, merge, then declare the backfill complete."""
        for pool in self.pools:
            await self.admin.finalise_for_serving(generation=self.generation, pool=pool)
        self._advance(GenerationState.BACKFILLED)

    async def verify(
        self, *, evaluate: ShadowEvaluator | None, baseline: dict[str, float] | None = None
    ) -> VerificationReport:
        """Reconcile counts, then shadow-evaluate. Both must pass for LIVE to be reachable."""
        self._advance(GenerationState.VERIFYING)
        report = VerificationReport(passed=True)

        # Gate one: does the new generation hold what Postgres says it should?
        report.counts = await self.admin.counts(generation=self.generation, pools=self.pools)
        for pool in self.pools:
            expected = await self.source.count(pool=pool)
            report.expected_counts[pool] = expected
            actual = report.counts.get(chunk_index_name(generation=self.generation, pool=pool), 0)
            actual += report.counts.get(parent_index_name(generation=self.generation, pool=pool), 0)
            if expected and (expected - actual) / expected > self.count_tolerance:
                report.failures.append(
                    f"pool {pool}: {actual} indexed against {expected} expected "
                    f"({(expected - actual) / expected:.2%} short, tolerance {self.count_tolerance:.2%})"
                )

        # Gate two: is it any good? A full count with a collapsed score is the dangerous case --
        # it looks healthy on every operational dashboard, which is exactly why this gate runs
        # against the generation fingerprint rather than against the alias.
        if evaluate is not None:
            report.metrics = await evaluate(self.spec.fingerprint)
            report.baseline = dict(baseline or {})
            report.failures.extend(compare_metrics(report.metrics, report.baseline, self.metric_tolerance))
        else:
            report.failures.append("no shadow evaluation was run; promotion refused")

        report.passed = not report.failures
        if not report.passed:
            # Back to BUILDING, not RETIRED: the usual cause is an incomplete backfill, and the
            # cursor means resuming costs minutes rather than the whole rebuild.
            self._advance(GenerationState.BUILDING)
            logger.error("generation.verification_failed", extra={"failures": report.failures})
        return report

    async def promote(self, report: VerificationReport) -> None:
        """One atomic ``_aliases`` call across every pool."""
        if not report.passed:
            raise ValueError("refusing to promote a generation that failed verification")
        if self.previous_generation is None:
            await self.admin.point_aliases(generation=self.generation, pools=self.pools)
        else:
            await self.admin.swap(
                from_generation=self.previous_generation, to_generation=self.generation, pools=self.pools
            )
        self._advance(GenerationState.LIVE)

    async def rollback(self) -> None:
        """Put the previous generation back. The same single call, reversed.

        This is why ``DRAINING`` exists and why nothing is deleted during it. A rollback that
        required a rebuild would take hours, which in practice means nobody rolls back -- they
        try to fix forward under pressure instead.
        """
        if self.previous_generation is None:
            raise ValueError("there is no previous generation to roll back to")
        await self.admin.swap(from_generation=self.generation, to_generation=self.previous_generation, pools=self.pools)
        self._advance(GenerationState.DRAINING)

    async def retire_previous(self) -> None:
        """Delete the old generation. Only after the soak window has elapsed."""
        if self.previous_generation is None:
            return
        await self.admin.drop_generation(generation=self.previous_generation, pools=self.pools)
        logger.info("generation.retired", extra={"generation": self.previous_generation})

    # ---------------------------------------------------------------------------------------

    async def run(
        self,
        *,
        evaluate: ShadowEvaluator | None,
        baseline: dict[str, float] | None = None,
        resume: dict[int, str] | None = None,
    ) -> VerificationReport:
        """The whole rebuild, up to LIVE. Retirement is a separate, later decision."""
        await self.prepare()
        await self.backfill(resume=resume)
        if self.state is not GenerationState.BACKFILLED:
            return VerificationReport(passed=False, failures=["backfill did not complete"])
        report = await self.verify(evaluate=evaluate, baseline=baseline)
        if report.passed:
            await self.promote(report)
        return report


def compare_metrics(metrics: dict[str, float], baseline: dict[str, float], tolerance: dict[str, float]) -> list[str]:
    """Which metrics regressed beyond their tolerance.

    A metric present in the baseline but missing from the run is a failure, not a pass: an eval
    that silently stopped reporting recall would otherwise promote anything.
    """
    failures: list[str] = []
    for name, allowed in tolerance.items():
        if name not in baseline:
            continue
        if name not in metrics:
            failures.append(f"{name} was not reported by the shadow evaluation")
            continue
        drop = baseline[name] - metrics[name]
        if drop > allowed:
            failures.append(
                f"{name} fell {drop:.3f} (from {baseline[name]:.3f} to {metrics[name]:.3f}, allowed {allowed:.3f})"
            )
    return failures
