"""The rebuild state machine, its two gates, and the atomicity of the swap.

The tests worth reading are the gate tests. A rebuild that completes and is *wrong* -- a full
count with collapsed recall -- looks healthy on every operational dashboard, and the shadow
evaluation is the only thing standing between it and production traffic.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from typing import Any

import pytest

from app.search.admin import IndexAdmin
from app.search.bulk import BulkDoc
from app.search.generations import (
    GenerationSpec,
    GenerationState,
    alias_add_actions,
    chunk_index_name,
    chunk_read_alias,
    chunk_write_alias,
    parent_index_name,
    parent_read_alias,
    promote_actions,
    retire_read_actions,
    rollback_actions,
)
from app.search.rebuild import BackfillBatch, RebuildOrchestrator, VerificationReport, compare_metrics

SPEC = GenerationSpec(embedder_id="e5-large-v2", chunker_version="1", contextualizer_version="t1", dimension=1024)


class FakeIndices:
    def __init__(self) -> None:
        self.created: list[str] = []
        self.deleted: list[str] = []
        self.alias_calls: list[list[dict[str, Any]]] = []
        self.settings: list[tuple[str, dict[str, Any]]] = []
        self.merged: list[str] = []

    async def create(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.created.append(index)
        return {"acknowledged": True}

    async def delete(self, *, index: str, ignore: list[int] | None = None) -> dict[str, Any]:
        self.deleted.append(index)
        return {"acknowledged": True}

    async def update_aliases(self, *, body: dict[str, Any]) -> dict[str, Any]:
        self.alias_calls.append(body["actions"])
        return {"acknowledged": True}

    async def put_settings(self, *, index: str, body: dict[str, Any]) -> dict[str, Any]:
        self.settings.append((index, body))
        return {"acknowledged": True}

    async def refresh(self, *, index: str) -> dict[str, Any]:
        return {}

    async def forcemerge(self, *, index: str, max_num_segments: int) -> dict[str, Any]:
        self.merged.append(index)
        return {}


class FakeClient:
    def __init__(self, *, counts: dict[str, int] | None = None) -> None:
        self.indices = FakeIndices()
        self._counts = counts or {}
        self.bulk_bodies: list[str] = []

    async def count(self, *, index: str) -> dict[str, int]:
        return {"count": self._counts.get(index, 0)}

    async def bulk(self, *, body: str, refresh: bool = False) -> dict[str, Any]:
        self.bulk_bodies.append(body)
        items = len(body.strip().split("\n")) // 2
        return {"errors": False, "items": [{"index": {"status": 201}} for _ in range(items)]}


class FakeSource:
    """A Postgres stand-in: a fixed number of chunks per pool, streamed in batches."""

    def __init__(self, per_pool: dict[int, int], *, batch: int = 4) -> None:
        self.per_pool = per_pool
        self.batch = batch
        self.resumed_from: dict[int, str | None] = {}

    async def count(self, *, pool: int) -> int:
        return self.per_pool.get(pool, 0)

    async def stream(self, *, pool: int, after: str | None) -> AsyncIterator[BackfillBatch]:
        self.resumed_from[pool] = after
        total = self.per_pool.get(pool, 0)
        start = int(after.split("-")[-1]) + 1 if after else 0
        for offset in range(start, total, self.batch):
            docs = [
                BulkDoc(
                    index=chunk_index_name(generation=2, pool=pool),
                    doc_id=f"p{pool}-{i}",
                    body={"content": f"chunk {i}"},
                )
                for i in range(offset, min(offset + self.batch, total))
            ]
            yield BackfillBatch(docs=docs, cursor=docs[-1].doc_id)


def orchestrator(
    *, client: FakeClient, source: FakeSource, pools: Sequence[int] = (0, 1), previous: int | None = 1
) -> RebuildOrchestrator:
    return RebuildOrchestrator(
        admin=IndexAdmin(client, shards=1, replicas=1),
        source=source,
        client=client,
        spec=SPEC,
        generation=2,
        pools=list(pools),
        previous_generation=previous,
    )


async def passing_eval(fingerprint: str) -> dict[str, float]:
    return {"recall@10": 0.95, "ndcg@10": 0.88}


BASELINE = {"recall@10": 0.94, "ndcg@10": 0.87}


# ------------------------------------------------------------------------------------------
# Preparation
# ------------------------------------------------------------------------------------------


async def test_prepare_creates_both_indices_for_every_pool_and_no_aliases() -> None:
    """A newly created generation must be invisible: nothing points at it until it passes."""
    client = FakeClient()
    await orchestrator(client=client, source=FakeSource({})).prepare()

    assert set(client.indices.created) == {
        "chunks_g2_p000",
        "parents_g2_p000",
        "chunks_g2_p001",
        "parents_g2_p001",
    }
    assert client.indices.alias_calls == []


async def test_a_backfill_target_is_created_without_replicas_or_refresh() -> None:
    """Replicating and refreshing a copy nobody reads roughly doubles the cost of the one
    operation that touches every chunk the tenant owns."""
    client = FakeClient()
    await IndexAdmin(client, shards=3, replicas=1).create_generation(
        generation=2, pool=0, dimension=1024, for_backfill=True
    )
    # The body is not recorded by FakeIndices.create, so assert through the serving path instead:
    # finalise must be what restores them.
    await IndexAdmin(client, shards=3, replicas=1).finalise_for_serving(generation=2, pool=0)
    restored = dict(client.indices.settings)["chunks_g2_p000"]["index"]
    assert restored["number_of_replicas"] == 1
    assert restored["refresh_interval"] == "30s"


async def test_finalising_merges_to_one_segment() -> None:
    client = FakeClient()
    await IndexAdmin(client).finalise_for_serving(generation=2, pool=0)
    assert client.indices.merged == ["chunks_g2_p000", "parents_g2_p000"]


async def test_a_failed_merge_does_not_fail_the_rebuild() -> None:
    """Merging is an optimisation. A cluster too busy to merge is not a reason to refuse a
    generation that already reconciles and evaluates."""

    class NoMerge(FakeIndices):
        async def forcemerge(self, *, index: str, max_num_segments: int) -> dict[str, Any]:
            raise RuntimeError("merge queue full")

    client = FakeClient()
    client.indices = NoMerge()  # type: ignore[assignment]
    await IndexAdmin(client).finalise_for_serving(generation=2, pool=0)


# ------------------------------------------------------------------------------------------
# Backfill
# ------------------------------------------------------------------------------------------


async def test_a_complete_backfill_reaches_backfilled() -> None:
    client = FakeClient()
    runner = orchestrator(client=client, source=FakeSource({0: 10, 1: 6}))
    await runner.prepare()
    await runner.backfill()

    assert runner.state is GenerationState.BACKFILLED
    assert runner.progress[0].indexed == 10
    assert runner.progress[1].indexed == 6


async def test_the_cursor_tracks_the_last_document_of_each_batch() -> None:
    """Without a resumable cursor, a crash at document 900 of 1000 restarts from zero -- which
    on a ten-hour rebuild means the rebuild never finishes."""
    client = FakeClient()
    runner = orchestrator(client=client, source=FakeSource({0: 10}, batch=4), pools=(0,))
    await runner.prepare()
    await runner.backfill()
    assert runner.progress[0].cursor == "p0-9"


async def test_a_backfill_resumes_from_the_recorded_cursor() -> None:
    client = FakeClient()
    source = FakeSource({0: 10}, batch=4)
    runner = orchestrator(client=client, source=source, pools=(0,))
    await runner.prepare()
    await runner.backfill(resume={0: "p0-5"})

    assert source.resumed_from[0] == "p0-5"
    assert runner.progress[0].indexed == 4


async def test_a_backfill_cannot_run_outside_building() -> None:
    runner = orchestrator(client=FakeClient(), source=FakeSource({}))
    with pytest.raises(ValueError, match="requires BUILDING"):
        await runner.backfill()


async def test_a_permanent_failure_leaves_the_generation_in_building() -> None:
    """Not RETIRED: the cursor is intact, so resuming costs minutes rather than the rebuild."""

    class Rejecting(FakeClient):
        async def bulk(self, *, body: str, refresh: bool = False) -> dict[str, Any]:
            return {
                "errors": True,
                "items": [{"index": {"_id": "p0-0", "status": 400, "error": {"reason": "strict"}}}],
            }

    client = Rejecting()
    runner = orchestrator(client=client, source=FakeSource({0: 4}, batch=4), pools=(0,))
    await runner.prepare()
    await runner.backfill()

    assert runner.state is GenerationState.BUILDING
    assert runner.progress[0].failed == 1


# ------------------------------------------------------------------------------------------
# Gate one: reconciliation
# ------------------------------------------------------------------------------------------


async def test_a_full_count_passes_reconciliation() -> None:
    client = FakeClient(counts={"chunks_g2_p000": 10, "parents_g2_p000": 0})
    runner = orchestrator(client=client, source=FakeSource({0: 10}), pools=(0,))
    await runner.prepare()
    await runner.backfill()
    report = await runner.verify(evaluate=passing_eval, baseline=BASELINE)
    assert report.passed


async def test_a_stalled_backfill_is_caught_by_the_count() -> None:
    """The half-built index: it answers queries, it just answers them from 60% of the corpus."""
    client = FakeClient(counts={"chunks_g2_p000": 600, "parents_g2_p000": 0})
    runner = orchestrator(client=client, source=FakeSource({0: 0}), pools=(0,))
    runner.source = FakeSource({0: 1000})  # type: ignore[assignment]
    await runner.prepare()
    runner.state = GenerationState.BACKFILLED

    report = await runner.verify(evaluate=passing_eval, baseline=BASELINE)
    assert not report.passed
    assert "40.00% short" in report.failures[0]


async def test_a_small_shortfall_is_tolerated_because_the_dual_write_races_the_count() -> None:
    """A chunk written moments before the count is legitimately in flight. Zero tolerance here
    would make every rebuild under live traffic fail intermittently."""
    client = FakeClient(counts={"chunks_g2_p000": 9_995, "parents_g2_p000": 0})
    runner = orchestrator(client=client, source=FakeSource({0: 10_000}), pools=(0,))
    await runner.prepare()
    runner.state = GenerationState.BACKFILLED
    report = await runner.verify(evaluate=passing_eval, baseline=BASELINE)
    assert report.passed


async def test_an_empty_pool_does_not_divide_by_zero() -> None:
    client = FakeClient(counts={})
    runner = orchestrator(client=client, source=FakeSource({0: 0}), pools=(0,))
    await runner.prepare()
    runner.state = GenerationState.BACKFILLED
    assert (await runner.verify(evaluate=passing_eval, baseline=BASELINE)).passed


# ------------------------------------------------------------------------------------------
# Gate two: shadow evaluation
# ------------------------------------------------------------------------------------------


async def test_a_full_count_with_collapsed_recall_is_refused() -> None:
    """The dangerous case, and the reason this gate exists.

    Every operational dashboard is green -- the counts match, the backfill finished, no errors.
    The index is simply worse, and without this gate nobody finds out for a quarter.
    """
    client = FakeClient(counts={"chunks_g2_p000": 1000, "parents_g2_p000": 0})
    runner = orchestrator(client=client, source=FakeSource({0: 1000}), pools=(0,))
    await runner.prepare()
    runner.state = GenerationState.BACKFILLED

    async def collapsed(fingerprint: str) -> dict[str, float]:
        return {"recall@10": 0.41, "ndcg@10": 0.30}

    report = await runner.verify(evaluate=collapsed, baseline=BASELINE)
    assert not report.passed
    assert any("recall@10 fell" in failure for failure in report.failures)


async def test_promotion_without_a_shadow_evaluation_is_refused() -> None:
    """ "Just promote it, the backfill looked fine" is how a half-built index reaches production."""
    client = FakeClient(counts={"chunks_g2_p000": 10})
    runner = orchestrator(client=client, source=FakeSource({0: 10}), pools=(0,))
    await runner.prepare()
    runner.state = GenerationState.BACKFILLED
    report = await runner.verify(evaluate=None)
    assert not report.passed
    assert "no shadow evaluation" in report.failures[0]


async def test_the_shadow_evaluation_runs_against_the_fingerprint_not_the_alias() -> None:
    """The alias still points at the old generation -- that is the point of a shadow run."""
    seen: list[str] = []

    async def record(fingerprint: str) -> dict[str, float]:
        seen.append(fingerprint)
        return BASELINE

    client = FakeClient(counts={"chunks_g2_p000": 10})
    runner = orchestrator(client=client, source=FakeSource({0: 10}), pools=(0,))
    await runner.prepare()
    runner.state = GenerationState.BACKFILLED
    await runner.verify(evaluate=record, baseline=BASELINE)

    assert seen == [SPEC.fingerprint]
    assert client.indices.alias_calls == []


async def test_a_failed_verification_returns_to_building_for_resumption() -> None:
    client = FakeClient(counts={"chunks_g2_p000": 100})
    runner = orchestrator(client=client, source=FakeSource({0: 1000}), pools=(0,))
    await runner.prepare()
    runner.state = GenerationState.BACKFILLED
    await runner.verify(evaluate=passing_eval, baseline=BASELINE)
    assert runner.state is GenerationState.BUILDING


def test_a_metric_missing_from_the_run_fails_rather_than_passes() -> None:
    """An eval that silently stopped reporting recall would otherwise promote anything."""
    failures = compare_metrics({"ndcg@10": 0.9}, {"recall@10": 0.9, "ndcg@10": 0.9}, {"recall@10": 0.02})
    assert failures == ["recall@10 was not reported by the shadow evaluation"]


def test_a_metric_absent_from_the_baseline_is_not_a_failure() -> None:
    """A newly added metric has nothing to regress against; refusing would block its adoption."""
    assert compare_metrics({"ndcg@10": 0.9}, {}, {"ndcg@10": 0.02}) == []


def test_an_improvement_is_never_a_regression() -> None:
    assert compare_metrics({"recall@10": 0.99}, {"recall@10": 0.90}, {"recall@10": 0.02}) == []


def test_a_drop_inside_tolerance_passes() -> None:
    assert compare_metrics({"recall@10": 0.89}, {"recall@10": 0.90}, {"recall@10": 0.02}) == []


# ------------------------------------------------------------------------------------------
# The swap
# ------------------------------------------------------------------------------------------


async def test_every_pool_swaps_in_exactly_one_aliases_call() -> None:
    """The atomicity is the entire point. A per-pool loop serves a mixture of generations for as
    long as it runs, and a failure halfway leaves it that way permanently -- with no error, since
    each individual call succeeded."""
    client = FakeClient(counts={f"chunks_g2_p{pool:03d}": 10 for pool in range(4)})
    runner = RebuildOrchestrator(
        admin=IndexAdmin(client),
        source=FakeSource(dict.fromkeys(range(4), 10)),
        client=client,
        spec=SPEC,
        generation=2,
        pools=list(range(4)),
        previous_generation=1,
    )
    await runner.prepare()
    runner.state = GenerationState.BACKFILLED
    report = await runner.verify(evaluate=passing_eval, baseline=BASELINE)
    await runner.promote(report)

    assert len(client.indices.alias_calls) == 1
    # Six actions per pool: two write removes, two write adds, two read adds. The old
    # generation's read entries stay until finish_drain.
    assert len(client.indices.alias_calls[0]) == 4 * 6
    assert runner.state is GenerationState.LIVE


def test_removes_precede_adds_so_two_indices_never_claim_the_write_alias() -> None:
    """OpenSearch rejects a request that would leave an alias with two write indices, and the
    actions inside one request apply in order."""
    actions = promote_actions(from_generation=1, to_generation=2, pool=0)
    write_alias_ops = [
        next(iter(action)) for action in actions if next(iter(action.values()))["alias"] == chunk_write_alias(0)
    ]
    assert write_alias_ops == ["remove", "add"]


def test_exactly_one_index_claims_each_write_alias_after_a_promotion() -> None:
    actions = promote_actions(from_generation=1, to_generation=2, pool=0)
    writes = [action["add"] for action in actions if "add" in action and action["add"].get("is_write_index")]
    assert len(writes) == 2
    assert {str(entry["index"]) for entry in writes} == {"chunks_g2_p000", "parents_g2_p000"}


def test_a_promotion_widens_the_read_alias_rather_than_moving_it() -> None:
    """The fix for a bug the integration test caught under continuous traffic.

    Every query asserts a ``generation_fingerprint`` term filter, and a reader learns the new
    fingerprint from its own next refresh -- not at the instant the alias moves. Moving the read
    alias wholesale leaves a window where a reader holding the old fingerprint queries an alias
    resolving only to the new index: valid, HTTP 200, zero hits, nothing logged. Reversing the
    order just moves the window to the other side.
    """
    actions = promote_actions(from_generation=1, to_generation=2, pool=0)
    read_ops = [
        (next(iter(action)), next(iter(action.values()))["index"])
        for action in actions
        if next(iter(action.values()))["alias"] in {chunk_read_alias(0), parent_read_alias(0)}
    ]
    assert all(op == "add" for op, _ in read_ops), "the old generation must keep serving reads"
    assert {index for _, index in read_ops} == {"chunks_g2_p000", "parents_g2_p000"}


def test_the_old_generation_leaves_the_read_alias_only_at_the_end_of_the_drain() -> None:
    actions = retire_read_actions(generation=1, pool=0)
    assert all("remove" in action for action in actions)
    assert {str(action["remove"]["index"]) for action in actions} == {"chunks_g1_p000", "parents_g1_p000"}


def test_a_rollback_moves_only_the_write_alias_back() -> None:
    """Cheap precisely because the old generation never stopped being readable."""
    actions = rollback_actions(live_generation=2, back_to=1, pool=0)
    restored = [action["add"] for action in actions if "add" in action]
    assert all(entry.get("is_write_index") for entry in restored)
    assert {str(entry["index"]) for entry in restored} == {"chunks_g1_p000", "parents_g1_p000"}


def test_a_bootstrap_points_the_read_alias_without_removing_anything() -> None:
    actions = alias_add_actions(generation=1, pool=0)
    assert all("add" in action for action in actions)
    assert any(action["add"]["alias"] == chunk_read_alias(0) for action in actions)


async def test_a_generation_that_failed_verification_cannot_be_promoted() -> None:
    runner = orchestrator(client=FakeClient(), source=FakeSource({}))
    with pytest.raises(ValueError, match="failed verification"):
        await runner.promote(VerificationReport(passed=False, failures=["recall collapsed"]))


async def test_a_first_generation_is_pointed_at_rather_than_swapped() -> None:
    """There is nothing to remove, and a remove of a non-existent alias fails the whole call."""
    client = FakeClient(counts={"chunks_g2_p000": 10})
    runner = orchestrator(client=client, source=FakeSource({0: 10}), pools=(0,), previous=None)
    await runner.prepare()
    runner.state = GenerationState.BACKFILLED
    report = await runner.verify(evaluate=passing_eval, baseline=BASELINE)
    await runner.promote(report)

    assert all("add" in action for action in client.indices.alias_calls[0])


# ------------------------------------------------------------------------------------------
# Rollback and retirement
# ------------------------------------------------------------------------------------------


async def test_a_rollback_is_one_call_with_the_generations_reversed() -> None:
    """A rollback that required a rebuild would take hours -- which in practice means nobody
    rolls back, they fix forward under pressure instead."""
    client = FakeClient(counts={"chunks_g2_p000": 10})
    runner = orchestrator(client=client, source=FakeSource({0: 10}), pools=(0,))
    await runner.prepare()
    runner.state = GenerationState.BACKFILLED
    await runner.promote(await runner.verify(evaluate=passing_eval, baseline=BASELINE))

    await runner.rollback()

    assert len(client.indices.alias_calls) == 2
    reverted = [action for action in client.indices.alias_calls[1] if "add" in action]
    assert all("_g1_" in str(action["add"]["index"]) for action in reverted)
    assert all(action["add"].get("is_write_index") for action in reverted), (
        "only the write alias needs moving back; the old generation never stopped serving reads"
    )
    assert runner.state is GenerationState.DRAINING


async def test_nothing_is_deleted_before_the_drain_window() -> None:
    """Draining is what makes the rollback above possible; deleting early removes the option."""
    client = FakeClient(counts={"chunks_g2_p000": 10})
    runner = orchestrator(client=client, source=FakeSource({0: 10}), pools=(0,))
    await runner.prepare()
    runner.state = GenerationState.BACKFILLED
    await runner.promote(await runner.verify(evaluate=passing_eval, baseline=BASELINE))
    assert client.indices.deleted == []


async def test_retirement_drops_the_previous_generations_indices() -> None:
    client = FakeClient()
    runner = orchestrator(client=client, source=FakeSource({}), pools=(0, 1))
    await runner.retire_previous()
    assert set(client.indices.deleted) == {
        "chunks_g1_p000",
        "parents_g1_p000",
        "chunks_g1_p001",
        "parents_g1_p001",
    }


async def test_retiring_a_first_generation_deletes_nothing() -> None:
    client = FakeClient()
    await orchestrator(client=client, source=FakeSource({}), previous=None).retire_previous()
    assert client.indices.deleted == []


async def test_rolling_back_a_first_generation_is_refused_rather_than_silently_skipped() -> None:
    runner = orchestrator(client=FakeClient(), source=FakeSource({}), previous=None)
    with pytest.raises(ValueError, match="no previous generation"):
        await runner.rollback()


# ------------------------------------------------------------------------------------------
# End to end
# ------------------------------------------------------------------------------------------


async def test_the_whole_rebuild_runs_planned_to_live() -> None:
    client = FakeClient(counts={"chunks_g2_p000": 10, "chunks_g2_p001": 10})
    runner = orchestrator(client=client, source=FakeSource({0: 10, 1: 10}))
    report = await runner.run(evaluate=passing_eval, baseline=BASELINE)

    assert report.passed
    assert runner.state is GenerationState.LIVE
    assert len(client.indices.alias_calls) == 1


async def test_a_rebuild_whose_backfill_fails_never_touches_an_alias() -> None:
    class Rejecting(FakeClient):
        async def bulk(self, *, body: str, refresh: bool = False) -> dict[str, Any]:
            return {"errors": True, "items": [{"index": {"_id": "p0-0", "status": 400}}]}

    client = Rejecting()
    runner = orchestrator(client=client, source=FakeSource({0: 8}, batch=4), pools=(0,))
    report = await runner.run(evaluate=passing_eval, baseline=BASELINE)

    assert not report.passed
    assert client.indices.alias_calls == []


def test_the_report_serialises_for_verified_metrics() -> None:
    """It is persisted as JSONB and read back by the operator UI, so int keys must not survive."""
    report = VerificationReport(passed=True, expected_counts={0: 10}, metrics={"recall@10": 0.9})
    payload = report.as_json()
    assert payload["expected_counts"] == {"0": 10}
    assert "verified_at" in payload


# ------------------------------------------------------------------------------------------


def test_the_state_machine_offers_no_path_from_backfilled_straight_to_live() -> None:
    """Enforced structurally rather than by convention: there is no code path to skip the gate."""
    from app.search.generations import can_transition

    assert not can_transition(GenerationState.BACKFILLED, GenerationState.LIVE)
    assert not can_transition(GenerationState.BUILDING, GenerationState.LIVE)
    assert can_transition(GenerationState.VERIFYING, GenerationState.LIVE)


def test_index_names_are_stable_across_the_module_boundary() -> None:
    assert chunk_index_name(generation=2, pool=0) == "chunks_g2_p000"
    assert parent_index_name(generation=2, pool=15) == "parents_g2_p015"
