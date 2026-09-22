"""The bulk writer: error partitioning, backpressure, batching, dual-write asymmetry.

The first test in this file is the one that matters. OpenSearch answers a partially-failed bulk
request with HTTP 200, so a writer that checks only the status code drops documents silently --
and under ``dynamic: strict`` that is precisely how a mapping drift becomes an index missing a
tenth of its chunks with nothing in any log.
"""

from __future__ import annotations

from typing import Any

import pytest

from app.search.bulk import (
    BulkDoc,
    BulkIndexer,
    BulkIndexError,
    BulkResult,
    approx_bytes,
    batched,
    dual_write,
    partition_response,
    to_ndjson,
)


class FakeClient:
    """Replays scripted bulk responses and records what it was sent."""

    def __init__(self, responses: list[dict[str, Any]] | None = None) -> None:
        self.responses = responses or []
        self.bodies: list[str] = []
        self.refreshes: list[bool] = []

    async def bulk(self, *, body: str, refresh: bool = False) -> dict[str, Any]:
        self.bodies.append(body)
        self.refreshes.append(refresh)
        if self.responses:
            return self.responses.pop(0)
        return _all_ok(body)


def _all_ok(body: str) -> dict[str, Any]:
    count = len(body.strip().split("\n")) // 2
    return {"errors": False, "items": [{"index": {"status": 201}} for _ in range(count)]}


def docs(n: int, *, index: str = "chunks_g1_p000") -> list[BulkDoc]:
    return [BulkDoc(index=index, doc_id=f"c{i}", body={"content": f"text {i}"}) for i in range(n)]


# ------------------------------------------------------------------------------------------
# A 200 is not a success
# ------------------------------------------------------------------------------------------


def test_a_partial_failure_inside_a_200_is_not_counted_as_indexed() -> None:
    """The failure this whole module exists to prevent."""
    response = {
        "errors": True,
        "items": [
            {"index": {"_id": "c0", "status": 201}},
            {"index": {"_id": "c1", "status": 400, "error": {"type": "strict_dynamic_mapping_exception"}}},
        ],
    }
    succeeded, retryable, permanent = partition_response(response, docs(2))
    assert succeeded == 1
    assert retryable == []
    assert permanent == [("c1", 400, "strict_dynamic_mapping_exception")]


def test_a_clean_response_short_circuits_the_per_item_walk() -> None:
    succeeded, retryable, permanent = partition_response({"errors": False}, docs(3))
    assert (succeeded, retryable, permanent) == (3, [], [])


def test_the_mapping_rejection_reason_survives_into_the_failure() -> None:
    """Without the reason, a strict-mapping rejection is indistinguishable from a timeout."""
    response = {
        "errors": True,
        "items": [{"index": {"_id": "c0", "status": 400, "error": {"reason": "mapping set to strict"}}}],
    }
    _, _, permanent = partition_response(response, docs(1))
    assert permanent[0][2] == "mapping set to strict"


@pytest.mark.parametrize("status", [429, 502, 503, 504])
def test_backpressure_statuses_are_retried_not_failed(status: int) -> None:
    response = {"errors": True, "items": [{"index": {"_id": "c0", "status": status}}]}
    _, retryable, permanent = partition_response(response, docs(1))
    assert [doc.doc_id for doc in retryable] == ["c0"]
    assert permanent == []


@pytest.mark.parametrize("status", [400, 403, 404, 409])
def test_client_errors_are_permanent(status: int) -> None:
    """Retrying a document our own mapping rejects wastes four round trips to fail identically."""
    response = {"errors": True, "items": [{"index": {"_id": "c0", "status": status}}]}
    _, retryable, permanent = partition_response(response, docs(1))
    assert retryable == []
    assert len(permanent) == 1


# ------------------------------------------------------------------------------------------
# Retries
# ------------------------------------------------------------------------------------------


async def test_a_rejected_item_is_resent_and_only_that_item() -> None:
    client = FakeClient(
        [
            {
                "errors": True,
                "items": [{"index": {"_id": "c0", "status": 201}}, {"index": {"_id": "c1", "status": 429}}],
            },
            {"errors": False, "items": [{"index": {"_id": "c1", "status": 201}}]},
        ]
    )
    result = await BulkIndexer(client, base_backoff_s=0.0).index(docs(2))

    assert result.ok
    assert result.indexed == 2
    assert result.retried == 1
    assert "c1" in client.bodies[1] and "c0" not in client.bodies[1]


async def test_retries_are_bounded_and_the_survivors_are_reported() -> None:
    """An unbounded retry on a saturated cluster is a rebuild that never ends."""
    always_busy = {"errors": True, "items": [{"index": {"_id": "c0", "status": 429}}]}
    client = FakeClient([always_busy] * 10)

    result = await BulkIndexer(client, max_retries=2, base_backoff_s=0.0).index(docs(1))

    assert not result.ok
    assert result.failures == [("c0", 429, "retries exhausted")]
    assert len(client.bodies) == 3


async def test_raise_for_failures_names_the_scale_and_a_sample() -> None:
    result = BulkResult(indexed=8, failures=[("c1", 400, "bad field"), ("c2", 400, "bad field")])
    with pytest.raises(BulkIndexError, match=r"2 of 10 items failed.*c1: 400 bad field"):
        result.raise_for_failures(context="test")


async def test_a_clean_run_raises_nothing() -> None:
    BulkResult(indexed=10).raise_for_failures(context="test")


# ------------------------------------------------------------------------------------------
# Batching
# ------------------------------------------------------------------------------------------


def test_batches_respect_the_count_ceiling() -> None:
    batches = list(batched(docs(250), size=100, max_bytes=10**9))
    assert [len(batch) for batch in batches] == [100, 100, 50]


def test_a_batch_of_large_vectors_splits_on_bytes_not_count() -> None:
    """500 table chunks with 1024-d vectors are an order of magnitude bigger than 500 sentences.

    Count alone produces a request that times out, and the whole batch is then re-sent.
    """
    heavy = [BulkDoc(index="i", doc_id=f"c{i}", body={"content": "x", "embedding": [0.1] * 1024}) for i in range(100)]
    batches = list(batched(heavy, size=500, max_bytes=20_000))
    assert len(batches) > 1
    assert all(sum(approx_bytes(doc) for doc in batch) <= 20_000 or len(batch) == 1 for batch in batches)


def test_an_oversized_single_document_is_still_emitted() -> None:
    """Dropping it would lose a chunk; the request is the only thing that can complain."""
    huge = [BulkDoc(index="i", doc_id="c0", body={"content": "x" * 100_000})]
    assert list(batched(huge, size=500, max_bytes=1000)) == [huge]


def test_no_documents_yields_no_batches() -> None:
    assert list(batched([], size=100, max_bytes=1000)) == []


# ------------------------------------------------------------------------------------------
# NDJSON
# ------------------------------------------------------------------------------------------


def test_ndjson_is_action_then_body_and_ends_with_a_newline() -> None:
    payload = to_ndjson(docs(2))
    lines = payload.split("\n")
    assert payload.endswith("\n")
    assert lines[0].startswith('{"index":') and lines[1].startswith('{"content":')


def test_routing_reaches_the_action_line() -> None:
    """Missing routing broadcasts the write and lands the chunk on the wrong shard."""
    doc = BulkDoc(index="i", doc_id="c0", body={}, routing="tenant-a")
    assert '"routing":"tenant-a"' in to_ndjson([doc]).split("\n")[0]


def test_an_update_wraps_the_body_and_upserts() -> None:
    """doc_as_upsert keeps a retried metadata update idempotent against the initial index."""
    lines = to_ndjson(docs(1), op="update").split("\n")
    assert lines[0].startswith('{"update":')
    assert '"doc":' in lines[1] and '"doc_as_upsert":true' in lines[1]


async def test_update_metadata_never_resends_the_vector() -> None:
    """The point of the incremental path: a moved chunk costs metadata, not 4 KB of float."""
    client = FakeClient()
    moved = [BulkDoc(index="i", doc_id="c0", body={"ordinal": 4, "page_from": 2})]
    await BulkIndexer(client).update_metadata(moved)
    assert "embedding" not in client.bodies[0]


# ------------------------------------------------------------------------------------------
# Dual-write
# ------------------------------------------------------------------------------------------


async def test_both_generations_are_written_while_building() -> None:
    live, building = FakeClient(), FakeClient()
    live_result, building_result = await dual_write(docs(3), live=BulkIndexer(live), building=BulkIndexer(building))
    assert live_result.indexed == 3
    assert building_result is not None and building_result.indexed == 3


async def test_without_a_rebuild_in_flight_only_the_live_index_is_written() -> None:
    live = FakeClient()
    live_result, building_result = await dual_write(docs(2), live=BulkIndexer(live), building=None)
    assert live_result.indexed == 2
    assert building_result is None
    assert len(live.bodies) == 1


async def test_a_failing_rebuild_target_does_not_fail_the_users_upload() -> None:
    """The asymmetry is the design: a struggling rebuild index must never block ingestion.

    Documents it drops are caught by reconciliation at BACKFILLED and by the shadow evaluation
    at VERIFYING, which is exactly why both gates exist.
    """

    class Broken:
        async def bulk(self, *, body: str, refresh: bool = False) -> dict[str, Any]:
            raise ConnectionError("rebuild index unavailable")

    live_result, building_result = await dual_write(
        docs(2), live=BulkIndexer(FakeClient()), building=BulkIndexer(Broken(), max_retries=0)
    )
    assert live_result.indexed == 2
    assert building_result is not None and not building_result.ok


async def test_a_failing_live_write_does_fail_the_upload() -> None:
    rejected = {"errors": True, "items": [{"index": {"_id": "c0", "status": 400, "error": {"reason": "strict"}}}]}
    with pytest.raises(BulkIndexError, match="live generation write"):
        await dual_write(docs(1), live=BulkIndexer(FakeClient([rejected])), building=None)


async def test_the_two_generations_may_receive_different_bodies() -> None:
    """A rebuild that changes the embedder produces a different vector for the same chunk."""
    live, building = FakeClient(), FakeClient()
    await dual_write(
        [BulkDoc(index="chunks_g1_p000", doc_id="c0", body={"embedding": [0.0] * 384})],
        live=BulkIndexer(live),
        building=BulkIndexer(building),
        building_docs=[BulkDoc(index="chunks_g2_p000", doc_id="c0", body={"embedding": [0.0] * 1024})],
    )
    assert "chunks_g1_p000" in live.bodies[0]
    assert "chunks_g2_p000" in building.bodies[0]


# ------------------------------------------------------------------------------------------


def test_a_backfill_does_not_refresh_per_batch() -> None:
    """Refreshing per batch spends the rebuild building segments it immediately merges away."""
    assert BulkIndexer(FakeClient()).refresh is False
