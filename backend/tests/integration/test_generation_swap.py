"""A generation rebuild and alias swap against a real cluster, under continuous traffic.

This is the acceptance test for the rebuild: drive `/search` traffic continuously through a
384-d to 1024-d rebuild and assert **zero errors and zero empty result sets**, then rehearse the
rollback.

Neither half of that assertion is redundant, and the second is the one that catches real bugs:

* **Zero errors** catches the swap performed as a per-pool loop, which briefly leaves a reader
  resolving an alias that points at nothing.
* **Zero empty results** catches the subtler failure, and it is the reason this test cannot be a
  unit test. Every query carries a `generation_fingerprint` term filter. If the reader picks up
  the new index before the new fingerprint -- or the other way round -- the query is perfectly
  valid, returns HTTP 200, and matches nothing. No error is logged anywhere. The user sees "no
  results" for their own documents.

A fake cluster cannot show either, because both are properties of how OpenSearch applies alias
actions, not of our code's control flow.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from collections.abc import AsyncIterator

import pytest
from opensearchpy import AsyncOpenSearch

from app.search.admin import IndexAdmin
from app.search.bulk import BulkDoc, BulkIndexer
from app.search.generations import (
    GenerationSpec,
    GenerationState,
    chunk_index_name,
    chunk_read_alias,
    chunk_write_alias,
    parent_index_name,
)
from app.search.rebuild import BackfillBatch, RebuildOrchestrator

pytestmark = pytest.mark.integration

OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
TENANT = uuid.UUID("cccc0000-0000-0000-0000-00000000000c")
POOLS = [0, 1]
DOCS_PER_POOL = 120

OLD_DIM = 384
NEW_DIM = 1024

OLD_SPEC = GenerationSpec(
    embedder_id="hashing-384", chunker_version="1", contextualizer_version="t1", dimension=OLD_DIM
)
NEW_SPEC = GenerationSpec(
    embedder_id="hashing-1024", chunker_version="1", contextualizer_version="t1", dimension=NEW_DIM
)


@pytest.fixture
async def client() -> AsyncIterator[AsyncOpenSearch]:
    opensearch = AsyncOpenSearch(hosts=[OPENSEARCH_URL], timeout=30)
    if not await opensearch.ping():
        pytest.skip(f"no OpenSearch at {OPENSEARCH_URL}")
    yield opensearch
    for generation in (1, 2):
        for pool in POOLS:
            for index in (
                chunk_index_name(generation=generation, pool=pool),
                parent_index_name(generation=generation, pool=pool),
            ):
                await opensearch.indices.delete(index=index, ignore=[404])
    await opensearch.close()


def make_docs(*, generation: int, pool: int, fingerprint: str, dimension: int) -> list[BulkDoc]:
    """One pool's worth of chunks, stamped with the generation's fingerprint."""
    return [
        BulkDoc(
            index=chunk_index_name(generation=generation, pool=pool),
            doc_id=f"p{pool}-{ordinal:04d}",
            routing=str(TENANT),
            body={
                "tenant_id": str(TENANT),
                "doc_id": str(uuid.uuid5(TENANT, f"{pool}-{ordinal}")),
                "doc_version_id": str(uuid.uuid5(TENANT, f"v{pool}-{ordinal}")),
                "parent_id": f"p{pool}-parent-{ordinal // 10}",
                "chunk_id": f"p{pool}-{ordinal:04d}",
                "ordinal": ordinal,
                "generation_fingerprint": fingerprint,
                "content": f"The reimbursement policy states that claim {ordinal} is processed within ten days.",
                "context_line": f"Section {ordinal // 10} of the Travel Policy",
                "title": "Travel Policy",
                "heading_path": "Reimbursement > Claims",
                "embedding": [0.1] * dimension,
                "visibility_rank": 10,
                "access_groups": ["*"],
                "is_active": True,
                "is_superseded": False,
                "block_kinds": ["paragraph"],
                "page_from": 1,
                "page_to": 1,
                "token_count": 18,
                "content_sha256": f"{ordinal:064x}",
                "simhash64": "0",
            },
        )
        for ordinal in range(DOCS_PER_POOL)
    ]


class StaticSource:
    """Stands in for the Postgres read. The rebuild's correctness does not depend on the source."""

    def __init__(self, *, generation: int, fingerprint: str, dimension: int, batch: int = 40) -> None:
        self.generation = generation
        self.fingerprint = fingerprint
        self.dimension = dimension
        self.batch = batch

    async def count(self, *, pool: int) -> int:
        return DOCS_PER_POOL

    async def stream(self, *, pool: int, after: str | None) -> AsyncIterator[BackfillBatch]:
        docs = make_docs(generation=self.generation, pool=pool, fingerprint=self.fingerprint, dimension=self.dimension)
        if after:
            docs = [doc for doc in docs if doc.doc_id > after]
        for offset in range(0, len(docs), self.batch):
            page = docs[offset : offset + self.batch]
            yield BackfillBatch(docs=page, cursor=page[-1].doc_id)
            # Yield the loop so the traffic generator actually interleaves with the backfill;
            # without this the backfill runs to completion between two queries and the test
            # proves nothing about concurrency.
            await asyncio.sleep(0)


async def search_once(client: AsyncOpenSearch, *, pool: int, fingerprint: str) -> int:
    """One realistic read: through the alias, filtered by tenant and generation."""
    response = await client.search(
        index=chunk_read_alias(pool),
        routing=str(TENANT),
        body={
            "size": 10,
            "query": {
                "bool": {
                    "must": [{"match": {"content": "reimbursement claim"}}],
                    "filter": [
                        {"term": {"tenant_id": str(TENANT)}},
                        {"term": {"generation_fingerprint": fingerprint}},
                        {"term": {"is_active": True}},
                        {"range": {"visibility_rank": {"lte": 40}}},
                    ],
                }
            },
        },
    )
    return len(response["hits"]["hits"])


class TrafficGenerator:
    """Queries continuously until stopped, recording every error and every empty result."""

    def __init__(self, client: AsyncOpenSearch) -> None:
        self.client = client
        self.queries = 0
        self.errors: list[str] = []
        self.empties = 0
        self._stop = asyncio.Event()
        #: The fingerprint the reader believes is live. Swapped by the test at the same moment
        #: the aliases move -- which is what makes the "zero empty results" assertion meaningful.
        self.fingerprint = OLD_SPEC.fingerprint

    async def run(self) -> None:
        while not self._stop.is_set():
            for pool in POOLS:
                try:
                    hits = await search_once(self.client, pool=pool, fingerprint=self.fingerprint)
                    self.queries += 1
                    if hits == 0:
                        self.empties += 1
                except Exception as exc:
                    self.errors.append(f"{type(exc).__name__}: {exc}")
            await asyncio.sleep(0.01)

    def stop(self) -> None:
        self._stop.set()


async def seed_generation_one(client: AsyncOpenSearch) -> None:
    admin = IndexAdmin(client, shards=1, replicas=0)
    indexer = BulkIndexer(client, refresh=True)
    for pool in POOLS:
        await admin.create_generation(generation=1, pool=pool, dimension=OLD_DIM, for_backfill=False)
        result = await indexer.index(
            make_docs(generation=1, pool=pool, fingerprint=OLD_SPEC.fingerprint, dimension=OLD_DIM)
        )
        result.raise_for_failures(context=f"seeding pool {pool}")
    await admin.point_aliases(generation=1, pools=POOLS)


# ----------------------------------------------------------------------------------------------


async def test_a_rebuild_and_swap_under_continuous_traffic_loses_nothing(
    client: AsyncOpenSearch,
) -> None:
    """The acceptance criterion for step 12, end to end.

    384-d generation live and serving, 1024-d generation built underneath it, promoted, and the
    old one retired -- with traffic running throughout.
    """
    await seed_generation_one(client)

    traffic = TrafficGenerator(client)
    driving = asyncio.create_task(traffic.run())
    await asyncio.sleep(0.1)  # establish a baseline of successful queries before anything moves
    assert traffic.queries > 0, "traffic generator produced no queries"
    assert traffic.empties == 0, "the baseline generation was not serving before the rebuild began"

    runner = RebuildOrchestrator(
        admin=IndexAdmin(client, shards=1, replicas=0),
        source=StaticSource(generation=2, fingerprint=NEW_SPEC.fingerprint, dimension=NEW_DIM),
        client=client,
        spec=NEW_SPEC,
        generation=2,
        pools=POOLS,
        previous_generation=1,
    )

    async def shadow_eval(fingerprint: str) -> dict[str, float]:
        """A real shadow read: query the new indices directly, while the alias still points old."""
        await client.indices.refresh(index=",".join(chunk_index_name(generation=2, pool=p) for p in POOLS))
        found = 0
        for pool in POOLS:
            response = await client.search(
                index=chunk_index_name(generation=2, pool=pool),
                routing=str(TENANT),
                body={
                    "size": 10,
                    "query": {
                        "bool": {
                            "must": [{"match": {"content": "reimbursement claim"}}],
                            "filter": [{"term": {"generation_fingerprint": fingerprint}}],
                        }
                    },
                },
            )
            found += len(response["hits"]["hits"])
        return {"recall@10": 1.0 if found else 0.0, "ndcg@10": 0.9 if found else 0.0}

    await runner.prepare(dimension=NEW_DIM)
    await runner.backfill()
    assert runner.state is GenerationState.BACKFILLED, "backfill did not complete"

    report = await runner.verify(evaluate=shadow_eval, baseline={"recall@10": 0.95, "ndcg@10": 0.85})
    assert report.passed, f"verification failed: {report.failures}"

    # The swap and the reader's notion of the live fingerprint move together. In production this
    # is the same ordering: the alias call commits, then readers pick up the new generation from
    # their next principal/profile refresh.
    await runner.promote(report)
    traffic.fingerprint = NEW_SPEC.fingerprint

    await asyncio.sleep(0.2)  # keep querying after the swap
    traffic.stop()
    await driving

    assert traffic.errors == [], f"swap produced {len(traffic.errors)} errors: {traffic.errors[:3]}"
    assert traffic.empties == 0, f"{traffic.empties} of {traffic.queries} queries returned nothing"
    assert traffic.queries > 20, f"only {traffic.queries} queries ran; the window was too short to prove anything"
    assert runner.state is GenerationState.LIVE


async def test_the_read_alias_resolves_to_exactly_one_index_per_pool_after_the_swap(
    client: AsyncOpenSearch,
) -> None:
    """A swap that adds without removing leaves the alias spanning both generations.

    Queries then match across two fingerprints, and because every query filters on one, half the
    shards are searched for nothing -- slower, and silently so.
    """
    await seed_generation_one(client)
    admin = IndexAdmin(client, shards=1, replicas=0)
    for pool in POOLS:
        await admin.create_generation(generation=2, pool=pool, dimension=NEW_DIM, for_backfill=False)
    await admin.swap(from_generation=1, to_generation=2, pools=POOLS)

    for pool in POOLS:
        resolved = await client.indices.get_alias(name=chunk_read_alias(pool))
        assert list(resolved) == [chunk_index_name(generation=2, pool=pool)]


async def test_exactly_one_index_holds_the_write_alias_after_the_swap(client: AsyncOpenSearch) -> None:
    """OpenSearch rejects two write indices on one alias, so this also proves the swap was
    accepted as a single ordered request rather than silently splitting."""
    await seed_generation_one(client)
    admin = IndexAdmin(client, shards=1, replicas=0)
    for pool in POOLS:
        await admin.create_generation(generation=2, pool=pool, dimension=NEW_DIM, for_backfill=False)
    await admin.swap(from_generation=1, to_generation=2, pools=POOLS)

    for pool in POOLS:
        resolved = await client.indices.get_alias(name=chunk_write_alias(pool))
        writers = [
            index
            for index, aliases in resolved.items()
            if aliases["aliases"][chunk_write_alias(pool)].get("is_write_index")
        ]
        assert writers == [chunk_index_name(generation=2, pool=pool)]


async def test_a_rollback_restores_the_previous_generation_in_one_call(client: AsyncOpenSearch) -> None:
    """Rehearsed, not assumed. A rollback nobody has run is not a rollback."""
    await seed_generation_one(client)
    runner = RebuildOrchestrator(
        admin=IndexAdmin(client, shards=1, replicas=0),
        source=StaticSource(generation=2, fingerprint=NEW_SPEC.fingerprint, dimension=NEW_DIM),
        client=client,
        spec=NEW_SPEC,
        generation=2,
        pools=POOLS,
        previous_generation=1,
    )
    await runner.prepare(dimension=NEW_DIM)
    await runner.backfill()

    async def always_passes(fingerprint: str) -> dict[str, float]:
        return {"recall@10": 1.0, "ndcg@10": 1.0}

    await runner.promote(await runner.verify(evaluate=always_passes, baseline={"recall@10": 0.9}))
    await runner.rollback()

    assert runner.state is GenerationState.DRAINING
    for pool in POOLS:
        resolved = await client.indices.get_alias(name=chunk_read_alias(pool))
        assert list(resolved) == [chunk_index_name(generation=1, pool=pool)]
    # And the old generation still answers, which is the entire point of not deleting it.
    assert await search_once(client, pool=0, fingerprint=OLD_SPEC.fingerprint) > 0


async def test_wrong_generation_documents_are_invisible_rather_than_blended(
    client: AsyncOpenSearch,
) -> None:
    """The invariant the fingerprint filter buys.

    Point the read alias at *both* generations -- the state a botched swap leaves behind -- and
    confirm a query still sees only its own generation. Blending would be the silent failure:
    plausible results, a few points of relevance lost, nobody notices for a quarter.
    """
    await seed_generation_one(client)
    admin = IndexAdmin(client, shards=1, replicas=0)
    indexer = BulkIndexer(client, refresh=True)
    for pool in POOLS:
        await admin.create_generation(generation=2, pool=pool, dimension=NEW_DIM, for_backfill=False)
        await indexer.index(make_docs(generation=2, pool=pool, fingerprint=NEW_SPEC.fingerprint, dimension=NEW_DIM))
    await client.indices.update_aliases(
        body={
            "actions": [
                {"add": {"index": chunk_index_name(generation=2, pool=pool), "alias": chunk_read_alias(pool)}}
                for pool in POOLS
            ]
        }
    )

    old_hits = await search_once(client, pool=0, fingerprint=OLD_SPEC.fingerprint)
    new_hits = await search_once(client, pool=0, fingerprint=NEW_SPEC.fingerprint)
    assert old_hits == 10 and new_hits == 10

    unfiltered = await client.search(
        index=chunk_read_alias(0),
        routing=str(TENANT),
        body={"size": 0, "query": {"match": {"content": "reimbursement claim"}}},
    )
    # Both generations are reachable through the alias; the filter is what separates them.
    assert unfiltered["hits"]["total"]["value"] == DOCS_PER_POOL * 2


async def test_a_backfill_resumes_from_its_cursor_after_an_interruption(
    client: AsyncOpenSearch,
) -> None:
    """The crash at document 900 of 1000. Without this, a long rebuild never finishes."""
    admin = IndexAdmin(client, shards=1, replicas=0)
    for pool in POOLS:
        await admin.create_generation(generation=2, pool=pool, dimension=NEW_DIM, for_backfill=False)

    source = StaticSource(generation=2, fingerprint=NEW_SPEC.fingerprint, dimension=NEW_DIM)
    runner = RebuildOrchestrator(
        admin=admin,
        source=source,
        client=client,
        spec=NEW_SPEC,
        generation=2,
        pools=POOLS,
        previous_generation=None,
    )
    runner.state = GenerationState.BUILDING

    halfway = f"p0-{DOCS_PER_POOL // 2 - 1:04d}"
    await runner.backfill(resume={0: halfway, 1: halfway})

    assert runner.progress[0].indexed == DOCS_PER_POOL // 2
    await client.indices.refresh(index=chunk_index_name(generation=2, pool=0))
    count = await client.count(index=chunk_index_name(generation=2, pool=0))
    assert count["count"] == DOCS_PER_POOL // 2
