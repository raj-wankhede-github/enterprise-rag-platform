"""Ingestion through to search, against a real cluster.

This is the test that proves the pieces fit: the chunker's output survives ``dynamic: strict``,
the index action writes only mapped fields, the analyzers find what the design says they find,
and -- most importantly -- the filter from ``search/dsl.py`` actually constrains a kNN query
rather than being applied after the fact.

Two tenants are given documents containing the *same distinctive tokens*, so nothing but the
filter can separate them. If the tenant term were dropped, these tests would fail loudly rather
than pass with a plausible-looking result set.

The client fixture is function-scoped on purpose: pytest-asyncio gives each test its own event
loop, and an ``AsyncClient`` built on a previous loop fails at connection time with an error that
points at httpcore rather than at the fixture.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest

from app.embeddings.hashing import HashingEmbedder
from app.ingestion.chunker import StructureAwareChunker
from app.ingestion.contextualize import DocumentContext, TemplateContextualizer
from app.ingestion.loaders import default_registry
from app.ingestion.pipeline import IngestionPipeline, index_action
from app.retrieval.types import TenantScope
from app.search import dsl
from app.search.generations import chunk_document_id
from app.search.mappings import chunk_index_body

pytestmark = pytest.mark.integration

OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
DIM = 64

TENANT_A = uuid.UUID("aaaaaaaa-0000-0000-0000-000000000001")
TENANT_B = uuid.UUID("bbbbbbbb-0000-0000-0000-000000000002")

POLICY = """# Travel Policy

## Per Diem

Employees travelling on company business may claim a per diem allowance under reference
SEC-4.2.1. The allowance depends on destination grade and is paid without receipts. Claims are
submitted within thirty days of return through the expense portal and approved by a line manager.

| Grade | Per diem |
| --- | --- |
| A | 120 |
| B | 90 |

## Receipts

Receipts are required for any single item over 50 EUR. Ticket TKT-99812 tracks the exception
process for lost receipts, and questions go to the finance team.
"""

CONFIDENTIAL = """# Board Compensation

## Executive Pay

The zarquon executive compensation band for the coming year is set by the remuneration
committee and is not published to staff. Reference SEC-4.2.1 applies to expense claims made by
board members while travelling on company business.
"""


def pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        registry=default_registry(),
        chunker=StructureAwareChunker(),
        contextualizer=TemplateContextualizer(),
        embedder=HashingEmbedder(DIM),
    )


@pytest.fixture
async def client() -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(base_url=OPENSEARCH_URL, timeout=30.0) as session:
        try:
            (await session.get("/_cluster/health")).raise_for_status()
        except httpx.HTTPError as exc:  # pragma: no cover - env dependent
            pytest.skip(f"OpenSearch not reachable at {OPENSEARCH_URL}: {exc}")
        yield session


async def _ingest(
    client: httpx.AsyncClient,
    index: str,
    *,
    markdown: str,
    filename: str,
    tenant_id: uuid.UUID,
    title: str,
    visibility_rank: int,
    fingerprint: str,
) -> int:
    result = await pipeline().run(markdown.encode(), filename, context=DocumentContext(title=title, doc_type="policy"))
    doc_id, version_id = uuid.uuid4(), uuid.uuid4()
    lines: list[str] = []
    for item in result.prepared:
        chunk_id = chunk_document_id(tenant_id=tenant_id, doc_version_id=version_id, ordinal=item.chunk.ordinal)
        body = index_action(
            item,
            tenant_id=tenant_id,
            doc_id=doc_id,
            doc_version_id=version_id,
            parent_id=f"{version_id}:{item.chunk.parent_ordinal}",
            chunk_id=chunk_id,
            generation_fingerprint=fingerprint,
            title=title,
            visibility_rank=visibility_rank,
            access_groups=[],
        )
        lines.append(json.dumps({"index": {"_index": index, "_id": chunk_id}}))
        lines.append(json.dumps(body))

    response = await client.post(
        "/_bulk?refresh=true",
        content="\n".join(lines) + "\n",
        headers={"Content-Type": "application/x-ndjson"},
    )
    assert response.status_code == 200, response.text
    assert not response.json()["errors"], response.text
    return len(result.prepared)


@pytest.fixture
async def corpus(client: httpx.AsyncClient) -> AsyncIterator[dict[str, Any]]:
    """Two tenants, overlapping vocabulary, differing ranks."""
    index = f"test-e2e-{uuid.uuid4().hex[:8]}"
    await client.put(
        f"/{index}",
        json=chunk_index_body(dimension=DIM, shards=1, replicas=0, refresh_interval="1s"),
    )
    fingerprint = pipeline().generation.fingerprint

    counts = {
        "a_public": await _ingest(
            client,
            index,
            markdown=POLICY,
            filename="travel.md",
            tenant_id=TENANT_A,
            title="Travel Policy",
            visibility_rank=10,
            fingerprint=fingerprint,
        ),
        "a_confidential": await _ingest(
            client,
            index,
            markdown=CONFIDENTIAL,
            filename="board.md",
            tenant_id=TENANT_A,
            title="Board Compensation",
            visibility_rank=40,
            fingerprint=fingerprint,
        ),
        "b_public": await _ingest(
            client,
            index,
            markdown=POLICY,
            filename="travel.md",
            tenant_id=TENANT_B,
            title="Travel Policy",
            visibility_rank=10,
            fingerprint=fingerprint,
        ),
    }
    try:
        yield {"index": index, "fingerprint": fingerprint, "counts": counts}
    finally:
        await client.delete(f"/{index}")


def scope(tenant_id: uuid.UUID, rank: int, fingerprint: str) -> TenantScope:
    return TenantScope(
        tenant_id=tenant_id,
        visibility_rank=rank,
        access_groups=(),
        generation_fingerprint=fingerprint,
        user_id=None,
    )


async def search(client: httpx.AsyncClient, index: str, body: dict[str, Any]) -> dict[str, Any]:
    response = await client.post(f"/{index}/_search", json=body)
    assert response.status_code == 200, response.text
    result: dict[str, Any] = response.json()
    return result


async def total(client: httpx.AsyncClient, index: str, body: dict[str, Any]) -> int:
    hits = await search(client, index, body)
    count: int = hits["hits"]["total"]["value"]
    return count


def bm25(query: str, scope_: TenantScope) -> dict[str, Any]:
    return {
        "size": 20,
        "_source": ["tenant_id", "title", "visibility_rank", "content"],
        "query": {"bool": {"must": [{"match": {"content": query}}], "filter": dsl.build_filter(scope_)}},
    }


def exact(query: str, scope_: TenantScope) -> dict[str, Any]:
    return {
        "size": 20,
        "_source": ["tenant_id", "visibility_rank"],
        "query": {
            "bool": {
                "must": [{"match_phrase": {"content.exact": query}}],
                "filter": dsl.build_filter(scope_),
            }
        },
    }


# --------------------------------------------------------------------------------------------
# The pipeline writes something the mapping accepts
# --------------------------------------------------------------------------------------------


async def test_ingestion_indexes_without_mapping_errors(corpus: dict[str, Any]) -> None:
    assert all(count > 0 for count in corpus["counts"].values())


async def test_the_table_is_searchable_by_a_cell_value(client: httpx.AsyncClient, corpus: dict[str, Any]) -> None:
    """Enterprise answers are frequently a number that appears only in a cell."""
    hits = await search(client, corpus["index"], bm25("120", scope(TENANT_A, 10, corpus["fingerprint"])))
    assert hits["hits"]["total"]["value"] >= 1
    assert any("| A | 120 |" in hit["_source"]["content"] for hit in hits["hits"]["hits"])


@pytest.mark.parametrize("query", ["TKT-99812", "tkt99812", "SEC-4.2.1"])
async def test_identifiers_are_findable_through_the_exact_field(
    client: httpx.AsyncClient, corpus: dict[str, Any], query: str
) -> None:
    found = await total(client, corpus["index"], exact(query, scope(TENANT_A, 10, corpus["fingerprint"])))
    assert found >= 1, query


async def test_the_context_line_is_searchable_separately(client: httpx.AsyncClient, corpus: dict[str, Any]) -> None:
    """Contextual retrieval only helps BM25 if the generated line is actually indexed."""
    body = {
        "size": 10,
        "query": {
            "bool": {
                "must": [{"match": {"context_line": "Per Diem"}}],
                "filter": dsl.build_filter(scope(TENANT_A, 10, corpus["fingerprint"])),
            }
        },
    }
    assert await total(client, corpus["index"], body) >= 1


# --------------------------------------------------------------------------------------------
# Tenant isolation, on identical content
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tenant", [TENANT_A, TENANT_B])
async def test_bm25_never_crosses_a_tenant(
    client: httpx.AsyncClient, corpus: dict[str, Any], tenant: uuid.UUID
) -> None:
    """Both tenants hold the same document, so only the filter can separate them."""
    hits = await search(
        client,
        corpus["index"],
        bm25("per diem allowance", scope(tenant, 40, corpus["fingerprint"])),
    )
    assert hits["hits"]["hits"]
    assert {hit["_source"]["tenant_id"] for hit in hits["hits"]["hits"]} == {str(tenant)}


async def test_knn_never_crosses_a_tenant(client: httpx.AsyncClient, corpus: dict[str, Any]) -> None:
    """The filter must be inside knn, not applied to its output."""
    vector = await HashingEmbedder(DIM).embed_query("per diem allowance for travel")
    body = {
        "size": 20,
        "_source": ["tenant_id"],
        "query": {
            "knn": {
                "embedding": {
                    "vector": vector,
                    "k": 50,
                    "filter": {"bool": {"filter": dsl.build_filter(scope(TENANT_B, 40, corpus["fingerprint"]))}},
                }
            }
        },
    }
    hits = await search(client, corpus["index"], body)
    assert hits["hits"]["hits"]
    assert {hit["_source"]["tenant_id"] for hit in hits["hits"]["hits"]} == {str(TENANT_B)}


async def test_hit_count_does_not_leak_other_tenants(client: httpx.AsyncClient, corpus: dict[str, Any]) -> None:
    """``total`` must reflect the filtered set, or it discloses that other documents exist."""
    scoped = await total(client, corpus["index"], bm25("per diem", scope(TENANT_A, 10, corpus["fingerprint"])))
    unfiltered = await total(client, corpus["index"], {"size": 0, "query": {"match": {"content": "per diem"}}})
    assert scoped >= 1
    assert scoped < unfiltered, "an unfiltered query must see strictly more than one tenant's share"


# --------------------------------------------------------------------------------------------
# Visibility rank
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("rank", [10, 20, 30])
async def test_below_admin_rank_cannot_see_the_confidential_document(
    client: httpx.AsyncClient, corpus: dict[str, Any], rank: int
) -> None:
    found = await total(client, corpus["index"], bm25("zarquon", scope(TENANT_A, rank, corpus["fingerprint"])))
    assert found == 0


async def test_admin_rank_can_see_the_confidential_document(client: httpx.AsyncClient, corpus: dict[str, Any]) -> None:
    found = await total(client, corpus["index"], bm25("zarquon", scope(TENANT_A, 40, corpus["fingerprint"])))
    assert found >= 1


async def test_rank_filter_does_not_hide_lower_ranked_documents(
    client: httpx.AsyncClient, corpus: dict[str, Any]
) -> None:
    """At or below: an ADMIN still sees everything a PROD user sees."""
    admin = await total(client, corpus["index"], bm25("per diem", scope(TENANT_A, 40, corpus["fingerprint"])))
    prod = await total(client, corpus["index"], bm25("per diem", scope(TENANT_A, 10, corpus["fingerprint"])))
    assert admin >= prod >= 1


async def test_a_shared_identifier_is_still_rank_filtered(client: httpx.AsyncClient, corpus: dict[str, Any]) -> None:
    """SEC-4.2.1 appears in both the public and the confidential document."""
    hits = await search(client, corpus["index"], exact("SEC-4.2.1", scope(TENANT_A, 10, corpus["fingerprint"])))
    assert hits["hits"]["hits"]
    assert all(hit["_source"]["visibility_rank"] <= 10 for hit in hits["hits"]["hits"])


# --------------------------------------------------------------------------------------------
# Generation fingerprint
# --------------------------------------------------------------------------------------------


async def test_a_wrong_generation_sees_nothing(client: httpx.AsyncClient, corpus: dict[str, Any]) -> None:
    """A half-built index must be invisible, not blended into results."""
    found = await total(client, corpus["index"], bm25("per diem", scope(TENANT_A, 40, "0" * 16)))
    assert found == 0
