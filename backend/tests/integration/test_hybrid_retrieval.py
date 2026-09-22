"""Hybrid retrieval against a real cluster.

Covers what only a live engine can show: that four legs really do go out as one ``_msearch``,
that each contributes candidates, that fusion produces a sensible order, and that the whole
thing still cannot cross a tenant boundary or a visibility rank.

It also closes the loop on abstention -- a question whose answer is genuinely absent from the
corpus must come back as "I don't know", not as the nearest-looking paragraph.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
from opensearchpy import AsyncOpenSearch

from app.answer.abstain import abstain
from app.answer.evidence import assess
from app.answer.types import AbstentionReason
from app.embeddings.hashing import HashingEmbedder
from app.ingestion.chunker import StructureAwareChunker
from app.ingestion.contextualize import DocumentContext, TemplateContextualizer
from app.ingestion.loaders import default_registry
from app.ingestion.pipeline import IngestionPipeline, index_action
from app.retrieval.hybrid import OpenSearchHybridRetriever
from app.retrieval.types import RetrievalProfile, RetrievalRequest, TenantScope
from app.search.client import SearchClient
from app.search.generations import chunk_document_id
from app.search.mappings import chunk_index_body, parent_index_body

pytestmark = pytest.mark.integration

OPENSEARCH_URL = os.environ.get("OPENSEARCH_URL", "http://localhost:9200")
DIM = 64

TENANT_A = uuid.UUID("aaaa0000-0000-0000-0000-00000000000a")
TENANT_B = uuid.UUID("bbbb0000-0000-0000-0000-00000000000b")

POLICY = """# Travel Policy

## Per Diem

Employees travelling on company business may claim a per diem allowance under reference
SEC-4.2.1. The allowance depends on destination grade and is paid without receipts.

| Grade | Per diem |
| --- | --- |
| A | 120 |
| B | 90 |

## Escalation Procedure

If a claim is rejected the employee may escalate to their line manager within ten working days.
The line manager consults the finance business partner, who reviews the original submission and
the rejection note. Where the two disagree the matter passes to the head of finance, whose
decision closes the escalation. Ticket TKT-99812 tracks exceptions to this procedure.
"""

CONFIDENTIAL = """# Board Compensation

## Executive Pay

The zarquon executive compensation band is set by the remuneration committee and is not
published to staff.
"""


def pipeline() -> IngestionPipeline:
    return IngestionPipeline(
        registry=default_registry(),
        chunker=StructureAwareChunker(),
        contextualizer=TemplateContextualizer(),
        embedder=HashingEmbedder(DIM),
    )


@pytest.fixture
async def raw_client() -> AsyncIterator[AsyncOpenSearch]:
    client = AsyncOpenSearch(hosts=[OPENSEARCH_URL], timeout=30)
    try:
        if not await client.ping():  # pragma: no cover - env dependent
            pytest.skip(f"OpenSearch not reachable at {OPENSEARCH_URL}")
        yield client
    finally:
        await client.close()


async def _index_document(
    client: AsyncOpenSearch,
    *,
    chunk_index: str,
    parent_index: str,
    markdown: str,
    filename: str,
    tenant_id: uuid.UUID,
    title: str,
    visibility_rank: int,
    fingerprint: str,
) -> None:
    result = await pipeline().run(markdown.encode(), filename, context=DocumentContext(title=title, doc_type="policy"))
    doc_id, version_id = uuid.uuid4(), uuid.uuid4()
    parent_ids = {parent.ordinal: f"{version_id}:{parent.ordinal}" for parent in result.parents}

    lines: list[str] = []
    for item in result.prepared:
        chunk_id = chunk_document_id(tenant_id=tenant_id, doc_version_id=version_id, ordinal=item.chunk.ordinal)
        body = index_action(
            item,
            tenant_id=tenant_id,
            doc_id=doc_id,
            doc_version_id=version_id,
            parent_id=parent_ids[item.chunk.parent_ordinal],
            chunk_id=chunk_id,
            generation_fingerprint=fingerprint,
            title=title,
            visibility_rank=visibility_rank,
            access_groups=[],
        )
        lines.append(json.dumps({"index": {"_index": chunk_index, "_id": chunk_id}}))
        lines.append(json.dumps(body))

    # Parents carry the same ACL fields so the parent leg is filtered exactly like the others.
    for parent in result.parents:
        parent_body = {
            "tenant_id": str(tenant_id),
            "doc_id": str(doc_id),
            "doc_version_id": str(version_id),
            "parent_id": parent_ids[parent.ordinal],
            "parent_ordinal": parent.ordinal,
            "content": parent.text,
            "title": title,
            "heading_path": parent.heading_path,
            "generation_fingerprint": fingerprint,
            "visibility_rank": visibility_rank,
            "access_groups": ["*"],
            "is_active": True,
            "is_superseded": False,
            "child_count": sum(1 for c in result.chunked.children if c.parent_ordinal == parent.ordinal),
        }
        lines.append(json.dumps({"index": {"_index": parent_index, "_id": parent_ids[parent.ordinal]}}))
        lines.append(json.dumps(parent_body))

    response = await client.bulk(body="\n".join(lines) + "\n", refresh=True)
    assert not response["errors"], json.dumps(response)[:2000]


@pytest.fixture
async def corpus(raw_client: AsyncOpenSearch) -> AsyncIterator[dict[str, Any]]:
    suffix = uuid.uuid4().hex[:8]
    chunk_index, parent_index = f"test-hyb-c-{suffix}", f"test-hyb-p-{suffix}"
    await raw_client.indices.create(
        index=chunk_index,
        body=chunk_index_body(dimension=DIM, shards=1, replicas=0, refresh_interval="1s"),
    )
    await raw_client.indices.create(
        index=parent_index, body=parent_index_body(shards=1, replicas=0, refresh_interval="1s")
    )
    fingerprint = pipeline().generation.fingerprint

    for tenant in (TENANT_A, TENANT_B):
        await _index_document(
            raw_client,
            chunk_index=chunk_index,
            parent_index=parent_index,
            markdown=POLICY,
            filename="travel.md",
            tenant_id=tenant,
            title="Travel Policy",
            visibility_rank=10,
            fingerprint=fingerprint,
        )
    await _index_document(
        raw_client,
        chunk_index=chunk_index,
        parent_index=parent_index,
        markdown=CONFIDENTIAL,
        filename="board.md",
        tenant_id=TENANT_A,
        title="Board Compensation",
        visibility_rank=40,
        fingerprint=fingerprint,
    )

    try:
        yield {"chunk_index": chunk_index, "parent_index": parent_index, "fingerprint": fingerprint}
    finally:
        await raw_client.indices.delete(index=chunk_index, ignore=[404])
        await raw_client.indices.delete(index=parent_index, ignore=[404])


def retriever(raw_client: AsyncOpenSearch, corpus: dict[str, Any], **kwargs: Any) -> OpenSearchHybridRetriever:
    return OpenSearchHybridRetriever(
        SearchClient(raw_client),
        chunk_index=corpus["chunk_index"],
        parent_index=corpus["parent_index"],
        embedder=HashingEmbedder(DIM),
        **kwargs,
    )


def request(
    query: str,
    corpus: dict[str, Any],
    *,
    tenant: uuid.UUID = TENANT_A,
    rank: int = 10,
    **kwargs: Any,
) -> RetrievalRequest:
    return RetrievalRequest(
        query=query,
        scope=TenantScope(
            tenant_id=tenant,
            visibility_rank=rank,
            access_groups=(),
            generation_fingerprint=corpus["fingerprint"],
        ),
        **kwargs,
    )


# --------------------------------------------------------------------------------------------
# The legs run, and each contributes
# --------------------------------------------------------------------------------------------


async def test_hybrid_retrieval_returns_candidates(raw_client: AsyncOpenSearch, corpus: dict[str, Any]) -> None:
    outcome = await retriever(raw_client, corpus).retrieve(request("per diem allowance for travel", corpus))
    assert outcome.candidates
    assert all(candidate.fused_score > 0 for candidate in outcome.candidates)


async def test_all_four_legs_execute_in_one_request(raw_client: AsyncOpenSearch, corpus: dict[str, Any]) -> None:
    outcome = await retriever(raw_client, corpus).retrieve(
        request("per diem allowance", corpus, exact_tokens=("SEC-4.2.1",))
    )
    names = {leg.name for leg in outcome.diagnostics.legs}
    assert names == {"bm25", "exact", "dense", "parent"}
    assert all(leg.error is None for leg in outcome.diagnostics.legs)


async def test_no_leg_reports_an_error(raw_client: AsyncOpenSearch, corpus: dict[str, Any]) -> None:
    """A malformed leg body shows up here rather than as quietly reduced recall."""
    outcome = await retriever(raw_client, corpus).retrieve(
        request("escalation procedure", corpus, exact_tokens=("TKT-99812",))
    )
    failures = [leg for leg in outcome.diagnostics.legs if leg.error]
    assert not failures, failures


async def test_candidates_record_which_leg_found_them(raw_client: AsyncOpenSearch, corpus: dict[str, Any]) -> None:
    outcome = await retriever(raw_client, corpus).retrieve(request("per diem", corpus, exact_tokens=("SEC-4.2.1",)))
    assert any(candidate.legs for candidate in outcome.candidates)
    found_by = {name for candidate in outcome.candidates for name in candidate.legs}
    assert "bm25" in found_by


async def test_the_exact_leg_finds_an_identifier_the_others_may_miss(
    raw_client: AsyncOpenSearch, corpus: dict[str, Any]
) -> None:
    outcome = await retriever(raw_client, corpus).retrieve(request("TKT-99812", corpus, exact_tokens=("TKT-99812",)))
    assert outcome.candidates
    assert any("TKT-99812" in candidate.text for candidate in outcome.candidates)


async def test_the_parent_leg_contributes_children(raw_client: AsyncOpenSearch, corpus: dict[str, Any]) -> None:
    """A question about a whole section is what the parent leg exists for."""
    outcome = await retriever(raw_client, corpus).retrieve(request("summarise the escalation procedure", corpus))
    parent_leg = next(leg for leg in outcome.diagnostics.legs if leg.name == "parent")
    assert parent_leg.hits > 0, "the parent leg should have matched a section and projected it"
    assert any("parent" in candidate.legs for candidate in outcome.candidates)


async def test_ablation_can_disable_legs_without_a_second_code_path(
    raw_client: AsyncOpenSearch, corpus: dict[str, Any]
) -> None:
    outcome = await retriever(raw_client, corpus).retrieve(request("per diem", corpus, legs=frozenset({"bm25"})))
    assert {leg.name for leg in outcome.diagnostics.legs} == {"bm25"}
    assert outcome.candidates


async def test_top_k_is_respected(raw_client: AsyncOpenSearch, corpus: dict[str, Any]) -> None:
    outcome = await retriever(raw_client, corpus).retrieve(request("per diem", corpus, top_k=2))
    assert len(outcome.candidates) <= 2


async def test_profile_weights_reach_the_fused_scores(raw_client: AsyncOpenSearch, corpus: dict[str, Any]) -> None:
    """Per-tenant weighted RRF is a data change, not a deploy.

    Asserted on the scores rather than the ordering: this corpus is small enough that every leg
    returns the same handful of chunks in the same order, so no weighting can reorder them. The
    claim being tested is that weights are applied at all, and scores show that unambiguously.
    """
    base = RetrievalProfile()
    exact_heavy = RetrievalProfile(leg_weights={"bm25": 0.1, "exact": 5.0, "dense": 0.1, "parent": 0.1})
    common: dict[str, Any] = {"exact_tokens": ("TKT-99812",)}

    first = await retriever(raw_client, corpus).retrieve(request("escalation", corpus, profile=base, **common))
    second = await retriever(raw_client, corpus).retrieve(request("escalation", corpus, profile=exact_heavy, **common))

    by_id_first = {c.chunk_id: c.fused_score for c in first.candidates}
    by_id_second = {c.chunk_id: c.fused_score for c in second.candidates}
    shared = set(by_id_first) & set(by_id_second)
    assert shared, "the two runs should retrieve overlapping chunks"
    assert any(by_id_first[cid] != by_id_second[cid] for cid in shared)


async def test_every_returned_candidate_has_a_positive_fused_score(
    raw_client: AsyncOpenSearch, corpus: dict[str, Any]
) -> None:
    """A zero score means a leg contributed a candidate without contributing its rank."""
    outcome = await retriever(raw_client, corpus).retrieve(
        request("summarise the escalation procedure", corpus, exact_tokens=("TKT-99812",))
    )
    assert outcome.candidates
    zero_scored = [c.chunk_id for c in outcome.candidates if c.fused_score <= 0]
    assert not zero_scored, f"candidates fused to zero: {zero_scored}"


# --------------------------------------------------------------------------------------------
# Isolation still holds through the whole retriever
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("tenant", [TENANT_A, TENANT_B])
async def test_hybrid_retrieval_never_crosses_a_tenant(
    raw_client: AsyncOpenSearch, corpus: dict[str, Any], tenant: uuid.UUID
) -> None:
    """Both tenants hold the same document, so only the filter separates them."""
    outcome = await retriever(raw_client, corpus).retrieve(
        request("per diem allowance", corpus, tenant=tenant, rank=40, exact_tokens=("SEC-4.2.1",))
    )
    assert outcome.candidates
    doc_ids = {candidate.doc_id for candidate in outcome.candidates}
    other = await retriever(raw_client, corpus).retrieve(
        request(
            "per diem allowance",
            corpus,
            tenant=TENANT_B if tenant == TENANT_A else TENANT_A,
            rank=40,
            exact_tokens=("SEC-4.2.1",),
        )
    )
    assert doc_ids.isdisjoint({candidate.doc_id for candidate in other.candidates})


async def test_hybrid_retrieval_respects_visibility_rank(raw_client: AsyncOpenSearch, corpus: dict[str, Any]) -> None:
    low = await retriever(raw_client, corpus).retrieve(request("zarquon compensation", corpus, rank=10))
    high = await retriever(raw_client, corpus).retrieve(request("zarquon compensation", corpus, rank=40))
    assert not any("zarquon" in candidate.text for candidate in low.candidates)
    assert any("zarquon" in candidate.text for candidate in high.candidates)


async def test_a_stale_generation_retrieves_nothing(raw_client: AsyncOpenSearch, corpus: dict[str, Any]) -> None:
    stale = dict(corpus, fingerprint="0" * 16)
    outcome = await retriever(raw_client, corpus).retrieve(request("per diem", stale))
    assert outcome.candidates == []


# --------------------------------------------------------------------------------------------
# Abstention, end to end
# --------------------------------------------------------------------------------------------


async def test_a_question_the_corpus_cannot_answer_abstains(
    raw_client: AsyncOpenSearch, corpus: dict[str, Any]
) -> None:
    """The behaviour the product is sold on: no answer rather than a plausible one.

    The corpus talks about travel and expenses at length. It says nothing about parental leave,
    and a system without an evidence gate will happily answer from the nearest paragraph.
    """
    question = "how many weeks of paid parental leave am I entitled to"
    outcome = await retriever(raw_client, corpus).retrieve(request(question, corpus))
    assessment = assess(question, outcome.candidates)

    assert not assessment.sufficient
    answer = abstain(assessment.reason or AbstentionReason.NO_RESULTS, near_misses=assessment.near_misses)
    assert answer.answerable is False
    assert "do not have an answer" in answer.text or "would rather not guess" in answer.text
    # It must not have invented a number.
    assert "weeks" not in answer.text.lower().replace("parental leave", "")


async def test_a_question_the_corpus_can_answer_is_not_abstained(
    raw_client: AsyncOpenSearch, corpus: dict[str, Any]
) -> None:
    question = "what is the per diem allowance"
    outcome = await retriever(raw_client, corpus).retrieve(request(question, corpus))
    assessment = assess(question, outcome.candidates)
    assert assessment.sufficient
    assert assessment.supporting


async def test_abstains_when_the_only_matching_document_is_out_of_reach(
    raw_client: AsyncOpenSearch, corpus: dict[str, Any]
) -> None:
    """A PROD user asking about board pay must get "I don't know", not a permission error.

    Revealing that a matching document exists but is restricted is itself a disclosure.
    """
    question = "what is the zarquon executive compensation band"
    outcome = await retriever(raw_client, corpus).retrieve(request(question, corpus, rank=10))
    assessment = assess(question, outcome.candidates)
    assert not assessment.sufficient
    answer = abstain(assessment.reason or AbstentionReason.NO_RESULTS)
    assert "zarquon" not in answer.text.lower()
    assert "restricted" not in answer.text.lower()
    assert "permission" not in answer.text.lower() or "your administrator" in answer.text.lower()
