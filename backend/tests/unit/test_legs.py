"""Leg query construction.

These assert the *shape of the request that would be sent*. That matters more than it sounds:
the kNN filter placement is invisible in a response -- a post-filtered query returns plausible
results, just fewer of them than it should, for the users with the narrowest access.
"""

from __future__ import annotations

import uuid

import pytest

from app.retrieval.legs.base import candidate_from_hit, parse_hits, response_error
from app.retrieval.legs.dense import DenseKnnLeg
from app.retrieval.legs.lexical import Bm25Leg, ExactTokenLeg
from app.retrieval.legs.parent import ParentBm25Leg, project_to_children
from app.retrieval.types import (
    Candidate,
    LegHit,
    MetadataFilter,
    RetrievalProfile,
    RetrievalRequest,
    TenantScope,
)
from app.search import dsl

TENANT = uuid.UUID("33333333-3333-3333-3333-333333333333")
FINGERPRINT = "fingerprint000000"[:16]


def request(**kwargs: object) -> RetrievalRequest:
    base: dict[str, object] = {
        "query": "per diem allowance",
        "scope": TenantScope(
            tenant_id=TENANT,
            visibility_rank=30,
            access_groups=("g1",),
            generation_fingerprint=FINGERPRINT,
        ),
    }
    base.update(kwargs)
    return RetrievalRequest(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------
# Every leg carries the filter
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("leg", "index"),
    [
        (Bm25Leg(), "chunks"),
        (ExactTokenLeg(), "chunks"),
        (DenseKnnLeg(), "chunks"),
        (ParentBm25Leg(), "parents"),
    ],
)
def test_every_leg_carries_exactly_one_tenant_term(leg: object, index: str) -> None:
    req = request(exact_tokens=("TKT-1",), embedding=(0.1, 0.2, 0.3))
    _, body = leg.build(req, index=index)  # type: ignore[attr-defined]
    assert dsl.count_tenant_terms(body) == 1


def test_dense_leg_puts_the_filter_inside_knn_not_outside() -> None:
    """Post-filtering silently destroys recall for narrow-ACL users."""
    _, body = DenseKnnLeg().build(request(embedding=(0.1, 0.2)), index="chunks")
    knn = body["query"]["knn"]["embedding"]
    assert "filter" in knn, "the filter must be a kNN parameter, applied during traversal"
    assert dsl.count_tenant_terms(knn["filter"]) == 1
    # And nowhere else: an outer bool filter would be the post-filter we are avoiding.
    assert "bool" not in body["query"]


def filter_clauses(body: dict) -> list:  # type: ignore[type-arg]
    """The filter list a leg actually sends, wherever that leg puts it."""
    query = body["query"]
    if "knn" in query:
        return query["knn"]["embedding"]["filter"]["bool"]["filter"]
    return query["bool"]["filter"]


def test_metadata_filters_reach_every_leg() -> None:
    """A user-supplied narrowing must apply to the dense leg too, not only the lexical ones."""
    filters = MetadataFilter(doc_types=("policy",))
    expected = {"terms": {"doc_type": ["policy"]}}
    for leg, index in (
        (Bm25Leg(), "chunks"),
        (ExactTokenLeg(), "chunks"),
        (DenseKnnLeg(), "chunks"),
        (ParentBm25Leg(), "parents"),
    ):
        req = request(filters=filters, embedding=(0.1,), exact_tokens=("X-1",))
        _, body = leg.build(req, index=index)  # type: ignore[attr-defined]
        assert expected in filter_clauses(body), f"{leg.name} dropped the metadata filter"  # type: ignore[attr-defined]


# --------------------------------------------------------------------------------------------
# Enablement
# --------------------------------------------------------------------------------------------


def test_exact_leg_is_off_without_identifiers() -> None:
    """It costs nothing on natural-language questions."""
    assert not ExactTokenLeg().enabled(request())
    assert ExactTokenLeg().enabled(request(exact_tokens=("SEC-4.2",)))


def test_dense_leg_is_off_without_an_embedding() -> None:
    """This is how the fast path and the bm25-only ablation row are expressed."""
    assert not DenseKnnLeg().enabled(request())
    assert DenseKnnLeg().enabled(request(embedding=(0.1,)))


def test_lexical_legs_are_off_for_an_empty_query() -> None:
    assert not Bm25Leg().enabled(request(query="   "))
    assert not ParentBm25Leg().enabled(request(query=""))


# --------------------------------------------------------------------------------------------
# Query shape
# --------------------------------------------------------------------------------------------


def test_bm25_boosts_phrase_matches_over_scattered_terms() -> None:
    _, body = Bm25Leg().build(request(), index="chunks")
    clauses = body["query"]["bool"]["should"]
    phrase = next(c for c in clauses if c["multi_match"]["type"] == "phrase")
    best = next(c for c in clauses if c["multi_match"]["type"] == "best_fields")
    assert phrase["multi_match"]["boost"] > 1.0
    assert "boost" not in best["multi_match"]


def test_bm25_weights_headings_above_body() -> None:
    _, body = Bm25Leg().build(request(), index="chunks")
    fields = body["query"]["bool"]["should"][0]["multi_match"]["fields"]
    weights = {field.split("^")[0]: float(field.split("^")[1]) for field in fields}
    assert weights["title"] > weights["content"]
    assert weights["heading_path"] > weights["content"]


def test_exact_leg_matches_any_identifier_not_all() -> None:
    """A question naming two references is relevant to a document containing either."""
    _, body = ExactTokenLeg().build(request(exact_tokens=("TKT-1", "SEC-2")), index="chunks")
    assert body["query"]["bool"]["minimum_should_match"] == 1
    rendered = repr(body["query"]["bool"]["should"])
    assert "TKT-1" in rendered and "SEC-2" in rendered


def test_exact_leg_uses_the_exact_subfields_only() -> None:
    _, body = ExactTokenLeg().build(request(exact_tokens=("TKT-1",)), index="chunks")
    for clause in body["query"]["bool"]["should"]:
        field = next(iter(clause["match_phrase"]))
        assert field.endswith(".exact")


def test_leg_sizes_come_from_the_profile() -> None:
    profile = RetrievalProfile(leg_sizes={"bm25": 42, "exact": 7, "dense": 13, "parent": 5})
    req = request(profile=profile, exact_tokens=("X-1",), embedding=(0.1,))
    assert Bm25Leg().build(req, index="c")[1]["size"] == 42
    assert ExactTokenLeg().build(req, index="c")[1]["size"] == 7
    assert DenseKnnLeg().build(req, index="c")[1]["size"] == 13
    assert ParentBm25Leg().build(req, index="p")[1]["size"] == 5


def test_ef_search_and_oversample_come_from_the_profile() -> None:
    profile = RetrievalProfile(ef_search=256, oversample_factor=2.0)
    _, body = DenseKnnLeg().build(request(profile=profile, embedding=(0.1,)), index="c")
    knn = body["query"]["knn"]["embedding"]
    assert knn["method_parameters"]["ef_search"] == 256
    assert knn["rescore"]["oversample_factor"] == 2.0


def test_oversample_is_omitted_when_not_quantized() -> None:
    _, body = DenseKnnLeg().build(request(embedding=(0.1,)), index="c")
    assert "rescore" not in body["query"]["knn"]["embedding"]


def test_parent_leg_queries_the_parent_index() -> None:
    header, _ = ParentBm25Leg().build(request(), index="parents")
    assert header == {"index": "parents"}


def test_parent_projection_filters_by_parent_id_and_keeps_the_acl() -> None:
    _, body = ParentBm25Leg().projection_query(request(), parent_ids=["p1", "p2"], index="chunks")
    assert {"terms": {"parent_id": ["p1", "p2"]}} in body["query"]["bool"]["filter"]
    assert dsl.count_tenant_terms(body) == 1
    assert body["sort"] == [{"ordinal": "asc"}]


# --------------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------------


def hit(chunk_id: str, score: float = 1.0, **source: object) -> dict[str, object]:
    return {
        "_id": chunk_id,
        "_score": score,
        "_source": {"chunk_id": chunk_id, "content": f"text {chunk_id}", **source},
    }


def test_parse_hits_records_rank_and_leg() -> None:
    candidates = parse_hits({"hits": {"hits": [hit("a"), hit("b")]}}, leg="bm25")
    assert [c.chunk_id for c in candidates] == ["a", "b"]
    assert candidates[0].rank_in("bm25") == 1
    assert candidates[1].rank_in("bm25") == 2


def test_a_failed_leg_yields_no_candidates_rather_than_raising() -> None:
    """Losing one leg costs recall; raising costs the answer."""
    assert parse_hits({"error": {"type": "search_phase_execution_exception"}}, leg="dense") == []


def test_response_error_extracts_a_readable_type() -> None:
    assert response_error({"error": {"type": "boom"}}) == "boom"
    assert response_error({"hits": {"hits": []}}) is None


def test_candidate_carries_metadata_needed_by_priors_and_citations() -> None:
    candidate = candidate_from_hit(
        hit("x", 2.0, visibility_rank=30, is_superseded=True, authority_rank=0.8, page_from=4, page_to=5),
        leg="bm25",
        rank=1,
    )
    assert candidate.meta.visibility_rank == 30
    assert candidate.meta.is_superseded is True
    assert candidate.meta.authority_rank == 0.8
    assert candidate.meta.page_from == 4


# --------------------------------------------------------------------------------------------
# Parent projection
# --------------------------------------------------------------------------------------------


def child(chunk_id: str, parent_id: str) -> Candidate:
    return Candidate(chunk_id=chunk_id, parent_id=parent_id, doc_id="d", doc_version_id="v", text=chunk_id)


def parent_hit(parent_id: str, rank: int) -> Candidate:
    candidate = Candidate(chunk_id="", parent_id=parent_id, doc_id="d", doc_version_id="v", text="")
    candidate.legs["parent"] = LegHit(rank=rank, score=1.0 / rank)
    return candidate


def test_children_inherit_their_parents_ordering() -> None:
    parents = [parent_hit("p2", 1), parent_hit("p1", 2)]
    children = [child("c1", "p1"), child("c2", "p2")]
    ranked = project_to_children(parents, children)
    assert [c.chunk_id for c in ranked] == ["c2", "c1"]
    assert ranked[0].rank_in("parent") == 1


def test_projection_caps_children_per_parent() -> None:
    """A long section must not contribute twenty candidates and crowd out other sources."""
    parents = [parent_hit("p1", 1)]
    children = [child(f"c{i}", "p1") for i in range(10)]
    ranked = project_to_children(parents, children, max_children=3)
    assert len(ranked) == 3
    assert [c.chunk_id for c in ranked] == ["c0", "c1", "c2"]


def test_projection_reranks_contiguously() -> None:
    parents = [parent_hit("p1", 1), parent_hit("p2", 2)]
    children = [child("a", "p1"), child("b", "p1"), child("c", "p2")]
    ranked = project_to_children(parents, children)
    assert [c.rank_in("parent") for c in ranked] == [1, 2, 3]


def test_projection_of_nothing_is_empty() -> None:
    assert project_to_children([], []) == []
