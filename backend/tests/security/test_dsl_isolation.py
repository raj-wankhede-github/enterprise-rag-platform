"""The tenant-isolation contract, asserted on the serialized query body.

These tests inspect the DSL that *would be sent*, not the results that come back. That matters:
a filter can look applied in a response and still be a post-filter, which silently loses results
for narrow-ACL users and leaks document existence through hit counts.
"""

from __future__ import annotations

import itertools
import uuid
from datetime import date

import pytest

from app.retrieval.types import MetadataFilter, TenantScope
from app.search import dsl

TENANT_A = uuid.UUID("00000000-0000-0000-0000-0000000000aa")
TENANT_B = uuid.UUID("00000000-0000-0000-0000-0000000000bb")
USER = uuid.UUID("00000000-0000-0000-0000-000000000001")
FINGERPRINT = "abc123def456"


def scope(**kwargs: object) -> TenantScope:
    base: dict[str, object] = {
        "tenant_id": TENANT_A,
        "visibility_rank": 30,
        "access_groups": ("g-eng",),
        "generation_fingerprint": FINGERPRINT,
        "user_id": USER,
    }
    base.update(kwargs)
    return TenantScope(**base)  # type: ignore[arg-type]


def test_tenant_term_is_always_first() -> None:
    clauses = dsl.build_filter(scope())
    assert clauses[0] == {"term": {"tenant_id": str(TENANT_A)}}


def test_exactly_one_tenant_term_in_every_permutation() -> None:
    """Every combination of scope options and metadata filters carries exactly one tenant term."""
    filters = [
        MetadataFilter(),
        MetadataFilter(doc_types=("policy",)),
        MetadataFilter(source_systems=("sharepoint",), authors=("a@x.com",)),
        MetadataFilter(published_after=date(2024, 1, 1), published_before=date(2025, 1, 1)),
        MetadataFilter(doc_ids=("d1", "d2"), collection_ids=("c1",), languages=("en",)),
    ]
    for rank, superseded, as_of, user_id, mf in itertools.product(
        (10, 20, 30, 40),
        (True, False),
        (None, date(2024, 6, 1)),
        (USER, None),
        filters,
    ):
        clauses = dsl.build_filter(
            scope(visibility_rank=rank, include_superseded=superseded, as_of=as_of, user_id=user_id),
            mf,
        )
        assert dsl.count_tenant_terms(clauses) == 1, (rank, superseded, as_of, user_id, mf)


def test_generation_fingerprint_is_always_present() -> None:
    clauses = dsl.build_filter(scope())
    assert {"term": {"generation_fingerprint": FINGERPRINT}} in clauses


@pytest.mark.parametrize("rank", [10, 20, 30, 40])
def test_visibility_is_an_lte_range_on_the_principals_rank(rank: int) -> None:
    clauses = dsl.build_filter(scope(visibility_rank=rank))
    assert {"range": {"visibility_rank": {"lte": rank}}} in clauses


def test_a_prod_user_can_never_match_a_higher_rank() -> None:
    clauses = dsl.build_filter(scope(visibility_rank=10))
    rng = next(c for c in clauses if "range" in c and "visibility_rank" in c["range"])
    assert rng["range"]["visibility_rank"]["lte"] == 10


def test_group_clause_uses_a_terms_lookup_not_an_inline_list() -> None:
    """A user in 300 directory groups would otherwise produce a ~12 KB query."""
    clauses = dsl.build_filter(scope())
    group_clause = next(c for c in clauses if "bool" in c and "should" in c["bool"])
    lookups = [
        s for s in group_clause["bool"]["should"] if "terms" in s and "index" in s["terms"].get("access_groups", {})
    ]
    assert len(lookups) == 1
    assert lookups[0]["terms"]["access_groups"]["index"] == dsl.USER_ACL_INDEX
    assert lookups[0]["terms"]["access_groups"]["id"] == f"{TENANT_A}:{USER}"


def test_public_documents_match_without_group_membership() -> None:
    clauses = dsl.build_filter(scope(access_groups=(), user_id=None))
    group_clause = next(c for c in clauses if "bool" in c and "should" in c["bool"])
    assert {"term": {"access_groups": dsl.PUBLIC_GROUP}} in group_clause["bool"]["should"]


def test_deny_beats_allow() -> None:
    clauses = dsl.build_filter(scope())
    deny = [c for c in clauses if "bool" in c and "must_not" in c["bool"]]
    assert deny, "a deny clause must be present"
    rendered = repr(deny)
    assert "denied_groups" in rendered
    assert "denied_user_ids" in rendered


def test_superseded_documents_are_excluded_by_default() -> None:
    assert {"term": {"is_superseded": False}} in dsl.build_filter(scope())


def test_historical_questions_may_opt_into_superseded() -> None:
    clauses = dsl.build_filter(scope(include_superseded=True))
    assert {"term": {"is_superseded": False}} not in clauses


def test_effective_window_tolerates_missing_bounds() -> None:
    """An open-ended effective window must still match; a missing field is not an exclusion."""
    clauses = dsl.build_filter(scope(as_of=date(2025, 6, 1)))
    rendered = repr(clauses)
    assert "effective_from" in rendered
    assert "effective_to" in rendered
    assert "must_not" in rendered  # the "field absent" branch of each window


def test_metadata_filters_only_narrow() -> None:
    """Adding metadata filters must not remove any ACL clause."""
    bare = dsl.build_filter(scope())
    filtered = dsl.build_filter(scope(), MetadataFilter(doc_types=("policy",)))
    for clause in bare:
        assert clause in filtered
    assert {"terms": {"doc_type": ["policy"]}} in filtered


def test_two_tenants_produce_different_filters() -> None:
    a = dsl.build_filter(scope(tenant_id=TENANT_A))
    b = dsl.build_filter(scope(tenant_id=TENANT_B))
    assert a != b
    assert {"term": {"tenant_id": str(TENANT_B)}} in b
    assert {"term": {"tenant_id": str(TENANT_A)}} not in b


def test_count_tenant_terms_finds_nested_occurrences() -> None:
    """The counter must see through the knn.filter nesting, which is where post-filtering hides."""
    body = {
        "query": {"bool": {"filter": [{"term": {"tenant_id": "x"}}]}},
        "knn": {"embedding": {"filter": {"bool": {"filter": [{"term": {"tenant_id": "x"}}]}}}},
    }
    assert dsl.count_tenant_terms(body) == 2
    assert dsl.count_tenant_terms({"query": {"match_all": {}}}) == 0
