"""THE filter chokepoint.

Every OpenSearch query in this product gets its filter clauses from this module and nowhere else.
That is not a style preference -- it is the mechanism that makes multi-tenancy correct:

* A shared, pooled index means one missing ``tenant_id`` term is a cross-tenant data breach.
* The same clauses must go into BOTH the BM25 ``bool.filter`` AND the ``knn.filter``. Putting
  them only in the outer ``bool`` *post-filters* the kNN results, which silently destroys recall
  for narrow-ACL users (their legitimate hits are pruned after selection) and leaks the existence
  of documents they may not see through ``total`` and aggregations.
* Group membership is resolved by a **terms lookup** against the tiny ``user-acl`` index rather
  than an inline list. A user in 300 directory groups would otherwise produce a ~12 KB query and
  trip ``index.max_terms_count``.

``tests/security/test_dsl_isolation.py`` asserts, over every leg x profile x filter permutation,
that the serialized DSL contains exactly one tenant term and that it sits in a filter context.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Final

from app.retrieval.types import MetadataFilter, TenantScope

#: Documents with no restriction carry this sentinel so that "unrestricted" is a positive match
#: rather than the absence of a field -- ``must_not exists`` is far slower and harder to cache.
PUBLIC_GROUP: Final[str] = "*"

#: One tiny document per user: {tenant_id, user_id, groups: [...]}, written at login and on SCIM
#: sync. OpenSearch resolves the lookup server-side and caches it.
USER_ACL_INDEX: Final[str] = "erp-user-acl"


def user_acl_doc_id(scope: TenantScope) -> str:
    return f"{scope.tenant_id}:{scope.user_id}"


def _group_clause(scope: TenantScope) -> dict[str, Any]:
    """Match documents that are public, or share a group with the principal, or name them.

    Uses a terms lookup when the principal is a real user; falls back to an inline list for
    machine principals (API keys carry a small, fixed group set) and for tests.
    """
    should: list[dict[str, Any]] = [{"term": {"access_groups": PUBLIC_GROUP}}]
    if scope.user_id is not None:
        should.append(
            {
                "terms": {
                    "access_groups": {
                        "index": USER_ACL_INDEX,
                        "id": user_acl_doc_id(scope),
                        "path": "groups",
                    }
                }
            }
        )
        should.append({"term": {"allowed_user_ids": str(scope.user_id)}})
    elif scope.access_groups:
        should.append({"terms": {"access_groups": list(scope.access_groups)}})
    return {"bool": {"should": should, "minimum_should_match": 1}}


def _deny_clause(scope: TenantScope) -> dict[str, Any] | None:
    """Deny wins, mirroring SharePoint and Drive semantics."""
    must_not: list[dict[str, Any]] = []
    if scope.user_id is not None:
        must_not.append(
            {
                "terms": {
                    "denied_groups": {
                        "index": USER_ACL_INDEX,
                        "id": user_acl_doc_id(scope),
                        "path": "groups",
                    }
                }
            }
        )
        must_not.append({"term": {"denied_user_ids": str(scope.user_id)}})
    elif scope.access_groups:
        must_not.append({"terms": {"denied_groups": list(scope.access_groups)}})
    if not must_not:
        return None
    return {"bool": {"must_not": must_not}}


def acl_filter(scope: TenantScope) -> list[dict[str, Any]]:
    """The non-negotiable clauses. Always index 0 == the tenant term.

    Order matters only for the isolation test's readability, but keeping the tenant term first
    makes a malformed query obvious at a glance in the retrieval debugger.
    """
    clauses: list[dict[str, Any]] = [
        {"term": {"tenant_id": str(scope.tenant_id)}},
        # Wrong-generation documents are invisible rather than blended into results. A stalled
        # backfill then surfaces as missing results in a shadow eval, not as quietly degraded
        # relevance in production.
        {"term": {"generation_fingerprint": scope.generation_fingerprint}},
        {"term": {"is_active": True}},
        {"range": {"visibility_rank": {"lte": scope.visibility_rank}}},
        _group_clause(scope),
    ]
    deny = _deny_clause(scope)
    if deny is not None:
        clauses.append(deny)

    as_of = (scope.as_of or date.today()).isoformat()
    if not scope.include_superseded:
        clauses.append({"term": {"is_superseded": False}})
        # Effective-date windows are open-ended at both ends, so a missing bound must match.
        clauses.append(
            {
                "bool": {
                    "should": [
                        {"bool": {"must_not": {"exists": {"field": "effective_from"}}}},
                        {"range": {"effective_from": {"lte": as_of}}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        )
        clauses.append(
            {
                "bool": {
                    "should": [
                        {"bool": {"must_not": {"exists": {"field": "effective_to"}}}},
                        {"range": {"effective_to": {"gte": as_of}}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        )
    return clauses


def metadata_filter(filters: MetadataFilter) -> list[dict[str, Any]]:
    """User-supplied narrowing. Never widens what ``acl_filter`` permits."""
    clauses: list[dict[str, Any]] = []
    for field_name, values in (
        ("doc_type", filters.doc_types),
        ("source_system", filters.source_systems),
        ("author", filters.authors),
        ("doc_id", filters.doc_ids),
        ("collection_id", filters.collection_ids),
        ("language", filters.languages),
    ):
        if values:
            clauses.append({"terms": {field_name: list(values)}})

    published: dict[str, str] = {}
    if filters.published_after:
        published["gte"] = filters.published_after.isoformat()
    if filters.published_before:
        published["lte"] = filters.published_before.isoformat()
    if published:
        clauses.append({"range": {"publication_date": published}})
    return clauses


def build_filter(scope: TenantScope, filters: MetadataFilter | None = None) -> list[dict[str, Any]]:
    """The only public way to obtain filter clauses. Both halves of a hybrid query use this."""
    clauses = acl_filter(scope)
    if filters is not None:
        clauses.extend(metadata_filter(filters))
    return clauses


def count_tenant_terms(dsl: Any) -> int:
    """Walk a serialized query and count ``{"term": {"tenant_id": ...}}`` occurrences.

    Used by the isolation tests: every leg must carry exactly one, including the kNN half.
    """
    found = 0
    stack: list[Any] = [dsl]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            term = node.get("term")
            if isinstance(term, dict) and "tenant_id" in term:
                found += 1
            stack.extend(node.values())
        elif isinstance(node, list):
            stack.extend(node)
    return found
