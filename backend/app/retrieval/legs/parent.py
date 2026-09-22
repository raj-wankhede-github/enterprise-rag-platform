"""The parent leg: BM25 over whole sections.

A child chunk is ~300 tokens, which is the right size for a vector and the wrong size for a
question like "summarise the escalation procedure". Such a question matches *a section* -- its
terms are spread across several children, so no single child scores well and the query fails on
all three of the other legs at once.

Running BM25 over the parent index fixes that, but its hits are sections while fusion happens in
child space. So parent hits are projected down to their children with a second query, batched
across every winning parent: one extra round trip for the request, not one per parent.

The projection keeps the first ``MAX_CHILDREN_PER_PARENT`` children of each section rather than
all of them. A long section would otherwise contribute twenty candidates and crowd the fused list
with one source, which is precisely what the diversity cap at assembly time then has to undo.
"""

from __future__ import annotations

from typing import Any

from app.retrieval.legs.base import SOURCE_FIELDS, MsearchItem
from app.retrieval.types import Candidate, LegHit, RetrievalRequest
from app.search import dsl

#: Parent hits are ranked sections; this many children carry each one into the fused list.
MAX_CHILDREN_PER_PARENT: int = 3

PARENT_FIELDS: tuple[str, ...] = ("content^1.0", "heading_path^1.6", "title^2.0")


class ParentBm25Leg:
    """Matches whole sections, then hands back their children."""

    name = "parent"

    def enabled(self, request: RetrievalRequest) -> bool:
        return bool(request.query.strip())

    def build(self, request: RetrievalRequest, *, index: str) -> MsearchItem:
        size = request.profile.leg_sizes.get(self.name, 50)
        body: dict[str, Any] = {
            "size": size,
            # Only the identity is needed: the text that reaches the model comes from the
            # children, or from parent expansion later in the pipeline.
            "_source": ["tenant_id", "doc_id", "doc_version_id", "parent_id", "title", "heading_path"],
            "track_total_hits": False,
            "query": {
                "bool": {
                    "must": [
                        {
                            "multi_match": {
                                "query": request.query,
                                "fields": list(PARENT_FIELDS),
                                "type": "best_fields",
                                "minimum_should_match": "2<70%",
                            }
                        }
                    ],
                    "filter": dsl.build_filter(request.scope, request.filters),
                }
            },
        }
        return {"index": index}, body

    def projection_query(
        self,
        request: RetrievalRequest,
        *,
        parent_ids: list[str],
        index: str,
        max_children: int = MAX_CHILDREN_PER_PARENT,
    ) -> MsearchItem:
        """Fetch the children of the winning sections, in one query for all of them.

        Ordered by ``ordinal`` so the children come back in reading order, which makes the
        per-parent truncation take the start of a section rather than an arbitrary slice.
        """
        body: dict[str, Any] = {
            "size": len(parent_ids) * max_children,
            "_source": list(SOURCE_FIELDS),
            "track_total_hits": False,
            "sort": [{"ordinal": "asc"}],
            "query": {
                "bool": {
                    "filter": [
                        *dsl.build_filter(request.scope, request.filters),
                        {"terms": {"parent_id": parent_ids}},
                    ]
                }
            },
        }
        return {"index": index}, body


def project_to_children(
    parent_hits: list[Candidate],
    children: list[Candidate],
    *,
    max_children: int = MAX_CHILDREN_PER_PARENT,
) -> list[Candidate]:
    """Rank children by the rank of the section they came from.

    A child inherits its parent's position, so a section ranked first contributes the candidates
    that fuse as though they had been found first. Within a parent, reading order decides -- there
    is no independent signal to prefer one child of a matched section over another.
    """
    order = {hit.parent_id: hit.rank_in("parent") or 10**9 for hit in parent_hits}
    grouped: dict[str, list[Candidate]] = {}
    for child in children:
        grouped.setdefault(child.parent_id, []).append(child)

    score_by_parent = {
        hit.parent_id: (hit.legs["parent"].score if "parent" in hit.legs else 0.0) for hit in parent_hits
    }

    ranked: list[Candidate] = []
    for parent_id in sorted(grouped, key=lambda pid: order.get(pid, 10**9)):
        ranked.extend(grouped[parent_id][:max_children])

    for position, child in enumerate(ranked, start=1):
        child.legs["parent"] = LegHit(rank=position, score=score_by_parent.get(child.parent_id, 0.0))
    return ranked
