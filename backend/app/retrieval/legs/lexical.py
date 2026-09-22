"""The two lexical legs: analyzed BM25 and exact-token matching.

They are separate legs, not one query with a field boost, so the ablation table can say what
identifier matching is worth on its own. In an enterprise corpus that number is large and very
uneven: it is near zero for "what is the travel policy" and decisive for "TKT-99812".
"""

from __future__ import annotations

from typing import Any

from app.retrieval.legs.base import SOURCE_FIELDS, MsearchItem
from app.retrieval.types import RetrievalRequest
from app.search import dsl

#: Field weights for the analyzed leg. The heading path and title are boosted because a match
#: there is evidence about what the whole section is *about*, which a body match is not.
BM25_FIELDS: tuple[str, ...] = (
    "content^1.0",
    "context_line^0.7",
    "heading_path^1.6",
    "title^2.0",
)

#: Fields consulted by the exact leg, all analyzed with ``text_exact``.
EXACT_FIELDS: tuple[str, ...] = ("content.exact", "title.exact", "context_line.exact")


class Bm25Leg:
    """Analyzed, stemmed matching over the whole chunk."""

    name = "bm25"

    def enabled(self, request: RetrievalRequest) -> bool:
        return bool(request.query.strip())

    def build(self, request: RetrievalRequest, *, index: str) -> MsearchItem:
        size = request.profile.leg_sizes.get(self.name, 200)
        body: dict[str, Any] = {
            "size": size,
            "_source": list(SOURCE_FIELDS),
            "track_total_hits": False,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": request.query,
                                "fields": list(BM25_FIELDS),
                                "type": "best_fields",
                                # "2<70%" -- both terms of a two-word query must match, but a
                                # long query only needs 70%. A long query with a strict
                                # requirement returns nothing, which reads as "no results"
                                # rather than as an over-tight threshold.
                                "minimum_should_match": "2<70%",
                            }
                        },
                        {
                            # Phrase proximity, boosted: "per diem allowance" appearing together
                            # is much stronger evidence than the three words scattered.
                            "multi_match": {
                                "query": request.query,
                                "fields": list(BM25_FIELDS),
                                "type": "phrase",
                                "slop": 2,
                                "boost": 2.0,
                            }
                        },
                    ],
                    "minimum_should_match": 1,
                    "filter": dsl.build_filter(request.scope, request.filters),
                }
            },
        }
        return {"index": index}, body


class ExactTokenLeg:
    """Identifier matching: invoice numbers, section ids, ticket keys, product codes.

    Only runs when query understanding actually extracted an identifier, so it costs nothing on
    natural-language questions. Each token is its own phrase clause rather than one combined
    query, because a document containing *any* of the identifiers is relevant -- requiring all of
    them would drop the common case of a question naming two references.
    """

    name = "exact"

    def enabled(self, request: RetrievalRequest) -> bool:
        return bool(request.exact_tokens)

    def build(self, request: RetrievalRequest, *, index: str) -> MsearchItem:
        size = request.profile.leg_sizes.get(self.name, 50)
        should: list[dict[str, Any]] = []
        for token in request.exact_tokens:
            for field in EXACT_FIELDS:
                should.append({"match_phrase": {field: {"query": token}}})
        body: dict[str, Any] = {
            "size": size,
            "_source": list(SOURCE_FIELDS),
            "track_total_hits": False,
            "query": {
                "bool": {
                    "should": should,
                    "minimum_should_match": 1,
                    "filter": dsl.build_filter(request.scope, request.filters),
                }
            },
        }
        return {"index": index}, body
