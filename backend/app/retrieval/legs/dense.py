"""The dense kNN leg.

The one place where the ACL filter placement is genuinely subtle: it goes inside
``knn.filter``, where the engine applies it *during* graph traversal, not in the surrounding
``bool``. Filtering afterwards selects the k nearest neighbours from the whole corpus and then
discards the ones the user may not see -- so a user with narrow access gets a handful of results
or none, from a query that was working correctly. The symptom looks like bad relevance rather
than like a filter bug, which is why it survives review.
"""

from __future__ import annotations

from typing import Any

from app.retrieval.legs.base import SOURCE_FIELDS, MsearchItem
from app.retrieval.types import RetrievalRequest
from app.search import dsl


class DenseKnnLeg:
    """Approximate nearest neighbours over the chunk embeddings."""

    name = "dense"

    def enabled(self, request: RetrievalRequest) -> bool:
        return request.embedding is not None

    def build(self, request: RetrievalRequest, *, index: str) -> MsearchItem:
        if request.embedding is None:  # pragma: no cover - guarded by enabled()
            raise ValueError("DenseKnnLeg requires an embedded query")

        size = request.profile.leg_sizes.get(self.name, 150)
        knn: dict[str, Any] = {
            "vector": list(request.embedding),
            "k": size,
            # Pre-filtering, applied during traversal. See the module docstring.
            "filter": {"bool": {"filter": dsl.build_filter(request.scope, request.filters)}},
            "method_parameters": {"ef_search": request.profile.ef_search},
        }
        if request.profile.oversample_factor > 1.0:
            # Only meaningful once the index is quantized: the engine retrieves
            # k * oversample from the compressed graph and rescores those against the
            # full-precision vectors held on disk.
            knn["rescore"] = {"oversample_factor": request.profile.oversample_factor}

        body: dict[str, Any] = {
            "size": size,
            "_source": list(SOURCE_FIELDS),
            "track_total_hits": False,
            "query": {"knn": {"embedding": knn}},
        }
        return {"index": index}, body
