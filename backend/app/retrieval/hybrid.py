"""Hybrid retrieval: four legs, one ``_msearch``, RRF in application code.

The whole argument for doing fusion here rather than in OpenSearch's ``hybrid`` query is that it
costs nothing extra. ``_msearch`` dispatches all four legs together and the cluster runs them in
parallel, so the round-trip count is the same; RRF over ~450 candidates is microseconds. What we
get in exchange is per-leg depth control (``pagination_depth`` caps the in-engine path and that
cap lands on recall@50, which no later stage can recover), fusion across two indices, per-tenant
weights as a database row, and a fusion function that is unit-testable with no cluster at all.

A leg that fails does not fail the request. Losing the dense leg costs recall; raising would cost
the answer. Failures are recorded in the diagnostics so they show up in the retrieval debugger
rather than as an unexplained drop in quality.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from app.embeddings.base import Embedder
from app.retrieval.fusion import fuse_rrf
from app.retrieval.legs.base import MsearchItem, RetrievalLeg, parse_hits, response_error
from app.retrieval.legs.dense import DenseKnnLeg
from app.retrieval.legs.lexical import Bm25Leg, ExactTokenLeg
from app.retrieval.legs.parent import ParentBm25Leg, project_to_children
from app.retrieval.priors import apply_priors
from app.retrieval.types import (
    Candidate,
    LegDiagnostics,
    RetrievalDiagnostics,
    RetrievalOutcome,
    RetrievalRequest,
)
from app.search.client import SearchClient


def default_legs() -> list[RetrievalLeg]:
    return [Bm25Leg(), ExactTokenLeg(), DenseKnnLeg(), ParentBm25Leg()]


class OpenSearchHybridRetriever:
    """The shipping retriever."""

    name = "hybrid_rrf"

    def __init__(
        self,
        client: SearchClient,
        *,
        chunk_index: str,
        parent_index: str,
        embedder: Embedder | None = None,
        legs: Sequence[RetrievalLeg] | None = None,
        apply_priors_after_fusion: bool = True,
        debug: bool = False,
    ) -> None:
        self.client = client
        self.chunk_index = chunk_index
        self.parent_index = parent_index
        self.embedder = embedder
        self.legs = list(legs) if legs is not None else default_legs()
        self.apply_priors_after_fusion = apply_priors_after_fusion
        self.debug = debug

    # -- public --------------------------------------------------------------------------

    async def retrieve(self, request: RetrievalRequest) -> RetrievalOutcome:
        started = time.perf_counter()
        request = await self._ensure_embedding(request)

        active = self._active_legs(request)
        if not active:
            return RetrievalOutcome(candidates=[], diagnostics=RetrievalDiagnostics(total_ms=0.0))

        items: list[MsearchItem] = []
        for leg in active:
            index = self.parent_index if leg.name == ParentBm25Leg.name else self.chunk_index
            items.append(leg.build(request, index=index))

        # Routing co-locates a tenant's chunks on one shard, so a tenant query touches one shard
        # rather than fanning out across the pool.
        routing = str(request.scope.tenant_id)
        responses = await self.client.msearch(items, routing=routing)

        diagnostics = RetrievalDiagnostics(
            indices=[self.chunk_index, self.parent_index],
            raw_dsl=[body for _, body in items] if self.debug else None,
        )
        leg_results: dict[str, list[Candidate]] = {}

        for leg, response in zip(active, responses, strict=False):
            error = response_error(response)
            candidates = parse_hits(response, leg=leg.name)
            diagnostics.legs.append(
                LegDiagnostics(
                    name=leg.name,
                    hits=len(candidates),
                    took_ms=float(response.get("took", 0.0)),
                    size=request.profile.leg_sizes.get(leg.name, 0),
                    error=error,
                )
            )
            leg_results[leg.name] = candidates

        if ParentBm25Leg.name in leg_results:
            leg_results[ParentBm25Leg.name] = await self._project_parents(
                request, leg_results[ParentBm25Leg.name], routing=routing
            )
            for entry in diagnostics.legs:
                if entry.name == ParentBm25Leg.name:
                    entry.hits = len(leg_results[ParentBm25Leg.name])

        fused = fuse_rrf(
            leg_results,
            k=request.profile.rrf_k,
            weights=request.profile.leg_weights,
        )
        if self.apply_priors_after_fusion:
            fused = apply_priors(fused)

        diagnostics.total_ms = (time.perf_counter() - started) * 1000.0
        return RetrievalOutcome(candidates=fused[: request.top_k], diagnostics=diagnostics)

    # -- internals -----------------------------------------------------------------------

    def _active_legs(self, request: RetrievalRequest) -> list[RetrievalLeg]:
        """Legs that are both enabled for this request and permitted by the profile.

        ``request.legs`` is how the ablation runner produces one row per configuration without a
        second code path -- ``bm25_only`` is this retriever with three legs switched off, not a
        different retriever that might diverge.
        """
        allowed = request.legs
        return [leg for leg in self.legs if leg.enabled(request) and (allowed is None or leg.name in allowed)]

    async def _ensure_embedding(self, request: RetrievalRequest) -> RetrievalRequest:
        """Embed the query unless the caller supplied a vector or disabled the dense leg."""
        if request.embedding is not None or self.embedder is None:
            return request
        if request.legs is not None and DenseKnnLeg.name not in request.legs:
            return request
        vector = await self.embedder.embed_query(request.query)
        return replace_embedding(request, tuple(vector))

    async def _project_parents(
        self, request: RetrievalRequest, parent_hits: list[Candidate], *, routing: str
    ) -> list[Candidate]:
        """Turn section hits into child candidates so everything fuses in one space."""
        parent_ids = [hit.parent_id for hit in parent_hits if hit.parent_id]
        if not parent_ids:
            return []
        leg = ParentBm25Leg()
        _, body = leg.projection_query(request, parent_ids=parent_ids, index=self.chunk_index)
        response = await self.client.search(index=self.chunk_index, body=body, routing=routing)
        children = parse_hits(response, leg=ParentBm25Leg.name)
        return project_to_children(parent_hits, children)


def replace_embedding(request: RetrievalRequest, embedding: tuple[float, ...]) -> RetrievalRequest:
    """``dataclasses.replace`` equivalent that keeps the frozen/slots contract explicit."""
    return RetrievalRequest(
        query=request.query,
        scope=request.scope,
        filters=request.filters,
        profile=request.profile,
        sub_queries=request.sub_queries,
        exact_tokens=request.exact_tokens,
        embedding=embedding,
        top_k=request.top_k,
        legs=request.legs,
    )


def merge_sub_query_results(
    per_sub_query: Sequence[dict[str, list[Candidate]]],
    *,
    k: int = 60,
    leg_weights: dict[str, float] | None = None,
) -> list[Candidate]:
    """Fuse across sub-queries *and* legs at once.

    A decomposed question contributes ``(sub_query, leg)`` pairs, each of which is just another
    ranked list as far as RRF is concerned. Fusing in one pass rather than per sub-query and then
    again across them avoids double-counting a chunk that several sub-queries agree on.
    """
    flattened: dict[str, list[Candidate]] = {}
    for index, legs in enumerate(per_sub_query):
        for leg_name, candidates in legs.items():
            flattened[f"{leg_name}#{index}"] = candidates

    weights: dict[str, float] | None = None
    if leg_weights:
        weights = {}
        for key in flattened:
            leg_name, _, _ = key.partition("#")
            weights[key] = leg_weights.get(leg_name, 1.0)
    return fuse_rrf(flattened, k=k, weights=weights)
