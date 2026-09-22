"""Reciprocal Rank Fusion.

BM25 scores and cosine similarities live on incomparable scales, so we fuse by *rank*:

    score(d) = sum_i  w_i / (k + rank_i(d))

summed over the legs in which ``d`` appears, with 1-based ranks and ``k = 60``.

``k = 60`` (Cormack, Clarke & Buettcher, 2009) damps the influence of any single leg's top
ranks: a document ranked ~5th by two legs beats one ranked 1st by a single leg, which is the
behaviour we want when the legs disagree because the query is ambiguous. No score normalization
is needed, and no leg can dominate because its scores happen to be larger.

Fusion lives in application code rather than in OpenSearch's ``hybrid`` query because it must
fuse four legs (and, in decomposed mode, legs x sub-queries) across *two* indices, stay unit
testable with no cluster, and let per-tenant weights be a Postgres row rather than a mutation to
a cluster-level search pipeline. See ``docs/decisions.md``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from app.retrieval.types import Candidate, LegHit


def fuse_rrf(
    leg_results: Mapping[str, Sequence[Candidate]],
    *,
    k: int = 60,
    weights: Mapping[str, float] | None = None,
) -> list[Candidate]:
    """Fuse ranked candidate lists.

    Input lists are assumed to be in descending relevance order per leg. Candidates are merged by
    ``chunk_id``; the first occurrence wins for the payload fields, and every leg contributes its
    rank to ``Candidate.legs`` so the retrieval debugger and the ablation table can attribute a
    result to the leg that found it.

    The ordering is fully deterministic -- ties break on best single-leg rank, then ``chunk_id`` --
    because the CI ablation gate compares runs byte for byte.
    """
    if k <= 0:
        raise ValueError("RRF k must be positive")

    merged: dict[str, Candidate] = {}
    scores: dict[str, float] = {}

    for leg_name, candidates in leg_results.items():
        weight = 1.0 if weights is None else weights.get(leg_name, 1.0)
        if weight == 0.0:
            continue

        # What THIS fusion has already counted for this leg. Deliberately not read from
        # ``candidate.legs``: a candidate may arrive with that entry already populated -- the
        # parent leg sets it while projecting sections down to children -- and treating a
        # pre-existing entry as "already counted" would skip the contribution entirely and leave
        # the chunk fused at zero. The guard is about double-counting within this pass, nothing else.
        counted: dict[str, int] = {}

        for zero_based, candidate in enumerate(candidates):
            rank = zero_based + 1
            incoming = candidate.legs.get(leg_name)
            leg_score = incoming.score if incoming is not None else 0.0

            existing = merged.get(candidate.chunk_id)
            if existing is None:
                merged[candidate.chunk_id] = candidate
                existing = candidate
                scores[candidate.chunk_id] = 0.0

            # The same chunk listed twice by one leg keeps its better (lower) rank.
            prior_rank = counted.get(candidate.chunk_id)
            if prior_rank is not None and prior_rank <= rank:
                continue
            if prior_rank is not None:
                scores[candidate.chunk_id] -= weight / (k + prior_rank)

            counted[candidate.chunk_id] = rank
            existing.legs[leg_name] = LegHit(rank=rank, score=leg_score)
            scores[candidate.chunk_id] += weight / (k + rank)

    for chunk_id, candidate in merged.items():
        candidate.fused_score = scores[chunk_id]

    def sort_key(candidate: Candidate) -> tuple[float, int, str]:
        best_rank = min((hit.rank for hit in candidate.legs.values()), default=10**9)
        return (-candidate.fused_score, best_rank, candidate.chunk_id)

    return sorted(merged.values(), key=sort_key)


def leg_contribution(leg_results: Mapping[str, Sequence[Candidate]]) -> dict[str, int]:
    """How many chunks each leg found that no other leg did.

    This is the number that justifies a leg's existence in the ablation table: a leg whose unique
    contribution is near zero is pure latency.
    """
    seen: dict[str, set[str]] = {}
    for leg_name, candidates in leg_results.items():
        seen[leg_name] = {candidate.chunk_id for candidate in candidates}

    unique: dict[str, int] = {}
    for leg_name, chunk_ids in seen.items():
        others: set[str] = set()
        for other_name, other_ids in seen.items():
            if other_name != leg_name:
                others |= other_ids
        unique[leg_name] = len(chunk_ids - others)
    return unique
