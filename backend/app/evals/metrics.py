"""Retrieval metrics.

Written rather than imported. ``recall@k``, ``nDCG@k`` and ``MRR`` are a page of well-understood
arithmetic, and importing a framework for them means inheriting its qrel format, its LLM client
and its release cadence in exchange for code we need to control anyway -- graded labels, logical
chunk resolution and per-leg attribution are all things the frameworks do differently from what
this harness needs.

The one place a framework genuinely helps is its LLM-judge prompt library, and that is exactly
the part that cannot run in CI. So Ragas appears in the nightly provider job as a *check on our
judge*, not as the source of the numbers that gate a pull request.

Grades are 0/1/2: irrelevant, related, directly answers. Binary labels would make nDCG a
rebranded recall, which is the most common way a retrieval evaluation ends up measuring nothing.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Fraction of relevant chunks appearing in the top k.

    **The primary gate after fusion.** A reranker reorders what it is given; it cannot conjure a
    chunk that retrieval never returned. If recall@50 is low, nothing downstream can fix it, and
    tuning the reranker is wasted effort.
    """
    wanted = {chunk_id for chunk_id, grade in relevant.items() if grade > 0}
    if not wanted:
        # An unanswerable question has nothing to recall. Returning 1.0 keeps it from dragging
        # the mean down for behaving correctly; abstention is scored separately.
        return 1.0
    found = wanted.intersection(retrieved[:k])
    return len(found) / len(wanted)


def precision_at_k(retrieved: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    top = retrieved[:k]
    if not top:
        return 0.0
    hits = sum(1 for chunk_id in top if relevant.get(chunk_id, 0) > 0)
    return hits / len(top)


def reciprocal_rank(retrieved: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """1/rank of the first relevant result, or 0.

    Answers what the user experiences: how far down the page they had to read.
    """
    for position, chunk_id in enumerate(retrieved[:k], start=1):
        if relevant.get(chunk_id, 0) > 0:
            return 1.0 / position
    return 0.0


def dcg(gains: Sequence[float]) -> float:
    return sum(gain / math.log2(position + 1) for position, gain in enumerate(gains, start=1))


def ndcg_at_k(retrieved: Sequence[str], relevant: Mapping[str, int], k: int) -> float:
    """Graded, position-weighted quality. **The primary gate after reranking.**

    Uses the exponential gain ``2^grade - 1``, so a grade-2 chunk is worth three times a grade-1
    rather than twice. That gap is deliberate: the difference between "directly answers" and
    "mentions the topic" is what reranking is being asked to learn, and a linear gain barely
    rewards getting it right.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not any(grade > 0 for grade in relevant.values()):
        return 1.0

    gains = [(2 ** relevant.get(chunk_id, 0)) - 1 for chunk_id in retrieved[:k]]
    ideal = sorted((2**grade) - 1 for grade in relevant.values() if grade > 0)[::-1][:k]

    ideal_dcg = dcg(ideal)
    return dcg(gains) / ideal_dcg if ideal_dcg else 0.0


def leg_contribution(per_leg: Mapping[str, Sequence[str]], relevant: Mapping[str, int]) -> dict[str, int]:
    """Relevant chunks each leg found that no other leg did.

    This is the number that justifies a leg's existence. A leg whose unique contribution is near
    zero across the golden set is pure latency, and the ablation table should say so plainly
    rather than leaving it in because removing things feels risky.
    """
    wanted = {chunk_id for chunk_id, grade in relevant.items() if grade > 0}
    found: dict[str, set[str]] = {leg: wanted.intersection(chunk_ids) for leg, chunk_ids in per_leg.items()}
    unique: dict[str, int] = {}
    for leg, hits in found.items():
        others: set[str] = set()
        for other_leg, other_hits in found.items():
            if other_leg != leg:
                others |= other_hits
        unique[leg] = len(hits - others)
    return unique


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile.

    Nearest-rank rather than interpolated so that a p95 over 20 questions is an actual observed
    latency rather than an average of two, which is what an operator expects when they compare it
    against a budget.
    """
    if not values:
        return 0.0
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    ordered = sorted(values)
    index = math.ceil(fraction * len(ordered)) - 1
    return ordered[max(0, min(index, len(ordered) - 1))]
