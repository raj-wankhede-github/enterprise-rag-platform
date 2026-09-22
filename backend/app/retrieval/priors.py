"""Authority and recency priors, applied after fusion.

Deliberately not folded into BM25 as ``rank_feature`` boosts, even though the mapping supports
it. Two reasons:

* each leg stays independently measurable, so the ablation table attributes recall to retrieval
  rather than to a ranking prior smuggled inside it;
* the prior becomes its own ablation row, and a prior that does not earn its place gets removed
  instead of quietly inflating every leg.

The effect is bounded and multiplicative. An unbounded prior turns into a de facto sort by date,
which is how a system starts confidently answering from last week's draft instead of the policy
that is actually in force.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Final

from app.retrieval.types import Candidate

#: Half-life for the recency term, in days. Two years: enterprise policy is not news, and a
#: shorter half-life starts pushing superseded-but-recent drafts above stable current policy.
RECENCY_HALF_LIFE_DAYS: Final[float] = 730.0


def _parse_date(value: object) -> date | None:
    if isinstance(value, date):
        return value
    if isinstance(value, str) and value:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _recency(candidate: Candidate, *, today: date) -> float:
    """1.0 for something published today, decaying to 0.5 at one half-life."""
    published = _parse_date(candidate.meta.effective_from) or _parse_date(candidate.meta.publication_date)
    if published is None:
        # No date is not the same as old. Returning 0 would penalise every undated document,
        # which in most corpora is the majority of them.
        return 0.0
    age_days = float(max(0, (today - published).days))
    decay: float = 0.5 ** (age_days / RECENCY_HALF_LIFE_DAYS)
    return decay


def apply_priors(
    candidates: Sequence[Candidate],
    *,
    authority_weight: float = 0.15,
    recency_weight: float = 0.10,
    superseded_penalty: float = 0.25,
    today: date | None = None,
) -> list[Candidate]:
    """Score each candidate with a bounded multiplicative prior and re-sort.

    ``final = fused * (1 + a*authority + b*recency) * penalty``

    With the default weights the prior can lift a result by at most 25%, which reorders
    near-ties without letting an old-but-authoritative document outrank a direct answer.
    """
    if not candidates:
        return []
    for weight, name in ((authority_weight, "authority_weight"), (recency_weight, "recency_weight")):
        if not 0.0 <= weight <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")

    stamp = today or date.today()
    scored = list(candidates)
    for candidate in scored:
        authority = max(0.0, min(1.0, candidate.meta.authority_rank))
        multiplier = 1.0 + authority_weight * authority + recency_weight * _recency(candidate, today=stamp)
        # A superseded document is still retrievable -- "what was the rule in 2023" is a real
        # question -- but it must not outrank the policy that replaced it.
        if candidate.meta.is_superseded:
            multiplier *= superseded_penalty
        candidate.prior_score = candidate.fused_score * multiplier

    scored.sort(key=lambda item: (-(item.prior_score or 0.0), item.chunk_id))
    return scored
