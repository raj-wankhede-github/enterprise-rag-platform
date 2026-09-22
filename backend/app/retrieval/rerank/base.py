"""Reranking: the second-stage scorer that reads query and passage together.

Fusion produces a candidate set; it does not produce an order anyone would want to read. A
cross-encoder does, because it attends over the query and the passage jointly rather than
comparing two vectors computed in ignorance of each other. That is also why it cannot run over
the corpus -- it scores pairs, so it only ever sees the fused top-N.

Every implementation is interchangeable behind this Protocol, and a failure is always a
degradation rather than an error: a reranker that times out returns fusion order, which costs
precision. Raising would cost the answer.

The same Protocol doubles as the entailment scorer for citation verification. A cross-encoder
asked "does this passage support this claim" is doing the task it was trained for, so the answer
path gets entailment checking for free once the model is deployed -- no second model, no second
container, no second thing to operate.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from app.retrieval.types import Candidate


@runtime_checkable
class Reranker(Protocol):
    name: str
    #: The most pairs this implementation will score in one call. Above it, the caller truncates
    #: rather than the implementation silently dropping the tail.
    max_pairs: int

    async def rerank(
        self, query: str, candidates: Sequence[Candidate], *, top_n: int, timeout_s: float = 0.4
    ) -> list[Candidate]: ...


class IdentityReranker:
    """Returns fusion order untouched.

    The CI default and the control arm: any nDCG movement between this and a real reranker is
    attributable to the reranker, because nothing else differs.
    """

    name = "identity"
    max_pairs = 10_000

    async def rerank(
        self, query: str, candidates: Sequence[Candidate], *, top_n: int, timeout_s: float = 0.4
    ) -> list[Candidate]:
        return list(candidates[:top_n])


def apply_scores(candidates: Sequence[Candidate], scores: Sequence[float], *, top_n: int) -> list[Candidate]:
    """Attach scores and re-sort.

    Ties break on ``chunk_id`` so the output is byte-identical across runs -- the CI ablation
    gate compares tables, and a reranker that shuffles equal scores would make every run a
    spurious regression.
    """
    for candidate, score in zip(candidates, scores, strict=True):
        candidate.rerank_score = score
    ordered = sorted(candidates, key=lambda item: (-(item.rerank_score or 0.0), item.chunk_id))
    return ordered[:top_n]
