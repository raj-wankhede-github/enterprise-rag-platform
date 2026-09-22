"""A deterministic reranker that needs no model.

It exists for two reasons, and neither is "it is good".

**It gives CI a non-trivial ``+reranker`` row.** With only ``IdentityReranker`` available
offline, the ablation table's reranking row would be a copy of the row above it, and the one
number the table exists to justify -- does reranking earn its latency -- would be unmeasurable
on every pull request. A weak but real reranker makes that row move.

**It is an honest floor.** It scores query-passage overlap with IDF-ish weighting and a few
structural signals. It has no idea what a sentence means, so whatever a cross-encoder achieves
above this line is the model's genuine contribution rather than the effect of reordering by
term overlap, which is most of what a naive "heuristic reranker" measures.

Deterministic, dependency-free, and about 0.2 ms for 24 passages.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from app.retrieval.rerank.base import apply_scores
from app.retrieval.types import Candidate
from app.utils.text import tokens

_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "may",
        "must",
        "will",
        "shall",
        "can",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "as",
        "if",
        "not",
        "no",
        "any",
        "all",
        "which",
        "what",
        "when",
        "where",
        "who",
        "how",
        "why",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
    }
)


class LexicalReranker:
    name = "lexical"
    max_pairs = 512

    def __init__(self, *, title_weight: float = 0.15, value_weight: float = 0.10) -> None:
        self.title_weight = title_weight
        self.value_weight = value_weight

    async def rerank(
        self, query: str, candidates: Sequence[Candidate], *, top_n: int, timeout_s: float = 0.4
    ) -> list[Candidate]:
        if not candidates:
            return []

        query_terms = [term for term in tokens(query) if term not in _STOPWORDS]
        if not query_terms:
            return list(candidates[:top_n])

        pool = list(candidates[: self.max_pairs])
        idf = _inverse_document_frequency(query_terms, pool)
        wants_value = any(character.isdigit() for character in query) or _asks_for_a_value(query_terms)

        scores = [self._score(candidate, query_terms, idf, wants_value) for candidate in pool]
        return apply_scores(pool, scores, top_n=top_n)

    def _score(
        self,
        candidate: Candidate,
        query_terms: Sequence[str],
        idf: dict[str, float],
        wants_value: bool,
    ) -> float:
        body = set(tokens(candidate.text))
        heading = set(tokens(f"{candidate.title} {candidate.heading_path or ''}"))

        total = sum(idf.values()) or 1.0
        matched = sum(idf[term] for term in query_terms if term in body)
        coverage = matched / total

        # A heading match is evidence about what the whole section is *about*, which a body
        # match is not -- the same reason the BM25 leg boosts heading fields.
        heading_hits = sum(idf[term] for term in query_terms if term in heading) / total

        # Questions asking for a figure should surface the passage that states one.
        value_bonus = self.value_weight if wants_value and _contains_digit(candidate.text) else 0.0

        # Keep a trace of the fused order so the reranker refines rather than replaces it; a
        # second stage that discards the first stage entirely is not reranking, it is a
        # different retriever with a worse recall budget.
        return coverage + self.title_weight * heading_hits + value_bonus + 0.05 * candidate.fused_score


def _inverse_document_frequency(query_terms: Sequence[str], candidates: Sequence[Candidate]) -> dict[str, float]:
    """Weight a term by how rare it is *within the candidate set*.

    Corpus statistics are not available here, and the candidate set is the right denominator
    anyway: a term present in every candidate cannot discriminate between them, whatever its
    frequency in the corpus.
    """
    total = len(candidates) or 1
    idf: dict[str, float] = {}
    for term in query_terms:
        occurrences = sum(1 for candidate in candidates if term in tokens(candidate.text))
        idf[term] = math.log(1 + total / (1 + occurrences))
    return idf


_VALUE_WORDS = frozenset({"how", "many", "much", "rate", "limit", "cost", "amount", "threshold", "period", "long"})


def _asks_for_a_value(query_terms: Sequence[str]) -> bool:
    return bool(_VALUE_WORDS.intersection(query_terms))


def _contains_digit(text: str) -> bool:
    return any(character.isdigit() for character in text)


class LexicalEntailmentScorer:
    """Entailment proxy for citation verification when no cross-encoder is deployed.

    Deliberately weak and deliberately lenient: it measures whether the claim's content words
    appear in the passage, which catches a citation pointing somewhere unrelated and nothing
    subtler. It is a placeholder that keeps the verification interface exercised offline, not a
    substitute for a model -- the failure it cannot see is a passage that shares every term with
    the claim and still does not support it.
    """

    name = "lexical-entailment"

    async def score(self, claim: str, passage: str) -> float:
        claim_terms = {term for term in tokens(claim) if term not in _STOPWORDS}
        if not claim_terms:
            return 1.0
        passage_terms = set(tokens(passage))
        return len(claim_terms & passage_terms) / len(claim_terms)
