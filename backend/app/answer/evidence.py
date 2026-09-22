"""Deciding whether the retrieved evidence can support an answer at all.

This runs **before** the generator, and it is the cheapest, most reliable part of the abstention
story: no model is involved, so it cannot be talked out of its verdict by a confident-sounding
draft. A model handed thin evidence will usually produce something fluent anyway -- the whole
point of gating here is that it never gets the chance.

Three signals, in increasing cost:

1. **Anything at all?** An empty result set after the ACL filter is a clean ``NO_RESULTS``.
2. **Is it about the question?** Relevance thresholds on the fused score.
3. **Does it cover the question?** Term coverage -- specifically, whether the distinctive terms
   of the question appear in the evidence. This is what catches the dangerous case: a corpus that
   discusses the topic at length and never states the figure asked for. Retrieval looks healthy,
   scores look fine, and the model fills the gap. Coverage is the check that notices.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

from app.answer.types import AbstentionReason, NearMiss
from app.retrieval.types import Candidate
from app.utils.text import tokens

#: Words that carry no discriminating power, so their absence from the evidence means nothing.
#: Kept separate from the analyzer's stop list: this one is about *question* words.
_QUESTION_WORDS: frozenset[str] = frozenset(
    {
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "when",
        "where",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "am",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
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
        "about",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "that",
        "this",
        "these",
        "those",
        "i",
        "we",
        "you",
        "they",
        "it",
        "my",
        "our",
        "your",
        "their",
        "its",
        "me",
        "us",
        "them",
        "there",
        "here",
        "any",
        "some",
        "all",
        "not",
        "no",
        "please",
        "tell",
        "show",
        "give",
        "given",
        "explain",
        "describe",
        "many",
        "much",
        "long",
        "have",
        "has",
        "had",
        "get",
        "got",
        "need",
        "want",
        "allowed",
        "entitled",
        "apply",
        "applies",
    }
)

#: An identifier in the question is a request about that specific thing. Matching the pattern
#: used by the retrieval planner so the two agree on what counts as one.
_IDENTIFIER_RE: Final[re.Pattern[str]] = re.compile(r"[A-Z]{2,5}[-_]\d{2,}(?:[-_]\d+)*|\d+\.\d+(?:\.\d+)+")


@dataclass(frozen=True, slots=True)
class EvidenceThresholds:
    """Tunables, all with a stated reason.

    These are the numbers a tenant most wants to move: a compliance team wants a system that
    abstains readily, a helpdesk wants one that tries. They live in the retrieval profile so the
    trade is a per-tenant setting rather than a redeploy.
    """

    #: A fused RRF score below this means no leg ranked the chunk anywhere near the top. With
    #: k=60 a single leg's first place scores 1/61 = 0.0164, so this is "roughly one leg, top 20".
    min_top_score: float = 0.012
    #: How many candidates must clear the bar. One lucky hit is not a corpus that knows the answer.
    min_supporting: int = 1
    #: Fraction of the question's distinctive terms that must appear somewhere in the evidence.
    #:
    #: Calibrated, not chosen. ``bench/abstention_calibration.py`` sweeps this against the golden
    #: set (51 answerable, 10 unanswerable) and the distributions overlap:
    #:
    #:     threshold  refused_ok  made_up  wrongly_refused
    #:          0.50           5        5                1
    #:          0.60           7        3                4     <- shipped
    #:          0.70           8        2               12
    #:          0.85          10        0               20
    #:
    #: 0.60 is the knee. Going to 0.70 buys one fewer invented answer at the cost of eight more
    #: refused real ones; reaching zero invented answers means refusing 39% of questions the
    #: corpus can actually answer, which is a system nobody would use.
    #:
    #: The honest conclusion is that **coverage alone cannot close this gap**. This gate is the
    #: cheap pre-filter, not the whole abstention story: the residue is questions whose topic
    #: terms are all present and only the asked-for *value* is absent ("the per diem for grade
    #: D", where the table lists A, B and C). No lexical measure can see that. Citation
    #: verification at build step 7 can, because a model required to cite a chunk stating grade D
    #: will not find one.
    min_term_coverage: float = 0.6
    #: How many near misses to show the user when abstaining.
    max_near_misses: int = 3


@dataclass(frozen=True, slots=True)
class EvidenceAssessment:
    sufficient: bool
    reason: AbstentionReason | None
    supporting: tuple[Candidate, ...]
    near_misses: tuple[NearMiss, ...]
    term_coverage: float
    missing_terms: tuple[str, ...]
    top_score: float

    @property
    def has_any_results(self) -> bool:
        return bool(self.supporting or self.near_misses)


def distinctive_terms(question: str) -> tuple[str, ...]:
    """The words in a question that a document would actually have to contain.

    Numbers and identifiers are kept whatever their length -- "50", "4.2" and "TKT-99812" are
    frequently the entire question, and dropping them as short tokens is how a coverage check
    concludes that a document about travel answers "what is the limit".
    """
    result: list[str] = [token for token in tokens(question) if token not in _QUESTION_WORDS]
    return tuple(dict.fromkeys(result))


def _coverage(terms: Sequence[str], candidates: Sequence[Candidate]) -> tuple[float, tuple[str, ...]]:
    if not terms:
        # A question with no distinctive terms ("what about it?") cannot be assessed this way.
        # Treating that as full coverage hands the decision to the score thresholds instead of
        # abstaining on a technicality.
        return 1.0, ()
    haystack: set[str] = set()
    for candidate in candidates:
        haystack.update(tokens(candidate.text))
        if candidate.context_line:
            haystack.update(tokens(candidate.context_line))
        haystack.update(tokens(candidate.title))
        if candidate.heading_path:
            haystack.update(tokens(candidate.heading_path))

    missing = tuple(term for term in terms if term not in haystack)
    return (len(terms) - len(missing)) / len(terms), missing


def _missing_identifiers(question: str, candidates: Sequence[Candidate]) -> tuple[str, ...]:
    """Identifiers named in the question that appear nowhere in the evidence."""
    named = _IDENTIFIER_RE.findall(question)
    if not named:
        return ()
    haystack = " ".join(
        [c.text for c in candidates] + [c.title for c in candidates] + [c.context_line or "" for c in candidates]
    ).casefold()
    return tuple(item for item in dict.fromkeys(named) if item.casefold() not in haystack)


def _near_misses(candidates: Sequence[Candidate], limit: int) -> tuple[NearMiss, ...]:
    seen: dict[str, NearMiss] = {}
    for candidate in candidates:
        if candidate.doc_id in seen:
            continue
        seen[candidate.doc_id] = NearMiss(
            doc_id=candidate.doc_id,
            title=candidate.title or "Untitled document",
            heading_path=candidate.heading_path,
            score=candidate.final_score,
        )
        if len(seen) >= limit:
            break
    return tuple(seen.values())


def assess(
    question: str,
    candidates: Sequence[Candidate],
    *,
    thresholds: EvidenceThresholds | None = None,
) -> EvidenceAssessment:
    """Decide whether these candidates can support an answer."""
    limits = thresholds or EvidenceThresholds()

    if not candidates:
        return EvidenceAssessment(
            sufficient=False,
            reason=AbstentionReason.NO_RESULTS,
            supporting=(),
            near_misses=(),
            term_coverage=0.0,
            missing_terms=distinctive_terms(question),
            top_score=0.0,
        )

    ranked = sorted(candidates, key=lambda item: -item.final_score)
    top_score = ranked[0].final_score
    supporting = tuple(item for item in ranked if item.final_score >= limits.min_top_score)

    if len(supporting) < limits.min_supporting:
        return EvidenceAssessment(
            sufficient=False,
            reason=AbstentionReason.WEAK_EVIDENCE,
            supporting=(),
            near_misses=_near_misses(ranked, limits.max_near_misses),
            term_coverage=0.0,
            missing_terms=distinctive_terms(question),
            top_score=top_score,
        )

    # Everything relevant is superseded, and the question did not ask for history. Answering from
    # a withdrawn policy is the single most damaging failure in this product category.
    if all(item.meta.is_superseded for item in supporting):
        return EvidenceAssessment(
            sufficient=False,
            reason=AbstentionReason.ONLY_SUPERSEDED,
            supporting=(),
            near_misses=_near_misses(supporting, limits.max_near_misses),
            term_coverage=0.0,
            missing_terms=(),
            top_score=top_score,
        )

    # An identifier the question names and the evidence does not contain is not "partial
    # coverage" -- it is a miss. "How long is the contract with SUP-7777" scored 0.67 because
    # the tokenizer split the code and "sup" matched SUP-4471, which is a different supplier.
    missing_identifiers = _missing_identifiers(question, supporting)
    if missing_identifiers:
        return EvidenceAssessment(
            sufficient=False,
            reason=AbstentionReason.INCOMPLETE_EVIDENCE,
            supporting=(),
            near_misses=_near_misses(supporting, limits.max_near_misses),
            term_coverage=0.0,
            missing_terms=missing_identifiers,
            top_score=top_score,
        )

    terms = distinctive_terms(question)
    coverage, missing = _coverage(terms, supporting)
    if coverage < limits.min_term_coverage:
        return EvidenceAssessment(
            sufficient=False,
            reason=AbstentionReason.INCOMPLETE_EVIDENCE,
            supporting=(),
            near_misses=_near_misses(supporting, limits.max_near_misses),
            term_coverage=coverage,
            missing_terms=missing,
            top_score=top_score,
        )

    return EvidenceAssessment(
        sufficient=True,
        reason=None,
        supporting=supporting,
        near_misses=(),
        term_coverage=coverage,
        missing_terms=missing,
        top_score=top_score,
    )
