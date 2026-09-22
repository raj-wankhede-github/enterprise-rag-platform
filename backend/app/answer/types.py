"""The answer contract.

The product's core invariant: **an answer is returned only if retrieved evidence supports it.**
Everything else abstains. That is not a safety garnish -- it is the feature enterprise buyers are
actually paying for, because a system that invents a per-diem rate is worse than no system at
all, and the first time it does so in front of a compliance team the pilot is over.

So "I don't know" is a **first-class success**, not an error. It carries a reason code, whatever
the system *did* find, and a next step. A bare "I could not find that" is a dead end; a response
naming the closest documents and the filters in force is something a user can act on.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class AbstentionReason(StrEnum):
    """Why no answer was produced. Each maps to a distinct message and a distinct next step."""

    #: Retrieval returned nothing at all within the user's permissions.
    NO_RESULTS = "no_results"
    #: Results came back, but none scored well enough to be about the question.
    WEAK_EVIDENCE = "weak_evidence"
    #: Evidence exists but does not cover the specific thing asked (the entity, the date, the
    #: figure). This is the one that separates a real RAG system from a plausible one.
    INCOMPLETE_EVIDENCE = "incomplete_evidence"
    #: Sources disagree and nothing resolves the conflict, so asserting either would be a guess.
    CONFLICTING_EVIDENCE = "conflicting_evidence"
    #: The model produced text that the citation check could not tie to the evidence.
    UNGROUNDED_DRAFT = "ungrounded_draft"
    #: Everything relevant is superseded and the question was about current rules.
    ONLY_SUPERSEDED = "only_superseded"
    #: The question is ambiguous enough that answering would mean picking an interpretation.
    NEEDS_CLARIFICATION = "needs_clarification"


@dataclass(frozen=True, slots=True)
class NearMiss:
    """Something retrieval found that was not good enough to answer with.

    Returned with an abstention so the user can see the search worked and judge for themselves
    whether the right document is simply missing or merely not indexed yet.
    """

    doc_id: str
    title: str
    heading_path: str | None = None
    score: float = 0.0


@dataclass(frozen=True, slots=True)
class AnswerSentence:
    """One sentence of an answer.

    ``claim`` sentences carry at least one evidence id. ``connective`` sentences ("Additionally,"
    "However,") carry none, because requiring a citation on connective text pushes the model into
    attaching an arbitrary one, which corrupts the citation signal precisely where it is checked.
    """

    text: str
    evidence_ids: tuple[str, ...] = ()
    kind: Literal["claim", "connective"] = "claim"

    def __post_init__(self) -> None:
        if self.kind == "claim" and not self.evidence_ids:
            raise ValueError("a claim sentence must cite at least one chunk")


@dataclass(frozen=True, slots=True)
class Conflict:
    """Two sources that disagree, surfaced rather than silently resolved."""

    summary: str
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class Answer:
    answerable: bool
    sentences: tuple[AnswerSentence, ...] = ()
    caveats: tuple[str, ...] = ()
    conflicts: tuple[Conflict, ...] = ()

    # --- abstention fields, set only when answerable is False -------------------------
    reason: AbstentionReason | None = None
    message: str = ""
    near_misses: tuple[NearMiss, ...] = ()
    suggestions: tuple[str, ...] = ()
    #: Sub-questions that *were* answerable, when a multi-part question was only partly covered.
    partial: tuple[AnswerSentence, ...] = field(default=())

    def __post_init__(self) -> None:
        if self.answerable and not self.sentences:
            raise ValueError("an answerable result must contain at least one sentence")
        if not self.answerable and self.reason is None:
            raise ValueError("an abstention must carry a reason")

    @property
    def text(self) -> str:
        """The rendered answer, or the abstention message."""
        if not self.answerable:
            return self.message
        return " ".join(sentence.text for sentence in self.sentences)

    @property
    def cited_ids(self) -> tuple[str, ...]:
        seen: dict[str, None] = {}
        for sentence in self.sentences:
            for evidence_id in sentence.evidence_ids:
                seen.setdefault(evidence_id, None)
        return tuple(seen)
