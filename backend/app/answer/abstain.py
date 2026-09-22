"""Saying "I don't know", professionally.

Rules these messages follow, each learned from a way that abstention messages go wrong:

* **State plainly that no answer was found.** No "I was unable to locate information that would
  allow me to provide a definitive response" -- hedged length reads as evasion, and a user
  skimming cannot tell whether they got an answer or not.
* **Never apologise more than once, and never grovel.** "I'm so sorry, unfortunately I'm afraid
  I cannot..." is three apologies for a correct outcome. Not finding something that is not in the
  corpus is the system working.
* **Say what was searched.** "across the documents you have access to" tells the user the search
  happened and hints that permissions or filters may be why.
* **Give a next step.** A dead end invites the user to rephrase at random; a concrete suggestion
  converts an abstention into a successful second query.
* **Never imply the answer exists.** "I couldn't find it, but generally speaking..." is exactly
  the failure abstention exists to prevent, wearing a disclaimer.
* **Never name a document the user cannot see.** Near misses are drawn from the same
  permission-filtered result set as everything else.

The messages are plain strings rather than an LLM call on purpose: a model asked to phrase an
abstention will sometimes answer the question instead, which is the one thing this path must not
do. It is also the only part of the pipeline that is guaranteed to run when the model is
unavailable.
"""

from __future__ import annotations

from collections.abc import Sequence

from app.answer.types import AbstentionReason, Answer, AnswerSentence, NearMiss

#: The core sentence per reason. Kept short; detail is added by the builder below.
_MESSAGES: dict[AbstentionReason, str] = {
    AbstentionReason.NO_RESULTS: (
        "I could not find anything about that in the documents you have access to, so I do not have an answer."
    ),
    AbstentionReason.WEAK_EVIDENCE: (
        "I found some related material, but nothing that actually answers your question, so I would rather not guess."
    ),
    AbstentionReason.INCOMPLETE_EVIDENCE: (
        "The documents I can see cover this topic but do not state the specific detail you asked "
        "for, so I do not have an answer."
    ),
    AbstentionReason.CONFLICTING_EVIDENCE: (
        "The sources I found disagree, and nothing in them settles which one currently applies. "
        "Rather than pick one, here is what each says."
    ),
    AbstentionReason.UNGROUNDED_DRAFT: (
        "I could not produce an answer that the retrieved documents actually support, so I am not going to offer one."
    ),
    AbstentionReason.ONLY_SUPERSEDED: (
        "Everything I found on this has been superseded, and the replacement is not in the "
        "documents I can see. I do not have the current position."
    ),
    AbstentionReason.NEEDS_CLARIFICATION: (
        "I can find material on this, but the question could be read more than one way and the "
        "readings have different answers."
    ),
}

#: Default next steps. Deliberately actionable and specific to the failure.
_SUGGESTIONS: dict[AbstentionReason, tuple[str, ...]] = {
    AbstentionReason.NO_RESULTS: (
        "Check whether the document has been uploaded to this workspace.",
        "Try naming the document, policy or reference number directly.",
        "If you expect access to it, your administrator can confirm your permissions.",
    ),
    AbstentionReason.WEAK_EVIDENCE: (
        "Try more specific wording, or the exact term used in the document.",
        "If you know the reference number or section, include it.",
    ),
    AbstentionReason.INCOMPLETE_EVIDENCE: (
        "The detail may live in an annexe or a linked document that is not indexed here.",
        "Ask about the surrounding section and I can tell you what it does cover.",
    ),
    AbstentionReason.CONFLICTING_EVIDENCE: (
        "Check the effective dates on the sources below.",
        "Your document owner can confirm which version is in force.",
    ),
    AbstentionReason.UNGROUNDED_DRAFT: (
        "Try asking about one thing at a time.",
        "Naming the document or section usually produces a grounded answer.",
    ),
    AbstentionReason.ONLY_SUPERSEDED: (
        "Ask for the historical position and I can quote the superseded version.",
        "The replacement document may need to be uploaded.",
    ),
    AbstentionReason.NEEDS_CLARIFICATION: (),
}


def _format_near_misses(near_misses: Sequence[NearMiss]) -> str:
    if not near_misses:
        return ""
    lines = ["", "The closest things I found were:"]
    for miss in near_misses:
        where = f" - {miss.heading_path}" if miss.heading_path else ""
        lines.append(f"  - {miss.title}{where}")
    return "\n".join(lines)


def _format_suggestions(suggestions: Sequence[str]) -> str:
    if not suggestions:
        return ""
    return "\n".join(["", "What might help:", *(f"  - {item}" for item in suggestions)])


def build_message(
    reason: AbstentionReason,
    *,
    near_misses: Sequence[NearMiss] = (),
    suggestions: Sequence[str] | None = None,
    clarifying_question: str | None = None,
) -> str:
    """Compose the user-facing text for an abstention.

    Sections are joined with a blank line between them and empty ones are dropped, so an
    abstention with no near misses does not open with a gap where a list would have been.
    """
    sections: list[str] = [_MESSAGES[reason]]

    if reason is AbstentionReason.NEEDS_CLARIFICATION and clarifying_question:
        sections.append(clarifying_question)

    near_miss_block = _format_near_misses(near_misses).strip("\n")
    if near_miss_block:
        sections.append(near_miss_block)

    chosen = _SUGGESTIONS[reason] if suggestions is None else tuple(suggestions)
    suggestion_block = _format_suggestions(chosen).strip("\n")
    if suggestion_block:
        sections.append(suggestion_block)

    return "\n\n".join(sections)


def abstain(
    reason: AbstentionReason,
    *,
    near_misses: Sequence[NearMiss] = (),
    suggestions: Sequence[str] | None = None,
    clarifying_question: str | None = None,
    partial: Sequence[AnswerSentence] = (),
    caveats: Sequence[str] = (),
) -> Answer:
    """Build a complete abstention.

    ``partial`` carries the sub-questions that *were* answerable. A multi-part question where
    three of four parts are covered should return those three plus an honest gap, not discard
    correct work because one part failed.
    """
    return Answer(
        answerable=False,
        reason=reason,
        message=build_message(
            reason,
            near_misses=near_misses,
            suggestions=suggestions,
            clarifying_question=clarifying_question,
        ),
        near_misses=tuple(near_misses),
        suggestions=tuple(_SUGGESTIONS[reason] if suggestions is None else suggestions),
        partial=tuple(partial),
        caveats=tuple(caveats),
    )
