"""Saying "I don't know" instead of making something up.

The product invariant these enforce: an answer is returned only when retrieved evidence supports
it. The tests are split between *when* to abstain (the evidence gate) and *how* the refusal
reads, because both fail in different ways -- a gate that is too lenient invents answers, and a
message that hedges leaves the user unsure whether they got one.
"""

from __future__ import annotations

import pytest

from app.answer.abstain import abstain, build_message
from app.answer.evidence import (
    EvidenceThresholds,
    assess,
    distinctive_terms,
)
from app.answer.types import (
    AbstentionReason,
    Answer,
    AnswerSentence,
    NearMiss,
)
from app.retrieval.types import Candidate, ChunkMeta


def candidate(
    chunk_id: str,
    text: str,
    *,
    score: float = 0.05,
    title: str = "Travel Policy",
    superseded: bool = False,
    heading: str | None = "Travel Policy > Per Diem",
) -> Candidate:
    item = Candidate(
        chunk_id=chunk_id,
        parent_id=f"p-{chunk_id}",
        doc_id=f"d-{chunk_id}",
        doc_version_id="v1",
        text=text,
        title=title,
        heading_path=heading,
        meta=ChunkMeta(is_superseded=superseded),
    )
    item.fused_score = score
    return item


# --------------------------------------------------------------------------------------------
# When to abstain
# --------------------------------------------------------------------------------------------


def test_no_results_at_all() -> None:
    result = assess("what is the per diem rate", [])
    assert not result.sufficient
    assert result.reason is AbstentionReason.NO_RESULTS


def test_weak_scores_abstain_rather_than_answering_from_noise() -> None:
    weak = [candidate("a", "unrelated text about parking", score=0.001)]
    result = assess("what is the per diem rate", weak)
    assert not result.sufficient
    assert result.reason is AbstentionReason.WEAK_EVIDENCE
    assert result.near_misses, "the user should still see what was found"


def test_strong_and_covering_evidence_is_sufficient() -> None:
    good = [candidate("a", "The per diem rate for grade A is 120 EUR per night.", score=0.05)]
    result = assess("what is the per diem rate", good)
    assert result.sufficient
    assert result.reason is None
    assert result.term_coverage == 1.0


def test_topic_present_but_the_asked_detail_missing() -> None:
    """The dangerous case: plenty of on-topic text, and the specific figure is simply absent.

    Retrieval looks healthy and scores look fine. Without a coverage check the model fills the
    gap, which is exactly the failure that ends a pilot.
    """
    on_topic = [
        candidate("a", "Employees may claim a per diem allowance when travelling.", score=0.05),
        candidate("b", "Per diem claims are submitted through the expense portal.", score=0.04),
    ]
    result = assess("what is the per diem rate for Reykjavik", on_topic)
    assert not result.sufficient
    assert result.reason is AbstentionReason.INCOMPLETE_EVIDENCE
    assert "reykjavik" in result.missing_terms


def test_only_superseded_evidence_abstains() -> None:
    """Answering from a withdrawn policy is the most damaging failure in this category."""
    old = [candidate("a", "The per diem rate is 100 EUR.", score=0.05, superseded=True)]
    result = assess("what is the per diem rate", old)
    assert not result.sufficient
    assert result.reason is AbstentionReason.ONLY_SUPERSEDED


def test_one_current_source_among_superseded_ones_is_enough() -> None:
    mixed = [
        candidate("old", "The per diem rate is 100 EUR.", score=0.05, superseded=True),
        candidate("new", "The per diem rate is 120 EUR.", score=0.04, superseded=False),
    ]
    assert assess("what is the per diem rate", mixed).sufficient


def test_thresholds_are_tunable_per_tenant() -> None:
    """A compliance team wants a system that abstains readily; a helpdesk wants one that tries."""
    marginal = [candidate("a", "per diem rate is 120", score=0.013)]
    assert assess("per diem rate", marginal).sufficient
    strict = EvidenceThresholds(min_top_score=0.02)
    assert not assess("per diem rate", marginal, thresholds=strict).sufficient


def test_requiring_more_supporting_chunks_abstains_on_a_single_hit() -> None:
    single = [candidate("a", "the per diem rate is 120 EUR", score=0.05)]
    strict = EvidenceThresholds(min_supporting=2)
    assert not assess("per diem rate", single, thresholds=strict).sufficient


# --------------------------------------------------------------------------------------------
# Distinctive terms
# --------------------------------------------------------------------------------------------


def test_question_words_are_not_treated_as_content() -> None:
    assert distinctive_terms("what is the per diem rate") == ("per", "diem", "rate")


def test_numbers_and_identifiers_survive_the_short_token_filter() -> None:
    """ "50" and "4.2" are frequently the entire question."""
    terms = distinctive_terms("does section 4.2 apply to claims over 50 EUR")
    assert "4" in terms or "42" in terms or "4.2" in " ".join(terms)
    assert "50" in terms


def test_duplicate_terms_are_collapsed() -> None:
    assert distinctive_terms("policy policy policy") == ("policy",)


def test_a_question_with_no_distinctive_terms_does_not_abstain_on_coverage() -> None:
    """ "What about it?" cannot be coverage-checked; the score thresholds decide instead."""
    good = [candidate("a", "Some substantive content here.", score=0.05)]
    assert assess("what about it", good).sufficient


# --------------------------------------------------------------------------------------------
# How the refusal reads
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("reason", list(AbstentionReason))
def test_every_reason_has_a_message(reason: AbstentionReason) -> None:
    message = build_message(reason)
    assert message
    assert message[0].isupper()


@pytest.mark.parametrize("reason", list(AbstentionReason))
def test_no_message_grovels(reason: AbstentionReason) -> None:
    """Not finding something absent from the corpus is the system working, not a failure."""
    lowered = build_message(reason).lower()
    for grovel in ("i'm sorry", "i am sorry", "apologi", "unfortunately", "i'm afraid", "regret"):
        assert grovel not in lowered, f"{reason} message grovels: {grovel!r}"


@pytest.mark.parametrize("reason", list(AbstentionReason))
def test_no_message_hedges_toward_answering_anyway(reason: AbstentionReason) -> None:
    """ "I couldn't find it, but generally speaking..." is the failure this path prevents."""
    lowered = build_message(reason).lower()
    for hedge in ("generally speaking", "typically", "usually it", "in most cases", "i believe"):
        assert hedge not in lowered, f"{reason} message hedges toward a guess: {hedge!r}"


def test_the_refusal_says_plainly_that_there_is_no_answer() -> None:
    message = build_message(AbstentionReason.NO_RESULTS).lower()
    assert "could not find" in message
    assert "do not have an answer" in message


def test_the_refusal_says_what_was_searched() -> None:
    """Hints that permissions or filters may be the reason, without naming a hidden document."""
    assert "documents you have access to" in build_message(AbstentionReason.NO_RESULTS)


def test_the_refusal_offers_a_next_step() -> None:
    message = build_message(AbstentionReason.NO_RESULTS)
    assert "What might help:" in message
    assert message.count("  - ") >= 2


def test_near_misses_are_listed_so_the_user_can_judge() -> None:
    message = build_message(
        AbstentionReason.WEAK_EVIDENCE,
        near_misses=[NearMiss(doc_id="1", title="Travel Policy", heading_path="Per Diem")],
    )
    assert "The closest things I found were:" in message
    assert "Travel Policy" in message
    assert "Per Diem" in message


def test_no_blank_gap_when_there_are_no_near_misses() -> None:
    assert "\n\n\n" not in build_message(AbstentionReason.NO_RESULTS)


def test_clarification_includes_the_question_to_answer() -> None:
    message = build_message(
        AbstentionReason.NEEDS_CLARIFICATION,
        clarifying_question="Do you mean the 2024 or the 2025 policy?",
    )
    assert "2024 or the 2025" in message


def test_custom_suggestions_replace_the_defaults() -> None:
    message = build_message(AbstentionReason.NO_RESULTS, suggestions=["Ask your line manager."])
    assert "Ask your line manager." in message
    assert "uploaded to this workspace" not in message


# --------------------------------------------------------------------------------------------
# The Answer contract
# --------------------------------------------------------------------------------------------


def test_abstention_is_a_first_class_result_not_an_error() -> None:
    answer = abstain(AbstentionReason.NO_RESULTS)
    assert isinstance(answer, Answer)
    assert answer.answerable is False
    assert answer.text == answer.message
    assert answer.cited_ids == ()


def test_an_abstention_must_carry_a_reason() -> None:
    with pytest.raises(ValueError, match="must carry a reason"):
        Answer(answerable=False)


def test_an_answerable_result_must_have_sentences() -> None:
    with pytest.raises(ValueError, match="at least one sentence"):
        Answer(answerable=True)


def test_a_claim_sentence_must_cite_something() -> None:
    """The invariant that makes citation accuracy measurable at all."""
    with pytest.raises(ValueError, match="must cite at least one chunk"):
        AnswerSentence(text="The rate is 120 EUR.", evidence_ids=())


def test_a_connective_sentence_needs_no_citation() -> None:
    """Forcing one pushes the model to attach an arbitrary id, corrupting the signal."""
    sentence = AnswerSentence(text="However,", kind="connective")
    assert sentence.evidence_ids == ()


def test_partial_answers_survive_an_abstention() -> None:
    """Three of four sub-questions answered should not be discarded because one failed."""
    answered = AnswerSentence(text="Claims are filed within 30 days.", evidence_ids=("c1",))
    answer = abstain(AbstentionReason.INCOMPLETE_EVIDENCE, partial=[answered])
    assert answer.partial == (answered,)
    assert answer.answerable is False


def test_cited_ids_are_deduplicated_in_order() -> None:
    answer = Answer(
        answerable=True,
        sentences=(
            AnswerSentence(text="A.", evidence_ids=("c1", "c2")),
            AnswerSentence(text="B.", evidence_ids=("c2", "c3")),
        ),
    )
    assert answer.cited_ids == ("c1", "c2", "c3")


def test_abstention_carries_its_near_misses_and_suggestions() -> None:
    answer = abstain(
        AbstentionReason.WEAK_EVIDENCE,
        near_misses=[NearMiss(doc_id="1", title="Handbook")],
    )
    assert answer.near_misses[0].title == "Handbook"
    assert answer.suggestions
