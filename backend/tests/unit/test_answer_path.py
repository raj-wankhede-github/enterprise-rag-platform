"""Assembly, generation and citation verification.

The verification tests carry the most weight. They are the backstop for the failure the product
is sold on preventing, and they encode the specific prediction made when the evaluation harness
landed: that a value present in the answer but absent from its cited evidence is catchable
deterministically, with no model involved.
"""

from __future__ import annotations

import pytest

from app.answer.assemble import AssemblyConfig, EvidenceBlock, assemble
from app.answer.generator import ExtractiveAnswerGenerator, render
from app.answer.pipeline import AnswerPipeline
from app.answer.types import AbstentionReason, Answer, AnswerSentence
from app.answer.verify import (
    VerificationConfig,
    extract_values,
    strip_unsupported,
    surviving_blocks,
    verify,
)
from app.retrieval.types import Candidate, ChunkMeta

# --------------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------------


def candidate(
    chunk_id: str,
    text: str,
    *,
    score: float = 0.05,
    doc_id: str | None = None,
    parent_id: str | None = None,
    title: str = "Travel Policy",
    superseded: bool = False,
) -> Candidate:
    item = Candidate(
        chunk_id=chunk_id,
        parent_id=parent_id or f"p-{chunk_id}",
        doc_id=doc_id or f"d-{chunk_id}",
        doc_version_id="v1",
        text=text,
        title=title,
        heading_path="Travel Policy > Per Diem",
        meta=ChunkMeta(is_superseded=superseded),
    )
    item.fused_score = score
    return item


def block(block_id: str, text: str) -> EvidenceBlock:
    return EvidenceBlock(id=block_id, text=text, title="Doc", heading_path=None, doc_id="d", token_count=20, score=1.0)


def context_of(*blocks: EvidenceBlock):  # type: ignore[no-untyped-def]
    from app.answer.assemble import AssembledContext

    return AssembledContext(blocks=list(blocks), total_tokens=sum(b.token_count for b in blocks))


def claim(text: str, *ids: str) -> AnswerSentence:
    return AnswerSentence(text=text, evidence_ids=ids)


def answer_of(*sentences: AnswerSentence) -> Answer:
    return Answer(answerable=True, sentences=sentences)


# --------------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------------


def test_identical_text_from_several_documents_collapses() -> None:
    """A top-10 that is one paragraph twelve times looks corroborated and has one source."""
    boilerplate = "This document is confidential and may not be redistributed outside the company."
    result = assemble([candidate(f"c{i}", boilerplate, doc_id=f"d{i}") for i in range(5)])
    assert len(result.blocks) == 1
    assert result.dropped_duplicates == 4


def test_near_duplicates_differing_only_in_a_date_collapse() -> None:
    base = "Employees may claim a per diem allowance when travelling on company business in {}."
    result = assemble(
        [candidate("a", base.format("2024"), doc_id="d1"), candidate("b", base.format("2025"), doc_id="d2")]
    )
    assert len(result.blocks) == 1


def test_genuinely_different_text_is_kept() -> None:
    result = assemble(
        [
            candidate("a", "The per diem for grade A is 120 EUR per night.", doc_id="d1"),
            candidate("b", "Receipts are required for any item over 50 EUR.", doc_id="d2"),
        ]
    )
    assert len(result.blocks) == 2


def test_a_winning_child_expands_to_its_section() -> None:
    result = assemble(
        [candidate("a", "The rate is 120 EUR.", parent_id="p1")],
        parent_texts={"p1": "Per Diem\n\nThe rate is 120 EUR. Claims are filed within 30 days."},
    )
    assert "30 days" in result.blocks[0].text
    assert result.blocks[0].expanded


def test_a_section_expands_once_even_with_several_winning_children() -> None:
    result = assemble(
        [
            candidate("a", "first chunk about rates", parent_id="p1", doc_id="d1"),
            candidate("b", "second chunk about claims", parent_id="p1", doc_id="d1"),
        ],
        parent_texts={"p1": "first chunk about rates and second chunk about claims together"},
    )
    assert len(result.blocks) == 1


def test_a_missing_parent_degrades_to_the_child_rather_than_failing() -> None:
    result = assemble([candidate("a", "the child text", parent_id="p-missing")], parent_texts={})
    assert result.blocks[0].text == "the child text"
    assert not result.blocks[0].expanded


def test_one_document_cannot_fill_the_context() -> None:
    """A single verbose source must not become a consensus of one."""
    topics = [
        "Receipts are required for any single item over 50 EUR.",
        "Business class requires written approval from a department head.",
        "Alcohol is not reimbursable under any destination grade.",
        "Claims must be submitted within thirty days of return.",
        "Mileage in a private car is reimbursed per kilometre.",
        "The corporate travel desk books every flight and hotel.",
        "Escalations reach the head of finance within ten working days.",
        "Per diem allowances are paid without supporting documentation.",
    ]
    crowd = [candidate(f"c{i}", text, doc_id="same") for i, text in enumerate(topics)]
    result = assemble(crowd, config=AssemblyConfig(max_per_document=3))
    assert len(result.blocks) == 3
    assert result.dropped_for_diversity == 5


def test_budget_is_respected() -> None:
    long_text = " ".join(f"word{i}" for i in range(400))
    result = assemble(
        [candidate(f"c{i}", f"{long_text} unique{i}", doc_id=f"d{i}") for i in range(6)],
        config=AssemblyConfig(token_budget=600, max_per_document=10),
    )
    assert result.total_tokens <= 600


def test_a_block_that_does_not_fit_is_truncated_rather_than_dropped() -> None:
    """A truncated source still contributes a citable fact; a dropped one contributes nothing."""
    first = "First. " + " ".join(f"alpha{i}" for i in range(200))
    second = "Second. " + " ".join(f"beta{i}" for i in range(200))
    result = assemble(
        [candidate("a", first, doc_id="d1"), candidate("b", second, doc_id="d2")],
        config=AssemblyConfig(token_budget=260, max_per_document=10, min_block_tokens=20),
    )
    assert len(result.blocks) == 2
    assert result.blocks[1].truncated


def test_a_truncated_block_is_marked_so_the_model_knows() -> None:
    long_text = ". ".join(f"Sentence number {i} with content" for i in range(100))
    result = assemble([candidate("a", long_text)], config=AssemblyConfig(token_budget=100, min_block_tokens=20))
    assert result.blocks[0].truncated
    assert "[...]" in result.blocks[0].text
    assert 'truncated="true"' in result.blocks[0].render()


def test_max_blocks_caps_the_context() -> None:
    result = assemble(
        [candidate(f"c{i}", f"Distinct content {i} here.", doc_id=f"d{i}") for i in range(20)],
        config=AssemblyConfig(max_blocks=4),
    )
    assert len(result.blocks) == 4


def test_render_marks_evidence_as_data_and_indexes_the_ids() -> None:
    result = assemble(
        [candidate("abcdef12", "The rate is 120 EUR.", doc_id="d1"), candidate("bbcdef12", "Other fact.", doc_id="d2")]
    )
    rendered = result.render()
    assert "<evidence " in rendered and "</evidence>" in rendered
    assert "Evidence ids, strongest first:" in rendered
    for block_id in result.ids:
        assert block_id in rendered


def test_superseded_evidence_is_labelled_in_the_prompt() -> None:
    result = assemble([candidate("a", "The rate is 100 EUR.", superseded=True)])
    assert 'superseded="true"' in result.blocks[0].render()


def test_assembly_of_nothing_is_empty_not_an_error() -> None:
    result = assemble([])
    assert result.blocks == []
    assert result.render() == ""


# --------------------------------------------------------------------------------------------
# Extractive generation
# --------------------------------------------------------------------------------------------


async def test_extractive_generator_quotes_verbatim() -> None:
    """Its defining property: it cannot invent, so it is the control arm of the ablation."""
    ctx = context_of(block("b1", "The per diem for grade A is 120 EUR per night. Claims need receipts."))
    answer = await ExtractiveAnswerGenerator().generate("what is the per diem for grade A", ctx)
    assert answer.answerable
    assert all(sentence.text in ctx.blocks[0].text for sentence in answer.sentences)


async def test_extractive_generator_cites_every_claim() -> None:
    ctx = context_of(block("b1", "The rate is 120 EUR per night."))
    answer = await ExtractiveAnswerGenerator().generate("what is the rate", ctx)
    assert all(sentence.evidence_ids for sentence in answer.sentences if sentence.kind == "claim")


async def test_extractive_generator_prefers_sentences_carrying_a_value() -> None:
    ctx = context_of(
        block("b1", "Rates are reviewed annually by the finance team."),
        block("b2", "The per diem rate is 120 EUR per night."),
    )
    answer = await ExtractiveAnswerGenerator(max_sentences=1).generate("what is the per diem rate", ctx)
    assert "120" in answer.text


async def test_extractive_generator_warns_when_quoting_superseded_text() -> None:
    from app.answer.assemble import AssembledContext

    stale = EvidenceBlock(
        id="b1",
        text="The per diem rate is 100 EUR.",
        title="Old",
        heading_path=None,
        doc_id="d",
        token_count=10,
        score=1.0,
        is_superseded=True,
    )
    ctx = AssembledContext(blocks=[stale], total_tokens=10)
    answer = await ExtractiveAnswerGenerator().generate("what is the per diem rate", ctx)
    assert any("superseded" in caveat for caveat in answer.caveats)


async def test_extractive_generator_refuses_when_nothing_overlaps() -> None:
    ctx = context_of(block("b1", "Completely unrelated content about parking."))
    with pytest.raises(ValueError, match="overlapped"):
        await ExtractiveAnswerGenerator().generate("parental leave entitlement weeks", ctx)


def test_render_emits_citation_markers_and_never_a_remote_url() -> None:
    """Markdown images are the classic exfiltration channel for injected instructions."""
    rendered = render(answer_of(claim("The rate is 120 EUR.", "b1")))
    assert "[^b1]" in rendered
    assert "![" not in rendered
    assert "http" not in rendered


# --------------------------------------------------------------------------------------------
# Value extraction
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("The rate is 120 EUR.", "120 EUR"),
        ("Submit within 60 days.", "60 days"),
        ("Reference TE-2025-01 applies.", "TE-2025-01"),
        ("See section 4.2.1.", "4.2.1"),
        ("Core hours are 10:00 to 16:00.", "10:00"),
        ("Interest of 5% applies.", "5%"),
        ("Effective 1 January 2025.", "1 January 2025"),
    ],
)
def test_values_are_extracted(text: str, expected: str) -> None:
    assert expected in extract_values(text)


def test_a_long_date_is_not_also_split_into_its_parts() -> None:
    values = extract_values("Effective 1 January 2025.")
    assert "1 January 2025" in values
    assert "2025" not in values


def test_spelled_out_numbers_are_extracted_with_their_digits() -> None:
    values = extract_values("Probation lasts six months.")
    assert "six" in values and "6" in values


# --------------------------------------------------------------------------------------------
# Verification -- the backstop
# --------------------------------------------------------------------------------------------


async def test_a_grounded_answer_passes() -> None:
    ctx = context_of(block("b1", "The per diem for grade A is 120 EUR per night."))
    result = await verify(answer_of(claim("The per diem for grade A is 120 EUR.", "b1")), ctx)
    assert result.ok
    assert result.citation_support == 1.0


async def test_a_wrong_number_with_a_real_citation_is_caught() -> None:
    """The dominant real citation failure: a plausible source attached to a slightly wrong value.

    A reviewer skims the citation, sees a real document, and moves on. Exact matching does not.
    """
    ctx = context_of(block("b1", "The per diem for grade A is 120 EUR per night."))
    result = await verify(answer_of(claim("The per diem for grade A is 150 EUR.", "b1")), ctx)
    assert not result.ok
    assert "150 EUR" in result.unsupported[0].missing_values


async def test_the_case_the_evidence_gate_provably_cannot_catch() -> None:
    """A value absent from a corpus that discusses the topic at length.

    This is the specific prediction the evaluation harness recorded: coverage cannot separate
    "per diem for grade D" from an answerable question, because every topic term is present and
    only the asked-for value is missing. Verification catches it, with no model involved.
    """
    ctx = context_of(block("b1", "| Grade | Per diem |\n| --- | --- |\n| A | 120 |\n| B | 90 |\n| C | 65 |"))
    invented = answer_of(claim("The per diem for grade D is 45 EUR.", "b1"))
    result = await verify(invented, ctx)
    assert not result.ok
    assert "45 EUR" in result.unsupported[0].missing_values


async def test_a_citation_to_an_unknown_block_is_caught() -> None:
    ctx = context_of(block("b1", "The rate is 120 EUR."))
    result = await verify(answer_of(claim("The rate is 120 EUR.", "invented")), ctx)
    assert not result.ok
    assert "invented" in result.unknown_ids


async def test_a_claim_with_no_resolvable_citation_is_caught() -> None:
    ctx = context_of(block("b1", "The rate is 120 EUR."))
    result = await verify(answer_of(claim("The rate is 120 EUR.", "gone")), ctx)
    assert result.uncited_claims == 1


async def test_a_connective_sentence_needs_no_support() -> None:
    ctx = context_of(block("b1", "The rate is 120 EUR."))
    answer = Answer(
        answerable=True,
        sentences=(
            AnswerSentence(text="Additionally,", kind="connective"),
            claim("The rate is 120 EUR.", "b1"),
        ),
    )
    assert (await verify(answer, ctx)).ok


async def test_a_value_in_a_table_cell_supports_a_prose_claim() -> None:
    """Retrieval returns tables; answers are prose. The two must be allowed to agree."""
    ctx = context_of(block("b1", "| Grade | Per diem |\n| --- | --- |\n| A | 120 |"))
    result = await verify(answer_of(claim("Grade A is 120 EUR.", "b1")), ctx)
    assert result.ok


async def test_spelled_out_and_numeric_forms_agree() -> None:
    ctx = context_of(block("b1", "The standard probation period is six months."))
    assert (await verify(answer_of(claim("Probation is 6 months.", "b1")), ctx)).ok


async def test_thousands_separators_do_not_break_matching() -> None:
    ctx = context_of(block("b1", "Purchases over 25,000 EUR need CFO approval."))
    assert (await verify(answer_of(claim("Purchases over 25000 EUR need approval.", "b1")), ctx)).ok


async def test_a_claim_is_checked_against_its_own_citation_not_all_evidence() -> None:
    """Citing block A for a fact that only block B contains is still a citation failure."""
    ctx = context_of(
        block("b1", "Receipts are required over 50 EUR."),
        block("b2", "The per diem is 120 EUR."),
    )
    result = await verify(answer_of(claim("The per diem is 120 EUR.", "b1")), ctx)
    assert not result.ok


async def test_a_cross_encoder_can_reject_a_numerically_valid_claim() -> None:
    class Rejecting:
        async def score(self, claim_text: str, passage: str) -> float:
            return 0.01

    ctx = context_of(block("b1", "The per diem is 120 EUR."))
    result = await verify(answer_of(claim("The per diem is 120 EUR.", "b1")), ctx, scorer=Rejecting())
    assert not result.ok
    assert result.unsupported[0].entailment == 0.01


async def test_entailment_is_skipped_when_no_scorer_is_available() -> None:
    """A missing reranker degrades the check; it does not fail the answer."""
    ctx = context_of(block("b1", "The per diem is 120 EUR."))
    assert (await verify(answer_of(claim("The per diem is 120 EUR.", "b1")), ctx, scorer=None)).ok


async def test_value_checking_can_be_disabled_for_a_tenant() -> None:
    ctx = context_of(block("b1", "The rate is 120 EUR."))
    lenient = VerificationConfig(check_values=False)
    assert (await verify(answer_of(claim("The rate is 150 EUR.", "b1")), ctx, config=lenient)).ok


async def test_an_abstention_is_not_verified() -> None:
    from app.answer.abstain import abstain

    assert (await verify(abstain(AbstentionReason.NO_RESULTS), context_of())).ok


async def test_citation_support_is_a_fraction_not_a_boolean() -> None:
    ctx = context_of(block("b1", "The per diem is 120 EUR and receipts are needed over 50 EUR."))
    answer = answer_of(claim("The per diem is 120 EUR.", "b1"), claim("The cap is 999 EUR.", "b1"))
    result = await verify(answer, ctx)
    assert result.citation_support == 0.5


async def test_surviving_blocks_narrows_the_retry_context() -> None:
    ctx = context_of(block("b1", "The per diem is 120 EUR."), block("b2", "Unrelated content."))
    answer = answer_of(claim("The per diem is 120 EUR.", "b1"), claim("The cap is 999 EUR.", "b2"))
    result = await verify(answer, ctx)
    assert [b.id for b in surviving_blocks(result, ctx)] == ["b1"]


async def test_strip_unsupported_keeps_the_good_sentences() -> None:
    ctx = context_of(block("b1", "The per diem is 120 EUR."))
    answer = answer_of(claim("The per diem is 120 EUR.", "b1"), claim("The cap is 999 EUR.", "b1"))
    stripped = strip_unsupported(answer, await verify(answer, ctx))
    assert stripped.answerable
    assert len(stripped.sentences) == 1


async def test_strip_unsupported_abstains_when_nothing_survives() -> None:
    ctx = context_of(block("b1", "The per diem is 120 EUR."))
    answer = answer_of(claim("The cap is 999 EUR.", "b1"))
    stripped = strip_unsupported(answer, await verify(answer, ctx))
    assert not stripped.answerable


# --------------------------------------------------------------------------------------------
# The pipeline
# --------------------------------------------------------------------------------------------


class StubRetriever:
    def __init__(self, candidates: list[Candidate]) -> None:
        self.candidates = candidates

    async def retrieve(self, request: object):  # type: ignore[no-untyped-def]
        from app.retrieval.types import RetrievalOutcome

        return RetrievalOutcome(candidates=self.candidates)


class FabricatingGenerator:
    """A generator that cites a real block and states a number that is not in it."""

    name = "fabricating"

    async def generate(self, question: str, context, *, strict: bool = False) -> Answer:  # type: ignore[no-untyped-def]
        return answer_of(claim("The per diem for grade D is 45 EUR.", context.blocks[0].id))


def request_for(query: str):  # type: ignore[no-untyped-def]
    import uuid

    from app.retrieval.types import RetrievalRequest, TenantScope

    return RetrievalRequest(
        query=query,
        scope=TenantScope(
            tenant_id=uuid.uuid4(),
            visibility_rank=30,
            access_groups=(),
            generation_fingerprint="f" * 16,
        ),
    )


async def test_pipeline_answers_when_evidence_supports_it() -> None:
    pipeline = AnswerPipeline(
        retriever=StubRetriever([candidate("a", "The per diem for grade A is 120 EUR per night.")]),
        generator=ExtractiveAnswerGenerator(),
    )
    outcome = await pipeline.answer(request_for("what is the per diem for grade A"))
    assert outcome.answer.answerable
    assert "120" in outcome.answer.text
    assert outcome.trace.citation_support == 1.0


async def test_pipeline_abstains_when_retrieval_finds_nothing() -> None:
    pipeline = AnswerPipeline(retriever=StubRetriever([]), generator=ExtractiveAnswerGenerator())
    outcome = await pipeline.answer(request_for("parental leave"))
    assert not outcome.answer.answerable
    assert outcome.answer.reason is AbstentionReason.NO_RESULTS


async def test_pipeline_rejects_a_fabricated_value_and_abstains() -> None:
    """End to end: the generator invents, verification catches it, the user gets no answer."""
    table = candidate("a", "| Grade | Per diem |\n| --- | --- |\n| A | 120 |\n| B | 90 |")
    pipeline = AnswerPipeline(retriever=StubRetriever([table]), generator=FabricatingGenerator())
    outcome = await pipeline.answer(request_for("what is the per diem for grade D destination"))

    assert not outcome.answer.answerable
    assert outcome.answer.reason is AbstentionReason.UNGROUNDED_DRAFT
    assert "45" not in outcome.answer.text
    assert outcome.trace.verification_failure is not None


async def test_pipeline_can_return_a_partial_answer_when_a_tenant_opts_in() -> None:
    class Mixed:
        name = "mixed"

        async def generate(self, question: str, context, *, strict: bool = False) -> Answer:  # type: ignore[no-untyped-def]
            bid = context.blocks[0].id
            return answer_of(claim("The per diem is 120 EUR.", bid), claim("The cap is 999 EUR.", bid))

    pipeline = AnswerPipeline(
        retriever=StubRetriever([candidate("a", "The per diem for grade A is 120 EUR per night.")]),
        generator=Mixed(),
        allow_partial=True,
    )
    outcome = await pipeline.answer(request_for("what is the per diem"))
    assert outcome.answer.answerable
    assert len(outcome.answer.sentences) == 1
    assert "999" not in outcome.answer.text


async def test_a_generator_crash_abstains_rather_than_raising() -> None:
    class Broken:
        name = "broken"

        async def generate(self, question: str, context, *, strict: bool = False) -> Answer:  # type: ignore[no-untyped-def]
            raise RuntimeError("model unavailable")

    pipeline = AnswerPipeline(
        retriever=StubRetriever([candidate("a", "The per diem for grade A is 120 EUR.")]),
        generator=Broken(),
    )
    outcome = await pipeline.answer(request_for("what is the per diem for grade A"))
    assert not outcome.answer.answerable
    assert outcome.answer.reason is AbstentionReason.UNGROUNDED_DRAFT


async def test_the_trace_records_every_stage() -> None:
    pipeline = AnswerPipeline(
        retriever=StubRetriever([candidate("a", "The per diem for grade A is 120 EUR per night.")]),
        generator=ExtractiveAnswerGenerator(),
    )
    trace = (await pipeline.answer(request_for("what is the per diem for grade A"))).trace
    assert {"retrieve", "assemble", "generate", "verify"} <= set(trace.stage_ms)
    assert trace.blocks >= 1
    assert trace.total_ms >= 0
