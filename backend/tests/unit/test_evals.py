"""The evaluation harness's own tests.

A harness that gates the build has to be at least as trustworthy as the thing it gates. These
cover the arithmetic, the label resolution that has already silently failed once, and the gate
that decides whether a pull request passes.
"""

from __future__ import annotations

import pytest

from app.evals.dataset import (
    GoldenQuestion,
    IndexedChunk,
    RelevanceLabel,
    load_manifest,
    load_questions,
    resolve,
)
from app.evals.metrics import (
    leg_contribution,
    mean,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.evals.report import check, corpus_warning, load_thresholds, render_table, to_baseline
from app.evals.runner import DEFAULT_ABLATIONS, ConfigReport

# --------------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------------


def test_recall_counts_only_graded_chunks() -> None:
    assert recall_at_k(["a", "b"], {"a": 2, "c": 1}, 10) == 0.5
    assert recall_at_k(["a", "c"], {"a": 2, "c": 1}, 10) == 1.0


def test_recall_respects_the_cutoff() -> None:
    assert recall_at_k(["x", "y", "a"], {"a": 2}, 2) == 0.0
    assert recall_at_k(["x", "y", "a"], {"a": 2}, 3) == 1.0


def test_recall_of_an_unanswerable_question_is_one() -> None:
    """Nothing to recall. Scoring it zero would punish correct behaviour."""
    assert recall_at_k([], {}, 50) == 1.0


def test_ndcg_rewards_the_right_order() -> None:
    relevant = {"a": 2, "b": 1}
    assert ndcg_at_k(["a", "b"], relevant, 10) == 1.0
    assert ndcg_at_k(["b", "a"], relevant, 10) < 1.0


def test_ndcg_uses_exponential_gain_so_grade_2_clearly_beats_grade_1() -> None:
    """Grade 2 is worth 3 and grade 1 is worth 1, not 2 and 1."""
    top_grade_first = ndcg_at_k(["a", "b"], {"a": 2, "b": 1}, 10)
    top_grade_second = ndcg_at_k(["b", "a"], {"a": 2, "b": 1}, 10)
    assert top_grade_first - top_grade_second > 0.1


def test_ndcg_of_nothing_relevant_is_one() -> None:
    assert ndcg_at_k(["x"], {}, 10) == 1.0


def test_ndcg_ideal_is_capped_at_k() -> None:
    """A question with more relevant chunks than k must still be able to score 1.0."""
    relevant = {"a": 2, "b": 2, "c": 2}
    assert ndcg_at_k(["a", "b"], relevant, 2) == pytest.approx(1.0)


def test_reciprocal_rank_is_the_position_of_the_first_hit() -> None:
    assert reciprocal_rank(["x", "a"], {"a": 2}, 10) == 0.5
    assert reciprocal_rank(["a"], {"a": 2}, 10) == 1.0
    assert reciprocal_rank(["x", "y"], {"a": 2}, 10) == 0.0


def test_precision_measures_the_top_slice() -> None:
    assert precision_at_k(["a", "x"], {"a": 2}, 2) == 0.5
    assert precision_at_k([], {"a": 2}, 5) == 0.0


def test_leg_contribution_counts_only_uniquely_found_relevant_chunks() -> None:
    contribution = leg_contribution(
        {"bm25": ["a", "shared"], "dense": ["b", "shared"]},
        {"a": 2, "b": 2, "shared": 2},
    )
    assert contribution == {"bm25": 1, "dense": 1}


def test_leg_contribution_ignores_irrelevant_hits() -> None:
    assert leg_contribution({"bm25": ["noise"]}, {"a": 2}) == {"bm25": 0}


def test_percentile_is_nearest_rank() -> None:
    values = [10.0, 20.0, 30.0, 40.0]
    assert percentile(values, 0.5) == 20.0
    assert percentile(values, 1.0) == 40.0
    assert percentile([], 0.5) == 0.0


@pytest.mark.parametrize("bad", [0.0, -1.0, 1.5])
def test_percentile_rejects_an_impossible_fraction(bad: float) -> None:
    with pytest.raises(ValueError, match="fraction"):
        percentile([1.0], bad)


@pytest.mark.parametrize("metric", [recall_at_k, precision_at_k, ndcg_at_k])
def test_metrics_reject_a_non_positive_k(metric: object) -> None:
    if metric is recall_at_k:
        pytest.skip("recall tolerates any k because it slices a list")
    with pytest.raises(ValueError, match="k must be positive"):
        metric(["a"], {"a": 2}, 0)  # type: ignore[operator]


def test_mean_of_nothing_is_zero() -> None:
    assert mean([]) == 0.0


# --------------------------------------------------------------------------------------------
# The golden set itself
# --------------------------------------------------------------------------------------------


def test_the_corpus_manifest_matches_the_files_on_disk() -> None:
    documents = load_manifest()
    assert len(documents) >= 10
    assert all(document.text.strip() for document in documents)


def test_the_corpus_contains_what_the_strata_need() -> None:
    """A stratum with nothing in the corpus to test against measures nothing."""
    documents = {document.slug: document for document in load_manifest()}
    assert any(d.is_superseded for d in documents.values()), "need a superseded document"
    assert any(d.superseded_by for d in documents.values()), "need a supersession link"
    assert any(d.visibility_rank >= 40 for d in documents.values()), "need a restricted document"
    assert any("| ---" in d.text for d in documents.values()), "need a table"


def test_golden_questions_load_and_cover_every_stratum() -> None:
    questions = load_questions()
    assert len(questions) >= 50
    categories = {question.category for question in questions}
    for required in (
        "exact_lookup",
        "factual",
        "table",
        "procedural",
        "policy",
        "multi_hop",
        "temporal_supersession",
        "unanswerable",
        "permission_scoped",
    ):
        assert required in categories, f"stratum {required} has no questions"


def test_question_ids_are_unique() -> None:
    ids = [question.id for question in load_questions()]
    assert len(ids) == len(set(ids))


def test_answerable_questions_carry_labels_and_unanswerable_ones_do_not() -> None:
    for question in load_questions():
        if question.answerable:
            assert question.relevant, f"{question.id} is answerable but has no labels"
        else:
            assert not question.relevant, f"{question.id} is unanswerable but carries labels"


def test_every_label_points_at_a_document_in_the_manifest() -> None:
    slugs = {document.slug for document in load_manifest()}
    for question in load_questions():
        for label in question.relevant:
            assert label.doc in slugs, f"{question.id} references unknown document {label.doc}"


def test_every_label_marker_exists_in_its_source_document() -> None:
    """Catches a stale marker at author time rather than as a silently missing label."""
    documents = {d.slug: " ".join(d.text.casefold().split()) for d in load_manifest()}
    for question in load_questions():
        for label in question.relevant:
            needle = " ".join(label.contains.casefold().split())
            assert needle in documents[label.doc], f"{question.id}: {label.contains!r} is not in {label.doc}.md"


def test_a_grade_outside_the_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="grade must be 1 or 2"):
        RelevanceLabel(doc="x", contains="y", grade=3)


# --------------------------------------------------------------------------------------------
# Label resolution
# --------------------------------------------------------------------------------------------


def question(**kwargs: object) -> GoldenQuestion:
    base: dict[str, object] = {
        "id": "q",
        "category": "factual",
        "question": "how long",
        "answerable": True,
    }
    base.update(kwargs)
    return GoldenQuestion(**base)  # type: ignore[arg-type]


def test_resolution_finds_the_chunk_holding_the_marker() -> None:
    chunks = [
        IndexedChunk("c1", "doc", "irrelevant text"),
        IndexedChunk("c2", "doc", "the rate is 120 EUR per night"),
    ]
    resolved = resolve(question(relevant=(RelevanceLabel(doc="doc", contains="120 EUR", grade=2),)), chunks)
    assert resolved.relevant == {"c2": 2}
    assert not resolved.unresolved


def test_resolution_survives_a_line_break_in_the_source() -> None:
    """Regression: hard-wrapped markdown put a newline inside the marker and it matched nothing.

    The label silently resolved to zero chunks, so the question stopped contributing to recall
    while the harness still counted it -- an evaluation measuring less than it claimed to.
    """
    chunks = [IndexedChunk("c1", "doc", "entitled to 18 weeks of paid parental\nleave at full salary")]
    resolved = resolve(
        question(relevant=(RelevanceLabel(doc="doc", contains="18 weeks of paid parental leave", grade=2),)),
        chunks,
    )
    assert resolved.relevant == {"c1": 2}


def test_resolution_is_case_insensitive() -> None:
    chunks = [IndexedChunk("c1", "doc", "Payroll Records are kept")]
    resolved = resolve(question(relevant=(RelevanceLabel(doc="doc", contains="payroll records", grade=2),)), chunks)
    assert resolved.relevant == {"c1": 2}


def test_a_marker_spanning_two_chunks_marks_both() -> None:
    """A chunker change that splits a sentence should not invalidate the label."""
    chunks = [IndexedChunk("c1", "doc", "the limit is 50"), IndexedChunk("c2", "doc", "the limit is 50")]
    resolved = resolve(question(relevant=(RelevanceLabel(doc="doc", contains="limit is 50", grade=2),)), chunks)
    assert set(resolved.relevant) == {"c1", "c2"}


def test_a_chunk_labelled_twice_keeps_the_higher_grade() -> None:
    chunks = [IndexedChunk("c1", "doc", "alpha and beta")]
    resolved = resolve(
        question(
            relevant=(
                RelevanceLabel(doc="doc", contains="alpha", grade=1),
                RelevanceLabel(doc="doc", contains="beta", grade=2),
            )
        ),
        chunks,
    )
    assert resolved.relevant == {"c1": 2}


def test_an_unmatched_marker_is_reported_rather_than_ignored() -> None:
    resolved = resolve(
        question(relevant=(RelevanceLabel(doc="doc", contains="nowhere", grade=2),)),
        [IndexedChunk("c1", "doc", "something else")],
    )
    assert resolved.relevant == {}
    assert len(resolved.unresolved) == 1


def test_a_marker_only_matches_within_its_own_document() -> None:
    chunks = [IndexedChunk("c1", "other-doc", "the rate is 120 EUR")]
    resolved = resolve(question(relevant=(RelevanceLabel(doc="doc", contains="120 EUR", grade=2),)), chunks)
    assert resolved.unresolved


def test_question_rank_defaults_and_overrides() -> None:
    assert question().rank == 30
    assert question(max_rank=10).rank == 10
    assert question(min_rank=40).rank == 40


# --------------------------------------------------------------------------------------------
# The gate
# --------------------------------------------------------------------------------------------


def report(name: str = "hybrid_rrf + contextual", **kwargs: float) -> ConfigReport:
    defaults: dict[str, float] = {
        "recall_at_10": 0.95,
        "recall_at_50": 0.99,
        "recall_at_100": 0.99,
        "ndcg_at_10": 0.85,
        "mrr_at_10": 0.85,
        "precision_at_5": 0.5,
        "abstention_precision": 1.0,
        "abstention_recall": 0.8,
        "unsupported_answer_rate": 0.2,
        "permission_leak_rate": 0.0,
        "citation_support": 1.0,
        "p50_ms": 20.0,
        "p95_ms": 40.0,
    }
    defaults.update(kwargs)
    return ConfigReport(name=name, questions=61, corpus_chunks=71, **defaults)  # type: ignore[arg-type]


def test_a_healthy_run_passes_the_shipped_thresholds() -> None:
    assert check([report()], load_thresholds()) == []


def test_a_recall_collapse_fails_absolutely() -> None:
    failures = check([report(recall_at_10=0.1)], load_thresholds())
    assert any(failure.metric == "recall@10" and failure.kind == "absolute" for failure in failures)


def test_any_permission_leak_fails() -> None:
    """Zero tolerance: a chunk above the asker's rank in a candidate list is a breach."""
    failures = check([report(permission_leak_rate=0.01)], load_thresholds())
    assert any(failure.metric == "permission_leak_rate" for failure in failures)


def test_a_slow_bleed_fails_the_regression_gate() -> None:
    """The failure that actually happens: a dozen changes each costing half a point."""
    thresholds = {"max_regression": 0.02, "absolute": {}}
    baseline = to_baseline([report(ndcg_at_10=0.85)])
    failures = check([report(ndcg_at_10=0.80)], thresholds, baseline=baseline)
    assert any(failure.metric == "ndcg@10" and failure.kind == "regression" for failure in failures)


def test_a_small_wobble_inside_tolerance_passes() -> None:
    thresholds = {"max_regression": 0.02, "absolute": {}}
    baseline = to_baseline([report(ndcg_at_10=0.85)])
    assert check([report(ndcg_at_10=0.84)], thresholds, baseline=baseline) == []


def test_an_improvement_never_fails_the_regression_gate() -> None:
    thresholds = {"max_regression": 0.02, "absolute": {}}
    baseline = to_baseline([report(ndcg_at_10=0.70)])
    assert check([report(ndcg_at_10=0.95)], thresholds, baseline=baseline) == []


def test_lower_is_better_metrics_regress_in_the_other_direction() -> None:
    thresholds = {"max_regression": 0.02, "absolute": {}}
    baseline = to_baseline([report(unsupported_answer_rate=0.10)])
    failures = check([report(unsupported_answer_rate=0.30)], thresholds, baseline=baseline)
    assert any(failure.metric == "unsupported_answer_rate" for failure in failures)


def test_latency_is_exempt_from_the_regression_gate() -> None:
    """Too machine-dependent to gate on; it has a dedicated benchmark instead."""
    thresholds = {"max_regression": 0.02, "absolute": {}}
    baseline = to_baseline([report(p95_ms=10.0)])
    assert check([report(p95_ms=900.0)], thresholds, baseline=baseline) == []


def test_every_ablation_row_has_thresholds() -> None:
    absolute = load_thresholds()["absolute"]
    for config in DEFAULT_ABLATIONS:
        assert config.name in absolute, f"{config.name} has no floor and would never fail"


def test_a_failure_message_names_the_config_metric_and_limit() -> None:
    failure = check([report(recall_at_10=0.1)], load_thresholds())[0]
    rendered = str(failure)
    assert "hybrid_rrf + contextual" in rendered
    assert "recall@10" in rendered
    assert "below" in rendered


# --------------------------------------------------------------------------------------------
# Reporting honesty
# --------------------------------------------------------------------------------------------


def test_a_small_corpus_warns_that_recall_at_50_cannot_discriminate() -> None:
    """The most common way a retrieval evaluation flatters itself."""
    warning = corpus_warning(report(), cutoff=50)
    assert warning is not None
    assert "does not discriminate" in warning
    assert "recall@10" in warning


def test_a_large_corpus_produces_no_warning() -> None:
    big = report()
    big.corpus_chunks = 100_000
    assert corpus_warning(big, cutoff=50) is None


def test_the_table_renders_every_config_and_the_headline_metrics() -> None:
    rendered = render_table([report("bm25_only"), report()])
    assert "bm25_only" in rendered
    assert "recall@10" in rendered
    assert "made_up" in rendered
    assert "cite_ok" in rendered
    assert "leak" in rendered


def test_ablation_configs_differ_in_exactly_one_dimension_where_they_claim_to() -> None:
    """A row that changes two things at once measures neither."""
    by_name = {config.name: config for config in DEFAULT_ABLATIONS}
    plain, contextual = by_name["hybrid_rrf"], by_name["hybrid_rrf + contextual"]
    assert plain.legs == contextual.legs
    assert plain.contextual is not contextual.contextual

    # The verified arm differs from its neighbour in exactly one way: it runs the answer path.
    verified = by_name["hybrid + contextual + verified"]
    assert verified.legs == contextual.legs
    assert verified.contextual is contextual.contextual
    assert verified.answer is not contextual.answer
