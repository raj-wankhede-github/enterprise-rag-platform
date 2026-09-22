from __future__ import annotations

import pytest

from app.retrieval.fusion import fuse_rrf, leg_contribution
from app.retrieval.types import Candidate, LegHit


def make(chunk_id: str) -> Candidate:
    return Candidate(
        chunk_id=chunk_id,
        parent_id=f"p-{chunk_id}",
        doc_id=f"d-{chunk_id}",
        doc_version_id=f"v-{chunk_id}",
        text=chunk_id,
    )


def test_rrf_formula_is_exact() -> None:
    fused = fuse_rrf({"bm25": [make("a"), make("b")]}, k=60)
    assert fused[0].chunk_id == "a"
    assert fused[0].fused_score == pytest.approx(1 / 61)
    assert fused[1].fused_score == pytest.approx(1 / 62)


def test_agreement_across_legs_beats_a_single_first_place() -> None:
    """The central property of RRF with k=60, and the reason we use rank fusion at all."""
    leg_a = [make("solo"), make("agreed"), make("x"), make("y"), make("z")]
    leg_b = [make("q"), make("agreed"), make("r"), make("s"), make("t")]
    fused = fuse_rrf({"bm25": leg_a, "dense": leg_b}, k=60)
    assert fused[0].chunk_id == "agreed"
    # agreed: 1/62 + 1/62 > solo: 1/61
    assert fused[0].fused_score == pytest.approx(2 / 62)


def test_weights_are_applied() -> None:
    fused = fuse_rrf(
        {"bm25": [make("a")], "exact": [make("b")]},
        k=60,
        weights={"bm25": 1.0, "exact": 2.0},
    )
    assert fused[0].chunk_id == "b"
    assert fused[0].fused_score == pytest.approx(2 / 61)


def test_zero_weight_drops_a_leg_entirely() -> None:
    fused = fuse_rrf({"bm25": [make("a")], "dense": [make("b")]}, weights={"dense": 0.0})
    assert [c.chunk_id for c in fused] == ["a"]


def test_per_leg_ranks_are_retained_for_the_debugger() -> None:
    fused = fuse_rrf({"bm25": [make("x"), make("y")], "dense": [make("y")]})
    by_id = {c.chunk_id: c for c in fused}
    assert by_id["y"].rank_in("bm25") == 2
    assert by_id["y"].rank_in("dense") == 1
    assert by_id["x"].rank_in("dense") is None


def test_duplicate_within_one_leg_keeps_the_better_rank() -> None:
    fused = fuse_rrf({"bm25": [make("a"), make("b"), make("a")]}, k=60)
    a = next(c for c in fused if c.chunk_id == "a")
    assert a.rank_in("bm25") == 1
    assert a.fused_score == pytest.approx(1 / 61)


def test_output_is_deterministic_regardless_of_leg_iteration_order() -> None:
    legs_1 = {"bm25": [make("a"), make("b")], "dense": [make("b"), make("a")]}
    legs_2 = {"dense": [make("b"), make("a")], "bm25": [make("a"), make("b")]}
    assert [c.chunk_id for c in fuse_rrf(legs_1)] == [c.chunk_id for c in fuse_rrf(legs_2)]


def test_empty_input_is_empty_output() -> None:
    assert fuse_rrf({}) == []
    assert fuse_rrf({"bm25": []}) == []


def test_invalid_k_is_rejected() -> None:
    with pytest.raises(ValueError, match="positive"):
        fuse_rrf({"bm25": [make("a")]}, k=0)


def test_leg_contribution_counts_unique_finds() -> None:
    contribution = leg_contribution(
        {
            "bm25": [make("shared"), make("only_bm25")],
            "dense": [make("shared"), make("only_dense")],
        }
    )
    assert contribution == {"bm25": 1, "dense": 1}


def test_a_candidate_arriving_with_its_leg_already_set_still_scores() -> None:
    """Regression: the parent leg pre-populates legs["parent"] while projecting sections.

    An earlier version treated any pre-existing entry as "already counted" and skipped the
    contribution, so every parent-only candidate fused to exactly zero -- invisible in unit tests
    that build candidates from scratch, and caught only by running the real retriever.
    """
    pre_set = make("from_parent")
    pre_set.legs["parent"] = LegHit(rank=1, score=3.5)

    fused = fuse_rrf({"parent": [pre_set]}, k=60)
    assert fused[0].fused_score == pytest.approx(1 / 61)
    # The engine score survives for the retrieval debugger.
    assert fused[0].legs["parent"].score == 3.5


def test_pre_set_leg_entries_do_not_suppress_other_legs() -> None:
    shared = make("shared")
    shared.legs["parent"] = LegHit(rank=2, score=1.0)
    fused = fuse_rrf({"bm25": [make("shared")], "parent": [shared]}, k=60)
    assert fused[0].fused_score == pytest.approx(1 / 61 + 1 / 61)
