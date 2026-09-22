"""Reranking behaviour and its degradation guarantees.

The degradation tests matter as much as the ranking ones. A reranker is a quality stage, and a
quality stage that can fail a request has been mis-designed: losing it costs precision, raising
costs the answer.
"""

from __future__ import annotations

import httpx
import pytest

from app.core.config import Settings
from app.retrieval.rerank.base import IdentityReranker, apply_scores
from app.retrieval.rerank.factory import build_entailment_scorer, build_reranker
from app.retrieval.rerank.lexical import LexicalEntailmentScorer, LexicalReranker
from app.retrieval.rerank.onnx_client import CrossEncoderReranker
from app.retrieval.types import Candidate


def candidate(
    chunk_id: str, text: str, *, score: float = 0.01, title: str = "Doc", heading: str | None = None
) -> Candidate:
    item = Candidate(
        chunk_id=chunk_id,
        parent_id=f"p-{chunk_id}",
        doc_id=f"d-{chunk_id}",
        doc_version_id="v",
        text=text,
        title=title,
        heading_path=heading,
    )
    item.fused_score = score
    return item


# --------------------------------------------------------------------------------------------
# Identity: the control arm
# --------------------------------------------------------------------------------------------


async def test_identity_preserves_fusion_order() -> None:
    """Any nDCG movement against this row is attributable to reranking alone."""
    items = [candidate("a", "one"), candidate("b", "two"), candidate("c", "three")]
    result = await IdentityReranker().rerank("q", items, top_n=3)
    assert [c.chunk_id for c in result] == ["a", "b", "c"]


async def test_identity_respects_top_n() -> None:
    items = [candidate(str(i), "text") for i in range(10)]
    assert len(await IdentityReranker().rerank("q", items, top_n=3)) == 3


# --------------------------------------------------------------------------------------------
# Lexical reranker
# --------------------------------------------------------------------------------------------


async def test_lexical_promotes_the_passage_that_answers() -> None:
    items = [
        candidate("vague", "Rates are reviewed annually by the finance committee."),
        candidate("answer", "The per diem allowance for grade A is 120 EUR per night."),
    ]
    result = await LexicalReranker().rerank("what is the per diem allowance for grade A", items, top_n=2)
    assert result[0].chunk_id == "answer"


async def test_lexical_uses_the_heading_as_evidence_of_topic() -> None:
    items = [
        candidate("body", "The committee meets quarterly to review submissions.", heading="Governance"),
        candidate("titled", "The committee meets quarterly to review submissions.", heading="Per Diem Allowance"),
    ]
    result = await LexicalReranker().rerank("per diem allowance", items, top_n=2)
    assert result[0].chunk_id == "titled"


async def test_lexical_prefers_a_stated_figure_when_one_is_asked_for() -> None:
    items = [
        candidate("prose", "The per diem allowance depends on the destination grade."),
        candidate("figure", "The per diem allowance is 120 EUR."),
    ]
    result = await LexicalReranker().rerank("how much is the per diem allowance", items, top_n=2)
    assert result[0].chunk_id == "figure"


async def test_lexical_assigns_a_score_to_every_candidate() -> None:
    items = [candidate("a", "per diem"), candidate("b", "unrelated")]
    result = await LexicalReranker().rerank("per diem", items, top_n=2)
    assert all(c.rerank_score is not None for c in result)


async def test_lexical_is_deterministic() -> None:
    """The CI ablation gate compares tables; a shuffling reranker would fake a regression."""
    items = [candidate(c, "identical text everywhere") for c in ("c", "a", "b")]
    first = [c.chunk_id for c in await LexicalReranker().rerank("text", items, top_n=3)]
    items2 = [candidate(c, "identical text everywhere") for c in ("c", "a", "b")]
    second = [c.chunk_id for c in await LexicalReranker().rerank("text", items2, top_n=3)]
    assert first == second


async def test_lexical_handles_a_query_of_only_stopwords() -> None:
    items = [candidate("a", "content"), candidate("b", "more")]
    assert len(await LexicalReranker().rerank("what is it", items, top_n=2)) == 2


async def test_lexical_of_nothing_is_empty() -> None:
    assert await LexicalReranker().rerank("q", [], top_n=5) == []


def test_apply_scores_breaks_ties_deterministically() -> None:
    items = [candidate("c", "x"), candidate("a", "x"), candidate("b", "x")]
    result = apply_scores(items, [1.0, 1.0, 1.0], top_n=3)
    assert [c.chunk_id for c in result] == ["a", "b", "c"]


# --------------------------------------------------------------------------------------------
# Cross-encoder client: degradation, not failure
# --------------------------------------------------------------------------------------------


def transport(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


async def test_cross_encoder_applies_returned_scores() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"id": "b", "score": 0.9}, {"id": "a", "score": 0.1}]})

    async with transport(handler) as client:
        reranker = CrossEncoderReranker("http://models", client=client)
        result = await reranker.rerank("q", [candidate("a", "x"), candidate("b", "y")], top_n=2)
    assert [c.chunk_id for c in result] == ["b", "a"]
    assert result[0].rerank_score == 0.9


async def test_cross_encoder_degrades_to_fusion_order_when_the_service_is_down() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with transport(handler) as client:
        reranker = CrossEncoderReranker("http://models", client=client)
        result = await reranker.rerank("q", [candidate("a", "x"), candidate("b", "y")], top_n=2)
    assert [c.chunk_id for c in result] == ["a", "b"]


async def test_cross_encoder_degrades_on_a_server_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"detail": "boom"})

    async with transport(handler) as client:
        reranker = CrossEncoderReranker("http://models", client=client)
        result = await reranker.rerank("q", [candidate("a", "x")], top_n=1)
    assert [c.chunk_id for c in result] == ["a"]


async def test_cross_encoder_caps_the_pairs_it_sends() -> None:
    """The top-24 cap is a main reason the latency budget is achievable."""
    seen: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["pairs"] = len(json.loads(request.content)["passages"])
        return httpx.Response(200, json={"results": []})

    async with transport(handler) as client:
        reranker = CrossEncoderReranker("http://models", max_pairs=24, client=client)
        await reranker.rerank("q", [candidate(str(i), "x") for i in range(100)], top_n=10)
    assert seen["pairs"] == 24


async def test_cross_encoder_sends_the_token_cap() -> None:
    seen: dict[str, int] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["max_length"] = json.loads(request.content)["max_length"]
        return httpx.Response(200, json={"results": []})

    async with transport(handler) as client:
        await CrossEncoderReranker("http://models", max_length=288, client=client).rerank(
            "q", [candidate("a", "x")], top_n=1
        )
    assert seen["max_length"] == 288


async def test_cross_encoder_includes_the_context_line_in_the_pair() -> None:
    """The context line is the highest-signal text in a chunk, so it must reach the model."""
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen["text"] = json.loads(request.content)["passages"][0]["text"]
        return httpx.Response(200, json={"results": []})

    item = candidate("a", "body text")
    item.context_line = "Per Diem section of the Travel Policy"
    async with transport(handler) as client:
        await CrossEncoderReranker("http://models", client=client).rerank("q", [item], top_n=1)
    assert "Per Diem section" in seen["text"]
    assert "body text" in seen["text"]


async def test_cross_encoder_entailment_returns_a_score() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"id": "claim", "score": 0.72}]})

    async with transport(handler) as client:
        assert await CrossEncoderReranker("http://models", client=client).score("c", "p") == 0.72


async def test_entailment_of_an_unreachable_scorer_does_not_reject_the_answer() -> None:
    """A missing scorer degrades verification; it must not fail a valid answer."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with transport(handler) as client:
        assert await CrossEncoderReranker("http://models", client=client).score("c", "p") == 1.0


# --------------------------------------------------------------------------------------------
# Lexical entailment: honest about what it cannot see
# --------------------------------------------------------------------------------------------


async def test_lexical_entailment_rejects_an_unrelated_passage() -> None:
    score = await LexicalEntailmentScorer().score(
        "The per diem for grade A is 120 EUR.", "Meeting rooms are released after ten minutes."
    )
    assert score < 0.3


async def test_lexical_entailment_cannot_see_a_faithful_non_answer() -> None:
    """The documented limitation, asserted so nobody mistakes the proxy for a model.

    The passage shares every content word with the claim and still does not establish it. Only a
    model that reads the two together can tell; term overlap cannot, which is exactly why the
    entailment ablation row did not move the made-up rate.
    """
    score = await LexicalEntailmentScorer().score(
        "The per diem for grade D is 45 EUR.",
        "The per diem depends on the destination grade. Grade A is 120 EUR and grade B is 90 EUR.",
    )
    assert score > 0.5, "term overlap is high even though the passage does not support the claim"


# --------------------------------------------------------------------------------------------
# Factory
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("provider", "expected"),
    [("identity", "identity"), ("lexical", "lexical")],
)
def test_factory_builds_the_configured_reranker(provider: str, expected: str) -> None:
    assert build_reranker(Settings(reranker_provider=provider)).name == expected  # type: ignore[arg-type]


def test_onnx_without_a_service_url_falls_back_rather_than_failing_to_start() -> None:
    """A forgotten models profile should degrade quality with a log line, not refuse to boot."""
    assert build_reranker(Settings(reranker_provider="onnx", models_service_url=None)).name == "lexical"


def test_onnx_with_a_service_url_builds_the_cross_encoder() -> None:
    settings = Settings(reranker_provider="onnx", models_service_url="http://models:8000")
    assert build_reranker(settings).name == "onnx-cross-encoder"


def test_entailment_scorer_is_absent_when_nothing_trustworthy_is_configured() -> None:
    """Verification degrades to its deterministic layers rather than guessing."""
    assert build_entailment_scorer(Settings(reranker_provider="identity")) is None


def test_entailment_scorer_reuses_the_cross_encoder_when_deployed() -> None:
    settings = Settings(reranker_provider="onnx", models_service_url="http://models:8000")
    scorer = build_entailment_scorer(settings)
    assert isinstance(scorer, CrossEncoderReranker)
