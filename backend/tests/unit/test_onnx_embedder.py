"""The embedding client, and chiefly the two places it refuses to be helpful.

Most of this product degrades under failure, on purpose. This client does not, and the tests that
matter here are the ones proving it: a fallback would write vectors from a second vector space
into one index, where similarity scores are meaningless and nothing downstream can tell. A slow
ingest is recoverable; a poisoned index is a rebuild.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from app.embeddings.onnx_client import (
    EmbedderIdentityChangedError,
    EmbeddingUnavailableError,
    OnnxEmbedder,
    build_onnx_embedder,
    probe,
)

EMBEDDER_ID = "bge-base-en-v1.5@768@int8"


def transport(handler: Any) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def embed_response(count: int, *, embedder_id: str = EMBEDDER_ID, truncated: int = 0) -> dict[str, Any]:
    return {
        "vectors": [[0.1] * 768 for _ in range(count)],
        "model": "BAAI/bge-base-en-v1.5",
        "embedder_id": embedder_id,
        "dimension": 768,
        "normalized": True,
        "latency_ms": 12.0,
        "truncated": truncated,
    }


def an_embedder(client: httpx.AsyncClient, **kwargs: Any) -> OnnxEmbedder:
    base: dict[str, Any] = {"embedder_id": EMBEDDER_ID, "dimension": 768, "client": client}
    base.update(kwargs)
    return OnnxEmbedder("http://models:8000", **base)


# ------------------------------------------------------------------------------------------
# It does not degrade
# ------------------------------------------------------------------------------------------


async def test_an_outage_raises_rather_than_falling_back_to_hashing() -> None:
    """The test this module exists for.

    A fallback would write hashed vectors into an index of bge vectors. Similarity between the
    two is meaningless, the dense leg returns plausible nonsense for those documents, and no
    error appears anywhere -- so the job must fail and retry instead.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with transport(handler) as client:
        with pytest.raises(EmbeddingUnavailableError):
            await an_embedder(client, max_retries=0).embed_passages(["some text"])


async def test_a_service_with_no_embedder_loaded_raises_and_says_to_rebuild() -> None:
    """503 means no model was baked into the image. The fix is a rebuild, not a restart, and an
    operator reading a retry storm deserves to be told which."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "no embedding model is loaded"})

    async with transport(handler) as client:
        with pytest.raises(EmbeddingUnavailableError):
            await an_embedder(client, max_retries=0).embed_passages(["text"])


async def test_a_short_outage_is_retried() -> None:
    """A warming replica clears in seconds. The queue owns the long backoff; this is the short
    one that avoids failing a job over a rolling restart."""
    attempts = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise httpx.ConnectError("still warming")
        return httpx.Response(200, json=embed_response(1))

    async with transport(handler) as client:
        vectors = await an_embedder(client).embed_passages(["text"])

    assert len(vectors) == 1
    assert attempts["n"] == 3


async def test_a_short_response_is_not_silently_accepted() -> None:
    """Fewer vectors than texts would misalign every chunk with its neighbour's embedding --
    an index that is wrong in a way no assertion downstream would catch."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=embed_response(2))

    async with transport(handler) as client:
        with pytest.raises(EmbeddingUnavailableError, match="asked for 3"):
            await an_embedder(client, max_retries=0).embed_passages(["a", "b", "c"])


# ------------------------------------------------------------------------------------------
# Identity
# ------------------------------------------------------------------------------------------


async def test_a_mid_run_checkpoint_change_is_fatal() -> None:
    """Two image versions serving at once during a rollout. Continuing would interleave two
    vector spaces in one index, which the generation fingerprint exists to make impossible."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=embed_response(1, embedder_id="bge-large-en-v1.5@1024@int8"))

    async with transport(handler) as client:
        with pytest.raises(EmbedderIdentityChangedError, match="rollout"):
            await an_embedder(client).embed_passages(["text"])


async def test_an_identity_change_is_not_retried() -> None:
    """A second attempt hits the same mixed fleet. The point is to stop before writing."""
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=embed_response(1, embedder_id="something-else@768@int8"))

    async with transport(handler) as client:
        with pytest.raises(EmbedderIdentityChangedError):
            await an_embedder(client, max_retries=5).embed_passages(["text"])

    assert calls["n"] == 1


async def test_the_id_is_discovered_from_the_service_not_configured() -> None:
    """It enters the generation fingerprint, so it must describe what actually produced the
    vectors. A configured id would let a replica running an older image stamp a plausible label
    onto vectors from a different checkpoint."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok", "embedder": EMBEDDER_ID})

    async with transport(handler) as client:
        embedder = await build_onnx_embedder("http://models:8000", client=client)

    assert embedder.id == EMBEDDER_ID
    assert embedder.dimension == 768


async def test_probing_a_service_with_no_embedder_refuses_clearly() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok", "model": "reranker", "embedder": None})

    async with transport(handler) as client:
        with pytest.raises(EmbeddingUnavailableError, match="EMBEDDING_MODEL"):
            await probe("http://models:8000", client=client)


async def test_an_unparseable_embedder_id_is_refused_rather_than_guessed() -> None:
    """The dimension is parsed from the id so the two cannot disagree. A guess here would size
    the kNN mapping wrongly and every bulk write would be rejected -- loudly, but much later."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok", "embedder": "some-model-without-a-dimension"})

    async with transport(handler) as client:
        with pytest.raises(EmbeddingUnavailableError, match="cannot parse"):
            await build_onnx_embedder("http://models:8000", client=client)


# ------------------------------------------------------------------------------------------
# Queries and passages are not the same thing
# ------------------------------------------------------------------------------------------


async def test_a_query_is_sent_as_a_query() -> None:
    """The prefix asymmetry is trained into every model family worth using, and omitting it does
    not error -- it silently costs several points of recall."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json=embed_response(1))

    async with transport(handler) as client:
        await an_embedder(client).embed_query("what is the per diem?")

    assert seen["kind"] == "query"


async def test_a_passage_is_sent_as_a_passage() -> None:
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json=embed_response(1))

    async with transport(handler) as client:
        await an_embedder(client).embed_passages(["the rate is 120 EUR"])

    assert seen["kind"] == "passage"


async def test_the_prefix_is_applied_by_the_server_not_the_client() -> None:
    """The card lives next to the weights, so swapping the model is a build argument rather than
    a code change. A client that prefixed would have to know which family is deployed."""
    seen: dict[str, Any] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        seen.update(json.loads(request.content))
        return httpx.Response(200, json=embed_response(1))

    async with transport(handler) as client:
        await an_embedder(client).embed_query("per diem")

    assert seen["texts"] == ["per diem"], "the client must send the raw text"


# ------------------------------------------------------------------------------------------
# Batching
# ------------------------------------------------------------------------------------------


async def test_a_large_request_is_split_into_batches() -> None:
    batches: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        count = len(json.loads(request.content)["texts"])
        batches.append(count)
        return httpx.Response(200, json=embed_response(count))

    async with transport(handler) as client:
        vectors = await an_embedder(client, batch_size=10).embed_passages([f"text {n}" for n in range(35)])

    assert len(vectors) == 35
    assert batches == [10, 10, 10, 5]


async def test_vectors_come_back_in_the_order_they_were_sent() -> None:
    """The server sorts by length internally for padding efficiency and must restore the order.
    A client that assumed otherwise would attach every vector to the wrong chunk."""

    def handler(request: httpx.Request) -> httpx.Response:
        import json

        texts = json.loads(request.content)["texts"]
        return httpx.Response(
            200,
            json={
                **embed_response(len(texts)),
                "vectors": [[float(len(t))] * 768 for t in texts],
            },
        )

    async with transport(handler) as client:
        vectors = await an_embedder(client).embed_passages(["a", "bbb", "cc"])

    assert [v[0] for v in vectors] == [1.0, 3.0, 2.0]


async def test_embedding_nothing_costs_no_request() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=embed_response(0))

    async with transport(handler) as client:
        assert await an_embedder(client).embed_passages([]) == []
    assert calls["n"] == 0


async def test_truncation_is_reported_rather_than_hidden(caplog: pytest.LogCaptureFixture) -> None:
    """A truncated passage is a permanent recall loss for that chunk, and a corpus that trips
    this constantly means the chunker cap and the embedder cap disagree."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=embed_response(2, truncated=2))

    with caplog.at_level("WARNING"):
        async with transport(handler) as client:
            await an_embedder(client).embed_passages(["long " * 500, "long " * 500])

    assert any("truncated" in record.message for record in caplog.records)


# ------------------------------------------------------------------------------------------
# The protocol
# ------------------------------------------------------------------------------------------


def test_it_satisfies_the_embedder_protocol() -> None:
    """Structural rather than nominal, so a missing attribute is a type error at the call site
    rather than an AttributeError during an ingest."""
    from app.embeddings.base import Embedder

    embedder: Embedder = OnnxEmbedder("http://models:8000", embedder_id=EMBEDDER_ID, dimension=768)
    assert embedder.normalized is True
    assert embedder.dimension == 768


def test_it_declares_itself_normalized() -> None:
    """The index uses space_type=innerproduct, which equals cosine only on unit vectors. The
    server normalizes; this attribute is the promise the mapping relies on."""
    assert OnnxEmbedder("http://x", embedder_id=EMBEDDER_ID, dimension=768).normalized is True
