"""The ``/embed`` route, against a stubbed ONNX model.

The real model needs ``optimum`` and ``transformers``, which live in another container. What can
be tested from here is everything around the inference: that vectors come back unit length, that
the embedder id is stamped on every response, that a service with no model loaded says so rather
than returning something plausible, and that queries and passages take different paths.

Unit length is the one to notice. The chunk mapping uses ``space_type: innerproduct``, which
equals cosine *only* on unit vectors. A server that skipped normalization would return vectors
that index cleanly and rank wrongly, with no error anywhere.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

SERVER = Path(__file__).resolve().parents[3] / "deploy" / "models" / "server.py"


class StubTokenizer:
    """Tokenizes on whitespace, pads to the batch's longest, and marks padding in the mask."""

    def __call__(
        self,
        texts: list[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 512,
        return_tensors: str = "np",
    ) -> dict[str, Any]:
        lengths = [max(1, min(len(text.split()), max_length)) for text in texts]
        width = max(lengths)
        mask = np.zeros((len(texts), width), dtype=np.int64)
        for row, length in enumerate(lengths):
            mask[row, :length] = 1
        return {"input_ids": np.zeros((len(texts), width), dtype=np.int64), "attention_mask": mask}


class StubOutput:
    def __init__(self, hidden: np.ndarray) -> None:
        self.last_hidden_state = hidden


class StubModel:
    """Returns a distinct constant per row, so ordering and pooling are both observable."""

    def __call__(self, input_ids: Any = None, attention_mask: Any = None) -> StubOutput:
        rows, width = attention_mask.shape
        values = np.arange(1, rows + 1, dtype=np.float32)[:, None, None]
        return StubOutput(np.tile(values, (1, width, 4)))


@pytest.fixture
def server() -> ModuleType:
    if not SERVER.exists():  # pragma: no cover
        pytest.skip("deploy/models/server.py is not present")

    spec = importlib.util.spec_from_file_location("models_server_route_test", SERVER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    module._state.update(
        {
            "ready": True,
            "embed_ready": True,
            "embed_tokenizer": StubTokenizer(),
            "embed_model": StubModel(),
            "embed_card": {
                "pooling": "mean",
                "query_prefix": "Represent this sentence for searching relevant passages: ",
                "passage_prefix": "",
                "dimension": 4,
                "embedder_id": "stub-model@4@fp32",
            },
        }
    )
    return module


@pytest.fixture
def client(server: ModuleType) -> TestClient:
    return TestClient(server.app)


# ------------------------------------------------------------------------------------------
# Vectors
# ------------------------------------------------------------------------------------------


def test_vectors_come_back_unit_length(client: TestClient) -> None:
    """The mapping uses innerproduct, which equals cosine only on unit vectors. Skipping this
    would produce vectors that index cleanly and rank wrongly, silently."""
    body = client.post("/embed", json={"texts": ["alpha beta", "gamma delta epsilon"]}).json()

    for vector in body["vectors"]:
        assert abs(float(np.linalg.norm(vector)) - 1.0) < 1e-5


def test_the_response_declares_itself_normalized(client: TestClient) -> None:
    assert client.post("/embed", json={"texts": ["x"]}).json()["normalized"] is True


def test_one_vector_per_text(client: TestClient) -> None:
    body = client.post("/embed", json={"texts": ["a", "b", "c", "d"]}).json()
    assert len(body["vectors"]) == 4
    assert all(len(vector) == 4 for vector in body["vectors"])


def test_the_dimension_matches_the_card(client: TestClient) -> None:
    body = client.post("/embed", json={"texts": ["x"]}).json()
    assert body["dimension"] == 4 == len(body["vectors"][0])


# ------------------------------------------------------------------------------------------
# Identity
# ------------------------------------------------------------------------------------------


def test_every_response_stamps_the_embedder_id(client: TestClient) -> None:
    """It enters the generation fingerprint, so the caller records what actually served the
    request rather than what it believes is configured. A replica running an older image would
    otherwise poison an index invisibly."""
    assert client.post("/embed", json={"texts": ["x"]}).json()["embedder_id"] == "stub-model@4@fp32"


def test_health_reports_which_embedder_is_serving(client: TestClient) -> None:
    """So a deployment can tell at a glance whether the dense leg is real."""
    assert client.get("/healthz").json()["embedder"] == "stub-model@4@fp32"


def test_health_reports_no_embedder_when_none_is_loaded(server: ModuleType) -> None:
    server._state["embed_ready"] = False
    assert TestClient(server.app).get("/healthz").json()["embedder"] is None


# ------------------------------------------------------------------------------------------
# Queries are not passages
# ------------------------------------------------------------------------------------------


def test_a_query_is_prefixed_and_a_passage_is_not(server: ModuleType) -> None:
    """The asymmetry is trained in, and omitting it costs recall without erroring. The server
    applies it from the card so no client has to know which family is deployed."""
    seen: list[str] = []

    class Recording(StubTokenizer):
        def __call__(self, texts: list[str], **kwargs: Any) -> dict[str, Any]:
            seen.extend(texts)
            return super().__call__(texts, **kwargs)

    server._state["embed_tokenizer"] = Recording()
    client = TestClient(server.app)

    client.post("/embed", json={"texts": ["per diem"], "kind": "query"})
    assert seen[-1].startswith("Represent this sentence")

    client.post("/embed", json={"texts": ["per diem"], "kind": "passage"})
    assert seen[-1] == "per diem"


def test_an_unknown_kind_is_refused(client: TestClient) -> None:
    """Better a 422 than silently treating a query as a passage."""
    assert client.post("/embed", json={"texts": ["x"], "kind": "sideways"}).status_code == 422


# ------------------------------------------------------------------------------------------
# Absent model, and bad input
# ------------------------------------------------------------------------------------------


def test_a_service_with_no_embedder_answers_503_and_says_what_to_do(server: ModuleType) -> None:
    """503 rather than 500: the caller degrades its *job*, and the message names the fix, which
    is a rebuild rather than a restart."""
    server._state["embed_ready"] = False
    response = TestClient(server.app).post("/embed", json={"texts": ["x"]})

    assert response.status_code == 503
    assert "EMBEDDING_MODEL" in response.json()["detail"]


def test_an_empty_request_is_refused_rather_than_answered_with_nothing(client: TestClient) -> None:
    """An empty embed is always a caller bug, and answering it with [] hides that."""
    assert client.post("/embed", json={"texts": []}).status_code == 422


def test_an_over_long_max_length_is_refused(client: TestClient) -> None:
    """The model's position embeddings stop at 512; asking for more would truncate anyway, so
    the refusal is more honest than silently capping."""
    assert client.post("/embed", json={"texts": ["x"], "max_length": 4096}).status_code == 422


def test_truncation_is_reported(server: ModuleType) -> None:
    """A truncated passage is a permanent recall loss for that chunk. The count travels back so
    the client can warn rather than the loss being invisible."""
    client = TestClient(server.app)
    long_text = " ".join(["word"] * 200)
    body = client.post("/embed", json={"texts": [long_text], "max_length": 16}).json()

    assert body["truncated"] == 1


def test_nothing_is_truncated_when_everything_fits(client: TestClient) -> None:
    assert client.post("/embed", json={"texts": ["short text"]}).json()["truncated"] == 0
