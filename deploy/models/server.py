"""The models service: reranking and embeddings.

Two models, two jobs that cannot share one. A cross-encoder scores a (query, passage) *pair* and
has no per-text vector to extract; a bi-encoder embeds one text and never sees the query
alongside the passage, so it reranks poorly. One container because they share a runtime, a warmup
path and an artefact to ship into an air-gapped registry -- not because they share weights.

The answer path reuses the cross-encoder as an entailment scorer: "does this passage support this
claim" is the task a reranker was trained for.

Everything here exists to hold a latency budget that is easy to lose:

* **One batched call.** Per-pair calls spend more time in HTTP than in the model.
* **Length-sorted batching.** Padding is wasted compute, and a batch sorted by length wastes
  much less of it.
* **A hard token cap.** Attention is quadratic, so 288 tokens costs roughly a third of 512.
* **Warm on startup.** The first inference of an ONNX session is several times slower than the
  rest; serving that to a user makes p99 meaningless. ``/healthz`` stays 503 until it is done,
  so an orchestrator does not route traffic to a cold replica.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

MODEL_DIR = os.environ.get("MODEL_DIR", "/srv/model")
MODEL_NAME = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")
DEFAULT_MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "288"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "64"))

EMBED_DIR = os.environ.get("EMBED_MODEL_DIR", "/srv/embedder")
EMBED_MODEL_NAME = os.environ.get("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
#: 512 rather than the reranker's 288. A passage is embedded once at ingest and queried against
#: forever, so truncating it is a permanent loss of recall -- whereas reranking is per-query and
#: can afford a tighter cap.
EMBED_MAX_LENGTH = int(os.environ.get("EMBED_MAX_LENGTH", "512"))
EMBED_MAX_BATCH = int(os.environ.get("EMBED_MAX_BATCH", "32"))

_state: dict[str, Any] = {"ready": False, "embed_ready": False}


class Passage(BaseModel):
    id: str
    text: str


class RerankRequest(BaseModel):
    query: str
    passages: list[Passage]
    max_length: int = Field(default=DEFAULT_MAX_LENGTH, ge=16, le=512)
    top_n: int = Field(default=24, ge=1, le=256)


class ScoredPassage(BaseModel):
    id: str
    score: float


class RerankResponse(BaseModel):
    results: list[ScoredPassage]
    model: str
    latency_ms: float
    pairs: int


class EmbedRequest(BaseModel):
    texts: list[str] = Field(min_length=1, max_length=512)
    #: Passages and queries are embedded differently by every model family worth using. Getting
    #: this wrong does not error -- it silently costs recall, which is why the caller must say
    #: which it is rather than the server guessing.
    kind: str = Field(default="passage", pattern="^(passage|query)$")
    max_length: int = Field(default=EMBED_MAX_LENGTH, ge=16, le=512)


class EmbedResponse(BaseModel):
    vectors: list[list[float]]
    model: str
    #: The string that enters the generation fingerprint. Returned on every response so the
    #: caller stamps what actually served the request rather than what it believes is
    #: configured -- a replica running an older image would otherwise poison an index invisibly.
    embedder_id: str
    dimension: int
    normalized: bool
    latency_ms: float
    truncated: int


# ----------------------------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------------------------


def _load() -> None:
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = ORTModelForSequenceClassification.from_pretrained(MODEL_DIR)
    _state["tokenizer"] = tokenizer
    _state["model"] = model

    # Warm up at the shape we actually serve, not a single short pair: an ONNX session
    # specialises per input shape, so warming on a 1x32 batch leaves the 24x288 path cold.
    warm_query = "warmup query about a policy"
    warm_passages = ["warmup passage text " * 20] * 8
    _score(warm_query, warm_passages, DEFAULT_MAX_LENGTH)

    _load_embedder()
    _state["ready"] = True


def _load_embedder() -> None:
    """Load the bi-encoder, if one was baked in.

    Optional on purpose: a deployment that only reranks should not carry an embedder it never
    calls, and ``/embed`` then says so plainly rather than returning wrong vectors.
    """
    directory = Path(EMBED_DIR)
    if not directory.exists():
        _state["embed_ready"] = False
        return

    from optimum.onnxruntime import ORTModelForFeatureExtraction
    from transformers import AutoTokenizer

    card_path = directory / "embedding_card.json"
    # The card records pooling, prefixes and dimension next to the weights. Falling back to
    # guesses would mean a model swap silently produces worse vectors, so an absent card is a
    # hard failure rather than a default.
    if not card_path.exists():
        raise RuntimeError(f"{card_path} is missing; re-export the embedder with export_model.py")

    card = json.loads(card_path.read_text(encoding="utf-8"))
    _state["embed_card"] = card
    _state["embed_tokenizer"] = AutoTokenizer.from_pretrained(EMBED_DIR)
    _state["embed_model"] = ORTModelForFeatureExtraction.from_pretrained(EMBED_DIR)
    _state["embed_ready"] = True

    # Warm at the batch shape actually served, for the same reason the reranker does.
    _embed(["warmup passage text " * 30] * 8, kind="passage", max_length=EMBED_MAX_LENGTH)


# ----------------------------------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------------------------------


def _sigmoid(values: np.ndarray) -> np.ndarray:
    """Map raw logits into [0, 1].

    Callers treat the score as a probability -- the entailment threshold in citation
    verification is expressed that way -- so the squashing belongs here rather than being
    reinvented, differently, at each call site.
    """
    return 1.0 / (1.0 + np.exp(-values))


def _score(query: str, texts: list[str], max_length: int) -> list[float]:
    tokenizer = _state["tokenizer"]
    model = _state["model"]

    scores: list[float] = [0.0] * len(texts)
    # Sort by length so each batch pads to something close to its own longest member rather
    # than to the longest passage in the whole request.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))

    for start in range(0, len(order), MAX_BATCH):
        window = order[start : start + MAX_BATCH]
        batch = tokenizer(
            [query] * len(window),
            [texts[i] for i in window],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        logits = model(**batch).logits
        flat = logits[:, 0] if logits.ndim == 2 and logits.shape[1] == 1 else logits.max(axis=-1)
        for position, value in zip(window, _sigmoid(np.asarray(flat, dtype=np.float32)), strict=True):
            scores[position] = float(value)
    return scores


def _pool(last_hidden: np.ndarray, attention_mask: np.ndarray, mode: str) -> np.ndarray:
    """Collapse token vectors into one sentence vector.

    Mean pooling **must** exclude padding. Including it drags every vector toward the padding
    embedding by an amount that depends on batch composition -- so the same text embedded in two
    differently-shaped batches gets two different vectors, and an index built that way is subtly
    inconsistent with itself. That is a bug with no error message and no obvious symptom beyond
    "retrieval is a bit worse than it should be".
    """
    if mode == "cls":
        return np.asarray(last_hidden[:, 0], dtype=np.float32)

    mask = np.asarray(attention_mask, dtype=np.float32)[..., None]
    summed = (last_hidden * mask).sum(axis=1)
    counts = np.clip(mask.sum(axis=1), 1e-9, None)
    return np.asarray(summed / counts, dtype=np.float32)


def _embed(texts: list[str], *, kind: str, max_length: int) -> tuple[list[list[float]], int]:
    """Embed texts, returning unit-length vectors and how many were truncated."""
    tokenizer = _state["embed_tokenizer"]
    model = _state["embed_model"]
    card = _state["embed_card"]

    prefix = card["query_prefix"] if kind == "query" else card["passage_prefix"]
    prepared = [f"{prefix}{text}" for text in texts]

    vectors: list[list[float]] = [[] for _ in texts]
    truncated = 0
    # Length-sorted, as the reranker is: padding is wasted compute and a sorted batch wastes
    # much less of it.
    order = sorted(range(len(prepared)), key=lambda i: len(prepared[i]))

    for start in range(0, len(order), EMBED_MAX_BATCH):
        window = order[start : start + EMBED_MAX_BATCH]
        batch = tokenizer(
            [prepared[i] for i in window],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="np",
        )
        truncated += int((np.asarray(batch["attention_mask"]).sum(axis=1) >= max_length).sum())

        output = model(**batch)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None:
            hidden = output[0]

        pooled = _pool(np.asarray(hidden, dtype=np.float32), batch["attention_mask"], card["pooling"])
        # L2 normalize here, not in the client. The index uses space_type=innerproduct, which
        # equals cosine only on unit vectors -- so a caller that forgot would get silently wrong
        # rankings rather than an error.
        norms = np.clip(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12, None)
        unit = pooled / norms

        for position, vector in zip(window, unit, strict=True):
            vectors[position] = [float(value) for value in vector]

    return vectors, truncated


# ----------------------------------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _load()
    yield


app = FastAPI(title="models", version="1.1.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> JSONResponse:
    if not _state.get("ready"):
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse(
        {
            "status": "ok",
            "model": MODEL_NAME,
            "warm": True,
            # Surfaced so a deployment can tell at a glance whether the dense leg is real. A
            # service silently serving no embedder is the failure this answers.
            "embedder": _state.get("embed_card", {}).get("embedder_id") if _state.get("embed_ready") else None,
        }
    )


@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest) -> RerankResponse:
    started = time.perf_counter()
    if not request.passages:
        return RerankResponse(results=[], model=MODEL_NAME, latency_ms=0.0, pairs=0)

    scores = _score(request.query, [p.text for p in request.passages], request.max_length)
    ranked = sorted(
        (ScoredPassage(id=p.id, score=s) for p, s in zip(request.passages, scores, strict=True)),
        key=lambda item: -item.score,
    )
    return RerankResponse(
        results=ranked[: request.top_n],
        model=MODEL_NAME,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        pairs=len(request.passages),
    )


@app.post("/embed", response_model=EmbedResponse)
async def embed(request: EmbedRequest) -> EmbedResponse | JSONResponse:
    started = time.perf_counter()

    if not _state.get("embed_ready"):
        # 503, not 500: no embedder was baked into this image. The caller degrades to its
        # fallback rather than failing an ingest, and the message says what to do about it.
        return JSONResponse(
            {"detail": "no embedding model is loaded; build the image with EMBEDDING_MODEL set"},
            status_code=503,
        )

    vectors, truncated = _embed(request.texts, kind=request.kind, max_length=request.max_length)
    card = _state["embed_card"]

    return EmbedResponse(
        vectors=vectors,
        model=EMBED_MODEL_NAME,
        embedder_id=card["embedder_id"],
        dimension=card["dimension"],
        normalized=True,
        latency_ms=(time.perf_counter() - started) * 1000.0,
        truncated=truncated,
    )
