"""The models service.

One endpoint doing one job: score (query, passage) pairs with a cross-encoder. The answer path
reuses it as an entailment scorer, which is why there is no second model and no second service --
"does this passage support this claim" is the task a reranker was trained for.

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

import os
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from typing import Any

import numpy as np
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

MODEL_DIR = os.environ.get("MODEL_DIR", "/srv/model")
MODEL_NAME = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")
DEFAULT_MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "288"))
MAX_BATCH = int(os.environ.get("MAX_BATCH", "64"))

_state: dict[str, Any] = {"ready": False}


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
    _state["ready"] = True


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


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _load()
    yield


app = FastAPI(title="models", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> JSONResponse:
    if not _state.get("ready"):
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse({"status": "ok", "model": MODEL_NAME, "warm": True})


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
