"""Client for the ``models`` container: cross-encoder reranking and entailment.

The model runs in a separate process behind HTTP, not as a library in this image. Three reasons,
in order of how much they matter:

* **Dependency isolation.** torch CPU is 1-2.5 GB; ONNX Runtime is ~150 MB. Neither belongs in
  an API image that needs to start fast and scale horizontally. Keeping them out is also what
  lets the API pin Python 3.14 while the ML stack stays on 3.12.
* **The GIL.** Inference is CPU-bound and would block the event loop serving every other
  request. A separate process cannot.
* **Independent scaling.** Reranking is the most expensive stage per request; it should scale on
  its own axis, and in a GPU deployment it is the only thing that needs the GPU.

The latency budget is real and tight. A 278M-parameter cross-encoder at 512 tokens costs roughly
25-50 ms *per pair* on CPU; fifty passages is 1.5-2.5 seconds, which is not a product. The budget
is met by four levers held simultaneously -- top-24 rather than top-50, 288-token truncation,
int8 quantization, and one batched call -- and `bench/rerank_bench.py` exists because relaxing
any one of them silently triples p95.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import httpx

from app.retrieval.rerank.base import apply_scores
from app.retrieval.types import Candidate


class CrossEncoderReranker:
    """Scores pairs via the models service."""

    name = "onnx-cross-encoder"

    def __init__(
        self,
        base_url: str,
        *,
        max_pairs: int = 24,
        max_length: int = 288,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.max_pairs = max_pairs
        self.max_length = max_length
        self._client = client

    async def _post(self, path: str, payload: dict[str, Any], timeout_s: float) -> dict[str, Any]:
        if self._client is not None:
            response = await self._client.post(f"{self.base_url}{path}", json=payload, timeout=timeout_s)
            response.raise_for_status()
            result: dict[str, Any] = response.json()
            return result
        async with httpx.AsyncClient(timeout=timeout_s) as client:
            response = await client.post(f"{self.base_url}{path}", json=payload)
            response.raise_for_status()
            fresh: dict[str, Any] = response.json()
            return fresh

    async def rerank(
        self, query: str, candidates: Sequence[Candidate], *, top_n: int, timeout_s: float = 0.4
    ) -> list[Candidate]:
        if not candidates:
            return []

        pool = list(candidates[: self.max_pairs])
        payload = {
            "query": query,
            # The context line is the highest-signal text in a chunk, so it survives truncation
            # ahead of the body rather than being cut off with it.
            "passages": [{"id": c.chunk_id, "text": _passage_text(c)} for c in pool],
            "max_length": self.max_length,
            "top_n": top_n,
        }
        try:
            body = await self._post("/rerank", payload, timeout_s)
        except httpx.HTTPError:
            # Degrade to fusion order. A reranker outage costs precision; raising costs the
            # answer, and the caller records the status in the retrieval diagnostics.
            return list(candidates[:top_n])

        by_id = {item["id"]: float(item["score"]) for item in body.get("results", [])}
        if not by_id:
            return list(candidates[:top_n])
        return apply_scores(pool, [by_id.get(c.chunk_id, 0.0) for c in pool], top_n=top_n)

    async def score(self, claim: str, passage: str) -> float:
        """Entailment for citation verification.

        The same model, the same container, the same call. A cross-encoder asked whether a
        passage supports a claim is doing the task it was trained for, so verification gets a
        real judgement without a second model to deploy or operate.
        """
        try:
            body = await self._post(
                "/rerank",
                {
                    "query": claim,
                    "passages": [{"id": "claim", "text": passage}],
                    "max_length": self.max_length,
                    "top_n": 1,
                },
                timeout_s=1.0,
            )
        except httpx.HTTPError:
            # A missing scorer must not fail an otherwise valid answer; verification treats a
            # skipped entailment check as "not disproved" and relies on the value layer.
            return 1.0
        results = body.get("results", [])
        return float(results[0]["score"]) if results else 1.0

    async def health(self) -> bool:
        try:
            if self._client is not None:
                response = await self._client.get(f"{self.base_url}/healthz", timeout=2.0)
            else:
                async with httpx.AsyncClient(timeout=2.0) as client:
                    response = await client.get(f"{self.base_url}/healthz")
            return response.status_code == 200
        except httpx.HTTPError:
            return False


def _passage_text(candidate: Candidate) -> str:
    if candidate.context_line:
        return f"{candidate.context_line}\n{candidate.text}"
    return candidate.text


class HostedReranker:
    """Cohere, Voyage and Jina behind the same Protocol.

    The only place a tenant's query text leaves the deployment, so every call is gated on a
    per-tenant ``allow_external_processing`` flag and emits an audit event. A BYOC install sets
    ``RERANKER_PROVIDER=onnx`` and these modules are never constructed.

    ``jina-reranker-v1-turbo-en`` is deliberately absent: it is CC-BY-NC and cannot be used in a
    commercial product, whatever its latency.
    """

    def __init__(self, provider: str, api_key: str, *, model: str, max_pairs: int = 50) -> None:
        self.name = f"hosted-{provider}"
        self.provider = provider
        self.api_key = api_key
        self.model = model
        self.max_pairs = max_pairs

    async def rerank(
        self, query: str, candidates: Sequence[Candidate], *, top_n: int, timeout_s: float = 0.4
    ) -> list[Candidate]:
        raise NotImplementedError(
            "Hosted rerankers are configured per tenant and constructed by the container; "
            "see docs/decisions.md on external processing."
        )
