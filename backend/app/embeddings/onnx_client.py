"""Client for the ``models`` container's ``/embed`` route.

**This client does not degrade, and that is the whole design.** Every other remote dependency in
this product falls back when it fails: a reranker timeout drops to fusion order, an LLM outage
drops to the template contextualizer, a parser outage fails one job. Falling back here would
write vectors from a *different vector space* into the same index, and nothing downstream could
tell. Similarity scores between a hashed vector and a bge vector are meaningless, the dense leg
would return plausible-looking nonsense for those documents, and no error would appear anywhere.

So an outage raises, the job retries with backoff, and the index stays internally consistent.
A slow ingest is recoverable; a poisoned index is a rebuild.

Two more properties follow from the same reasoning:

**The embedder id comes from the server, never from configuration.** It enters the generation
fingerprint, so it has to describe what actually produced the vectors. A replica running an older
image would otherwise stamp the configured id onto vectors from a different checkpoint, and the
fingerprint -- the one mechanism that makes wrong-generation documents invisible -- would be
lying.

**A mid-run change of embedder id is fatal.** If the service starts answering with a different
id than it did at startup, a deployment is mid-rollout with two versions serving. Continuing
would interleave two vector spaces in one index, so the client refuses.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

import httpx

logger = logging.getLogger(__name__)

#: Texts per request. Bounded by the service's own ``EMBED_MAX_BATCH``; sending far more just
#: moves the queuing from HTTP into the server's loop while making a timeout lose more work.
DEFAULT_BATCH = 32

#: Generous, because a cold replica warming two models can take a while and an ingest worker has
#: no user waiting on it. The query path passes a much tighter timeout.
DEFAULT_TIMEOUT_S = 60.0


class EmbeddingUnavailableError(Exception):
    """The embedding service could not be reached, or refused.

    Deliberately *not* recoverable by falling back. See the module docstring: a fallback would
    write a second vector space into one index.
    """


class EmbedderIdentityChangedError(Exception):
    """The service is serving a different checkpoint than it was at startup.

    Means a rollout is in progress with two image versions live. Continuing would interleave two
    vector spaces in one index, which the generation fingerprint exists to make impossible.
    """


class OnnxEmbedder:
    """Embeds through the models service. Implements the ``Embedder`` protocol."""

    normalized = True

    def __init__(
        self,
        base_url: str,
        *,
        embedder_id: str,
        dimension: int,
        max_tokens: int = 512,
        batch_size: int = DEFAULT_BATCH,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        max_retries: int = 3,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        #: Discovered from the service by ``probe``, never configured. Part of the generation
        #: fingerprint.
        self.id = embedder_id
        self.dimension = dimension
        self.max_tokens = max_tokens
        self.batch_size = batch_size
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self._client = client

    # -- Embedder protocol -------------------------------------------------------------------

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return await self._embed(list(texts), kind="passage")

    async def embed_query(self, text: str) -> list[float]:
        """A query, embedded with the query prefix the model family expects.

        The asymmetry is trained into every model family worth using, and omitting it does not
        error -- it silently costs several points of recall. The server applies the prefix from
        the card baked next to the weights, so this client never has to know which family is
        deployed.
        """
        vectors = await self._embed([text], kind="query")
        return vectors[0]

    # ----------------------------------------------------------------------------------------

    async def _embed(self, texts: list[str], *, kind: str) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            vectors.extend(await self._one_batch(batch, kind=kind))
        return vectors

    async def _one_batch(self, texts: list[str], *, kind: str) -> list[list[float]]:
        payload = {"texts": texts, "kind": kind, "max_length": self.max_tokens}
        last: Exception | None = None

        for attempt in range(self.max_retries + 1):
            if attempt:
                # A warming replica or a momentary 503 clears in seconds. This is a short,
                # bounded wait -- the job queue owns the long backoff.
                await asyncio.sleep(min(0.5 * 2**attempt, 4.0))
            try:
                body = await self._post(payload)
            except httpx.HTTPError as exc:
                last = exc
                continue

            returned = str(body.get("embedder_id", ""))
            if returned and returned != self.id:
                # Never retried: a second attempt would hit the same mixed fleet, and the whole
                # point is to stop before writing into the index.
                raise EmbedderIdentityChangedError(
                    f"the embedding service is serving {returned!r} but this run started with "
                    f"{self.id!r}; a rollout is in progress and continuing would mix two vector "
                    "spaces in one index"
                )

            vectors = body.get("vectors") or []
            if len(vectors) != len(texts):
                last = EmbeddingUnavailableError(f"asked for {len(texts)} vectors and got {len(vectors)}")
                continue

            if body.get("truncated"):
                # Worth a warning rather than silence: a truncated passage is a permanent recall
                # loss for that chunk, and a corpus that trips this constantly means the chunker
                # cap and the embedder cap disagree.
                logger.warning(
                    "embed.truncated",
                    extra={"count": body["truncated"], "of": len(texts), "max_tokens": self.max_tokens},
                )

            return [[float(value) for value in vector] for vector in vectors]

        raise EmbeddingUnavailableError(
            f"the embedding service at {self.base_url} failed after {self.max_retries + 1} attempts: {last}"
        )

    async def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._client is not None:
            response = await self._client.post(f"{self.base_url}/embed", json=payload, timeout=self.timeout_s)
        else:
            async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                response = await client.post(f"{self.base_url}/embed", json=payload)

        if response.status_code == 503:
            # No embedder baked into the image. Named distinctly because the fix is a rebuild,
            # not a restart, and an operator reading a retry storm deserves to be told that.
            raise httpx.HTTPStatusError(
                "the models service has no embedding model loaded; rebuild the image with EMBEDDING_MODEL set",
                request=response.request,
                response=response,
            )
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result


async def probe(base_url: str, *, client: httpx.AsyncClient | None = None, timeout_s: float = 10.0) -> dict[str, Any]:
    """Ask the service which embedder it is serving.

    Called once at startup, because the embedder id and dimension are properties of the deployed
    artefact rather than of our configuration -- and both enter the generation fingerprint. A
    client that trusted its own settings would stamp a plausible id onto vectors from whatever
    checkpoint happened to be running.
    """
    url = f"{base_url.rstrip('/')}/healthz"
    try:
        if client is not None:
            response = await client.get(url, timeout=timeout_s)
        else:
            async with httpx.AsyncClient(timeout=timeout_s) as owned:
                response = await owned.get(url)
        response.raise_for_status()
        body: dict[str, Any] = response.json()
    except httpx.HTTPError as exc:
        raise EmbeddingUnavailableError(f"could not reach the models service at {base_url}: {exc}") from exc

    if not body.get("embedder"):
        raise EmbeddingUnavailableError(
            f"the models service at {base_url} serves no embedding model; rebuild the image with "
            "EMBEDDING_MODEL set, or run with EMBEDDING_PROVIDER=hashing"
        )
    return body


async def build_onnx_embedder(
    base_url: str, *, client: httpx.AsyncClient | None = None, max_tokens: int = 512
) -> OnnxEmbedder:
    """Probe the service, then build a client pinned to what it is actually serving.

    Raises rather than falling back. A caller that wants the hashing embedder must ask for it by
    name -- silently substituting one here is how an index ends up holding two vector spaces.
    """
    health = await probe(base_url, client=client)
    embedder_id = str(health["embedder"])

    # bge-base-en-v1.5@768@int8 -> 768. Parsed rather than requested separately so the id and the
    # dimension cannot disagree.
    try:
        dimension = int(embedder_id.split("@")[1])
    except (IndexError, ValueError) as exc:
        raise EmbeddingUnavailableError(
            f"the models service reported an embedder id this client cannot parse: {embedder_id!r}"
        ) from exc

    logger.info("embedder.connected", extra={"id": embedder_id, "dimension": dimension, "url": base_url})
    return OnnxEmbedder(
        base_url,
        embedder_id=embedder_id,
        dimension=dimension,
        max_tokens=max_tokens,
        client=client,
    )
