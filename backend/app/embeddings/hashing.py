"""A deterministic, dependency-free embedder.

This is the CI and offline default, and it is what makes the ablation table runnable on every
pull request: no API key, no model download, no network, byte-identical output across runs and
machines.

It is a hashed bag-of-words projection with sub-word character n-grams -- essentially the
"hashing trick". It has **no semantic understanding whatsoever**: it will match paraphrases only
by shared vocabulary. That is deliberate and worth being honest about. Its job is to exercise
the plumbing and to give the ablation table a *stable floor*, not to represent what the dense leg
contributes in production. A recall number measured with this embedder says something about the
pipeline, not about embedding quality.

Because its output is part of the generation fingerprint, changing the projection here forces a
reindex -- which is correct, since every stored vector would otherwise be from a different space.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence
from typing import Final

from app.embeddings.base import l2_normalize
from app.utils.text import tokens

#: Character n-gram width. 3 and 4 together give some robustness to inflection and typos without
#: exploding the number of features per token.
_NGRAM_SIZES: Final[tuple[int, ...]] = (3, 4)

#: Bumped whenever the projection changes, so ``id`` changes and a rebuild is forced.
_VARIANT: Final[str] = "v1"


class HashingEmbedder:
    """Signed hashing projection with L2 normalization and sub-linear term weighting."""

    normalized = True
    max_tokens = 100_000

    def __init__(self, dimension: int = 384) -> None:
        if dimension < 8:
            raise ValueError("dimension must be at least 8")
        self.dimension = dimension
        self.id = f"hashing@{dimension}@{_VARIANT}"

    # -- Embedder protocol -------------------------------------------------------------------

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        return [self.embed(text) for text in texts]

    async def embed_query(self, text: str) -> list[float]:
        # Symmetric: there is no passage/query prefix to apply, because there is no model.
        return self.embed(text)

    # -- implementation ----------------------------------------------------------------------

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        counts: dict[str, int] = {}
        for token in tokens(text):
            counts[token] = counts.get(token, 0) + 1
            for size in _NGRAM_SIZES:
                padded = f"^{token}$"
                for index in range(max(1, len(padded) - size + 1)):
                    gram = padded[index : index + size]
                    counts[gram] = counts.get(gram, 0) + 1

        for feature, count in counts.items():
            # Sub-linear weighting: a term repeated twenty times should not dominate a vector.
            weight = 1.0 + math.log(count)
            bucket, sign = self._bucket(feature)
            vector[bucket] += sign * weight

        return l2_normalize(vector)

    def _bucket(self, feature: str) -> tuple[int, float]:
        """Map a feature to a bucket and a sign.

        The sign is the standard signed-hashing trick: it makes collisions cancel in expectation
        rather than accumulate, which keeps an unrelated pair of documents from drifting together
        purely because two of their terms landed in the same bucket.
        """
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        value = int.from_bytes(digest, "big")
        return value % self.dimension, 1.0 if value >> 63 & 1 else -1.0
