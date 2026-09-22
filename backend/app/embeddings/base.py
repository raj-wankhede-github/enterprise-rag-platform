"""The embedder contract.

``Embedder.id`` is not cosmetic: it goes into the generation fingerprint, so it must change
whenever anything about the produced vectors changes -- model, revision, dimension, normalization,
or the query/passage prefix convention. Two embedders that produce different vectors and share an
id will silently blend incompatible geometry into one index.

Query and passage embedding are separate methods because asymmetric models (E5, BGE, Nomic) need
different prefixes for each, and using the passage prefix for a query costs several points of
recall for no visible error.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class Embedder(Protocol):
    #: Participates in the generation fingerprint. Convention: ``name@dimension[@variant]``.
    id: str
    dimension: int
    #: True when vectors are unit length, which lets the index use innerproduct as cosine.
    normalized: bool
    max_tokens: int

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def embed_query(self, text: str) -> list[float]: ...


def l2_normalize(vector: list[float]) -> list[float]:
    """Scale to unit length.

    The index uses ``space_type: innerproduct``, which equals cosine only for unit vectors. A
    non-normalized vector would make longer documents score higher purely for being longer --
    a bug that looks like a relevance problem rather than a normalization one.
    """
    norm = sum(value * value for value in vector) ** 0.5
    if norm == 0.0:
        # A zero vector has no direction. Returning it unchanged keeps the contract total; the
        # caller sees a vector that matches nothing rather than a NaN that poisons the index.
        return vector
    return [value / norm for value in vector]
