"""Generation fingerprinting and index naming.

Five independent things change what retrieval returns: the embedding model, the chunker, the
contextualizer prompt, the mapping, and the analysis chain. Any one of them differing between
two documents in the same index makes those documents incomparable -- and the failure is silent,
because a blended index still returns plausible results. Relevance drops a few points and nobody
notices for a quarter.

So all five collapse into one ``generation_fingerprint``, stamped on every indexed chunk and
asserted as a term filter in **every** query (see ``search/dsl.py``). Wrong-generation documents
are invisible rather than blended, which converts that silent degradation into a loud, testable
symptom: a stalled backfill shows up as missing results in a shadow evaluation.

Naming
------
``chunks_g{generation}_p{pool:03d}``   concrete index
``chunks_read_p{pool:03d}``            read alias, may point at several indices during a rebuild
``chunks_write_p{pool:03d}``           write alias, exactly one index has is_write_index

The application only ever talks to aliases. That is what makes promoting a large tenant to a
dedicated index, or swapping to a new generation, a zero-code-change operation.
"""

from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from app.search.mappings import analysis_digest, mapping_digest

CHUNK_PREFIX: Final[str] = "chunks"
PARENT_PREFIX: Final[str] = "parents"


class GenerationState(StrEnum):
    """PLANNED -> BUILDING -> BACKFILLED -> VERIFYING -> LIVE -> DRAINING -> RETIRED."""

    PLANNED = "PLANNED"
    BUILDING = "BUILDING"
    BACKFILLED = "BACKFILLED"
    VERIFYING = "VERIFYING"
    LIVE = "LIVE"
    DRAINING = "DRAINING"
    RETIRED = "RETIRED"


#: Legal transitions. LIVE is reachable only from VERIFYING -- there is deliberately no path
#: that skips the shadow evaluation, because "just promote it, the backfill looked fine" is
#: exactly how a half-built index reaches production.
_TRANSITIONS: Final[dict[GenerationState, frozenset[GenerationState]]] = {
    GenerationState.PLANNED: frozenset({GenerationState.BUILDING, GenerationState.RETIRED}),
    GenerationState.BUILDING: frozenset({GenerationState.BACKFILLED, GenerationState.RETIRED}),
    GenerationState.BACKFILLED: frozenset({GenerationState.VERIFYING, GenerationState.RETIRED}),
    # Back to BUILDING when verification fails and more backfill is needed.
    GenerationState.VERIFYING: frozenset({GenerationState.LIVE, GenerationState.BUILDING, GenerationState.RETIRED}),
    GenerationState.LIVE: frozenset({GenerationState.DRAINING}),
    # DRAINING can go back to LIVE: that is the rollback, and it is one alias call.
    GenerationState.DRAINING: frozenset({GenerationState.RETIRED, GenerationState.LIVE}),
    GenerationState.RETIRED: frozenset(),
}


def can_transition(current: GenerationState, target: GenerationState) -> bool:
    return target in _TRANSITIONS[current]


def assert_transition(current: GenerationState, target: GenerationState) -> None:
    if not can_transition(current, target):
        raise ValueError(
            f"Illegal generation transition {current} -> {target}. "
            f"Allowed from {current}: {sorted(_TRANSITIONS[current])}."
        )


@dataclass(frozen=True, slots=True)
class GenerationSpec:
    """Everything that makes two indexed chunks comparable."""

    embedder_id: str
    chunker_version: str
    contextualizer_version: str
    dimension: int
    engine: str = "lucene"
    quantize: bool = False

    @property
    def mapping_sha(self) -> str:
        return mapping_digest(
            dimension=self.dimension,
            engine=self.engine,  # type: ignore[arg-type]
            quantize=self.quantize,
        )

    @property
    def analysis_sha(self) -> str:
        return analysis_digest()

    @property
    def fingerprint(self) -> str:
        return compute_fingerprint(
            embedder_id=self.embedder_id,
            chunker_version=self.chunker_version,
            contextualizer_version=self.contextualizer_version,
            mapping_sha=self.mapping_sha,
            analysis_sha=self.analysis_sha,
        )


def compute_fingerprint(
    *,
    embedder_id: str,
    chunker_version: str,
    contextualizer_version: str,
    mapping_sha: str,
    analysis_sha: str,
) -> str:
    """A 16-hex digest over the five inputs.

    Field-delimited with a separator that cannot occur in the inputs, so that
    ``("ab", "c")`` and ``("a", "bc")`` cannot collide into the same fingerprint.
    """
    parts = (embedder_id, chunker_version, contextualizer_version, mapping_sha, analysis_sha)
    for part in parts:
        if "\x1f" in part:
            raise ValueError("Fingerprint inputs must not contain the unit-separator character")
    payload = "\x1f".join(parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def pool_for_tenant(tenant_id: uuid.UUID, *, pool_count: int) -> int:
    """Assign a tenant to a pool by consistent hash.

    Computed once at provisioning and then *recorded* in ``tenant_index_bindings``; never
    recomputed at query time. Changing ``pool_count`` would otherwise silently re-point existing
    tenants at a pool that does not hold their chunks, which reads as "all my documents
    disappeared".

    Uses the UUID bytes rather than ``hash()``, whose randomized seed differs per process.
    """
    if pool_count < 1:
        raise ValueError("pool_count must be at least 1")
    digest = hashlib.sha256(tenant_id.bytes).digest()
    return int.from_bytes(digest[:4], "big") % pool_count


def chunk_index_name(*, generation: int, pool: int) -> str:
    return f"{CHUNK_PREFIX}_g{generation}_p{pool:03d}"


def parent_index_name(*, generation: int, pool: int) -> str:
    return f"{PARENT_PREFIX}_g{generation}_p{pool:03d}"


def chunk_read_alias(pool: int) -> str:
    return f"{CHUNK_PREFIX}_read_p{pool:03d}"


def chunk_write_alias(pool: int) -> str:
    return f"{CHUNK_PREFIX}_write_p{pool:03d}"


def parent_read_alias(pool: int) -> str:
    return f"{PARENT_PREFIX}_read_p{pool:03d}"


def parent_write_alias(pool: int) -> str:
    return f"{PARENT_PREFIX}_write_p{pool:03d}"


AliasAction = dict[str, dict[str, object]]


def alias_add_actions(*, generation: int, pool: int) -> list[AliasAction]:
    """Point one pool's four aliases at one generation."""
    chunks = chunk_index_name(generation=generation, pool=pool)
    parents = parent_index_name(generation=generation, pool=pool)
    return [
        {"add": {"index": chunks, "alias": chunk_read_alias(pool)}},
        {"add": {"index": chunks, "alias": chunk_write_alias(pool), "is_write_index": True}},
        {"add": {"index": parents, "alias": parent_read_alias(pool)}},
        {"add": {"index": parents, "alias": parent_write_alias(pool), "is_write_index": True}},
    ]


def alias_remove_actions(*, generation: int, pool: int) -> list[AliasAction]:
    """Detach one pool's four aliases from one generation."""
    chunks = chunk_index_name(generation=generation, pool=pool)
    parents = parent_index_name(generation=generation, pool=pool)
    return [
        {"remove": {"index": chunks, "alias": chunk_read_alias(pool)}},
        {"remove": {"index": chunks, "alias": chunk_write_alias(pool)}},
        {"remove": {"index": parents, "alias": parent_read_alias(pool)}},
        {"remove": {"index": parents, "alias": parent_write_alias(pool)}},
    ]


def promote_actions(*, from_generation: int, to_generation: int, pool: int) -> list[AliasAction]:
    """Phase one of a swap: the new generation starts serving, the old one keeps serving.

    The read alias is **added** to, not moved. For the length of the drain window it spans both
    generations, and that is the whole point rather than an oversight.

    Every query carries a ``generation_fingerprint`` term filter, and a reader learns the new
    fingerprint from its own next refresh -- not at the instant the alias moves. A single-phase
    swap therefore leaves a window in which a reader still holding the old fingerprint queries
    an alias that now resolves only to the new index: the query is valid, returns HTTP 200, and
    matches nothing. No error is logged anywhere and the user sees "no results" for their own
    documents. Reversing the order just moves the window to the other side.

    With both generations readable, a reader holding either fingerprint finds its own documents
    and the window closes on its own. The integration test drives continuous traffic through a
    swap to hold this property; it is the reason that test exists.

    The *write* alias still moves atomically and exactly once -- OpenSearch permits only one
    ``is_write_index`` per alias, and two generations accepting writes is a different and worse
    problem. Removes precede adds because actions apply in order within the one request.
    """
    chunks_from = chunk_index_name(generation=from_generation, pool=pool)
    parents_from = parent_index_name(generation=from_generation, pool=pool)
    chunks_to = chunk_index_name(generation=to_generation, pool=pool)
    parents_to = parent_index_name(generation=to_generation, pool=pool)

    return [
        # Writes move wholesale.
        {"remove": {"index": chunks_from, "alias": chunk_write_alias(pool)}},
        {"remove": {"index": parents_from, "alias": parent_write_alias(pool)}},
        {"add": {"index": chunks_to, "alias": chunk_write_alias(pool), "is_write_index": True}},
        {"add": {"index": parents_to, "alias": parent_write_alias(pool), "is_write_index": True}},
        # Reads widen. The old generation is removed later, by retire_read_actions.
        {"add": {"index": chunks_to, "alias": chunk_read_alias(pool)}},
        {"add": {"index": parents_to, "alias": parent_read_alias(pool)}},
    ]


def retire_read_actions(*, generation: int, pool: int) -> list[AliasAction]:
    """Phase two: stop reading the old generation, once every reader has moved on.

    Separated from promotion by the drain window. Doing both at once is the bug described in
    ``promote_actions``; doing this too early reintroduces it.
    """
    return [
        {"remove": {"index": chunk_index_name(generation=generation, pool=pool), "alias": chunk_read_alias(pool)}},
        {"remove": {"index": parent_index_name(generation=generation, pool=pool), "alias": parent_read_alias(pool)}},
    ]


def rollback_actions(*, live_generation: int, back_to: int, pool: int) -> list[AliasAction]:
    """Undo a promotion during the drain window.

    Cheap precisely because the old generation never stopped being readable: only the write
    alias has to move back, and the new generation leaves the read alias. One call, no rebuild --
    which is what makes rolling back something a team will actually do under pressure.
    """
    chunks_live = chunk_index_name(generation=live_generation, pool=pool)
    parents_live = parent_index_name(generation=live_generation, pool=pool)
    chunks_back = chunk_index_name(generation=back_to, pool=pool)
    parents_back = parent_index_name(generation=back_to, pool=pool)

    return [
        {"remove": {"index": chunks_live, "alias": chunk_write_alias(pool)}},
        {"remove": {"index": parents_live, "alias": parent_write_alias(pool)}},
        {"add": {"index": chunks_back, "alias": chunk_write_alias(pool), "is_write_index": True}},
        {"add": {"index": parents_back, "alias": parent_write_alias(pool), "is_write_index": True}},
        {"remove": {"index": chunks_live, "alias": chunk_read_alias(pool)}},
        {"remove": {"index": parents_live, "alias": parent_read_alias(pool)}},
    ]


def chunk_document_id(*, tenant_id: uuid.UUID, doc_version_id: uuid.UUID, ordinal: int) -> str:
    """Deterministic ``_id`` for a chunk.

    Deterministic so that re-indexing is idempotent -- a retried bulk request overwrites rather
    than duplicating -- and so citations stay stable for a given document version.
    """
    payload = f"{tenant_id}\x1f{doc_version_id}\x1f{ordinal}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]
