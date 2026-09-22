"""The retrieval-leg contract.

A leg is one ranked list. Keeping them separate -- rather than folding the exact-token matching
into the BM25 query as a field boost -- is what lets the ablation table attribute recall to a
specific mechanism and drop anything that does not earn its latency.

Every leg gets its filter clauses from ``search/dsl.build_filter`` and from nowhere else. A leg
that builds its own filter is a leg that can forget the tenant term.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from app.retrieval.types import Candidate, ChunkMeta, LegHit, RetrievalRequest

#: An ``_msearch`` entry: the header line and the body line.
MsearchItem = tuple[dict[str, Any], dict[str, Any]]


class RetrievalLeg(Protocol):
    name: str

    def enabled(self, request: RetrievalRequest) -> bool: ...

    def build(self, request: RetrievalRequest, *, index: str) -> MsearchItem: ...


#: Fields every leg asks for. The vector is excluded by the mapping, so this is cheap.
SOURCE_FIELDS: tuple[str, ...] = (
    "tenant_id",
    "doc_id",
    "doc_version_id",
    "parent_id",
    "chunk_id",
    "ordinal",
    "content",
    "context_line",
    "title",
    "heading_path",
    "visibility_rank",
    "doc_type",
    "source_system",
    "source_uri",
    "author",
    "language",
    "publication_date",
    "effective_from",
    "effective_to",
    "is_superseded",
    "authority_rank",
    "page_from",
    "page_to",
    "token_count",
    "content_sha256",
    "simhash64",
)


def _as_date(value: Any) -> Any:
    """Dates arrive as ISO strings. Parsing is deferred to whoever needs a date object."""
    return value


def candidate_from_hit(hit: dict[str, Any], *, leg: str, rank: int) -> Candidate:
    """Turn one search hit into a candidate, recording which leg found it and where.

    The per-leg rank is retained rather than collapsed into a score, because it is what RRF
    consumes and what the retrieval debugger shows when someone asks why a chunk surfaced.
    """
    source: dict[str, Any] = hit.get("_source", {})
    meta = ChunkMeta(
        doc_type=source.get("doc_type"),
        source_system=source.get("source_system"),
        source_uri=source.get("source_uri"),
        author=source.get("author"),
        language=source.get("language"),
        publication_date=_as_date(source.get("publication_date")),
        effective_from=_as_date(source.get("effective_from")),
        effective_to=_as_date(source.get("effective_to")),
        is_superseded=bool(source.get("is_superseded", False)),
        visibility_rank=int(source.get("visibility_rank", 10)),
        authority_rank=float(source.get("authority_rank") or 0.0),
        page_from=source.get("page_from"),
        page_to=source.get("page_to"),
        token_count=int(source.get("token_count") or 0),
    )
    return Candidate(
        chunk_id=str(source.get("chunk_id") or hit.get("_id", "")),
        parent_id=str(source.get("parent_id") or ""),
        doc_id=str(source.get("doc_id") or ""),
        doc_version_id=str(source.get("doc_version_id") or ""),
        text=str(source.get("content") or ""),
        title=str(source.get("title") or ""),
        context_line=source.get("context_line"),
        heading_path=source.get("heading_path"),
        meta=meta,
        legs={leg: LegHit(rank=rank, score=float(hit.get("_score") or 0.0))},
    )


def parse_hits(response: dict[str, Any], *, leg: str) -> list[Candidate]:
    """Read one msearch response into candidates.

    A response carrying ``error`` yields an empty list rather than raising: one degraded leg
    costs recall, a raised exception costs the whole answer. The caller records the failure in
    the diagnostics so it is visible rather than silent.
    """
    if "error" in response:
        return []
    hits: Sequence[dict[str, Any]] = response.get("hits", {}).get("hits", [])
    return [candidate_from_hit(hit, leg=leg, rank=index + 1) for index, hit in enumerate(hits)]


def response_error(response: dict[str, Any]) -> str | None:
    error = response.get("error")
    if error is None:
        return None
    if isinstance(error, dict):
        return str(error.get("type") or error.get("reason") or error)
    return str(error)
