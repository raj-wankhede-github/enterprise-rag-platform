"""Retrieval value objects.

``TenantScope`` is the security-relevant one: it is required (never ``None``) on every request,
and ``search/dsl.py`` turns it into the filter clauses that every leg must carry.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Protocol

from app.security.principal import Principal


@dataclass(frozen=True, slots=True)
class TenantScope:
    """Everything needed to constrain a search to what one principal may see."""

    tenant_id: uuid.UUID
    visibility_rank: int
    access_groups: tuple[str, ...]
    generation_fingerprint: str
    user_id: uuid.UUID | None = None
    include_superseded: bool = False
    as_of: date | None = None  # effective-date filtering; None => today

    @classmethod
    def from_principal(cls, principal: Principal, *, generation_fingerprint: str, **kwargs: Any) -> TenantScope:
        return cls(
            tenant_id=principal.tenant_id,
            visibility_rank=principal.rank,
            access_groups=tuple(sorted(principal.group_ids)),
            generation_fingerprint=generation_fingerprint,
            user_id=principal.user_id,
            **kwargs,
        )


@dataclass(frozen=True, slots=True)
class MetadataFilter:
    doc_types: tuple[str, ...] = ()
    source_systems: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()
    doc_ids: tuple[str, ...] = ()
    collection_ids: tuple[str, ...] = ()
    languages: tuple[str, ...] = ()
    published_after: date | None = None
    published_before: date | None = None

    def is_empty(self) -> bool:
        return self == MetadataFilter()


@dataclass(frozen=True, slots=True)
class LegHit:
    rank: int  # 1-based
    score: float


@dataclass(frozen=True, slots=True)
class ChunkMeta:
    doc_type: str | None = None
    source_system: str | None = None
    source_uri: str | None = None
    author: str | None = None
    language: str | None = None
    publication_date: date | None = None
    effective_from: date | None = None
    effective_to: date | None = None
    is_superseded: bool = False
    visibility_rank: int = 10
    authority_rank: float = 0.0
    page_from: int | None = None
    page_to: int | None = None
    token_count: int = 0


@dataclass(slots=True)
class Candidate:
    chunk_id: str
    parent_id: str
    doc_id: str
    doc_version_id: str
    text: str
    title: str = ""
    context_line: str | None = None
    heading_path: str | None = None
    meta: ChunkMeta = field(default_factory=ChunkMeta)
    legs: dict[str, LegHit] = field(default_factory=dict)
    fused_score: float = 0.0
    prior_score: float | None = None
    rerank_score: float | None = None

    @property
    def final_score(self) -> float:
        if self.rerank_score is not None:
            return self.rerank_score
        if self.prior_score is not None:
            return self.prior_score
        return self.fused_score

    def rank_in(self, leg: str) -> int | None:
        hit = self.legs.get(leg)
        return hit.rank if hit else None

    def debug(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "legs": {name: {"rank": h.rank, "score": h.score} for name, h in sorted(self.legs.items())},
            "fused_score": round(self.fused_score, 6),
            "prior_score": None if self.prior_score is None else round(self.prior_score, 6),
            "rerank_score": None if self.rerank_score is None else round(self.rerank_score, 6),
        }


@dataclass(frozen=True, slots=True)
class RetrievalProfile:
    name: str = "default"
    leg_weights: dict[str, float] = field(
        default_factory=lambda: {"bm25": 1.0, "exact": 1.0, "dense": 1.0, "parent": 1.0}
    )
    leg_sizes: dict[str, int] = field(default_factory=lambda: {"bm25": 200, "exact": 50, "dense": 150, "parent": 50})
    rrf_k: int = 60
    ef_search: int = 128
    oversample_factor: float = 1.0
    rerank_enabled: bool = True
    rerank_top_n: int = 24
    fusion_top_k: int = 50
    context_token_budget: int = 6000
    version: int = 1


DEFAULT_PROFILE = RetrievalProfile()


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    query: str
    scope: TenantScope
    filters: MetadataFilter = MetadataFilter()
    profile: RetrievalProfile = DEFAULT_PROFILE
    sub_queries: tuple[str, ...] = ()
    exact_tokens: tuple[str, ...] = ()
    #: The embedded query. None disables the dense leg, which is how the fast path and the
    #: bm25-only ablation row are expressed without a separate code path.
    embedding: tuple[float, ...] | None = None
    top_k: int = 50
    #: Legs to run. None means "every leg that reports itself enabled"; an explicit set is what
    #: the ablation runner uses to produce one row per configuration.
    legs: frozenset[str] | None = None


@dataclass(slots=True)
class LegDiagnostics:
    name: str
    hits: int
    took_ms: float
    size: int
    #: Set when the leg came back with an error. A failed leg degrades recall and is recorded
    #: here rather than raised, so the retrieval debugger shows it instead of it being invisible.
    error: str | None = None


@dataclass(slots=True)
class RetrievalDiagnostics:
    legs: list[LegDiagnostics] = field(default_factory=list)
    total_ms: float = 0.0
    fast_path: bool = False
    fast_path_fellthrough: bool = False
    rerank_status: str = "skipped"
    rerank_ms: float = 0.0
    indices: list[str] = field(default_factory=list)
    raw_dsl: list[dict[str, Any]] | None = None  # debug endpoint only


@dataclass(slots=True)
class RetrievalOutcome:
    candidates: list[Candidate]
    diagnostics: RetrievalDiagnostics = field(default_factory=RetrievalDiagnostics)


class Retriever(Protocol):
    name: str

    async def retrieve(self, req: RetrievalRequest) -> RetrievalOutcome: ...
