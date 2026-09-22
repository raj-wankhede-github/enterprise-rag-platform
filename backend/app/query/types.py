"""What the planner produces.

One rule governs this whole subsystem: **rules always run and are authoritative; the LLM may add
but never remove.** A model asked to "understand" a query will occasionally decide that
``TKT-99812`` is not an identifier, that a date filter is unnecessary, or that an injection
attempt is benign. Each of those is a silent, unreproducible failure. Deterministic extraction
runs first and its output cannot be overridden, so the worst a model can do is fail to add value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum


class Route(StrEnum):
    """Which pipeline a query takes."""

    #: Short, contains an identifier, no conversational history. Skips the LLM, the dense leg and
    #: the reranker entirely. In a real deployment this is 15-30% of traffic, and it is exactly
    #: the traffic where the full pipeline is both slowest and *worst* -- dense retrieval is
    #: unreliable on opaque codes.
    FAST_EXACT = "fast_exact"
    #: Everything else.
    STANDARD = "standard"
    #: A multi-part question, split into sub-queries that are retrieved separately and fused.
    DECOMPOSED = "decomposed"


@dataclass(frozen=True, slots=True)
class DateRange:
    after: date | None = None
    before: date | None = None

    def is_empty(self) -> bool:
        return self.after is None and self.before is None


@dataclass(frozen=True, slots=True)
class ExtractedFilters:
    """Filters pulled out of the query text.

    Separate from ``MetadataFilter`` because these are *proposals* from parsing, which the API
    layer reconciles with filters the user set explicitly in the UI. An explicit filter always
    wins: a user who ticked "policies only" did not ask the parser's opinion.
    """

    doc_types: tuple[str, ...] = ()
    source_systems: tuple[str, ...] = ()
    authors: tuple[str, ...] = ()
    published: DateRange = DateRange()
    #: True when the question is explicitly about the past, which is the only way superseded
    #: documents become retrievable.
    historical: bool = False

    def is_empty(self) -> bool:
        return (
            not (self.doc_types or self.source_systems or self.authors or self.historical) and self.published.is_empty()
        )


@dataclass(frozen=True, slots=True)
class QueryPlan:
    raw: str
    #: The query after follow-up resolution. Equal to ``raw`` when there is no history.
    standalone: str
    route: Route
    #: Identifiers, extracted by rules only. Never removable by a model.
    exact_tokens: tuple[str, ...] = ()
    filters: ExtractedFilters = ExtractedFilters()
    sub_queries: tuple[str, ...] = ()
    #: Rules-only, non-removable. An injection attempt in a query must not be reasoned away.
    injection_flags: tuple[str, ...] = ()
    needs_clarification: bool = False
    clarification: str | None = None
    #: Which analyser produced this plan, for the trace.
    analyzer: str = "rules"
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_fast(self) -> bool:
        return self.route is Route.FAST_EXACT

    @property
    def uses_dense(self) -> bool:
        """The fast path skips the dense leg: embeddings are unreliable on opaque identifiers."""
        return self.route is not Route.FAST_EXACT

    @property
    def uses_reranker(self) -> bool:
        return self.route is not Route.FAST_EXACT

    def legs(self) -> frozenset[str]:
        if self.route is Route.FAST_EXACT:
            return frozenset({"exact"})
        return frozenset({"bm25", "exact", "dense", "parent"})
