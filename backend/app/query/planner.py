"""Turning a raw query into a plan.

Rules run first and unconditionally. The LLM runs only when rules say it could add something --
a follow-up to resolve, a multi-part question to split, a fuzzy date to interpret -- and its
output is merged so that it can add but never remove. That asymmetry is the whole design: the
worst a model failure can do is leave the plan exactly as the rules produced it.

The fast path is the other half. A short query containing an identifier and no conversational
history skips the LLM, the dense leg and the reranker, because for that shape the full pipeline
is both the slowest option and the worst one.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date
from typing import Any, Protocol, runtime_checkable

from app.query import rules
from app.query.types import ExtractedFilters, QueryPlan, Route

#: A fast-path query must be at most this many words. Above it, an identifier is usually one
#: clause of a real question ("what does SEC-4.2.1 say about retention") which deserves the
#: full pipeline.
FAST_PATH_MAX_WORDS: int = 6


@runtime_checkable
class QueryUnderstanding(Protocol):
    """One call doing all four jobs: rewrite, filters, decomposition, route suggestion.

    Four separate calls would be clearer and would also put four model round trips in the
    latency budget of every non-trivial query, which is not a budget that exists.
    """

    version: str

    async def analyse(self, query: str, history: list[str], *, rules_plan: QueryPlan) -> dict[str, Any]: ...


class QueryPlanner:
    def __init__(
        self,
        understanding: QueryUnderstanding | None = None,
        *,
        fast_path_enabled: bool = True,
        max_sub_queries: int = 3,
    ) -> None:
        self.understanding = understanding
        self.fast_path_enabled = fast_path_enabled
        self.max_sub_queries = max_sub_queries

    async def plan(self, query: str, *, history: list[str] | None = None, today: date | None = None) -> QueryPlan:
        past = history or []
        plan = self._rules_plan(query, history=past, today=today)

        if self.understanding is None or not self._should_ask_model(plan, history=past):
            return plan

        try:
            payload = await self.understanding.analyse(query, past, rules_plan=plan)
        except Exception:
            # A model failure leaves the rules plan intact. Degrading to deterministic
            # understanding is a worse answer; failing the query is no answer.
            return replace(plan, notes=(*plan.notes, "llm_understanding_failed"))
        return self._merge(plan, payload)

    # -- rules ---------------------------------------------------------------------------

    def _rules_plan(self, query: str, *, history: list[str], today: date | None) -> QueryPlan:
        identifiers = rules.extract_identifiers(query)
        filters = rules.extract_filters(query, today=today)
        flags = rules.injection_flags(query)
        follow_up = rules.looks_like_follow_up(query, has_history=bool(history))

        route = Route.STANDARD
        if (
            self.fast_path_enabled
            and identifiers
            and not follow_up
            and rules.word_count(query) <= FAST_PATH_MAX_WORDS
            and not rules.is_interrogative(query)
        ):
            route = Route.FAST_EXACT
        elif rules.looks_multipart(query):
            route = Route.DECOMPOSED

        return QueryPlan(
            raw=query,
            standalone=query,
            route=route,
            exact_tokens=identifiers,
            filters=filters,
            injection_flags=flags,
            analyzer="rules",
            notes=("follow_up_detected",) if follow_up else (),
        )

    def _should_ask_model(self, plan: QueryPlan, *, history: list[str]) -> bool:
        """Only when the model could add something the rules could not."""
        if plan.route is Route.FAST_EXACT:
            return False
        if "follow_up_detected" in plan.notes:
            return True
        if plan.route is Route.DECOMPOSED:
            return True
        # A fuzzy temporal reference the date rules did not resolve.
        return plan.filters.published.is_empty() and _mentions_time(plan.raw)

    # -- merging -------------------------------------------------------------------------

    def _merge(self, plan: QueryPlan, payload: dict[str, Any]) -> QueryPlan:
        """Add, never remove.

        Every field here is a union or a fallback. The model cannot clear an identifier, drop an
        injection flag, or narrow a date range the rules established -- it can only supply what
        was missing.
        """
        standalone = str(payload.get("standalone") or "").strip() or plan.standalone

        model_ids = tuple(str(item) for item in payload.get("exact_tokens", ()) if str(item).strip())
        identifiers = tuple(dict.fromkeys((*plan.exact_tokens, *model_ids)))

        filters = plan.filters
        model_filters = payload.get("filters")
        if isinstance(model_filters, dict):
            filters = ExtractedFilters(
                doc_types=tuple(dict.fromkeys((*filters.doc_types, *_strings(model_filters.get("doc_types"))))),
                source_systems=tuple(
                    dict.fromkeys((*filters.source_systems, *_strings(model_filters.get("source_systems"))))
                ),
                authors=tuple(dict.fromkeys((*filters.authors, *_strings(model_filters.get("authors"))))),
                # A rules-derived range came from an explicit date in the text and wins.
                published=filters.published,
                # Historical intent can be added but never cleared -- clearing it would make
                # superseded documents invisible to a question explicitly about them.
                historical=filters.historical or bool(model_filters.get("historical")),
            )

        sub_queries = tuple(_strings(payload.get("sub_queries")))[: self.max_sub_queries]
        route = plan.route
        if len(sub_queries) > 1:
            route = Route.DECOMPOSED
        elif route is Route.DECOMPOSED and not sub_queries:
            # The model looked and found one question after all.
            route = Route.STANDARD

        return QueryPlan(
            raw=plan.raw,
            standalone=standalone,
            route=route,
            exact_tokens=identifiers,
            filters=filters,
            sub_queries=sub_queries,
            # Never removable, whatever the model concluded.
            injection_flags=plan.injection_flags,
            needs_clarification=bool(payload.get("needs_clarification")),
            clarification=payload.get("clarification"),
            analyzer="rules+llm",
            notes=plan.notes,
        )


def _strings(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item).strip().lower() for item in value if str(item).strip())


_TIME_WORDS = (
    "recent",
    "recently",
    "latest",
    "current",
    "now",
    "today",
    "yesterday",
    "ago",
    "since",
    "until",
    "last",
    "next",
    "this year",
    "upcoming",
)


def _mentions_time(text: str) -> bool:
    lowered = text.lower()
    return any(word in lowered for word in _TIME_WORDS)
