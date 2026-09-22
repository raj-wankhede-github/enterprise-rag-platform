"""The answering pipeline.

    retrieve -> gate -> assemble -> generate -> verify -> answer or abstain

Two gates, deliberately on either side of the generator, because they catch different things:

* the **evidence gate** runs first and is cheap. It stops a model ever seeing evidence too thin
  to answer from, which is the only reliable way to stop it being fluent about nothing.
* **verification** runs last and is the backstop. It catches what the first gate provably cannot:
  evidence that looks sufficient because every topic term is present, while the specific value
  asked for is absent.

Exactly one stricter retry between them. The retry is given only the evidence that survived
verification -- handing back the context that produced an ungrounded draft mostly produces
another one -- and there is no second retry, because at that point the latency is being spent on
an answer that has already failed twice.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from app.answer.abstain import abstain
from app.answer.assemble import AssembledContext, AssemblyConfig, assemble
from app.answer.evidence import EvidenceThresholds, assess
from app.answer.generator import AnswerGenerator
from app.answer.types import AbstentionReason, Answer, NearMiss
from app.answer.verify import (
    EntailmentScorer,
    VerificationConfig,
    VerificationResult,
    verify,
)
from app.retrieval.types import Candidate, RetrievalOutcome, RetrievalRequest


@dataclass(slots=True)
class AnswerTrace:
    """Per-stage record. Persisted as the product's own trace, independent of any ops tool."""

    question: str
    retrieved: int = 0
    assessed_sufficient: bool = False
    abstention_reason: str | None = None
    blocks: int = 0
    context_tokens: int = 0
    dropped_duplicates: int = 0
    dropped_for_budget: int = 0
    dropped_for_diversity: int = 0
    generated: bool = False
    retried: bool = False
    citation_support: float = 1.0
    verification_failure: str | None = None
    stage_ms: dict[str, float] = field(default_factory=dict)

    @property
    def total_ms(self) -> float:
        return sum(self.stage_ms.values())


@dataclass(slots=True)
class AnswerOutcome:
    answer: Answer
    context: AssembledContext
    trace: AnswerTrace
    verification: VerificationResult | None = None


class AnswerPipeline:
    def __init__(
        self,
        *,
        retriever: Any,
        generator: AnswerGenerator,
        parent_fetcher: Any | None = None,
        assembly: AssemblyConfig | None = None,
        thresholds: EvidenceThresholds | None = None,
        verification: VerificationConfig | None = None,
        scorer: EntailmentScorer | None = None,
        allow_partial: bool = False,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.parent_fetcher = parent_fetcher
        self.assembly = assembly or AssemblyConfig()
        self.thresholds = thresholds or EvidenceThresholds()
        self.verification = verification or VerificationConfig()
        self.scorer = scorer
        self.allow_partial = allow_partial

    async def answer(self, request: RetrievalRequest) -> AnswerOutcome:
        trace = AnswerTrace(question=request.query)

        outcome = await self._timed(trace, "retrieve", self.retriever.retrieve(request))
        trace.retrieved = len(outcome.candidates)

        assessment = assess(request.query, outcome.candidates, thresholds=self.thresholds)
        trace.assessed_sufficient = assessment.sufficient
        if not assessment.sufficient:
            trace.abstention_reason = assessment.reason.value if assessment.reason else None
            return AnswerOutcome(
                answer=abstain(
                    assessment.reason or AbstentionReason.NO_RESULTS,
                    near_misses=assessment.near_misses,
                ),
                context=AssembledContext(),
                trace=trace,
            )

        context = await self._assemble(trace, assessment.supporting)
        trace.blocks = len(context.blocks)
        trace.context_tokens = context.total_tokens
        trace.dropped_duplicates = context.dropped_duplicates
        trace.dropped_for_budget = context.dropped_for_budget
        trace.dropped_for_diversity = context.dropped_for_diversity

        if not context.blocks:
            # Everything was deduplicated or truncated away. Rare, but returning an answer with
            # no citable evidence would violate the core invariant.
            trace.abstention_reason = AbstentionReason.WEAK_EVIDENCE.value
            return AnswerOutcome(
                answer=abstain(
                    AbstentionReason.WEAK_EVIDENCE,
                    near_misses=_near_misses(assessment.supporting),
                ),
                context=context,
                trace=trace,
            )

        answer, result = await self._generate_and_verify(trace, request.query, context)

        if result is not None:
            trace.citation_support = result.citation_support

        if answer.answerable and result is not None and not result.ok:
            trace.verification_failure = result.failure_summary()
            if self.allow_partial:
                from app.answer.verify import strip_unsupported

                answer = strip_unsupported(answer, result)
            else:
                answer = abstain(
                    AbstentionReason.UNGROUNDED_DRAFT,
                    near_misses=_near_misses(assessment.supporting),
                )
            trace.abstention_reason = answer.reason.value if not answer.answerable and answer.reason else None

        return AnswerOutcome(answer=answer, context=context, trace=trace, verification=result)

    # -- stages ------------------------------------------------------------------------

    async def _assemble(self, trace: AnswerTrace, candidates: Any) -> AssembledContext:
        started = time.perf_counter()
        parents: dict[str, str] = {}
        if self.parent_fetcher is not None and self.assembly.expand_parents:
            parent_ids = list(dict.fromkeys(c.parent_id for c in candidates if c.parent_id))
            try:
                parents = await self.parent_fetcher.fetch(parent_ids)
            except Exception:
                parents = {}
        context = assemble(candidates, parent_texts=parents, config=self.assembly)
        trace.stage_ms["assemble"] = (time.perf_counter() - started) * 1000.0
        return context

    async def _generate_and_verify(
        self, trace: AnswerTrace, question: str, context: AssembledContext
    ) -> tuple[Answer, VerificationResult | None]:
        started = time.perf_counter()
        try:
            answer = await self.generator.generate(question, context)
            trace.generated = True
        except Exception:
            trace.stage_ms["generate"] = (time.perf_counter() - started) * 1000.0
            return abstain(AbstentionReason.UNGROUNDED_DRAFT), None
        trace.stage_ms["generate"] = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        result = await verify(answer, context, config=self.verification, scorer=self.scorer)
        trace.stage_ms["verify"] = (time.perf_counter() - started) * 1000.0

        if result.ok or not answer.answerable:
            return answer, result

        # One stricter retry, on the evidence that survived.
        from app.answer.verify import surviving_blocks

        survivors = surviving_blocks(result, context)
        if not survivors:
            return answer, result

        narrowed = AssembledContext(blocks=survivors, total_tokens=sum(block.token_count for block in survivors))
        started = time.perf_counter()
        try:
            retried = await self.generator.generate(question, narrowed, strict=True)
        except Exception:
            trace.stage_ms["retry"] = (time.perf_counter() - started) * 1000.0
            return answer, result
        trace.retried = True
        trace.stage_ms["retry"] = (time.perf_counter() - started) * 1000.0

        retried_result = await verify(retried, narrowed, config=self.verification, scorer=self.scorer)
        if retried_result.ok:
            return retried, retried_result
        return answer, result

    async def _timed(self, trace: AnswerTrace, stage: str, awaitable: Any) -> RetrievalOutcome:
        started = time.perf_counter()
        outcome: RetrievalOutcome = await awaitable
        trace.stage_ms[stage] = (time.perf_counter() - started) * 1000.0
        return outcome


def _near_misses(candidates: Any, limit: int = 3) -> tuple[NearMiss, ...]:
    seen: dict[str, NearMiss] = {}
    for candidate in candidates:
        if candidate.doc_id in seen:
            continue
        seen[candidate.doc_id] = NearMiss(
            doc_id=candidate.doc_id,
            title=candidate.title or "Untitled document",
            heading_path=candidate.heading_path,
            score=candidate.final_score,
        )
        if len(seen) >= limit:
            break
    return tuple(seen.values())


class OpenSearchParentFetcher:
    """Fetches parent section text by id, for expansion."""

    def __init__(self, client: Any, *, index: str, routing: str | None = None) -> None:
        self.client = client
        self.index = index
        self.routing = routing

    async def fetch(self, parent_ids: list[str]) -> dict[str, str]:
        if not parent_ids:
            return {}
        docs = await self.client.mget(index=self.index, ids=parent_ids, routing=self.routing)
        return {
            str(doc["_source"].get("parent_id") or doc["_id"]): str(doc["_source"].get("content", ""))
            for doc in docs
            if doc.get("_source")
        }


def candidates_for_display(candidates: list[Candidate]) -> list[dict[str, Any]]:
    return [candidate.debug() for candidate in candidates]
