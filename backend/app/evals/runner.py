"""The evaluation runner.

Ingests the authored corpus into a throwaway index, runs every configuration against the golden
set, and reports per-stage metrics. One code path, invoked identically from ``pytest -m evals``,
from ``python -m app.cli eval`` and from CI -- so the number in a pull-request comment is the
same number a developer sees locally.

Each ablation row is the *real* retriever with legs switched off, never a second implementation.
A separate "bm25_only" code path would drift from the shipping one and the table would slowly
start describing a system nobody runs.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from opensearchpy import AsyncOpenSearch

from app.answer.assemble import AssemblyConfig
from app.answer.evidence import EvidenceThresholds, assess
from app.answer.generator import ExtractiveAnswerGenerator
from app.answer.pipeline import AnswerPipeline, OpenSearchParentFetcher
from app.embeddings.hashing import HashingEmbedder
from app.evals.dataset import (
    CorpusDocument,
    GoldenQuestion,
    IndexedChunk,
    load_manifest,
    load_questions,
    resolve,
)
from app.evals.metrics import (
    leg_contribution,
    mean,
    ndcg_at_k,
    percentile,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)
from app.ingestion.chunker import StructureAwareChunker
from app.ingestion.contextualize import DocumentContext, TemplateContextualizer
from app.ingestion.loaders import default_registry
from app.ingestion.pipeline import IngestionPipeline, index_action
from app.query.planner import QueryPlanner
from app.query.types import QueryPlan, Route
from app.retrieval.hybrid import OpenSearchHybridRetriever
from app.retrieval.rerank.base import IdentityReranker, Reranker
from app.retrieval.rerank.lexical import LexicalEntailmentScorer, LexicalReranker
from app.retrieval.types import RetrievalProfile, RetrievalRequest, TenantScope
from app.search.client import SearchClient
from app.search.generations import chunk_document_id
from app.search.mappings import chunk_index_body, parent_index_body

EVAL_TENANT = uuid.UUID("e0000000-0000-0000-0000-0000000e7a15")
EVAL_DIM = 256


@dataclass(frozen=True, slots=True)
class AblationConfig:
    """One row of the table.

    ``contextual`` toggles whether the context line is prepended before embedding and indexing,
    which is the only reason a config needs its own index -- everything else is a query-time
    switch and can share one.
    """

    name: str
    legs: frozenset[str]
    contextual: bool = True
    rerank: bool = False
    leg_weights: dict[str, float] | None = None
    #: Which reranker to apply after fusion. "identity" is the control arm: any nDCG movement
    #: against it is attributable to reranking, because nothing else differs between the rows.
    reranker: str = "identity"
    #: Run the full answer path (assemble, generate, verify) rather than stopping at the
    #: evidence gate. The gate alone cannot catch a value absent from evidence that otherwise
    #: looks sufficient; verification can, so the two produce different made_up rates and the
    #: table should show both.
    answer: bool = False
    #: Let the query planner choose the route, including the fast path. Its own row because
    #: the fast path's claim is about latency, and a latency claim needs its own measurement.
    use_planner: bool = False
    #: Enable the entailment layer of citation verification. Its own row because it is the
    #: stage predicted to catch "evidence quoted faithfully that does not answer the question",
    #: and a prediction deserves its own measurement rather than being folded into another.
    entailment: bool = False

    @property
    def needs_own_index(self) -> bool:
        return not self.contextual


DEFAULT_ABLATIONS: tuple[AblationConfig, ...] = (
    AblationConfig("bm25_only", legs=frozenset({"bm25"}), contextual=False),
    AblationConfig("dense_only", legs=frozenset({"dense"}), contextual=False),
    AblationConfig("hybrid_rrf", legs=frozenset({"bm25", "exact", "dense", "parent"}), contextual=False),
    AblationConfig("hybrid_rrf + contextual", legs=frozenset({"bm25", "exact", "dense", "parent"}), contextual=True),
    AblationConfig(
        "hybrid + contextual + rerank",
        legs=frozenset({"bm25", "exact", "dense", "parent"}),
        contextual=True,
        reranker="lexical",
    ),
    AblationConfig(
        "hybrid + contextual + verified",
        legs=frozenset({"bm25", "exact", "dense", "parent"}),
        contextual=True,
        answer=True,
    ),
    AblationConfig(
        "planner (fast path on)",
        legs=frozenset({"bm25", "exact", "dense", "parent"}),
        contextual=True,
        reranker="lexical",
        use_planner=True,
    ),
    AblationConfig(
        "hybrid + rerank + verified + entail",
        legs=frozenset({"bm25", "exact", "dense", "parent"}),
        contextual=True,
        reranker="lexical",
        answer=True,
        entailment=True,
    ),
)


@dataclass(slots=True)
class QuestionResult:
    question_id: str
    category: str
    retrieved: list[str] = field(default_factory=list)
    relevant: dict[str, int] = field(default_factory=dict)
    per_leg: dict[str, list[str]] = field(default_factory=dict)
    unresolved_labels: int = 0
    latency_ms: float = 0.0
    abstained: bool = False
    abstention_reason: str | None = None
    leaked_chunk_ids: list[str] = field(default_factory=list)
    citation_support: float = 1.0
    answered_text: str = ""


@dataclass(slots=True)
class ConfigReport:
    name: str
    questions: int
    #: Total chunks in the index under test. Reported because it decides which recall cutoff is
    #: meaningful: recall@k discriminates only while k is a small fraction of the corpus.
    corpus_chunks: int
    recall_at_10: float
    recall_at_50: float
    recall_at_100: float
    ndcg_at_10: float
    mrr_at_10: float
    precision_at_5: float
    abstention_precision: float
    abstention_recall: float
    unsupported_answer_rate: float
    permission_leak_rate: float
    citation_support: float
    p50_ms: float
    p95_ms: float
    fast_path_questions: int = 0
    leg_contribution: dict[str, int] = field(default_factory=dict)
    per_category_recall: dict[str, float] = field(default_factory=dict)
    unresolved_labels: int = 0

    def as_row(self) -> dict[str, Any]:
        return {
            "config": self.name,
            "recall@10": round(self.recall_at_10, 4),
            "recall@50": round(self.recall_at_50, 4),
            "ndcg@10": round(self.ndcg_at_10, 4),
            "mrr@10": round(self.mrr_at_10, 4),
            "abstention_recall": round(self.abstention_recall, 4),
            "unsupported_answer_rate": round(self.unsupported_answer_rate, 4),
            "citation_support": round(self.citation_support, 4),
            "permission_leak_rate": round(self.permission_leak_rate, 4),
            "p95_ms": round(self.p95_ms, 1),
        }


class EvalRunner:
    def __init__(
        self,
        client: AsyncOpenSearch,
        *,
        documents: Sequence[CorpusDocument] | None = None,
        questions: Sequence[GoldenQuestion] | None = None,
        thresholds: EvidenceThresholds | None = None,
        dimension: int = EVAL_DIM,
    ) -> None:
        self.client = client
        self.search = SearchClient(client)
        self.documents = list(documents if documents is not None else load_manifest())
        self.questions = list(questions if questions is not None else load_questions())
        self.thresholds = thresholds or EvidenceThresholds()
        self.dimension = dimension
        self.embedder = HashingEmbedder(dimension)

    # -- indexing ----------------------------------------------------------------------

    def _pipeline(self, contextual: bool) -> IngestionPipeline:
        return IngestionPipeline(
            registry=default_registry(),
            chunker=StructureAwareChunker(),
            # Turning contextual retrieval off is expressed as an empty context line rather than
            # a different pipeline, so the two rows differ in exactly one input.
            contextualizer=TemplateContextualizer() if contextual else _NullContextualizer(),
            embedder=self.embedder,
        )

    async def build_index(self, *, contextual: bool) -> dict[str, Any]:
        suffix = uuid.uuid4().hex[:8]
        chunk_index = f"eval-chunks-{suffix}"
        parent_index = f"eval-parents-{suffix}"
        await self.client.indices.create(
            index=chunk_index,
            body=chunk_index_body(dimension=self.dimension, shards=1, replicas=0, refresh_interval="1s"),
        )
        await self.client.indices.create(
            index=parent_index, body=parent_index_body(shards=1, replicas=0, refresh_interval="1s")
        )

        pipeline = self._pipeline(contextual)
        fingerprint = pipeline.generation.fingerprint
        chunks: list[IndexedChunk] = []
        lines: list[str] = []

        for document in self.documents:
            result = await pipeline.run(
                document.text.encode("utf-8"),
                document.filename,
                context=DocumentContext(
                    title=document.title,
                    doc_type=document.doc_type,
                    effective_from=document.effective_from,
                ),
            )
            doc_id, version_id = uuid.uuid4(), uuid.uuid4()
            parent_ids = {p.ordinal: f"{version_id}:{p.ordinal}" for p in result.parents}

            for item in result.prepared:
                chunk_id = chunk_document_id(
                    tenant_id=EVAL_TENANT, doc_version_id=version_id, ordinal=item.chunk.ordinal
                )
                body = index_action(
                    item,
                    tenant_id=EVAL_TENANT,
                    doc_id=doc_id,
                    doc_version_id=version_id,
                    parent_id=parent_ids[item.chunk.parent_ordinal],
                    chunk_id=chunk_id,
                    generation_fingerprint=fingerprint,
                    title=document.title,
                    visibility_rank=document.visibility_rank,
                    access_groups=[],
                )
                body["doc_type"] = document.doc_type
                body["authority_rank"] = document.authority_rank or None
                body["is_superseded"] = document.is_superseded
                if document.effective_from:
                    body["effective_from"] = document.effective_from.isoformat()
                if document.effective_to:
                    body["effective_to"] = document.effective_to.isoformat()
                lines.append(_bulk_header(chunk_index, chunk_id))
                lines.append(_dumps(body))
                chunks.append(IndexedChunk(chunk_id=chunk_id, doc_slug=document.slug, text=item.chunk.text))

            for parent in result.parents:
                parent_body = {
                    "tenant_id": str(EVAL_TENANT),
                    "doc_id": str(doc_id),
                    "doc_version_id": str(version_id),
                    "parent_id": parent_ids[parent.ordinal],
                    "parent_ordinal": parent.ordinal,
                    "content": parent.text,
                    "title": document.title,
                    "heading_path": parent.heading_path,
                    "generation_fingerprint": fingerprint,
                    "visibility_rank": document.visibility_rank,
                    "access_groups": ["*"],
                    "is_active": True,
                    "is_superseded": document.is_superseded,
                    "doc_type": document.doc_type,
                    "child_count": sum(1 for c in result.chunked.children if c.parent_ordinal == parent.ordinal),
                }
                lines.append(_bulk_header(parent_index, parent_ids[parent.ordinal]))
                lines.append(_dumps(parent_body))

        response = await self.client.bulk(body="\n".join(lines) + "\n", refresh=True)
        if response["errors"]:
            failures = [item for item in response["items"] if item["index"].get("error")]
            raise RuntimeError(f"eval corpus failed to index: {_dumps(failures[:3])}")

        return {
            "chunk_index": chunk_index,
            "parent_index": parent_index,
            "fingerprint": fingerprint,
            "chunks": chunks,
        }

    async def drop_index(self, index: dict[str, Any]) -> None:
        await self.client.indices.delete(index=index["chunk_index"], ignore=[404])
        await self.client.indices.delete(index=index["parent_index"], ignore=[404])

    # -- running -----------------------------------------------------------------------

    async def run_config(self, config: AblationConfig, index: dict[str, Any]) -> ConfigReport:
        retriever = OpenSearchHybridRetriever(
            self.search,
            chunk_index=index["chunk_index"],
            parent_index=index["parent_index"],
            embedder=self.embedder,
            reranker=_reranker_for(config),
        )
        profile = RetrievalProfile(leg_weights=config.leg_weights or RetrievalProfile().leg_weights)

        # Built for every config, used only when the row asks for the answer path. Sharing the
        # retriever instance matters: the answer arm must see exactly the candidates the
        # retrieval arm measured, or the two rows are not comparable.
        pipeline = AnswerPipeline(
            retriever=retriever,
            generator=ExtractiveAnswerGenerator(),
            parent_fetcher=OpenSearchParentFetcher(self.search, index=index["parent_index"], routing=str(EVAL_TENANT)),
            assembly=AssemblyConfig(token_budget=3000, max_blocks=6),
            thresholds=self.thresholds,
            scorer=LexicalEntailmentScorer() if config.entailment else None,
        )
        results: list[QuestionResult] = []
        fast_path_questions = 0

        for question in self.questions:
            resolved = resolve(question, index["chunks"])
            plan = await _plan_for_question(question)
            request = _request_for(question, index, config, profile, plan)
            if plan.route is Route.FAST_EXACT:
                fast_path_questions += 1

            started = time.perf_counter()
            outcome = await retriever.retrieve(request)
            latency = (time.perf_counter() - started) * 1000.0

            assessment = assess(question.question, outcome.candidates, thresholds=self.thresholds)
            abstained = not assessment.sufficient
            reason = assessment.reason.value if assessment.reason else None
            citation_support = 1.0
            answered_text = ""

            if config.answer:
                # The full path: assemble, generate, verify. An answer that fails verification
                # becomes an abstention, which is what should move the made-up rate.
                started = time.perf_counter()
                produced = await pipeline.answer(request)
                latency = (time.perf_counter() - started) * 1000.0
                abstained = not produced.answer.answerable
                reason = produced.answer.reason.value if produced.answer.reason else None
                citation_support = produced.trace.citation_support
                answered_text = produced.answer.text if produced.answer.answerable else ""

            # Anything above the asker's rank must never appear. This is a build-breaking metric,
            # so it is measured on the raw candidate list rather than on what survives assembly.
            leaked = [
                candidate.chunk_id for candidate in outcome.candidates if candidate.meta.visibility_rank > question.rank
            ]

            results.append(
                QuestionResult(
                    question_id=question.id,
                    category=question.category,
                    retrieved=[candidate.chunk_id for candidate in outcome.candidates],
                    relevant=resolved.relevant,
                    per_leg=_per_leg(outcome.candidates),
                    unresolved_labels=len(resolved.unresolved),
                    latency_ms=latency,
                    abstained=abstained,
                    abstention_reason=reason,
                    leaked_chunk_ids=leaked,
                    citation_support=citation_support,
                    answered_text=answered_text,
                )
            )

        report = _summarise(config.name, self.questions, results, len(index["chunks"]))
        report.fast_path_questions = fast_path_questions
        return report


def _reranker_for(config: AblationConfig) -> Reranker:
    return LexicalReranker() if config.reranker == "lexical" else IdentityReranker()


def _request_for(
    question: GoldenQuestion,
    index: dict[str, Any],
    config: AblationConfig,
    profile: RetrievalProfile,
    plan: QueryPlan | None = None,
) -> RetrievalRequest:
    """One request shape for both the retrieval-only and the full answer arms.

    Built once so the two arms cannot drift: a difference between their numbers must come from
    the answer path, not from a subtly different query.
    """
    return RetrievalRequest(
        query=question.question,
        scope=TenantScope(
            tenant_id=EVAL_TENANT,
            visibility_rank=question.rank,
            access_groups=(),
            generation_fingerprint=index["fingerprint"],
            include_superseded=question.include_superseded,
        ),
        profile=profile,
        exact_tokens=plan.exact_tokens if plan else (),
        # The fast-path row lets the planner choose the legs; every other row pins them, so the
        # ablation still isolates one variable at a time.
        legs=(plan.legs() if plan and config.use_planner else config.legs),
        top_k=100,
    )


class _NullContextualizer:
    """The no-contextual-retrieval arm of the ablation."""

    version = "none-1"

    async def contextualize(self, document: Any, chunks: Sequence[Any]) -> list[str]:
        return ["" for _ in chunks]


def _bulk_header(index: str, doc_id: str) -> str:
    return _dumps({"index": {"_index": index, "_id": doc_id}})


def _dumps(payload: Any) -> str:
    return json.dumps(payload, default=str)


#: One planner for the whole run. Stateless, so sharing it costs nothing and guarantees every
#: row sees identical query understanding.
_PLANNER = QueryPlanner()


async def _plan_for_question(question: GoldenQuestion) -> QueryPlan:
    """The real planner, not an approximation of it.

    Earlier this was a local regex that mimicked what the planner would eventually do. Measuring
    an approximation of the shipping path is how an evaluation drifts away from the system it
    claims to describe, so now that the planner exists the evaluation calls it.
    """
    return await _PLANNER.plan(question.question)


def _per_leg(candidates: Sequence[Any]) -> dict[str, list[str]]:
    per_leg: dict[str, list[str]] = {}
    for candidate in candidates:
        for leg in candidate.legs:
            per_leg.setdefault(leg, []).append(candidate.chunk_id)
    return per_leg


def _summarise(
    name: str,
    questions: Sequence[GoldenQuestion],
    results: Sequence[QuestionResult],
    corpus_chunks: int,
) -> ConfigReport:
    answerable = [r for r in results if any(g > 0 for g in r.relevant.values())]
    unanswerable_ids = {q.id for q in questions if not q.answerable}
    unanswerable = [r for r in results if r.question_id in unanswerable_ids]

    per_category: dict[str, list[float]] = {}
    for result in answerable:
        per_category.setdefault(result.category, []).append(recall_at_k(result.retrieved, result.relevant, 50))

    contribution: dict[str, int] = {}
    for result in answerable:
        for leg, count in leg_contribution(result.per_leg, result.relevant).items():
            contribution[leg] = contribution.get(leg, 0) + count

    abstained_correctly = sum(1 for r in unanswerable if r.abstained)
    abstained_total = sum(1 for r in results if r.abstained)

    return ConfigReport(
        name=name,
        questions=len(results),
        corpus_chunks=corpus_chunks,
        recall_at_10=mean([recall_at_k(r.retrieved, r.relevant, 10) for r in answerable]),
        recall_at_50=mean([recall_at_k(r.retrieved, r.relevant, 50) for r in answerable]),
        recall_at_100=mean([recall_at_k(r.retrieved, r.relevant, 100) for r in answerable]),
        ndcg_at_10=mean([ndcg_at_k(r.retrieved, r.relevant, 10) for r in answerable]),
        mrr_at_10=mean([reciprocal_rank(r.retrieved, r.relevant, 10) for r in answerable]),
        precision_at_5=mean([precision_at_k(r.retrieved, r.relevant, 5) for r in answerable]),
        # Of everything we refused to answer, how much genuinely had no answer.
        abstention_precision=(abstained_correctly / abstained_total) if abstained_total else 1.0,
        # Of everything with no answer, how much we actually refused. This is the one that
        # corresponds to "did not make something up".
        abstention_recall=(abstained_correctly / len(unanswerable)) if unanswerable else 1.0,
        unsupported_answer_rate=(
            (len(unanswerable) - abstained_correctly) / len(unanswerable) if unanswerable else 0.0
        ),
        permission_leak_rate=(sum(1 for r in results if r.leaked_chunk_ids) / len(results) if results else 0.0),
        citation_support=mean([r.citation_support for r in results if not r.abstained]),
        p50_ms=percentile([r.latency_ms for r in results], 0.50),
        p95_ms=percentile([r.latency_ms for r in results], 0.95),
        leg_contribution=contribution,
        per_category_recall={key: mean(values) for key, values in sorted(per_category.items())},
        unresolved_labels=sum(r.unresolved_labels for r in results),
    )
