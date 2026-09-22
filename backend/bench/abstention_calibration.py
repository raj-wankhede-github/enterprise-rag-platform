"""Calibrate the evidence gate's coverage threshold against the golden set.

    uv run python bench/abstention_calibration.py

The threshold decides when the system says "I don't know". Picking it by argument is how a RAG
system ends up either inventing answers or refusing to answer anything; this script picks it from
the separation actually observed between answerable and unanswerable questions.

The errors are asymmetric and the output reflects that. Over-abstaining is visible to the user,
annoying, and fixed by rephrasing. Under-abstaining produces a confident wrong answer that nobody
notices until it matters -- and in a compliance-adjacent product, once. So the recommendation
prefers the threshold that drives ``made_up`` to zero, and reports what that costs in refused
answerable questions rather than hiding it.
"""

from __future__ import annotations

import asyncio
import sys

from opensearchpy import AsyncOpenSearch

from app.answer.evidence import EvidenceThresholds, assess
from app.core.config import get_settings
from app.evals.runner import EVAL_TENANT, EvalRunner
from app.retrieval.hybrid import OpenSearchHybridRetriever
from app.retrieval.types import RetrievalRequest, TenantScope
from app.search.client import SearchClient

CANDIDATE_THRESHOLDS = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0]


async def run() -> int:
    settings = get_settings()
    client = AsyncOpenSearch(hosts=[settings.opensearch_url], timeout=60)
    try:
        if not await client.ping():
            print(f"OpenSearch not reachable at {settings.opensearch_url}", file=sys.stderr)
            return 2

        runner = EvalRunner(client)
        index = await runner.build_index(contextual=True)
        try:
            retriever = OpenSearchHybridRetriever(
                SearchClient(client),
                chunk_index=index["chunk_index"],
                parent_index=index["parent_index"],
                embedder=runner.embedder,
            )

            # Coverage is computed once per question; only the threshold varies, so retrieval
            # runs once rather than once per candidate threshold.
            observed: list[tuple[str, bool, float, bool]] = []
            for question in runner.questions:
                outcome = await retriever.retrieve(
                    RetrievalRequest(
                        query=question.question,
                        scope=TenantScope(
                            tenant_id=EVAL_TENANT,
                            visibility_rank=question.rank,
                            access_groups=(),
                            generation_fingerprint=index["fingerprint"],
                            include_superseded=question.include_superseded,
                        ),
                        top_k=100,
                    )
                )
                # A threshold of 0 isolates the coverage number from the other gates, so the
                # sweep measures the coverage decision rather than the whole assessment.
                probe = assess(
                    question.question,
                    outcome.candidates,
                    thresholds=EvidenceThresholds(min_term_coverage=0.0),
                )
                # Whether the other gates (no results, weak scores, identifiers, supersession)
                # already abstained. Those fire regardless of the coverage threshold.
                gated_elsewhere = not probe.sufficient
                observed.append((question.id, question.answerable, probe.term_coverage, gated_elsewhere))

            answerable = [row for row in observed if row[1]]
            unanswerable = [row for row in observed if not row[1]]

            print(f"{len(answerable)} answerable, {len(unanswerable)} unanswerable")
            print()
            print("coverage distribution")
            for label, rows in (("answerable", answerable), ("unanswerable", unanswerable)):
                values = sorted(row[2] for row in rows)
                if not values:
                    continue
                print(
                    f"  {label:<14} min={values[0]:.2f}  p25={values[len(values) // 4]:.2f}  "
                    f"median={values[len(values) // 2]:.2f}  max={values[-1]:.2f}"
                )
            print()
            print(f"{'threshold':>9}  {'refused_ok':>10}  {'made_up':>8}  {'wrongly_refused':>16}")
            print(f"{'-' * 9}  {'-' * 10}  {'-' * 8}  {'-' * 16}")

            def abstains(row: tuple[str, bool, float, bool], threshold: float) -> bool:
                """True when this question would be refused at the given threshold.

                ``row[3]`` is whether another gate already abstained; those fire regardless of
                the coverage threshold and must count as refusals at every point in the sweep.
                """
                return row[3] or row[2] < threshold

            for threshold in CANDIDATE_THRESHOLDS:
                refused_ok = sum(1 for row in unanswerable if abstains(row, threshold))
                made_up = len(unanswerable) - refused_ok
                wrongly_refused = sum(1 for row in answerable if abstains(row, threshold))
                marker = "  <- drives made_up to zero" if made_up == 0 else ""
                print(f"{threshold:>9.2f}  {refused_ok:>10}  {made_up:>8}  {wrongly_refused:>16}{marker}")

            print()
            print("questions that no coverage threshold can separate:")
            for question_id, is_answerable, coverage, gated in observed:
                if not is_answerable and not gated and coverage >= 1.0:
                    print(
                        f"  {question_id}: unanswerable at full coverage -- every topic term is "
                        "present and only the value asked for is absent. Coverage cannot see "
                        "this; citation verification has to."
                    )
        finally:
            await runner.drop_index(index)
        return 0
    finally:
        await client.close()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(run()))
