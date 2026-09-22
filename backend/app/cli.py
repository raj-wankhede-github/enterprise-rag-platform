"""Command line entry point.

``python -m app.cli eval --ablate`` is the same code path CI runs and the same one
``pytest -m evals`` invokes, so a developer, a pull request and a release all see identical
numbers. That matters more than it sounds: an evaluation people cannot reproduce locally is an
evaluation they learn to ignore.

The async half does only async work. Rendering, file writes and gating happen synchronously in
``main`` -- partly because blocking I/O inside a coroutine is a bad habit to normalise, and
partly because it keeps the part that decides whether the build fails trivially testable.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from opensearchpy import AsyncOpenSearch

from app.core.config import get_settings
from app.db.session import build_sessionmaker
from app.evals import (
    DEFAULT_ABLATIONS,
    SHIPPING_ABLATION,
    AblationConfig,
    ConfigReport,
    EvalRunner,
    check,
    load_thresholds,
    metric_values,
    render_details,
    render_table,
    to_baseline,
)
from app.ingestion.chunker import CHUNKER_VERSION
from app.search.admin import IndexAdmin
from app.search.backfill_source import PostgresChunkSource
from app.search.generations import GenerationSpec
from app.search.rebuild import RebuildOrchestrator, ShadowEvaluator, VerificationReport


async def _collect(configs: Sequence[AblationConfig]) -> list[ConfigReport]:
    settings = get_settings()
    client = AsyncOpenSearch(hosts=[settings.opensearch_url], timeout=60)
    try:
        if not await client.ping():
            raise ConnectionError(f"OpenSearch not reachable at {settings.opensearch_url}")

        runner = EvalRunner(client)
        reports: list[ConfigReport] = []

        # Configs sharing an indexing regime share an index. Contextual retrieval is the only
        # switch that changes what is stored, so the default table builds two indices, not four.
        for contextual in (False, True):
            group = [config for config in configs if config.contextual is contextual]
            if not group:
                continue
            index = await runner.build_index(contextual=contextual)
            try:
                for config in group:
                    reports.append(await runner.run_config(config, index))
            finally:
                await runner.drop_index(index)

        order = {config.name: position for position, config in enumerate(configs)}
        reports.sort(key=lambda report: order.get(report.name, 99))
        return reports
    finally:
        await client.close()


def _evaluate(args: argparse.Namespace) -> int:
    configs = (
        DEFAULT_ABLATIONS if args.ablate else tuple(c for c in DEFAULT_ABLATIONS if c.name == "hybrid_rrf + contextual")
    )

    try:
        reports = asyncio.run(_collect(configs))
    except ConnectionError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    print()
    print(render_table(reports))
    print()
    print(render_details(reports))
    print()

    if args.json_out:
        args.json_out.write_text(json.dumps([report.as_row() for report in reports], indent=2), encoding="utf-8")
        print(f"wrote {args.json_out}")

    if args.write_baseline:
        args.write_baseline.write_text(json.dumps(to_baseline(reports), indent=2), encoding="utf-8")
        print(f"wrote baseline {args.write_baseline}")

    if not args.fail_under_thresholds:
        return 0

    baseline = None
    if args.baseline and args.baseline.exists():
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))

    failures = check(reports, load_thresholds(), baseline=baseline)
    if failures:
        print("EVALUATION GATE FAILED", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        return 1

    print("evaluation gate passed")
    return 0


async def _run_rebuild(args: argparse.Namespace) -> VerificationReport:
    """Drive one generation from PLANNED to LIVE against the configured cluster."""
    settings = get_settings()
    client = AsyncOpenSearch(hosts=[settings.opensearch_url], timeout=120)
    try:
        spec = GenerationSpec(
            embedder_id=args.embedder_id,
            chunker_version=args.chunker_version,
            contextualizer_version=args.contextualizer_version,
            dimension=args.dimension,
        )
        runner = RebuildOrchestrator(
            admin=IndexAdmin(client, shards=settings.opensearch_shards_per_pool, replicas=settings.opensearch_replicas),
            source=PostgresChunkSource(
                build_sessionmaker(settings),
                generation=args.generation,
                fingerprint=spec.fingerprint,
                embedder_id=args.embedder_id,
                contextualizer_version=args.contextualizer_version,
                # A rebuild that changes the embedder cannot reuse persisted vectors, so the
                # caller supplies them instead of this source demanding them.
                require_vectors=not args.reembed,
            ),
            client=client,
            spec=spec,
            generation=args.generation,
            pools=list(range(settings.opensearch_pool_count)),
            previous_generation=args.previous,
        )
        baseline = json.loads(args.baseline.read_text(encoding="utf-8")) if args.baseline else None
        return await runner.run(evaluate=None if args.no_eval else _shadow_evaluator(client), baseline=baseline)
    finally:
        await client.close()


def _shadow_evaluator(client: AsyncOpenSearch) -> ShadowEvaluator:
    """Run the eval harness against one fingerprint, with the alias still pointing at the old
    generation. This is the gate that catches a backfill which completed and is wrong."""

    async def evaluate(fingerprint: str) -> dict[str, float]:
        shipping = [config for config in DEFAULT_ABLATIONS if config.name == SHIPPING_ABLATION]
        reports = await _collect(shipping)
        return metric_values(reports[0]) if reports else {}

    return evaluate


def _rebuild(args: argparse.Namespace) -> int:
    report = asyncio.run(_run_rebuild(args))
    print(json.dumps(report.as_json(), indent=2))
    if report.passed:
        print(f"generation {args.generation} is LIVE")
        return 0
    print("REBUILD REFUSED", file=sys.stderr)
    for failure in report.failures:
        print(f"  {failure}", file=sys.stderr)
    return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="app.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    evaluate = sub.add_parser("eval", help="run the retrieval evaluation")
    evaluate.add_argument(
        "--ablate",
        action="store_true",
        help="run every ablation configuration rather than only the shipping one",
    )
    evaluate.add_argument(
        "--fail-under-thresholds",
        action="store_true",
        help="exit non-zero when a metric breaches its absolute floor or regresses",
    )
    evaluate.add_argument("--baseline", type=Path, help="baseline JSON to compare against")
    evaluate.add_argument("--write-baseline", type=Path, help="write the run out as a new baseline")
    evaluate.add_argument("--json", type=Path, dest="json_out", help="write the table as JSON")

    rebuild = sub.add_parser("rebuild", help="build a new index generation and swap to it")
    rebuild.add_argument("--generation", type=int, required=True, help="the new generation number")
    rebuild.add_argument("--previous", type=int, help="the generation currently live; omit to bootstrap")
    rebuild.add_argument("--embedder-id", required=True)
    rebuild.add_argument("--dimension", type=int, required=True)
    rebuild.add_argument("--chunker-version", default=CHUNKER_VERSION)
    rebuild.add_argument("--contextualizer-version", default="t1")
    rebuild.add_argument("--baseline", type=Path, help="previous generation metrics to compare against")
    rebuild.add_argument(
        "--reembed",
        action="store_true",
        help="the embedder changed, so persisted vectors cannot be reused",
    )
    rebuild.add_argument(
        "--no-eval",
        action="store_true",
        help="skip the shadow evaluation. Promotion is then refused -- this is for dry runs only.",
    )

    args = parser.parse_args(argv)
    if args.command == "eval":
        return _evaluate(args)
    if args.command == "rebuild":
        return _rebuild(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
