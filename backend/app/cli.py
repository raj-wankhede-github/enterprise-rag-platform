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
from app.evals import (
    DEFAULT_ABLATIONS,
    AblationConfig,
    ConfigReport,
    EvalRunner,
    check,
    load_thresholds,
    render_details,
    render_table,
    to_baseline,
)


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

    args = parser.parse_args(argv)
    if args.command == "eval":
        return _evaluate(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
