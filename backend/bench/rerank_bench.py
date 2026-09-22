"""Measure reranking latency against the budget it is supposed to hold.

    uv run python bench/rerank_bench.py [--url http://localhost:8002] [--pairs 24]

This file exists because the 100-300 ms budget is held by four levers at once -- top-24 rather
than top-50, 288-token truncation, int8 quantization, one batched call -- and relaxing any single
one silently triples p95. Someone raises ``rerank_top_n`` to 50 for a recall experiment, or the
chunker's ``child_max`` drifts up, or a model is swapped for a better one. Nothing in a
functional test suite notices; the first report is a user saying search got slow.

The output is deliberately a table of levers rather than a single number, so the person who blew
the budget can see which lever did it.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass

import httpx

#: The shipping budget, in milliseconds at p95.
BUDGET_MS = 300.0

PASSAGE = (
    "Employees travelling on company business may claim a per diem allowance under reference "
    "SEC-4.2.1. The allowance depends on destination grade and is paid without receipts. Claims "
    "are submitted within thirty days of return through the expense portal and approved by a "
    "line manager. Where a claim is rejected the employee may escalate to their line manager "
    "within ten working days of the rejection notice, and the line manager consults the finance "
    "business partner before the matter passes to the head of finance. "
)
QUERY = "what is the per diem allowance for a grade A destination"


@dataclass(frozen=True, slots=True)
class Measurement:
    pairs: int
    max_length: int
    p50: float
    p95: float
    within_budget: bool


async def measure(client: httpx.AsyncClient, url: str, *, pairs: int, max_length: int, runs: int) -> Measurement:
    passages = [{"id": f"c{i}", "text": (PASSAGE * (1 + i % 3))[: 200 * (1 + i % 4)]} for i in range(pairs)]
    payload = {"query": QUERY, "passages": passages, "max_length": max_length, "top_n": 10}

    # One untimed call: the first inference at a new input shape specialises the ONNX session
    # and is several times slower than the rest. Including it would measure warmup, not serving.
    await client.post(f"{url}/rerank", json=payload, timeout=60.0)

    timings: list[float] = []
    for _ in range(runs):
        started = time.perf_counter()
        response = await client.post(f"{url}/rerank", json=payload, timeout=60.0)
        response.raise_for_status()
        timings.append((time.perf_counter() - started) * 1000.0)

    timings.sort()
    p95 = timings[min(len(timings) - 1, int(0.95 * len(timings)))]
    return Measurement(
        pairs=pairs,
        max_length=max_length,
        p50=statistics.median(timings),
        p95=p95,
        within_budget=p95 <= BUDGET_MS,
    )


async def run(url: str, runs: int) -> int:
    async with httpx.AsyncClient() as client:
        try:
            health = await client.get(f"{url}/healthz", timeout=10.0)
        except httpx.HTTPError as exc:
            print(f"models service not reachable at {url}: {exc}", file=sys.stderr)
            print("start it with:  docker compose --profile models up -d", file=sys.stderr)
            return 2
        if health.status_code != 200:
            print(f"models service is not ready: {health.status_code} {health.text}", file=sys.stderr)
            return 2

        print(f"model: {health.json().get('model')}")
        print(f"budget: p95 <= {BUDGET_MS:.0f} ms")
        print()
        print(f"{'pairs':>6}  {'max_len':>8}  {'p50 ms':>8}  {'p95 ms':>8}  verdict")
        print(f"{'-' * 6}  {'-' * 8}  {'-' * 8}  {'-' * 8}  -------")

        # The shipping configuration first, then each lever relaxed one at a time, so a blown
        # budget can be attributed rather than merely observed.
        grid = [(24, 288), (24, 512), (50, 288), (50, 512), (100, 288)]
        shipping: Measurement | None = None
        for pairs, max_length in grid:
            result = await measure(client, url, pairs=pairs, max_length=max_length, runs=runs)
            if shipping is None:
                shipping = result
            verdict = "ok" if result.within_budget else "OVER BUDGET"
            marker = "  <- shipping config" if (pairs, max_length) == (24, 288) else ""
            print(
                f"{result.pairs:>6}  {result.max_length:>8}  {result.p50:>8.1f}  {result.p95:>8.1f}  {verdict}{marker}"
            )

        print()
        if shipping is None:
            return 2
        if not shipping.within_budget:
            print(
                f"FAIL: the shipping configuration is at {shipping.p95:.0f} ms p95, over the "
                f"{BUDGET_MS:.0f} ms budget.",
                file=sys.stderr,
            )
            print(
                "Drop rerank_top_n, drop rerank_max_tokens, or move to the 'fast' model tier "
                "before accepting a slower pipeline.",
                file=sys.stderr,
            )
            return 1
        print(f"shipping configuration is within budget at {shipping.p95:.0f} ms p95")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8002")
    parser.add_argument("--runs", type=int, default=15)
    args = parser.parse_args()
    return asyncio.run(run(args.url.rstrip("/"), args.runs))


if __name__ == "__main__":
    raise SystemExit(main())
