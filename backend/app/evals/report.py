"""Rendering and gating the ablation table.

Two gates, because they catch different failures:

* **Absolute thresholds** catch a collapse -- someone breaks the analyzer and recall halves.
* **A relative gate against the merge-base baseline** catches the slow bleed, which is the one
  that actually happens. Nobody ships a change that halves recall; people ship a dozen changes
  that each cost half a point, and a year later the system is materially worse with every
  individual pull request having looked fine.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.evals.runner import ConfigReport

THRESHOLDS_PATH = Path(__file__).parent / "thresholds.json"

#: Metrics where a lower number is better, so the gate compares in the other direction.
LOWER_IS_BETTER: frozenset[str] = frozenset({"unsupported_answer_rate", "permission_leak_rate", "p50_ms", "p95_ms"})


@dataclass(frozen=True, slots=True)
class GateFailure:
    config: str
    metric: str
    observed: float
    limit: float
    kind: str  # "absolute" | "regression"

    def __str__(self) -> str:
        direction = "above" if self.metric in LOWER_IS_BETTER else "below"
        return (
            f"{self.config}: {self.metric} = {self.observed:.4f} is {direction} the {self.kind} limit {self.limit:.4f}"
        )


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


def check(
    reports: Sequence[ConfigReport],
    thresholds: dict[str, Any],
    *,
    baseline: dict[str, dict[str, float]] | None = None,
) -> list[GateFailure]:
    failures: list[GateFailure] = []
    absolute: dict[str, dict[str, float]] = thresholds.get("absolute", {})
    tolerance: float = float(thresholds.get("max_regression", 0.02))

    for report in reports:
        limits = absolute.get(report.name, {})
        values = _metric_values(report)

        for metric, limit in limits.items():
            observed = values.get(metric)
            if observed is None:
                continue
            breached = observed > limit if metric in LOWER_IS_BETTER else observed < limit
            if breached:
                failures.append(GateFailure(report.name, metric, observed, limit, "absolute"))

        if baseline and report.name in baseline:
            for metric, previous in baseline[report.name].items():
                observed = values.get(metric)
                if observed is None or metric in ("p50_ms", "p95_ms"):
                    # Latency is too machine-dependent for a regression gate; it has its own
                    # dedicated benchmark where the hardware is controlled.
                    continue
                if metric in LOWER_IS_BETTER:
                    limit = previous + tolerance
                    if observed > limit:
                        failures.append(GateFailure(report.name, metric, observed, limit, "regression"))
                else:
                    limit = previous - tolerance
                    if observed < limit:
                        failures.append(GateFailure(report.name, metric, observed, limit, "regression"))

    return failures


def _metric_values(report: ConfigReport) -> dict[str, float]:
    return {
        "recall@10": report.recall_at_10,
        "recall@50": report.recall_at_50,
        "recall@100": report.recall_at_100,
        "ndcg@10": report.ndcg_at_10,
        "mrr@10": report.mrr_at_10,
        "precision@5": report.precision_at_5,
        "abstention_precision": report.abstention_precision,
        "abstention_recall": report.abstention_recall,
        "unsupported_answer_rate": report.unsupported_answer_rate,
        "citation_support": report.citation_support,
        "permission_leak_rate": report.permission_leak_rate,
        "p50_ms": report.p50_ms,
        "p95_ms": report.p95_ms,
    }


def render_table(reports: Sequence[ConfigReport]) -> str:
    """The ablation table. This is the artefact -- for tuning and for a buyer."""
    headers = [
        "config",
        "recall@10",
        "recall@50",
        "nDCG@10",
        "MRR@10",
        "abstain",
        "made_up",
        "cite_ok",
        "leak",
        "p95ms",
    ]
    rows = [
        [
            report.name,
            f"{report.recall_at_10:.3f}",
            f"{report.recall_at_50:.3f}",
            f"{report.ndcg_at_10:.3f}",
            f"{report.mrr_at_10:.3f}",
            f"{report.abstention_recall:.3f}",
            f"{report.unsupported_answer_rate:.3f}",
            f"{report.citation_support:.3f}",
            f"{report.permission_leak_rate:.3f}",
            f"{report.p95_ms:.0f}",
        ]
        for report in reports
    ]

    widths = [max(len(headers[i]), *(len(row[i]) for row in rows)) for i in range(len(headers))]
    lines = [
        "  ".join(header.ljust(widths[i]) for i, header in enumerate(headers)),
        "  ".join("-" * widths[i] for i in range(len(headers))),
    ]
    lines.extend("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)) for row in rows)
    return "\n".join(lines)


#: Above this, a recall cutoff covers so much of the corpus that it stops discriminating --
#: retrieving most of the index scores well no matter how bad the ranking is.
DISCRIMINATION_LIMIT = 0.25


def corpus_warning(report: ConfigReport, *, cutoff: int = 50) -> str | None:
    """Warn when the corpus is too small for a cutoff to mean anything.

    This is the most common way a retrieval evaluation flatters itself. On a 71-chunk corpus,
    recall@50 asks whether the answer is in 70% of the index; almost any ranking passes, and a
    semantically blind embedder scores ~0.96. The number is real, it just is not evidence.
    """
    if report.corpus_chunks <= 0:
        return None
    fraction = cutoff / report.corpus_chunks
    if fraction < DISCRIMINATION_LIMIT:
        return None
    return (
        f"recall@{cutoff} covers {fraction:.0%} of a {report.corpus_chunks}-chunk corpus and "
        f"does not discriminate at this size -- read recall@10, and grow the corpus before "
        f"trusting recall@{cutoff}"
    )


def render_details(reports: Sequence[ConfigReport]) -> str:
    """Per-category recall and leg contribution, for whoever is doing the tuning."""
    blocks: list[str] = []
    for report in reports:
        lines = [f"{report.name}:"]
        if report.per_category_recall:
            lines.append("  recall@50 by category")
            for category, value in report.per_category_recall.items():
                lines.append(f"    {category:<24} {value:.3f}")
        if report.leg_contribution:
            lines.append("  relevant chunks found by only this leg")
            for leg, count in sorted(report.leg_contribution.items(), key=lambda kv: -kv[1]):
                lines.append(f"    {leg:<24} {count}")
        warning = corpus_warning(report)
        if warning:
            lines.append(f"  NOTE {warning}")
        if report.fast_path_questions:
            share = report.fast_path_questions / max(1, report.questions)
            lines.append(
                f"  fast path taken on {report.fast_path_questions}/{report.questions} "
                f"questions ({share:.0%}) -- no LLM call, no dense leg, no reranker"
            )
        if report.unresolved_labels:
            lines.append(
                f"  WARNING {report.unresolved_labels} label(s) matched no chunk -- "
                "the golden set is stale or the chunker dropped text"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def to_baseline(reports: Sequence[ConfigReport]) -> dict[str, dict[str, float]]:
    return {report.name: _metric_values(report) for report in reports}
