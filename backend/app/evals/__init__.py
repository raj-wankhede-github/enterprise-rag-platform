"""Retrieval evaluation: the golden set, the metrics, and the ablation table."""

from __future__ import annotations

from app.evals.dataset import (
    CorpusDocument,
    GoldenQuestion,
    IndexedChunk,
    RelevanceLabel,
    load_manifest,
    load_questions,
    resolve,
)
from app.evals.report import GateFailure, check, load_thresholds, render_details, render_table, to_baseline
from app.evals.runner import DEFAULT_ABLATIONS, AblationConfig, ConfigReport, EvalRunner

__all__ = [
    "DEFAULT_ABLATIONS",
    "AblationConfig",
    "ConfigReport",
    "CorpusDocument",
    "EvalRunner",
    "GateFailure",
    "GoldenQuestion",
    "IndexedChunk",
    "RelevanceLabel",
    "check",
    "load_manifest",
    "load_questions",
    "load_thresholds",
    "render_details",
    "render_table",
    "resolve",
    "to_baseline",
]
