"""Answer assembly, generation, verification and abstention."""

from __future__ import annotations

from app.answer.abstain import abstain, build_message
from app.answer.assemble import AssembledContext, AssemblyConfig, EvidenceBlock, assemble
from app.answer.evidence import EvidenceAssessment, EvidenceThresholds, assess, distinctive_terms
from app.answer.generator import (
    AnswerGenerator,
    ExtractiveAnswerGenerator,
    LLMAnswerGenerator,
    render,
)
from app.answer.pipeline import AnswerOutcome, AnswerPipeline, AnswerTrace, OpenSearchParentFetcher
from app.answer.types import AbstentionReason, Answer, AnswerSentence, Conflict, NearMiss
from app.answer.verify import VerificationConfig, VerificationResult, extract_values, verify

__all__ = [
    "AbstentionReason",
    "Answer",
    "AnswerGenerator",
    "AnswerOutcome",
    "AnswerPipeline",
    "AnswerSentence",
    "AnswerTrace",
    "AssembledContext",
    "AssemblyConfig",
    "Conflict",
    "EvidenceAssessment",
    "EvidenceBlock",
    "EvidenceThresholds",
    "ExtractiveAnswerGenerator",
    "LLMAnswerGenerator",
    "NearMiss",
    "OpenSearchParentFetcher",
    "VerificationConfig",
    "VerificationResult",
    "abstain",
    "assemble",
    "assess",
    "build_message",
    "distinctive_terms",
    "extract_values",
    "render",
    "verify",
]
