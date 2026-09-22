"""Checking that each cited block actually supports the sentence attached to it.

The generator proposes; this disposes. Three layers, cheapest first, because the cheap ones
catch most of it:

1. **Structural** (free, always): every cited id exists, and every claim carries one. Catches a
   model that invented a citation or answered without one.
2. **Deterministic value checks** (free, always, no model): every number, amount, date, duration,
   percentage, code and quoted string in a claim must appear in one of *its own* cited blocks.

   This is the layer that matters most and it is the reason this module exists. The dominant
   real-world citation failure is not a fabricated document -- it is a real citation attached to
   a slightly wrong number: the 2023 rate quoted with the 2025 policy cited, or a figure
   borrowed from the adjacent row of a table. A human reviewer skims the citation, sees a
   plausible source, and moves on. An exact-match check does not.

   It is also what catches the case the pre-generation evidence gate provably cannot: a question
   whose topic terms are all present and only the asked-for *value* is absent. A model asked for
   "the per diem for grade D" against a table listing A, B and C has no block containing "D"'s
   rate, so whatever number it produces fails this check.

3. **Cross-encoder entailment** (optional): scores ``(claim, cited block)`` with the reranker
   already in the stack. No separate NLI model, no separate container. Skipped when the reranker
   is unavailable, which degrades the check rather than failing the answer.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.answer.assemble import AssembledContext, EvidenceBlock
from app.answer.types import Answer, AnswerSentence

#: Things whose exact value must be traceable to a cited block. Ordered longest-first where
#: patterns overlap, so "31 December 2026" is one token rather than three.
_VALUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("date_long", re.compile(r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b")),
    ("date_iso", re.compile(r"\b\d{4}-\d{2}-\d{2}\b")),
    ("identifier", re.compile(r"\b[A-Z]{2,6}[-_]\d{2,}(?:[-_]\d+)*\b")),
    ("section", re.compile(r"\b\d+\.\d+(?:\.\d+)*\b")),
    ("percentage", re.compile(r"\b\d+(?:\.\d+)?\s*%")),
    ("money", re.compile(r"\b\d[\d,]*(?:\.\d+)?\s*(?:EUR|USD|GBP|€|\$|£)\b", re.IGNORECASE)),
    ("duration", re.compile(r"\b\d+\s*(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?)\b", re.IGNORECASE)),
    ("time", re.compile(r"\b\d{1,2}:\d{2}\b")),
    ("number", re.compile(r"\b\d[\d,]*(?:\.\d+)?\b")),
)

_QUOTED = re.compile(r"[\"“]([^\"”]{4,})[\"”]")

#: Spelled-out numbers appear constantly in policy prose ("six months", "ten working days") and
#: a model substituting one for another is exactly the failure this catches.
_WORD_NUMBERS: dict[str, str] = {
    "zero": "0",
    "one": "1",
    "two": "2",
    "three": "3",
    "four": "4",
    "five": "5",
    "six": "6",
    "seven": "7",
    "eight": "8",
    "nine": "9",
    "ten": "10",
    "eleven": "11",
    "twelve": "12",
    "fourteen": "14",
    "fifteen": "15",
    "twenty": "20",
    "thirty": "30",
    "sixty": "60",
    "ninety": "90",
    "hundred": "100",
}


class EntailmentScorer(Protocol):
    """Anything that can score how well a passage supports a claim, in [0, 1]."""

    async def score(self, claim: str, passage: str) -> float: ...


@dataclass(frozen=True, slots=True)
class VerificationConfig:
    #: Below this, a cross-encoder says the block does not support the claim.
    min_entailment: float = 0.35
    #: How many unsupported claims are tolerated before the answer is rejected outright.
    #: Zero: a single unsupported claim is the failure mode the product exists to prevent, and
    #: "mostly grounded" is not a thing a compliance team can act on.
    max_unsupported: int = 0
    check_values: bool = True
    check_entailment: bool = True


@dataclass(frozen=True, slots=True)
class SentenceVerdict:
    sentence: AnswerSentence
    supported: bool
    missing_values: tuple[str, ...] = ()
    entailment: float | None = None
    reason: str = ""


@dataclass(slots=True)
class VerificationResult:
    verdicts: list[SentenceVerdict] = field(default_factory=list)
    unknown_ids: tuple[str, ...] = ()
    uncited_claims: int = 0

    @property
    def unsupported(self) -> list[SentenceVerdict]:
        return [verdict for verdict in self.verdicts if not verdict.supported]

    @property
    def ok(self) -> bool:
        return not self.unsupported and not self.unknown_ids and not self.uncited_claims

    @property
    def citation_support(self) -> float:
        """Fraction of claim sentences whose cited blocks actually support them."""
        claims = [v for v in self.verdicts if v.sentence.kind == "claim"]
        if not claims:
            return 1.0
        return sum(1 for verdict in claims if verdict.supported) / len(claims)

    def failure_summary(self) -> str:
        if self.unknown_ids:
            return f"cited unknown evidence ids: {', '.join(self.unknown_ids)}"
        if self.uncited_claims:
            return f"{self.uncited_claims} claim(s) carried no citation"
        missing = [value for verdict in self.unsupported for value in verdict.missing_values]
        if missing:
            return f"values not present in the cited evidence: {', '.join(sorted(set(missing))[:5])}"
        return f"{len(self.unsupported)} claim(s) were not supported by their cited evidence"


def extract_values(text: str) -> tuple[str, ...]:
    """Values whose exact presence in the cited evidence is checkable.

    Longer patterns are consumed first and their spans masked, so "31 December 2026" does not
    also yield "31" and "2026" as separate values that then fail independently.
    """
    found: list[str] = []
    remaining = text
    for _, pattern in _VALUE_PATTERNS:
        for match in pattern.finditer(remaining):
            found.append(match.group().strip())
        remaining = pattern.sub(lambda m: " " * len(m.group()), remaining)

    for match in _QUOTED.finditer(text):
        found.append(match.group(1).strip())

    for word, digits in _WORD_NUMBERS.items():
        if re.search(rf"\b{word}\b", text, re.IGNORECASE):
            found.append(word)
            found.append(digits)

    return tuple(dict.fromkeys(found))


def _normalize(text: str) -> str:
    collapsed = " ".join(text.split()).casefold()
    # Thousands separators and currency spacing differ between a table cell and prose.
    return collapsed.replace(",", "").replace(chr(0x00A0), " ")


def _digitize(text: str) -> str:
    """Rewrite spelled-out numbers as digits.

    Applied to the claim *and* the evidence, because policy prose and tables disagree about
    which form to use for the same fact: "six months" in a paragraph, "6" in a cell. Without
    this, "Probation is 6 months" fails against "the standard probation period is six months" --
    a correct answer rejected for formatting.
    """

    def replace(match: re.Match[str]) -> str:
        return _WORD_NUMBERS[match.group().casefold()]

    pattern = r"\b(?:" + "|".join(_WORD_NUMBERS) + r")\b"
    return re.sub(pattern, replace, text, flags=re.IGNORECASE)


def _value_present(value: str, haystack: str) -> bool:
    needle = _normalize(value)
    if needle in haystack:
        return True

    # Compare with both sides reduced to digits.
    if _digitize(needle) in _digitize(haystack):
        return True

    # "120 EUR" is supported by a table cell reading "| A | 120 |": the unit lives in the
    # column header, not the cell, so the bare number is the only thing that can match.
    digits_only = re.sub(r"[^\d.]", "", needle)
    return bool(digits_only) and len(digits_only) >= 2 and digits_only in haystack


async def verify(
    answer: Answer,
    context: AssembledContext,
    *,
    config: VerificationConfig | None = None,
    scorer: EntailmentScorer | None = None,
) -> VerificationResult:
    """Check every claim against the blocks it cites."""
    settings = config or VerificationConfig()
    result = VerificationResult()
    if not answer.answerable:
        return result

    blocks = context.by_id()
    unknown: list[str] = []

    for sentence in answer.sentences:
        if sentence.kind != "claim":
            result.verdicts.append(SentenceVerdict(sentence=sentence, supported=True))
            continue

        cited = [blocks[i] for i in sentence.evidence_ids if i in blocks]
        unknown.extend(i for i in sentence.evidence_ids if i not in blocks)

        if not cited:
            result.uncited_claims += 1
            result.verdicts.append(SentenceVerdict(sentence=sentence, supported=False, reason="no resolvable citation"))
            continue

        haystack = _normalize(" ".join(block.text for block in cited))

        missing: tuple[str, ...] = ()
        if settings.check_values:
            missing = tuple(value for value in extract_values(sentence.text) if not _value_present(value, haystack))
        if missing:
            result.verdicts.append(
                SentenceVerdict(
                    sentence=sentence,
                    supported=False,
                    missing_values=missing,
                    reason="cited evidence does not contain the stated value",
                )
            )
            continue

        entailment: float | None = None
        if settings.check_entailment and scorer is not None:
            entailment = max([await scorer.score(sentence.text, block.text) for block in cited])
            if entailment < settings.min_entailment:
                result.verdicts.append(
                    SentenceVerdict(
                        sentence=sentence,
                        supported=False,
                        entailment=entailment,
                        reason="cited evidence does not entail the claim",
                    )
                )
                continue

        result.verdicts.append(SentenceVerdict(sentence=sentence, supported=True, entailment=entailment))

    result.unknown_ids = tuple(dict.fromkeys(unknown))
    return result


def surviving_blocks(result: VerificationResult, context: AssembledContext) -> list[EvidenceBlock]:
    """Blocks cited by claims that passed. What the single stricter retry is given.

    Narrowing the evidence is the whole point of the retry: handing back the same context that
    produced an ungrounded draft mostly produces another one.
    """
    blocks = context.by_id()
    kept: list[EvidenceBlock] = []
    for verdict in result.verdicts:
        if not verdict.supported:
            continue
        for evidence_id in verdict.sentence.evidence_ids:
            block = blocks.get(evidence_id)
            if block and block not in kept:
                kept.append(block)
    return kept


def strip_unsupported(answer: Answer, result: VerificationResult) -> Answer:
    """Drop unsupported claims, keeping the rest.

    Used only where a tenant has opted into partial answers. The default is to abstain, because
    an answer silently missing its most important sentence is harder to notice than no answer.
    """
    kept: Sequence[Any] = [verdict.sentence for verdict in result.verdicts if verdict.supported]
    claims = [sentence for sentence in kept if sentence.kind == "claim"]
    if not claims:
        from app.answer.abstain import abstain
        from app.answer.types import AbstentionReason

        return abstain(AbstentionReason.UNGROUNDED_DRAFT)
    return Answer(answerable=True, sentences=tuple(kept), caveats=answer.caveats)
