"""Loading the golden set and resolving its labels against a live index.

The important design choice is how relevance is keyed. A label pointing at a chunk id breaks the
moment anything re-chunks; a label pointing at ``(doc, ordinal)`` breaks the moment the chunker
changes its boundaries -- which it will, because tuning the chunker is one of the things the
harness exists to measure. Either way the labels rot and people stop running the evaluation.

So a label is ``(doc_slug, contains)``: a distinctive phrase that must appear in the relevant
chunk. At run time the phrase is resolved to whichever chunks currently hold it. Re-chunk, and
the marker moves with the text.
"""

from __future__ import annotations

import json
import unicodedata
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

DATASET_DIR = Path(__file__).parent / "datasets"
CORPUS_DIR = DATASET_DIR / "corpus"
GOLDEN_PATH = DATASET_DIR / "golden.jsonl"
MANIFEST_PATH = DATASET_DIR / "corpus.json"


@dataclass(frozen=True, slots=True)
class CorpusDocument:
    slug: str
    title: str
    doc_type: str
    text: str
    effective_from: date | None = None
    effective_to: date | None = None
    visibility_rank: int = 10
    authority_rank: float = 0.5
    is_superseded: bool = False
    supersedes: tuple[str, ...] = ()
    superseded_by: tuple[str, ...] = ()

    @property
    def filename(self) -> str:
        return f"{self.slug}.md"


@dataclass(frozen=True, slots=True)
class RelevanceLabel:
    doc: str
    contains: str
    grade: int

    def __post_init__(self) -> None:
        if self.grade not in (1, 2):
            raise ValueError(f"grade must be 1 or 2, got {self.grade}")


@dataclass(frozen=True, slots=True)
class GoldenQuestion:
    id: str
    category: str
    question: str
    answerable: bool
    relevant: tuple[RelevanceLabel, ...] = ()
    expected_contains: tuple[str, ...] = ()
    must_not_contain: tuple[str, ...] = ()
    #: Set on historical questions, which must reach superseded documents on purpose.
    include_superseded: bool = False
    #: Permission-scoped questions: the rank at which this expectation holds.
    min_rank: int | None = None
    max_rank: int | None = None
    notes: str = ""

    @property
    def rank(self) -> int:
        """The visibility rank this question should be asked at."""
        if self.min_rank is not None:
            return self.min_rank
        if self.max_rank is not None:
            return self.max_rank
        return 30


@dataclass(slots=True)
class ResolvedQuestion:
    """A question with its labels bound to concrete chunk ids in the index under test."""

    question: GoldenQuestion
    #: chunk_id -> grade
    relevant: dict[str, int] = field(default_factory=dict)
    #: Labels whose marker matched nothing. A non-empty list means the label is stale or the
    #: chunker dropped text, and either way the metrics computed from it would be wrong.
    unresolved: list[RelevanceLabel] = field(default_factory=list)


def _parse_date(value: Any) -> date | None:
    return date.fromisoformat(value) if isinstance(value, str) and value else None


def load_manifest(path: Path = MANIFEST_PATH, corpus_dir: Path = CORPUS_DIR) -> list[CorpusDocument]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    documents: list[CorpusDocument] = []
    for entry in payload["documents"]:
        slug = entry["slug"]
        text_path = corpus_dir / f"{slug}.md"
        if not text_path.exists():
            raise FileNotFoundError(f"corpus manifest lists {slug} but {text_path} is missing")
        documents.append(
            CorpusDocument(
                slug=slug,
                title=entry["title"],
                doc_type=entry.get("doc_type", "document"),
                text=text_path.read_text(encoding="utf-8"),
                effective_from=_parse_date(entry.get("effective_from")),
                effective_to=_parse_date(entry.get("effective_to")),
                visibility_rank=int(entry.get("visibility_rank", 10)),
                authority_rank=float(entry.get("authority_rank", 0.5)),
                is_superseded=bool(entry.get("is_superseded", False)),
                supersedes=tuple(entry.get("supersedes", ())),
                superseded_by=tuple(entry.get("superseded_by", ())),
            )
        )
    return documents


def load_questions(path: Path = GOLDEN_PATH) -> list[GoldenQuestion]:
    questions: list[GoldenQuestion] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if "_comment" in payload:
                continue
            try:
                questions.append(
                    GoldenQuestion(
                        id=payload["id"],
                        category=payload["category"],
                        question=payload["question"],
                        answerable=bool(payload["answerable"]),
                        relevant=tuple(
                            RelevanceLabel(doc=item["doc"], contains=item["contains"], grade=int(item["grade"]))
                            for item in payload.get("relevant", ())
                        ),
                        expected_contains=tuple(payload.get("expected_contains", ())),
                        must_not_contain=tuple(payload.get("must_not_contain", ())),
                        include_superseded=bool(payload.get("include_superseded", False)),
                        min_rank=payload.get("min_rank"),
                        max_rank=payload.get("max_rank"),
                        notes=payload.get("notes", ""),
                    )
                )
            except (KeyError, ValueError) as exc:
                raise ValueError(f"{path}:{line_number} is not a valid golden question: {exc}") from exc
    return questions


def _normalize(text: str) -> str:
    """Casefold and collapse every whitespace run to a single space.

    Newlines matter here. Source markdown is hard-wrapped, so a marker like "18 weeks of paid
    parental leave" spans a line break in the file and therefore in the chunk text. Matching on
    the raw string silently fails, the label resolves to nothing, and the question quietly stops
    contributing to recall -- an evaluation that measures less than it claims to, which is worse
    than one that fails.
    """
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


@dataclass(frozen=True, slots=True)
class IndexedChunk:
    """Just enough of an indexed chunk to resolve a label against it."""

    chunk_id: str
    doc_slug: str
    text: str


def resolve(question: GoldenQuestion, chunks: Sequence[IndexedChunk]) -> ResolvedQuestion:
    """Bind a question's labels to the chunk ids currently holding their markers.

    A marker matching several chunks marks all of them relevant. That is correct rather than
    sloppy: if the chunker split a sentence across two chunks, either is a legitimate retrieval,
    and insisting on one would penalise a chunker change that is otherwise neutral.
    """
    resolved = ResolvedQuestion(question=question)
    by_doc: dict[str, list[IndexedChunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_slug, []).append(chunk)

    for label in question.relevant:
        needle = _normalize(label.contains)
        matches = [chunk for chunk in by_doc.get(label.doc, ()) if needle in _normalize(chunk.text)]
        if not matches:
            resolved.unresolved.append(label)
            continue
        for chunk in matches:
            # A chunk labelled by two markers keeps the higher grade.
            resolved.relevant[chunk.chunk_id] = max(resolved.relevant.get(chunk.chunk_id, 0), label.grade)
    return resolved


def iter_categories(questions: Sequence[GoldenQuestion]) -> Iterator[tuple[str, list[GoldenQuestion]]]:
    grouped: dict[str, list[GoldenQuestion]] = {}
    for question in questions:
        grouped.setdefault(question.category, []).append(question)
    yield from sorted(grouped.items())
