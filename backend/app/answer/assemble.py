"""Turning ranked candidates into the evidence block a model sees.

Retrieval optimises for finding things. Assembly optimises for *reading* them, and the two want
opposite shapes. A top-10 straight from the ranker is typically the same paragraph from twelve
documents, three consecutive chunks of one section, and a table split from its header -- all
technically relevant and collectively useless.

Order of operations, each step undoing a specific pathology:

1. **Near-duplicate removal.** Enterprise corpora repeat boilerplate across documents. Without
   this, a top-10 is one paragraph wearing twelve hats and the answer looks corroborated when it
   has a single source.
2. **Parent expansion.** A winning child is ~300 tokens and was embedded small on purpose; the
   model needs the section around it. This is also what makes zero chunk overlap safe.
3. **Containment removal.** After expansion a parent usually contains a lower-ranked sibling
   verbatim. Keeping both spends budget on text already present.
4. **Source diversity cap.** One exhaustive handbook must not fill the context, or the answer
   has a single point of failure that reads like consensus.
5. **Budget packing, strongest first**, with truncation rather than dropping -- a truncated
   source still contributes a citable fact; a dropped one contributes nothing.

Pure with respect to I/O: parent text arrives as a mapping, so the whole thing is testable with
no cluster.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from app.retrieval.types import Candidate
from app.utils.text import DEFAULT_TOKENIZER, Tokenizer, hamming_distance, jaccard, simhash64


@dataclass(frozen=True, slots=True)
class AssemblyConfig:
    token_budget: int = 6000
    #: How many distinct evidence blocks reach the model. Beyond roughly this many, added
    #: context reliably costs more in dilution than it buys in coverage.
    max_blocks: int = 8
    #: Blocks from any one document. The guard against a single verbose source dominating.
    max_per_document: int = 3
    #: SimHash distance at or below which two chunks are treated as the same text. 6 bits out of
    #: 64 tolerates a changed date or a reformatted list without merging genuinely distinct
    #: paragraphs.
    simhash_hamming: int = 6
    #: Token-set overlap above which two chunks are near-duplicates. Catches reorderings that
    #: SimHash misses.
    jaccard_threshold: float = 0.85
    expand_parents: bool = True
    #: A truncated block keeps at least this many tokens, or it is dropped instead -- a two-line
    #: fragment is not citable evidence, it is noise with an id.
    min_block_tokens: int = 40


@dataclass(frozen=True, slots=True)
class EvidenceBlock:
    """One citable unit in the prompt. ``id`` is what a claim cites."""

    id: str
    text: str
    title: str
    heading_path: str | None
    doc_id: str
    token_count: int
    score: float
    page_from: int | None = None
    page_to: int | None = None
    effective_from: str | None = None
    source_system: str | None = None
    is_superseded: bool = False
    truncated: bool = False
    expanded: bool = False

    def render(self) -> str:
        """Rendered as data, never as instructions.

        The attributes are what a citation resolves to in the UI, and the explicit closing tag
        matters: a model handed unterminated blocks will sometimes continue one rather than
        treat it as quoted material.
        """
        attributes = [f'id="{self.id}"', f'doc="{_escape(self.title)}"']
        if self.heading_path:
            attributes.append(f'section="{_escape(self.heading_path)}"')
        if self.effective_from:
            attributes.append(f'effective="{self.effective_from}"')
        if self.source_system:
            attributes.append(f'source="{_escape(self.source_system)}"')
        if self.page_from is not None:
            attributes.append(f'page="{self.page_from}"')
        if self.is_superseded:
            attributes.append('superseded="true"')
        if self.truncated:
            attributes.append('truncated="true"')
        return f"<evidence {' '.join(attributes)}>\n{self.text}\n</evidence>"


@dataclass(slots=True)
class AssembledContext:
    blocks: list[EvidenceBlock] = field(default_factory=list)
    total_tokens: int = 0
    dropped_duplicates: int = 0
    dropped_for_budget: int = 0
    dropped_for_diversity: int = 0

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(block.id for block in self.blocks)

    def by_id(self) -> dict[str, EvidenceBlock]:
        return {block.id: block for block in self.blocks}

    def render(self) -> str:
        """The evidence section of the prompt.

        Ends with a restatement of the ids in order. Models attend unevenly to the middle of a
        long context; a short index at the end costs a handful of tokens and measurably improves
        citation of blocks that are not first or last.
        """
        if not self.blocks:
            return ""
        body = "\n\n".join(block.render() for block in self.blocks)
        index = ", ".join(block.id for block in self.blocks)
        return f"{body}\n\nEvidence ids, strongest first: {index}"


def _escape(value: str) -> str:
    return value.replace('"', "'").replace("\n", " ").strip()


def assemble(
    candidates: Sequence[Candidate],
    *,
    parent_texts: Mapping[str, str] | None = None,
    config: AssemblyConfig | None = None,
    tokenizer: Tokenizer | None = None,
) -> AssembledContext:
    """Build the evidence block list.

    ``parent_texts`` maps ``parent_id`` to the full section text. A missing entry simply means
    that candidate is not expanded, so a failed parent fetch degrades the context rather than
    failing the answer.
    """
    settings = config or AssemblyConfig()
    counter = tokenizer or DEFAULT_TOKENIZER
    parents = parent_texts or {}

    ranked = sorted(candidates, key=lambda item: -item.final_score)
    survivors, duplicates = _drop_near_duplicates(ranked, settings)
    expanded = _expand(survivors, parents, settings)
    deduped, contained = _drop_contained(expanded)
    diverse, crowded_out = _cap_per_document(deduped, settings)
    blocks, over_budget, total = _pack(diverse, settings, counter)

    return AssembledContext(
        blocks=blocks,
        total_tokens=total,
        dropped_duplicates=duplicates + contained,
        dropped_for_budget=over_budget,
        dropped_for_diversity=crowded_out,
    )


# -- steps ----------------------------------------------------------------------------------


@dataclass(slots=True)
class _Working:
    candidate: Candidate
    text: str
    expanded: bool = False


def _drop_near_duplicates(candidates: Sequence[Candidate], settings: AssemblyConfig) -> tuple[list[Candidate], int]:
    """Exact hash, then SimHash, then Jaccard -- cheapest discriminator first."""
    kept: list[Candidate] = []
    seen_exact: set[str] = set()
    hashes: list[int] = []
    dropped = 0

    for candidate in candidates:
        normalized = " ".join(candidate.text.split()).casefold()
        if normalized in seen_exact:
            dropped += 1
            continue

        fingerprint = simhash64(candidate.text)
        if any(hamming_distance(fingerprint, other) <= settings.simhash_hamming for other in hashes):
            dropped += 1
            continue
        if any(jaccard(candidate.text, other.text) >= settings.jaccard_threshold for other in kept):
            dropped += 1
            continue

        seen_exact.add(normalized)
        hashes.append(fingerprint)
        kept.append(candidate)
    return kept, dropped


def _expand(candidates: Sequence[Candidate], parents: Mapping[str, str], settings: AssemblyConfig) -> list[_Working]:
    """Replace a winning child with its section, once per section.

    Only the highest-ranked child of a section triggers expansion; later siblings would expand to
    identical text and are handled by the containment step.
    """
    if not settings.expand_parents:
        return [_Working(candidate, candidate.text) for candidate in candidates]

    used_parents: set[str] = set()
    working: list[_Working] = []
    for candidate in candidates:
        parent_text = parents.get(candidate.parent_id)
        if parent_text and candidate.parent_id not in used_parents:
            used_parents.add(candidate.parent_id)
            working.append(_Working(candidate, parent_text, expanded=True))
        else:
            working.append(_Working(candidate, candidate.text))
    return working


def _drop_contained(working: Sequence[_Working]) -> tuple[list[_Working], int]:
    """Drop any block whose text is already inside a higher-ranked one."""
    kept: list[_Working] = []
    dropped = 0
    for item in working:
        needle = " ".join(item.text.split()).casefold()
        if any(needle in " ".join(other.text.split()).casefold() for other in kept):
            dropped += 1
            continue
        kept.append(item)
    return kept, dropped


def _cap_per_document(working: Sequence[_Working], settings: AssemblyConfig) -> tuple[list[_Working], int]:
    counts: dict[str, int] = {}
    kept: list[_Working] = []
    dropped = 0
    for item in working:
        doc_id = item.candidate.doc_id
        if counts.get(doc_id, 0) >= settings.max_per_document:
            dropped += 1
            continue
        counts[doc_id] = counts.get(doc_id, 0) + 1
        kept.append(item)
    return kept, dropped


def _pack(
    working: Sequence[_Working], settings: AssemblyConfig, counter: Tokenizer
) -> tuple[list[EvidenceBlock], int, int]:
    """Fill the budget strongest-first, truncating rather than dropping where possible."""
    blocks: list[EvidenceBlock] = []
    used = 0
    dropped = 0

    for item in working:
        if len(blocks) >= settings.max_blocks:
            dropped += 1
            continue

        remaining = settings.token_budget - used
        if remaining <= 0:
            dropped += 1
            continue

        text = item.text
        size = counter.count(text)
        truncated = False

        if size > remaining:
            if remaining < settings.min_block_tokens:
                dropped += 1
                continue
            text = _truncate(text, remaining, counter)
            size = counter.count(text)
            truncated = True
            if size < settings.min_block_tokens:
                dropped += 1
                continue

        candidate = item.candidate
        blocks.append(
            EvidenceBlock(
                id=candidate.chunk_id[:8],
                text=text,
                title=candidate.title,
                heading_path=candidate.heading_path,
                doc_id=candidate.doc_id,
                token_count=size,
                score=candidate.final_score,
                page_from=candidate.meta.page_from,
                page_to=candidate.meta.page_to,
                effective_from=_as_text(candidate.meta.effective_from),
                source_system=candidate.meta.source_system,
                is_superseded=candidate.meta.is_superseded,
                truncated=truncated,
                expanded=item.expanded,
            )
        )
        used += size

    return blocks, dropped, used


def _truncate(text: str, budget: int, counter: Tokenizer) -> str:
    """Cut at a paragraph boundary where possible, then a sentence, then hard.

    Cutting mid-sentence produces a fragment the model may quote as if complete, so the ellipsis
    is explicit and boundaries are preferred.
    """
    if counter.count(text) <= budget:
        return text

    for separator in ("\n\n", ". "):
        parts = text.split(separator)
        if len(parts) < 2:
            continue
        kept: list[str] = []
        used = 0
        for part in parts:
            size = counter.count(part) + 1
            if used + size > budget:
                break
            kept.append(part)
            used += size
        if kept:
            joined = separator.join(kept)
            candidate = f"{joined}{'' if joined.endswith('.') else '.'} [...]"
            # A boundary split can leave almost nothing when the text has no usable ones -- a
            # single 200-word "sentence" yields just its first clause. Falling through to the
            # word-level cut uses the budget that was actually available.
            if counter.count(candidate) >= budget * 0.5:
                return candidate

    words = text.split()
    while words and counter.count(" ".join(words)) > budget:
        words = words[: int(len(words) * 0.9)] if len(words) > 20 else words[:-1]
    return " ".join(words) + " [...]"


def _as_text(value: object) -> str | None:
    if value is None:
        return None
    return str(value)[:10]
