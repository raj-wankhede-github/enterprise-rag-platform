"""Structure-aware parent/child chunking.

Split on **sections, not character counts**. A fixed-width splitter cuts through the middle of a
clause and produces chunks whose first sentence is missing its subject; a section splitter
produces chunks that are about something.

Two sizes, because retrieval and generation want different things:

* **children** (~300 tokens) are embedded and searched. Small enough that a vector represents one
  idea rather than an average of six.
* **parents** (whole sections) are what reaches the LLM once a child wins. Large enough to carry
  the context a child lost by being small.

Overlap is **zero**. Overlap exists to recover context lost at a boundary; parent expansion
recovers it properly, and zero overlap keeps near-duplicate detection and token accounting
honest -- with overlap, every adjacent pair of chunks is a near-duplicate of itself and the
dedup step has to be taught to ignore exactly the case it exists to catch.

Tables are never split by the generic path. A table is kept whole as Markdown; one that exceeds
the cap is split **by rows with the header repeated**, because half a table with no header is
worse than useless -- it looks like data and cannot be read.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from app.ingestion.types import (
    Block,
    BlockKind,
    ChildChunk,
    ChunkedDocument,
    ExtractedDocument,
    ParentChunk,
)
from app.utils.text import DEFAULT_TOKENIZER, Tokenizer, normalize, split_sentences

#: Bumped whenever chunk boundaries can change. Part of the generation fingerprint, so a bump
#: forces a background reindex rather than silently mixing old and new chunks in one index.
CHUNKER_VERSION: Final[str] = "1"

_HEADING_SEPARATOR: Final[str] = " > "


@dataclass(frozen=True, slots=True)
class ChunkerConfig:
    child_target_tokens: int = 300
    child_max_tokens: int = 512
    #: Below this a chunk merges forward. Without it, a bare heading becomes a chunk that BM25
    #: scores highly on a title match and that contains no answer.
    child_min_tokens: int = 80
    parent_max_tokens: int = 1800

    def __post_init__(self) -> None:
        if not 0 < self.child_min_tokens <= self.child_target_tokens <= self.child_max_tokens:
            raise ValueError("chunk sizes must satisfy 0 < min <= target <= max")
        if self.parent_max_tokens < self.child_max_tokens:
            raise ValueError("parent_max_tokens must be at least child_max_tokens")


@dataclass(slots=True)
class _Section:
    """A heading and the blocks beneath it, before size limits are applied."""

    heading_path: tuple[str, ...]
    blocks: list[Block]

    @property
    def path_text(self) -> str:
        return _HEADING_SEPARATOR.join(self.heading_path)


class StructureAwareChunker:
    def __init__(self, config: ChunkerConfig | None = None, tokenizer: Tokenizer | None = None) -> None:
        self.config = config or ChunkerConfig()
        self.tokenizer = tokenizer or DEFAULT_TOKENIZER

    # -- public ----------------------------------------------------------------------------

    def chunk(self, document: ExtractedDocument) -> ChunkedDocument:
        result = ChunkedDocument(chunker_version=CHUNKER_VERSION)
        for section in self._sections(document):
            for parent in self._split_section(section, len(result.parents)):
                children = self._children_for(section, parent, start_ordinal=len(result.children))
                # A section that produced no usable children (an empty heading, a figure with no
                # OCR text) must not leave an orphan parent behind: the parent leg would match it
                # and expansion would hand the LLM a heading with nothing under it.
                if not children:
                    continue
                result.parents.append(parent)
                result.children.extend(children)
        return result

    # -- sectioning ------------------------------------------------------------------------

    def _sections(self, document: ExtractedDocument) -> list[_Section]:
        """Group blocks under their heading path.

        Content appearing before any heading gets the document title as its path, so it is never
        silently dropped -- preambles and executive summaries live there and are often the most
        quotable part of a document.
        """
        sections: list[_Section] = []
        stack: list[tuple[int, str]] = []
        root = document.title or "Document"
        current = _Section(heading_path=(root,), blocks=[])

        for block in document.blocks:
            if block.kind is BlockKind.HEADING:
                if current.blocks:
                    sections.append(current)
                while stack and stack[-1][0] >= block.level:
                    stack.pop()
                stack.append((block.level, block.text.strip()))
                path = (root, *(title for _, title in stack))
                current = _Section(heading_path=path, blocks=[])
            else:
                current.blocks.append(block)

        if current.blocks:
            sections.append(current)
        return sections

    def _split_section(self, section: _Section, start_ordinal: int) -> list[ParentChunk]:
        """Turn one section into one or more parents, splitting at block boundaries."""
        groups: list[list[Block]] = []
        current: list[Block] = []
        current_tokens = 0

        for block in section.blocks:
            block_tokens = self.tokenizer.count(block.text)
            if current and current_tokens + block_tokens > self.config.parent_max_tokens:
                groups.append(current)
                current, current_tokens = [], 0
            current.append(block)
            current_tokens += block_tokens
        if current:
            groups.append(current)

        parents: list[ParentChunk] = []
        for index, group in enumerate(groups):
            text = normalize("\n\n".join(block.text for block in group))
            if not text:
                continue
            pages = [block.page for block in group if block.page is not None]
            parents.append(
                ParentChunk(
                    ordinal=start_ordinal + index,
                    text=text,
                    heading_path=section.path_text,
                    token_count=self.tokenizer.count(text),
                    page_from=min(pages) if pages else None,
                    page_to=max(pages) if pages else None,
                    part=index,
                    part_count=len(groups),
                )
            )
        return parents

    # -- children --------------------------------------------------------------------------

    def _children_for(self, section: _Section, parent: ParentChunk, *, start_ordinal: int) -> list[ChildChunk]:
        blocks = [block for block in section.blocks if block.text.strip()]
        if parent.part_count > 1:
            blocks = self._blocks_for_part(section, parent)

        units = self._units(blocks)
        merged = self._merge_small(units)

        children: list[ChildChunk] = []
        for index, unit in enumerate(merged):
            children.append(
                ChildChunk(
                    ordinal=start_ordinal + index,
                    parent_ordinal=parent.ordinal,
                    text=unit.text,
                    heading_path=parent.heading_path,
                    token_count=self.tokenizer.count(unit.text),
                    block_kinds=unit.kinds,
                    page_from=unit.page_from,
                    page_to=unit.page_to,
                    table_part=unit.table_part,
                )
            )
        return children

    def _blocks_for_part(self, section: _Section, parent: ParentChunk) -> list[Block]:
        """Recover the blocks that belong to one part of an oversized section."""
        groups: list[list[Block]] = []
        current: list[Block] = []
        current_tokens = 0
        for block in section.blocks:
            block_tokens = self.tokenizer.count(block.text)
            if current and current_tokens + block_tokens > self.config.parent_max_tokens:
                groups.append(current)
                current, current_tokens = [], 0
            current.append(block)
            current_tokens += block_tokens
        if current:
            groups.append(current)
        return groups[parent.part] if parent.part < len(groups) else []

    def _units(self, blocks: list[Block]) -> list[_Unit]:
        """One unit per block, except tables too large to keep whole and prose over the cap."""
        units: list[_Unit] = []
        for block in blocks:
            text = normalize(block.text)
            if not text:
                continue
            size = self.tokenizer.count(text)

            if block.kind is BlockKind.TABLE:
                units.extend(self._table_units(block, text, size))
                continue

            if size <= self.config.child_max_tokens:
                units.append(_Unit.of(block, text))
                continue

            # Only prose over the hard cap is split, and only at sentence boundaries.
            buffer: list[str] = []
            buffer_tokens = 0
            for sentence in split_sentences(text):
                sentence_tokens = self.tokenizer.count(sentence)
                if buffer and buffer_tokens + sentence_tokens > self.config.child_target_tokens:
                    units.append(_Unit.of(block, " ".join(buffer)))
                    buffer, buffer_tokens = [], 0
                buffer.append(sentence)
                buffer_tokens += sentence_tokens
            if buffer:
                units.append(_Unit.of(block, " ".join(buffer)))
        return units

    def _table_units(self, block: Block, text: str, size: int) -> list[_Unit]:
        """Keep a table whole; split an oversized one by rows, repeating the header."""
        if size <= self.config.child_max_tokens:
            return [_Unit.of(block, text)]

        lines = [line for line in text.split("\n") if line.strip()]
        # A GFM table is: header row, separator row, then data rows. Anything else is not a
        # table we can split safely, so it stays whole and oversized rather than being mangled.
        if len(lines) < 3 or not set(lines[1].replace("|", "").strip()) <= set("-: "):
            return [_Unit.of(block, text)]

        header = lines[:2]
        header_tokens = self.tokenizer.count("\n".join(header))
        parts: list[list[str]] = []
        current: list[str] = []
        current_tokens = header_tokens
        for row in lines[2:]:
            row_tokens = self.tokenizer.count(row)
            if current and current_tokens + row_tokens > self.config.child_max_tokens:
                parts.append(current)
                current, current_tokens = [], header_tokens
            current.append(row)
            current_tokens += row_tokens
        if current:
            parts.append(current)

        return [
            _Unit(
                text="\n".join([*header, *rows]),
                kinds=(BlockKind.TABLE,),
                page_from=block.page,
                page_to=block.page,
                table_part=(index + 1, len(parts)),
            )
            for index, rows in enumerate(parts)
        ]

    def _merge_small(self, units: list[_Unit]) -> list[_Unit]:
        """Merge undersized units forward, so no chunk is too small to answer anything.

        Tables are never merged into prose: a table's value is that it is a table, and gluing a
        paragraph to it makes the Markdown unparseable for the model.
        """
        if not units:
            return []
        merged: list[_Unit] = []
        for unit in units:
            if (
                merged
                and BlockKind.TABLE not in unit.kinds
                and BlockKind.TABLE not in merged[-1].kinds
                and self.tokenizer.count(merged[-1].text) < self.config.child_min_tokens
                and self.tokenizer.count(merged[-1].text) + self.tokenizer.count(unit.text)
                <= self.config.child_max_tokens
            ):
                merged[-1] = merged[-1].merged_with(unit)
            else:
                merged.append(unit)

        # A trailing runt merges backwards, since there is nothing ahead of it.
        if (
            len(merged) > 1
            and self.tokenizer.count(merged[-1].text) < self.config.child_min_tokens
            and BlockKind.TABLE not in merged[-1].kinds
            and BlockKind.TABLE not in merged[-2].kinds
            and self.tokenizer.count(merged[-2].text) + self.tokenizer.count(merged[-1].text)
            <= self.config.child_max_tokens
        ):
            tail = merged.pop()
            merged[-1] = merged[-1].merged_with(tail)
        return merged


@dataclass(frozen=True, slots=True)
class _Unit:
    text: str
    kinds: tuple[BlockKind, ...]
    page_from: int | None
    page_to: int | None
    table_part: tuple[int, int] | None = None

    @classmethod
    def of(cls, block: Block, text: str) -> _Unit:
        return cls(text=text, kinds=(block.kind,), page_from=block.page, page_to=block.page)

    def merged_with(self, other: _Unit) -> _Unit:
        pages = [p for p in (self.page_from, self.page_to, other.page_from, other.page_to) if p is not None]
        return _Unit(
            text=f"{self.text}\n\n{other.text}",
            kinds=tuple(dict.fromkeys(self.kinds + other.kinds)),
            page_from=min(pages) if pages else None,
            page_to=max(pages) if pages else None,
        )
