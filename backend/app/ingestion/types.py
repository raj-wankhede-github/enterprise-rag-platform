"""The document model the chunker works on.

``ExtractedDocument`` is **our** type, not a parser's. Docling, pypdf and the cheap text loaders
all normalize into it, so the chunker never learns which parser produced a document and a parser
can be swapped without touching chunking. That indirection is the reason the Docling dependency
can live in a separate container behind an HTTP contract.

Blocks are deliberately coarse. The chunker needs to know "is this a table" and "is this a
heading, and at what level"; it does not need a faithful DOM. Anything finer would couple us to
one parser's idea of structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Literal


class BlockKind(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST = "list"
    TABLE = "table"
    CODE = "code"
    FOOTNOTE = "footnote"
    FIGURE = "figure"


@dataclass(frozen=True, slots=True)
class Block:
    """One structural unit of a document.

    ``text`` is always the renderable form: for a table that is GitHub-flavoured Markdown, so a
    table survives into the LLM context as a table rather than as flattened prose. Losing table
    structure is one of the most common ways a RAG system fails on enterprise documents, where
    the answer is frequently a number that appears only in a cell.
    """

    kind: BlockKind
    text: str
    level: int = 0  # heading level, 1-6; 0 for everything else
    page: int | None = None
    #: Footnote marker, list ordinal, table caption -- whatever the block kind needs.
    label: str | None = None
    #: True when the text came from OCR rather than a text layer, so it can be down-weighted.
    ocr: bool = False

    def __post_init__(self) -> None:
        if self.kind is BlockKind.HEADING and not 1 <= self.level <= 6:
            raise ValueError(f"heading level must be 1-6, got {self.level}")
        if self.kind is not BlockKind.HEADING and self.level != 0:
            raise ValueError(f"{self.kind} must have level 0, got {self.level}")


@dataclass(frozen=True, slots=True)
class ExtractedDocument:
    """A parsed document, before chunking."""

    blocks: tuple[Block, ...]
    title: str | None = None
    language: str = "en"
    parser: str = "unknown"
    parser_version: str = "0"
    page_count: int | None = None
    warnings: tuple[str, ...] = ()

    @property
    def text(self) -> str:
        """The full extracted text, used for ``doc_text_sha256``.

        Joined with a blank line so that two documents differing only in block boundaries hash
        differently -- if they chunk differently, they are not interchangeable.
        """
        return "\n\n".join(block.text for block in self.blocks)


@dataclass(frozen=True, slots=True)
class ParentChunk:
    """A heading section. Searched as its own leg, and what reaches the LLM on expansion."""

    ordinal: int
    text: str
    heading_path: str
    token_count: int
    page_from: int | None = None
    page_to: int | None = None
    #: Set when a section exceeded ``parent_max_tokens`` and had to be split.
    part: int = 0
    part_count: int = 1


@dataclass(frozen=True, slots=True)
class ChildChunk:
    """What gets embedded and searched.

    ``parent_ordinal`` rather than a parent id: chunking happens before anything is persisted,
    so the two are linked positionally and resolved to ids by the pipeline.
    """

    ordinal: int
    parent_ordinal: int
    text: str
    heading_path: str
    token_count: int
    block_kinds: tuple[BlockKind, ...] = ()
    page_from: int | None = None
    page_to: int | None = None
    #: Only set for a table that exceeded the chunk cap and was split by rows.
    table_part: tuple[int, int] | None = None


@dataclass(slots=True)
class ChunkedDocument:
    parents: list[ParentChunk] = field(default_factory=list)
    children: list[ChildChunk] = field(default_factory=list)
    chunker_version: str = "0"

    def child_texts(self) -> list[str]:
        return [child.text for child in self.children]


LoaderCost = Literal["cheap", "expensive"]


@dataclass(frozen=True, slots=True)
class DocumentProbe:
    """What the loader registry routes on.

    ``chars_per_page`` is the decisive signal for PDFs: a scanned page has a text layer of
    roughly nothing, and sending it to the cheap loader produces a document full of empty
    chunks that index cleanly and retrieve nothing. Routing on it rather than on the extension
    is what keeps scans out of the fast path.
    """

    filename: str
    mime_type: str
    size_bytes: int
    magic: bytes = b""
    page_count: int | None = None
    chars_per_page: float | None = None
    has_tables: bool = False

    @property
    def extension(self) -> str:
        _, _, ext = self.filename.rpartition(".")
        return ext.lower() if ext else ""
