"""Cheap in-process loaders: plain text, Markdown and HTML.

These carry no model dependency and no network call, which is what lets the whole ingestion path
run in CI. Between them they cover the formats an evaluation corpus is written in, so the golden
set never depends on the parser container being deployed.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser
from typing import Final

from app.ingestion.types import Block, BlockKind, DocumentProbe, ExtractedDocument, LoaderCost
from app.utils.text import normalize

_SETEXT_H1: Final[re.Pattern[str]] = re.compile(r"^=+$")
_SETEXT_H2: Final[re.Pattern[str]] = re.compile(r"^-{2,}$")
_ATX: Final[re.Pattern[str]] = re.compile(r"^(#{1,6})\s+(.*)$")
_LIST_ITEM: Final[re.Pattern[str]] = re.compile(r"^\s*([-*+]|\d+[.)])\s+")
_FENCE: Final[re.Pattern[str]] = re.compile(r"^\s*(```|~~~)")
_TABLE_ROW: Final[re.Pattern[str]] = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP: Final[re.Pattern[str]] = re.compile(r"^\s*\|[\s:|-]+\|\s*$")
_FOOTNOTE: Final[re.Pattern[str]] = re.compile(r"^\s*\[\^([^\]]+)\]:\s*(.*)$")


class PlainTextLoader:
    name: str = "text"
    version: str = "1"
    cost: LoaderCost = "cheap"

    _EXTENSIONS = frozenset({"txt", "log", "csv", "tsv", ""})

    def supports(self, probe: DocumentProbe) -> bool:
        return probe.extension in self._EXTENSIONS or probe.mime_type.startswith("text/plain")

    async def load(self, blob: bytes, probe: DocumentProbe) -> ExtractedDocument:
        text = normalize(_decode(blob))
        blocks = tuple(
            Block(kind=BlockKind.PARAGRAPH, text=part.strip()) for part in text.split("\n\n") if part.strip()
        )
        return ExtractedDocument(
            blocks=blocks,
            title=_title_from_filename(probe.filename),
            parser=self.name,
            parser_version=self.version,
        )


class MarkdownLoader:
    """Markdown, keeping headings as headings and tables as tables.

    Tables are the reason this is not a two-line splitter: flattening a Markdown table into prose
    destroys the row/column relationship, and enterprise answers are very often a single cell.
    """

    name: str = "markdown"
    version: str = "1"
    cost: LoaderCost = "cheap"

    _EXTENSIONS = frozenset({"md", "markdown", "mdown"})

    def supports(self, probe: DocumentProbe) -> bool:
        return probe.extension in self._EXTENSIONS or probe.mime_type == "text/markdown"

    async def load(self, blob: bytes, probe: DocumentProbe) -> ExtractedDocument:
        lines = normalize(_decode(blob)).split("\n")
        blocks: list[Block] = []
        buffer: list[str] = []
        buffer_kind = BlockKind.PARAGRAPH
        in_fence = False

        def flush() -> None:
            nonlocal buffer, buffer_kind
            text = "\n".join(buffer).strip()
            if text:
                blocks.append(Block(kind=buffer_kind, text=text))
            buffer = []
            buffer_kind = BlockKind.PARAGRAPH

        index = 0
        while index < len(lines):
            line = lines[index]

            if _FENCE.match(line):
                if in_fence:
                    buffer.append(line)
                    flush()
                    in_fence = False
                else:
                    flush()
                    in_fence = True
                    buffer_kind = BlockKind.CODE
                    buffer.append(line)
                index += 1
                continue
            if in_fence:
                buffer.append(line)
                index += 1
                continue

            atx = _ATX.match(line)
            if atx:
                flush()
                blocks.append(Block(kind=BlockKind.HEADING, text=atx.group(2).strip(), level=len(atx.group(1))))
                index += 1
                continue

            # Setext headings: the underline is on the following line.
            if index + 1 < len(lines) and line.strip() and not _TABLE_ROW.match(line):
                nxt = lines[index + 1].strip()
                if _SETEXT_H1.match(nxt) or _SETEXT_H2.match(nxt):
                    flush()
                    level = 1 if _SETEXT_H1.match(nxt) else 2
                    blocks.append(Block(kind=BlockKind.HEADING, text=line.strip(), level=level))
                    index += 2
                    continue

            footnote = _FOOTNOTE.match(line)
            if footnote:
                flush()
                blocks.append(Block(kind=BlockKind.FOOTNOTE, text=footnote.group(2).strip(), label=footnote.group(1)))
                index += 1
                continue

            if _TABLE_ROW.match(line):
                start = index
                while index < len(lines) and _TABLE_ROW.match(lines[index]):
                    index += 1
                rows = lines[start:index]
                # Only a real GFM table (header + separator + body) becomes a TABLE block;
                # anything else is prose that happens to contain pipes.
                if len(rows) >= 2 and _TABLE_SEP.match(rows[1]):
                    flush()
                    blocks.append(Block(kind=BlockKind.TABLE, text="\n".join(r.strip() for r in rows)))
                else:
                    buffer.extend(rows)
                continue

            if not line.strip():
                flush()
                index += 1
                continue

            kind = BlockKind.LIST if _LIST_ITEM.match(line) else BlockKind.PARAGRAPH
            if buffer and kind != buffer_kind:
                flush()
            buffer_kind = kind
            buffer.append(line)
            index += 1

        flush()
        title = next((b.text for b in blocks if b.kind is BlockKind.HEADING and b.level == 1), None)
        return ExtractedDocument(
            blocks=tuple(blocks),
            title=title or _title_from_filename(probe.filename),
            parser=self.name,
            parser_version=self.version,
        )


class _HTMLBlockParser(HTMLParser):
    """Collect block-level structure. Deliberately small: anything richer belongs in Docling."""

    # "head" is deliberately NOT here: script and style are skipped by their own tags, and
    # skipping the whole head would swallow <title>, which is the document name.
    _SKIP = frozenset({"script", "style", "noscript", "template"})
    _BLOCK = frozenset({"p", "div", "section", "article", "blockquote", "pre"})

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self.title: str | None = None
        self._buffer: list[str] = []
        self._kind = BlockKind.PARAGRAPH
        self._level = 0
        self._skip_depth = 0
        self._in_title = False
        self._row: list[str] = []
        self._table: list[list[str]] = []
        self._in_table = False
        self._in_cell = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
            return
        if tag == "title":
            self._in_title = True
            return
        if tag == "table":
            self._flush()
            self._in_table, self._table = True, []
            return
        if self._in_table:
            if tag == "tr":
                self._row = []
            elif tag in ("td", "th"):
                self._in_cell = True
                self._buffer = []
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6"):
            self._flush()
            self._kind, self._level = BlockKind.HEADING, int(tag[1])
        elif tag in ("li",):
            self._flush()
            self._kind, self._level = BlockKind.LIST, 0
        elif tag in self._BLOCK:
            self._flush()
            self._kind, self._level = BlockKind.PARAGRAPH, 0
        elif tag == "br":
            self._buffer.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag == "title":
            self._in_title = False
            self.title = "".join(self._buffer).strip() or None
            self._buffer = []
            return
        if self._in_table:
            if tag in ("td", "th"):
                self._row.append(" ".join("".join(self._buffer).split()))
                self._buffer, self._in_cell = [], False
            elif tag == "tr":
                if self._row:
                    self._table.append(self._row)
                self._row = []
            elif tag == "table":
                self._emit_table()
            return
        if tag in ("h1", "h2", "h3", "h4", "h5", "h6", "li", *self._BLOCK):
            self._flush()

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        if self._in_title or self._in_cell or not self._in_table:
            self._buffer.append(data)

    def _emit_table(self) -> None:
        """Render as GFM so the table survives chunking and reaches the model readable."""
        self._in_table = False
        rows = [row for row in self._table if row]
        self._table = []
        if not rows:
            return
        width = max(len(row) for row in rows)
        padded = [row + [""] * (width - len(row)) for row in rows]
        header, body = padded[0], padded[1:]
        lines = [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * width) + " |",
            *("| " + " | ".join(row) + " |" for row in body),
        ]
        self.blocks.append(Block(kind=BlockKind.TABLE, text="\n".join(lines)))

    def _flush(self) -> None:
        text = " ".join("".join(self._buffer).split())
        self._buffer = []
        if not text:
            self._kind, self._level = BlockKind.PARAGRAPH, 0
            return
        self.blocks.append(Block(kind=self._kind, text=text, level=self._level))
        self._kind, self._level = BlockKind.PARAGRAPH, 0

    def close(self) -> None:
        super().close()
        if self._in_table:
            self._emit_table()
        self._flush()


class HTMLLoader:
    name: str = "html"
    version: str = "1"
    cost: LoaderCost = "cheap"

    _EXTENSIONS = frozenset({"html", "htm", "xhtml"})

    def supports(self, probe: DocumentProbe) -> bool:
        return probe.extension in self._EXTENSIONS or probe.mime_type == "text/html"

    async def load(self, blob: bytes, probe: DocumentProbe) -> ExtractedDocument:
        parser = _HTMLBlockParser()
        parser.feed(_decode(blob))
        parser.close()
        return ExtractedDocument(
            blocks=tuple(parser.blocks),
            title=parser.title or _title_from_filename(probe.filename),
            parser=self.name,
            parser_version=self.version,
        )


def _decode(blob: bytes) -> str:
    """Decode with a BOM-aware fallback chain.

    Enterprise exports are full of UTF-16 and cp1252. Failing the whole ingest on a decode error
    is worse than a few replacement characters, so the last step never raises.
    """
    for encoding in ("utf-8-sig", "utf-8", "utf-16", "cp1252"):
        try:
            return blob.decode(encoding)
        except UnicodeDecodeError, LookupError:
            continue
    return blob.decode("utf-8", errors="replace")


def _title_from_filename(filename: str) -> str:
    stem = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    return stem.replace("_", " ").replace("-", " ").strip() or "Document"
