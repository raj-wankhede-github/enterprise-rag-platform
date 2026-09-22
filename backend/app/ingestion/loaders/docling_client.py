"""Client for the ``parser`` container: layout-aware parsing, OCR, table recovery.

Docling runs in its own container for the same reason the cross-encoder does: its dependency
tree (torch, vision models, OCR) is larger than the entire API image, and nothing in the request
path should carry it. A deployment that never ingests a scan never pays for it at all -- the
service sits behind a compose profile and the registry returns ``None`` when it is absent, so
cheap loaders still work and layout-heavy formats fail with a clean, actionable error rather
than an ImportError at startup.

What this converts is Docling's output into **our** ``ExtractedDocument``. That mapping is the
point of the module: the chunker must never learn which parser produced a document, or swapping
Docling out becomes a rewrite rather than a config change.

Tables are the reason any of this exists. A flattened table loses the row/column relationship,
and in enterprise documents the answer is very often a single cell -- a rate against a grade, a
retention period against a record type. Docling's TableFormer recovers that structure and we
render it as GitHub-flavoured Markdown, which survives chunking and reaches the model readable.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.ingestion.loaders.base import SCANNED_PAGE_CHAR_THRESHOLD
from app.ingestion.types import Block, BlockKind, DocumentProbe, ExtractedDocument, LoaderCost

logger = logging.getLogger(__name__)

#: Formats where layout genuinely carries meaning. Plain text and Markdown are handled by the
#: cheap in-process loaders and must never reach here -- a round trip to a vision model to parse
#: a .txt file is pure cost.
LAYOUT_FORMATS: frozenset[str] = frozenset({"pdf", "docx", "pptx", "xlsx", "png", "jpg", "jpeg", "tiff"})

#: Docling label -> our block kind. Unknown labels become paragraphs rather than being dropped:
#: losing text silently is worse than mislabelling it, because a missing paragraph is invisible
#: while a mislabelled one still retrieves.
_LABEL_MAP: dict[str, BlockKind] = {
    "title": BlockKind.HEADING,
    "section_header": BlockKind.HEADING,
    "paragraph": BlockKind.PARAGRAPH,
    "text": BlockKind.PARAGRAPH,
    "list_item": BlockKind.LIST,
    "table": BlockKind.TABLE,
    "code": BlockKind.CODE,
    "footnote": BlockKind.FOOTNOTE,
    "caption": BlockKind.FIGURE,
    "picture": BlockKind.FIGURE,
    "formula": BlockKind.PARAGRAPH,
}


class DoclingClient:
    """Routes layout-heavy documents to the parser service."""

    name = "docling"
    version = "1"
    cost: LoaderCost = "expensive"

    def __init__(
        self,
        base_url: str,
        *,
        timeout_s: float = 120.0,
        ocr: bool = True,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.ocr = ocr
        self._client = client

    def supports(self, probe: DocumentProbe) -> bool:
        """Claim layout formats, and text-layer PDFs only when the cheap loader cannot cope.

        ``chars_per_page`` is the decisive signal. A scanned page has a text layer of roughly
        nothing, and sending it to the cheap PDF loader produces a document of empty chunks that
        index cleanly and retrieve nothing -- an ingest that succeeds and a document that never
        comes back from a search.
        """
        if probe.extension in LAYOUT_FORMATS:
            return True
        if probe.mime_type == "application/pdf":
            return True
        return (
            probe.has_tables and probe.chars_per_page is not None and probe.chars_per_page < SCANNED_PAGE_CHAR_THRESHOLD
        )

    async def load(self, blob: bytes, probe: DocumentProbe) -> ExtractedDocument:
        payload = {
            "filename": probe.filename,
            "mime_type": probe.mime_type,
            "ocr": self.ocr,
        }
        files = {"file": (probe.filename, blob, probe.mime_type)}

        try:
            if self._client is not None:
                response = await self._client.post(
                    f"{self.base_url}/parse", data=payload, files=files, timeout=self.timeout_s
                )
            else:
                async with httpx.AsyncClient(timeout=self.timeout_s) as client:
                    response = await client.post(f"{self.base_url}/parse", data=payload, files=files)
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPError as exc:
            # Unlike a reranker outage, this cannot degrade: there is no cheaper way to read a
            # scanned page. Failing the job is correct -- it retries, and the document shows as
            # a failed ingest rather than as an empty one.
            raise ParserUnavailableError(f"the parser service at {self.base_url} failed: {exc}") from exc

        return to_extracted_document(body, filename=probe.filename, version=self.version)

    async def health(self) -> bool:
        try:
            if self._client is not None:
                response = await self._client.get(f"{self.base_url}/healthz", timeout=5.0)
            else:
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{self.base_url}/healthz")
            return response.status_code == 200
        except httpx.HTTPError:
            return False


class ParserUnavailableError(Exception):
    """The parser service could not be reached or could not read the document."""


def to_extracted_document(payload: dict[str, Any], *, filename: str, version: str) -> ExtractedDocument:
    """Map the parser's response onto our document model."""
    blocks: list[Block] = []
    warnings: list[str] = list(payload.get("warnings", ()))

    for item in payload.get("blocks", []):
        label = str(item.get("label", "text")).lower()
        text = str(item.get("text", "")).strip()
        if not text:
            continue

        kind = _LABEL_MAP.get(label, BlockKind.PARAGRAPH)
        if label not in _LABEL_MAP:
            warnings.append(f"unknown block label {label!r} treated as a paragraph")

        level = 0
        if kind is BlockKind.HEADING:
            # Clamp rather than reject: a parser reporting level 0 or 9 should not fail an
            # ingest, and Block's own validation would raise.
            level = max(1, min(6, int(item.get("level") or 1)))

        blocks.append(
            Block(
                kind=kind,
                text=text,
                level=level,
                page=item.get("page"),
                label=item.get("caption"),
                ocr=bool(item.get("ocr", False)),
            )
        )

    title = payload.get("title") or next(
        (block.text for block in blocks if block.kind is BlockKind.HEADING and block.level == 1),
        None,
    )
    return ExtractedDocument(
        blocks=tuple(blocks),
        title=title or _title_from_filename(filename),
        language=str(payload.get("language") or "en"),
        parser="docling",
        parser_version=str(payload.get("parser_version") or version),
        page_count=payload.get("page_count"),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _title_from_filename(filename: str) -> str:
    stem = filename.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    stem = stem.rsplit(".", 1)[0] if "." in stem else stem
    return stem.replace("_", " ").replace("-", " ").strip() or "Document"


def rows_to_markdown(rows: list[list[str]]) -> str:
    """Render a recovered table as GFM.

    Ragged rows are padded rather than dropped. A row the parser read short is still data, and
    discarding it loses a fact that may be the entire answer -- whereas an empty cell is visibly
    empty to both the model and the reader.
    """
    if not rows:
        return ""
    width = max(len(row) for row in rows)
    padded = [[cell.replace("|", "\\|").strip() for cell in row] + [""] * (width - len(row)) for row in rows]
    header, body = padded[0], padded[1:]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join(["---"] * width) + " |",
        *("| " + " | ".join(row) + " |" for row in body),
    ]
    return "\n".join(lines)
