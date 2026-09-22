"""Default loader wiring and probe construction."""

from __future__ import annotations

from app.ingestion.loaders.base import LoaderRegistry
from app.ingestion.loaders.text import HTMLLoader, MarkdownLoader, PlainTextLoader
from app.ingestion.types import DocumentProbe

#: Magic bytes -> mime type, for the cases where the extension lies. An uploaded "report.txt"
#: that is actually a PDF must not be handed to the plain-text loader, which would index its
#: binary header as prose.
_MAGIC: list[tuple[bytes, str]] = [
    (b"%PDF-", "application/pdf"),
    (b"PK\x03\x04", "application/zip"),  # docx/xlsx/pptx are zip containers
    (b"\xd0\xcf\x11\xe0", "application/x-ole-storage"),  # legacy doc/xls/ppt
    (b"\x89PNG", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
]

_EXTENSION_MIME: dict[str, str] = {
    "txt": "text/plain",
    "log": "text/plain",
    "csv": "text/csv",
    "md": "text/markdown",
    "markdown": "text/markdown",
    "html": "text/html",
    "htm": "text/html",
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
}


def sniff_mime(blob: bytes, filename: str) -> str:
    """Magic bytes win over the extension, because the extension is user-supplied."""
    head = blob[:8]
    for magic, mime in _MAGIC:
        if head.startswith(magic):
            return mime
    _, _, ext = filename.rpartition(".")
    return _EXTENSION_MIME.get(ext.lower(), "application/octet-stream")


def probe(blob: bytes, filename: str) -> DocumentProbe:
    return DocumentProbe(
        filename=filename,
        mime_type=sniff_mime(blob, filename),
        size_bytes=len(blob),
        magic=blob[:8],
    )


def default_registry() -> LoaderRegistry:
    """Cheap loaders only.

    The Docling client registers itself on top when ``PARSER_SERVICE_URL`` is configured, so a
    deployment without the parser container still handles text formats and fails clearly on
    everything else rather than importing a dependency it does not have.
    """
    return LoaderRegistry([PlainTextLoader(), MarkdownLoader(), HTMLLoader()])
