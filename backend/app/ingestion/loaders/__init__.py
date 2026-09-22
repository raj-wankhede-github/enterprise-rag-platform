"""Document loaders."""

from __future__ import annotations

from app.ingestion.loaders.base import (
    SCANNED_PAGE_CHAR_THRESHOLD,
    DocumentLoader,
    LoaderRegistry,
)
from app.ingestion.loaders.registry import default_registry, probe, sniff_mime
from app.ingestion.loaders.text import HTMLLoader, MarkdownLoader, PlainTextLoader

__all__ = [
    "SCANNED_PAGE_CHAR_THRESHOLD",
    "DocumentLoader",
    "HTMLLoader",
    "LoaderRegistry",
    "MarkdownLoader",
    "PlainTextLoader",
    "default_registry",
    "probe",
    "sniff_mime",
]
