"""The parser service: layout-aware parsing, OCR and table recovery.

Returns a flat, labelled block list rather than Docling's own document tree. The backend owns its
``ExtractedDocument`` model, and keeping the wire format dumb is what lets Docling be replaced
without touching the chunker.

Tables are rendered to GitHub-flavoured Markdown here, next to the model that recovered their
structure, so a table reaches the chunker already in the form that survives chunking.

The service is stateless: uploads land in a temp file that is removed on the way out. Nothing
customer-specific persists between requests, which is what lets it be scaled and restarted
freely, and what keeps it out of scope for most of a security questionnaire.
"""

from __future__ import annotations

import logging
import os
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
logger = logging.getLogger("parser")

MAX_BYTES = int(os.environ.get("MAX_BYTES", str(200 * 1024 * 1024)))

_state: dict[str, Any] = {"ready": False}


def _load() -> None:
    """Build the converter once, at startup.

    Docling loads several models on first use. Lazily would make the first real document pay a
    multi-second penalty and would let an orchestrator route traffic to a replica that cannot
    yet serve it, so /healthz stays 503 until this returns.
    """
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption

    options = PdfPipelineOptions()
    options.do_ocr = os.environ.get("ENABLE_OCR", "1") == "1"
    options.do_table_structure = True
    options.table_structure_options.do_cell_matching = True

    _state["converter"] = DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=options)}
    )
    _state["ready"] = True
    logger.info("parser ready (ocr=%s)", options.do_ocr)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    _load()
    yield


app = FastAPI(title="parser", version="1.0.0", lifespan=lifespan)


@app.get("/healthz")
async def healthz() -> JSONResponse:
    if not _state.get("ready"):
        return JSONResponse({"status": "loading"}, status_code=503)
    return JSONResponse({"status": "ok"})


@app.post("/parse")
async def parse(
    file: UploadFile = File(...),
    filename: str = Form(default=""),
    mime_type: str = Form(default=""),
    ocr: bool = Form(default=True),
) -> JSONResponse:
    started = time.perf_counter()
    blob = await file.read()
    if len(blob) > MAX_BYTES:
        return JSONResponse({"detail": "file too large"}, status_code=413)

    name = filename or file.filename or "document"
    suffix = Path(name).suffix or ".pdf"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as handle:
        handle.write(blob)
        path = Path(handle.name)

    try:
        result = _state["converter"].convert(str(path))
        blocks, warnings = _flatten(result.document)
        return JSONResponse(
            {
                "blocks": blocks,
                "title": getattr(result.document, "name", None) or None,
                "page_count": len(getattr(result.document, "pages", ()) or ()),
                "parser_version": _docling_version(),
                "warnings": warnings,
                "latency_ms": (time.perf_counter() - started) * 1000.0,
            }
        )
    except Exception as exc:
        # 422, not 500: the document could not be read, which is a property of the document.
        # The job retries and the document shows as a failed ingest rather than an empty one.
        logger.exception("parse failed for %s", name)
        return JSONResponse({"detail": f"parse failed: {exc}"}, status_code=422)
    finally:
        path.unlink(missing_ok=True)


def _flatten(document: Any) -> tuple[list[dict[str, Any]], list[str]]:
    """Walk the Docling document into a flat labelled list, in reading order."""
    blocks: list[dict[str, Any]] = []
    warnings: list[str] = []

    for item, _level in document.iterate_items():
        label = str(getattr(item, "label", "text"))
        page = _page_of(item)

        if label == "table":
            rows = _table_rows(item)
            if not rows:
                # Worth surfacing: a table detected but not recovered means the numbers in it
                # are simply absent from the index, and the ingest would otherwise look clean.
                warnings.append("a table was detected but no cells were recovered")
                continue
            blocks.append(
                {"label": "table", "text": _to_markdown(rows), "page": page, "caption": _caption_of(item)}
            )
            continue

        text = (getattr(item, "text", "") or "").strip()
        if not text:
            continue

        entry: dict[str, Any] = {"label": label, "text": text, "page": page}
        if label in ("title", "section_header"):
            entry["level"] = 1 if label == "title" else int(getattr(item, "level", 2) or 2)
        blocks.append(entry)

    return blocks, warnings


def _table_rows(item: Any) -> list[list[str]]:
    data = getattr(item, "data", None)
    grid = getattr(data, "grid", None)
    if not grid:
        return []
    return [[(getattr(cell, "text", "") or "").strip() for cell in row] for row in grid]


def _to_markdown(rows: list[list[str]]) -> str:
    width = max(len(row) for row in rows)
    padded = [[cell.replace("|", "\\|") for cell in row] + [""] * (width - len(row)) for row in rows]
    header, body = padded[0], padded[1:]
    return "\n".join(
        [
            "| " + " | ".join(header) + " |",
            "| " + " | ".join(["---"] * width) + " |",
            *("| " + " | ".join(row) + " |" for row in body),
        ]
    )


def _page_of(item: Any) -> int | None:
    provenance = getattr(item, "prov", None) or []
    return getattr(provenance[0], "page_no", None) if provenance else None


def _caption_of(item: Any) -> str | None:
    captions = getattr(item, "captions", None) or []
    if not captions:
        return None
    return (getattr(captions[0], "text", "") or "").strip() or None


def _docling_version() -> str:
    try:
        from importlib.metadata import version

        return version("docling")
    except Exception:
        return "unknown"
