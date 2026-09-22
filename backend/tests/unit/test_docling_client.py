"""The parser client: routing, mapping and failure behaviour.

The routing tests are the important ones. A scanned PDF sent to the cheap loader produces a
document of empty chunks that indexes cleanly and retrieves nothing -- an ingest that succeeds
and a document that never comes back from a search. That is the failure this routing exists to
prevent, and it is invisible to every other test in the suite.
"""

from __future__ import annotations

import httpx
import pytest

from app.ingestion.loaders import default_registry, probe
from app.ingestion.loaders.base import LoaderRegistry
from app.ingestion.loaders.docling_client import (
    DoclingClient,
    ParserUnavailableError,
    rows_to_markdown,
    to_extracted_document,
)
from app.ingestion.types import BlockKind, DocumentProbe


def transport(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def a_probe(**kwargs: object) -> DocumentProbe:
    base: dict[str, object] = {
        "filename": "doc.pdf",
        "mime_type": "application/pdf",
        "size_bytes": 1000,
    }
    base.update(kwargs)
    return DocumentProbe(**base)  # type: ignore[arg-type]


# --------------------------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("extension", ["pdf", "docx", "pptx", "xlsx", "png", "tiff"])
def test_layout_formats_are_claimed(extension: str) -> None:
    client = DoclingClient("http://parser")
    assert client.supports(a_probe(filename=f"doc.{extension}", mime_type="application/octet-stream"))


def test_plain_text_is_not_claimed() -> None:
    """A round trip to a vision model to read a .txt file is pure cost."""
    client = DoclingClient("http://parser")
    assert not client.supports(a_probe(filename="notes.txt", mime_type="text/plain"))
    assert not client.supports(a_probe(filename="notes.md", mime_type="text/markdown"))


def test_cheap_loaders_still_win_for_text_formats() -> None:
    registry = LoaderRegistry([*default_registry().loaders, DoclingClient("http://parser")])
    assert registry.select(probe(b"# Heading", "a.md")).name == "markdown"
    assert registry.select(probe(b"plain text", "a.txt")).name == "text"


def test_the_parser_picks_up_pdfs_the_cheap_loaders_cannot_handle() -> None:
    registry = LoaderRegistry([*default_registry().loaders, DoclingClient("http://parser")])
    assert registry.select(probe(b"%PDF-1.7 binary", "scan.pdf")).name == "docling"


def test_without_the_parser_a_pdf_fails_cleanly_rather_than_importing() -> None:
    """A deployment with the parse profile off must still start and still handle text."""
    assert default_registry().select(probe(b"%PDF-1.7", "scan.pdf")) is None


# --------------------------------------------------------------------------------------------
# Mapping the response onto our document model
# --------------------------------------------------------------------------------------------


def test_labels_map_onto_block_kinds() -> None:
    payload = {
        "blocks": [
            {"label": "title", "text": "Travel Policy", "level": 1},
            {"label": "section_header", "text": "Per Diem", "level": 2},
            {"label": "text", "text": "The rate is 120 EUR."},
            {"label": "list_item", "text": "First item"},
            {"label": "table", "text": "| A | B |\n| --- | --- |\n| 1 | 2 |"},
            {"label": "footnote", "text": "See annexe."},
        ]
    }
    document = to_extracted_document(payload, filename="doc.pdf", version="1")
    kinds = [block.kind for block in document.blocks]
    assert kinds == [
        BlockKind.HEADING,
        BlockKind.HEADING,
        BlockKind.PARAGRAPH,
        BlockKind.LIST,
        BlockKind.TABLE,
        BlockKind.FOOTNOTE,
    ]


def test_an_unknown_label_becomes_a_paragraph_and_is_warned_about() -> None:
    """Losing text silently is worse than mislabelling it: a missing paragraph is invisible."""
    payload = {"blocks": [{"label": "some_future_label", "text": "content here"}]}
    document = to_extracted_document(payload, filename="doc.pdf", version="1")
    assert document.blocks[0].kind is BlockKind.PARAGRAPH
    assert any("some_future_label" in warning for warning in document.warnings)


def test_an_out_of_range_heading_level_is_clamped_rather_than_raising() -> None:
    """Block validation would reject level 9; a parser quirk must not fail an ingest."""
    payload = {"blocks": [{"label": "section_header", "text": "Deep", "level": 9}]}
    document = to_extracted_document(payload, filename="doc.pdf", version="1")
    assert document.blocks[0].level == 6


def test_empty_blocks_are_dropped() -> None:
    payload = {"blocks": [{"label": "text", "text": "   "}, {"label": "text", "text": "real"}]}
    document = to_extracted_document(payload, filename="doc.pdf", version="1")
    assert len(document.blocks) == 1


def test_page_numbers_survive_for_citations() -> None:
    payload = {"blocks": [{"label": "text", "text": "content", "page": 7}]}
    assert to_extracted_document(payload, filename="d.pdf", version="1").blocks[0].page == 7


def test_ocr_text_is_marked_so_it_can_be_down_weighted() -> None:
    payload = {"blocks": [{"label": "text", "text": "scanned", "ocr": True}]}
    assert to_extracted_document(payload, filename="d.pdf", version="1").blocks[0].ocr


def test_the_title_falls_back_to_the_first_heading_then_the_filename() -> None:
    from_heading = to_extracted_document(
        {"blocks": [{"label": "title", "text": "Real Title", "level": 1}]},
        filename="doc.pdf",
        version="1",
    )
    assert from_heading.title == "Real Title"

    from_filename = to_extracted_document(
        {"blocks": [{"label": "text", "text": "body"}]}, filename="employee_handbook.pdf", version="1"
    )
    assert from_filename.title == "employee handbook"


def test_the_parser_and_its_version_are_recorded() -> None:
    """Both feed the generation fingerprint: a parser upgrade must force a rebuild."""
    document = to_extracted_document(
        {"blocks": [{"label": "text", "text": "x"}], "parser_version": "2.31.0"},
        filename="d.pdf",
        version="1",
    )
    assert document.parser == "docling"
    assert document.parser_version == "2.31.0"


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------


def test_rows_render_as_gfm() -> None:
    markdown = rows_to_markdown([["Grade", "Per diem"], ["A", "120"], ["B", "90"]])
    lines = markdown.split("\n")
    assert lines[0] == "| Grade | Per diem |"
    assert set(lines[1].replace("|", "").strip()) <= set("-: ")
    assert lines[2] == "| A | 120 |"


def test_a_ragged_row_is_padded_not_dropped() -> None:
    """A short row is still data, and it may be the entire answer."""
    markdown = rows_to_markdown([["A", "B", "C"], ["1"]])
    assert markdown.split("\n")[2].count("|") == 4


def test_a_pipe_inside_a_cell_is_escaped() -> None:
    markdown = rows_to_markdown([["Header"], ["a | b"]])
    assert "a \\| b" in markdown


def test_no_rows_renders_nothing() -> None:
    assert rows_to_markdown([]) == ""


# --------------------------------------------------------------------------------------------
# Failure behaviour
# --------------------------------------------------------------------------------------------


async def test_a_successful_parse_is_mapped() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "blocks": [{"label": "title", "text": "Scanned Policy", "level": 1}],
                "page_count": 4,
            },
        )

    async with transport(handler) as http:
        document = await DoclingClient("http://parser", client=http).load(b"%PDF", a_probe())
    assert document.title == "Scanned Policy"
    assert document.page_count == 4


async def test_an_unreachable_parser_raises_rather_than_returning_an_empty_document() -> None:
    """There is no cheaper way to read a scan, so this must fail the job and retry.

    Degrading here would produce a document that ingested successfully and contains nothing --
    exactly the silent failure the routing exists to prevent.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with transport(handler) as http:
        with pytest.raises(ParserUnavailableError, match="parser service"):
            await DoclingClient("http://parser", client=http).load(b"%PDF", a_probe())


async def test_an_unreadable_document_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(422, json={"detail": "parse failed: encrypted"})

    async with transport(handler) as http:
        with pytest.raises(ParserUnavailableError):
            await DoclingClient("http://parser", client=http).load(b"%PDF", a_probe())


async def test_health_reports_availability() -> None:
    def ok(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"status": "ok"})

    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    async with transport(ok) as http:
        assert await DoclingClient("http://parser", client=http).health()
    async with transport(down) as http:
        assert not await DoclingClient("http://parser", client=http).health()
