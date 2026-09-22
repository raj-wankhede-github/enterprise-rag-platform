"""Loaders, routing and the offline embedder."""

from __future__ import annotations

import pytest

from app.embeddings.base import l2_normalize
from app.embeddings.hashing import HashingEmbedder
from app.ingestion.loaders import default_registry, probe, sniff_mime
from app.ingestion.loaders.base import LoaderRegistry
from app.ingestion.loaders.text import HTMLLoader, MarkdownLoader, PlainTextLoader
from app.ingestion.types import BlockKind, DocumentProbe

# --------------------------------------------------------------------------------------------
# Routing
# --------------------------------------------------------------------------------------------


def test_magic_bytes_beat_a_lying_extension() -> None:
    """A PDF uploaded as report.txt must not be indexed as prose."""
    assert sniff_mime(b"%PDF-1.7\nstuff", "report.txt") == "application/pdf"
    assert sniff_mime(b"PK\x03\x04rest", "notes.txt") == "application/zip"


def test_extension_is_used_when_there_is_no_magic() -> None:
    assert sniff_mime(b"# Title\n", "notes.md") == "text/markdown"
    assert sniff_mime(b"hello", "unknown.bin") == "application/octet-stream"


def test_registry_routes_by_probe() -> None:
    registry = default_registry()
    assert registry.select(probe(b"# H\n", "a.md")).name == "markdown"
    assert registry.select(probe(b"<p>x</p>", "a.html")).name == "html"
    assert registry.select(probe(b"plain", "a.txt")).name == "text"


def test_registry_returns_none_for_a_format_it_cannot_handle() -> None:
    """A deployment without the parser container must fail clearly, not import-error."""
    assert default_registry().select(probe(b"%PDF-1.7", "scan.pdf")) is None


def test_cheap_loaders_are_preferred_over_expensive_ones() -> None:
    class ExpensiveEverything:
        name, version, cost = "docling", "1", "expensive"

        def supports(self, probe: DocumentProbe) -> bool:
            return True

        async def load(self, blob: bytes, probe: DocumentProbe):  # pragma: no cover
            raise AssertionError("should not be selected")

    registry = LoaderRegistry([ExpensiveEverything(), MarkdownLoader()])
    assert registry.select(probe(b"# H", "a.md")).name == "markdown"


# --------------------------------------------------------------------------------------------
# Markdown
# --------------------------------------------------------------------------------------------


async def load_md(text: str, name: str = "doc.md"):
    return await MarkdownLoader().load(text.encode("utf-8"), probe(text.encode("utf-8"), name))


async def test_atx_headings_carry_their_level() -> None:
    doc = await load_md("# One\n\ntext\n\n### Three\n\nmore")
    headings = [(b.text, b.level) for b in doc.blocks if b.kind is BlockKind.HEADING]
    assert headings == [("One", 1), ("Three", 3)]


async def test_setext_headings_are_recognised() -> None:
    doc = await load_md("Title\n=====\n\nbody text\n\nSub\n---\n\nmore")
    headings = [(b.text, b.level) for b in doc.blocks if b.kind is BlockKind.HEADING]
    assert headings == [("Title", 1), ("Sub", 2)]


async def test_title_comes_from_the_first_h1() -> None:
    assert (await load_md("# Travel Policy\n\nbody")).title == "Travel Policy"


async def test_title_falls_back_to_the_filename() -> None:
    assert (await load_md("body only", "employee_handbook.md")).title == "employee handbook"


async def test_gfm_table_becomes_one_table_block() -> None:
    doc = await load_md("| A | B |\n| --- | --- |\n| 1 | 2 |\n| 3 | 4 |")
    tables = [b for b in doc.blocks if b.kind is BlockKind.TABLE]
    assert len(tables) == 1
    assert "| 3 | 4 |" in tables[0].text


async def test_pipes_without_a_separator_row_are_not_a_table() -> None:
    """Prose containing pipes must not be misread as data."""
    doc = await load_md("| this is just | text with pipes |\n| and another line |")
    assert not [b for b in doc.blocks if b.kind is BlockKind.TABLE]


async def test_code_fence_is_kept_intact() -> None:
    doc = await load_md("```python\nif x:\n    pass\n```")
    code = [b for b in doc.blocks if b.kind is BlockKind.CODE]
    assert len(code) == 1
    assert "if x:" in code[0].text


async def test_headings_inside_a_code_fence_are_not_headings() -> None:
    doc = await load_md("```\n# not a heading\n```")
    assert not [b for b in doc.blocks if b.kind is BlockKind.HEADING]


async def test_list_items_group_into_a_list_block() -> None:
    doc = await load_md("- one\n- two\n- three")
    lists = [b for b in doc.blocks if b.kind is BlockKind.LIST]
    assert len(lists) == 1
    assert "three" in lists[0].text


async def test_footnotes_are_captured_with_their_label() -> None:
    doc = await load_md("Body text here.\n\n[^1]: The footnote body.")
    notes = [b for b in doc.blocks if b.kind is BlockKind.FOOTNOTE]
    assert len(notes) == 1
    assert notes[0].label == "1"


# --------------------------------------------------------------------------------------------
# HTML
# --------------------------------------------------------------------------------------------


async def load_html(text: str):
    return await HTMLLoader().load(text.encode("utf-8"), probe(text.encode("utf-8"), "a.html"))


async def test_html_headings_and_title() -> None:
    doc = await load_html("<html><head><title>Doc</title></head><body><h2>Sec</h2><p>text</p></body></html>")
    assert doc.title == "Doc"
    assert [(b.text, b.level) for b in doc.blocks if b.kind is BlockKind.HEADING] == [("Sec", 2)]


async def test_html_script_and_style_are_dropped() -> None:
    doc = await load_html("<body><script>var secret=1</script><style>p{}</style><p>real</p></body>")
    assert "secret" not in doc.text
    assert "real" in doc.text


async def test_html_table_is_rendered_as_gfm() -> None:
    """The table must survive chunking readable, since answers are often a single cell."""
    doc = await load_html("<table><tr><th>Grade</th><th>Rate</th></tr><tr><td>A</td><td>120</td></tr></table>")
    tables = [b for b in doc.blocks if b.kind is BlockKind.TABLE]
    assert len(tables) == 1
    lines = tables[0].text.split("\n")
    assert lines[0] == "| Grade | Rate |"
    assert set(lines[1].replace("|", "").strip()) <= set("-: ")
    assert lines[2] == "| A | 120 |"


async def test_html_ragged_table_rows_are_padded() -> None:
    doc = await load_html("<table><tr><th>A</th><th>B</th></tr><tr><td>1</td></tr></table>")
    table = next(b for b in doc.blocks if b.kind is BlockKind.TABLE)
    assert table.text.split("\n")[2].count("|") == 3


# --------------------------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize("encoding", ["utf-8", "utf-8-sig", "utf-16", "cp1252"])
async def test_common_enterprise_encodings_decode(encoding: str) -> None:
    text = "Ubung: naive cafe resume"
    blob = text.encode(encoding)
    doc = await PlainTextLoader().load(blob, probe(blob, "a.txt"))
    assert "cafe" in doc.text


async def test_undecodable_bytes_do_not_fail_the_ingest() -> None:
    """A few replacement characters beat losing the whole document."""
    blob = b"good text \xff\xfe\x00 more text"
    doc = await PlainTextLoader().load(blob, probe(blob, "a.txt"))
    assert "good text" in doc.text


# --------------------------------------------------------------------------------------------
# Hashing embedder
# --------------------------------------------------------------------------------------------


async def test_embeddings_are_deterministic() -> None:
    embedder = HashingEmbedder(64)
    first = await embedder.embed_query("travel expense policy")
    second = await embedder.embed_query("travel expense policy")
    assert first == second


async def test_vectors_are_unit_length() -> None:
    """The index uses innerproduct, which equals cosine only for unit vectors."""
    embedder = HashingEmbedder(128)
    vector = await embedder.embed_query("some text here")
    assert sum(v * v for v in vector) == pytest.approx(1.0, abs=1e-9)


async def test_dimension_is_respected() -> None:
    for dimension in (16, 64, 384):
        vector = await HashingEmbedder(dimension).embed_query("x")
        assert len(vector) == dimension


async def test_id_encodes_the_dimension_so_the_fingerprint_moves() -> None:
    assert HashingEmbedder(384).id != HashingEmbedder(768).id


async def test_similar_text_scores_higher_than_unrelated_text() -> None:
    embedder = HashingEmbedder(512)
    [target, near, far] = await embedder.embed_passages(
        [
            "the travel expense reimbursement policy for employees",
            "employee reimbursement policy covering travel expenses",
            "quarterly revenue forecast for the manufacturing division",
        ]
    )
    dot = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))  # noqa: E731
    assert dot(target, near) > dot(target, far)


async def test_empty_text_yields_a_zero_vector_not_a_crash() -> None:
    vector = await HashingEmbedder(32).embed_query("")
    assert vector == [0.0] * 32


async def test_batch_matches_single() -> None:
    embedder = HashingEmbedder(64)
    batch = await embedder.embed_passages(["alpha", "beta"])
    assert batch[0] == await embedder.embed_query("alpha")


def test_tiny_dimension_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 8"):
        HashingEmbedder(4)


def test_l2_normalize_handles_a_zero_vector() -> None:
    """A NaN here would poison every vector in the index."""
    assert l2_normalize([0.0, 0.0]) == [0.0, 0.0]
