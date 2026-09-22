"""Chunker behaviour.

Chunking decides what a vector can possibly represent, so these tests are about *boundaries*:
that sections are respected, that tables survive, that nothing useful is dropped, and that no
chunk is too small to answer anything.
"""

from __future__ import annotations

import pytest

from app.ingestion.chunker import CHUNKER_VERSION, ChunkerConfig, StructureAwareChunker
from app.ingestion.types import Block, BlockKind, ExtractedDocument
from app.utils.text import DEFAULT_TOKENIZER


def para(text: str, page: int | None = None) -> Block:
    return Block(kind=BlockKind.PARAGRAPH, text=text, page=page)


def heading(text: str, level: int = 1, page: int | None = None) -> Block:
    return Block(kind=BlockKind.HEADING, text=text, level=level, page=page)


def table(text: str, page: int | None = None) -> Block:
    return Block(kind=BlockKind.TABLE, text=text, page=page)


def doc(*blocks: Block, title: str = "Handbook") -> ExtractedDocument:
    return ExtractedDocument(blocks=blocks, title=title)


def words(count: int, stem: str = "word") -> str:
    """Prose long enough to exceed a size threshold, as whole sentences."""
    sentences = []
    made = 0
    while made < count:
        chunk = min(12, count - made)
        sentences.append(" ".join(f"{stem}{made + i}" for i in range(chunk)).capitalize() + ".")
        made += chunk
    return " ".join(sentences)


@pytest.fixture
def chunker() -> StructureAwareChunker:
    return StructureAwareChunker()


# --------------------------------------------------------------------------------------------
# Sectioning
# --------------------------------------------------------------------------------------------


def test_heading_path_is_built_from_the_heading_stack(chunker: StructureAwareChunker) -> None:
    result = chunker.chunk(
        doc(
            heading("Travel Policy", 1),
            heading("Reimbursement", 2),
            para(words(120)),
        )
    )
    assert result.parents[0].heading_path == "Handbook > Travel Policy > Reimbursement"


def test_sibling_heading_pops_the_stack(chunker: StructureAwareChunker) -> None:
    result = chunker.chunk(
        doc(
            heading("A", 1),
            heading("A1", 2),
            para(words(120)),
            heading("A2", 2),
            para(words(120)),
        )
    )
    paths = [parent.heading_path for parent in result.parents]
    assert paths == ["Handbook > A > A1", "Handbook > A > A2"]


def test_content_before_any_heading_is_kept(chunker: StructureAwareChunker) -> None:
    """Executive summaries live here and are often the most quotable part of a document."""
    result = chunker.chunk(doc(para(words(120)), heading("Body", 1), para(words(120))))
    assert len(result.parents) == 2
    assert result.parents[0].heading_path == "Handbook"


def test_a_heading_with_no_content_produces_no_parent(chunker: StructureAwareChunker) -> None:
    """An orphan parent would match the parent leg and expand to a heading with nothing under it."""
    result = chunker.chunk(doc(heading("Empty", 1), heading("Real", 1), para(words(120))))
    assert len(result.parents) == 1
    assert result.parents[0].heading_path == "Handbook > Real"


def test_children_never_cross_a_section_boundary(chunker: StructureAwareChunker) -> None:
    result = chunker.chunk(doc(heading("A", 1), para(words(100)), heading("B", 1), para(words(100))))
    by_parent: dict[int, set[str]] = {}
    for child in result.children:
        by_parent.setdefault(child.parent_ordinal, set()).add(child.heading_path)
    assert all(len(paths) == 1 for paths in by_parent.values())
    assert len(by_parent) == 2


# --------------------------------------------------------------------------------------------
# Sizes
# --------------------------------------------------------------------------------------------


def test_no_child_exceeds_the_hard_cap(chunker: StructureAwareChunker) -> None:
    result = chunker.chunk(doc(heading("Long", 1), para(words(4000))))
    assert result.children
    for child in result.children:
        assert child.token_count <= chunker.config.child_max_tokens


def test_small_blocks_merge_forward(chunker: StructureAwareChunker) -> None:
    """Ten one-line paragraphs must not become ten useless chunks."""
    result = chunker.chunk(doc(heading("S", 1), *[para(f"Line number {i}.") for i in range(10)]))
    assert len(result.children) == 1


def test_a_trailing_runt_merges_backwards(chunker: StructureAwareChunker) -> None:
    result = chunker.chunk(doc(heading("S", 1), para(words(280)), para("Short tail.")))
    assert all(
        child.token_count >= chunker.config.child_min_tokens or len(result.children) == 1 for child in result.children
    )


def test_oversized_section_splits_into_parts(chunker: StructureAwareChunker) -> None:
    result = chunker.chunk(doc(heading("Huge", 1), *[para(words(400)) for _ in range(12)]))
    assert len(result.parents) > 1
    assert {parent.part_count for parent in result.parents} == {len(result.parents)}
    for parent in result.parents:
        assert parent.token_count <= chunker.config.parent_max_tokens * 1.1


def test_every_child_points_at_a_real_parent(chunker: StructureAwareChunker) -> None:
    result = chunker.chunk(doc(heading("Huge", 1), *[para(words(400)) for _ in range(12)]))
    ordinals = {parent.ordinal for parent in result.parents}
    assert {child.parent_ordinal for child in result.children} <= ordinals


def test_ordinals_are_contiguous_and_ordered(chunker: StructureAwareChunker) -> None:
    result = chunker.chunk(doc(heading("A", 1), para(words(600)), heading("B", 1), para(words(600))))
    assert [c.ordinal for c in result.children] == list(range(len(result.children)))
    assert [p.ordinal for p in result.parents] == list(range(len(result.parents)))


def test_zero_overlap_between_adjacent_children(chunker: StructureAwareChunker) -> None:
    """With overlap, every adjacent pair is a near-duplicate of itself and dedup breaks."""
    result = chunker.chunk(doc(heading("S", 1), para(words(1500))))
    assert len(result.children) > 2
    for left, right in zip(result.children, result.children[1:], strict=False):
        assert not set(left.text.split()) & set(right.text.split()) or left.text != right.text
        assert right.text not in left.text


# --------------------------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------------------------

SMALL_TABLE = "| Grade | Per diem |\n| --- | --- |\n| A | 120 |\n| B | 90 |"


def test_a_small_table_is_kept_whole(chunker: StructureAwareChunker) -> None:
    result = chunker.chunk(doc(heading("Rates", 1), table(SMALL_TABLE), para(words(200))))
    table_children = [c for c in result.children if BlockKind.TABLE in c.block_kinds]
    assert len(table_children) == 1
    assert table_children[0].text == SMALL_TABLE


def test_a_table_is_never_merged_into_prose(chunker: StructureAwareChunker) -> None:
    """Gluing a paragraph onto Markdown makes the table unparseable for the model."""
    result = chunker.chunk(doc(heading("Rates", 1), para("Short intro."), table(SMALL_TABLE)))
    for child in result.children:
        if BlockKind.TABLE in child.block_kinds:
            assert child.text.startswith("| Grade")


def test_a_large_table_splits_by_rows_and_repeats_the_header() -> None:
    """Half a table with no header looks like data and cannot be read."""
    rows = "\n".join(f"| Item{i} | {i * 10} | description number {i} |" for i in range(200))
    big = f"| Item | Cost | Notes |\n| --- | --- | --- |\n{rows}"
    chunker = StructureAwareChunker(
        ChunkerConfig(child_min_tokens=40, child_target_tokens=150, child_max_tokens=200, parent_max_tokens=100_000)
    )
    result = chunker.chunk(doc(heading("Catalogue", 1), table(big)))

    parts = [c for c in result.children if BlockKind.TABLE in c.block_kinds]
    assert len(parts) > 1
    for part in parts:
        lines = part.text.split("\n")
        assert lines[0] == "| Item | Cost | Notes |", "every part must repeat the header"
        assert set(lines[1].replace("|", "").strip()) <= set("-: ")
        assert part.table_part is not None
    assert [p.table_part[0] for p in parts] == list(range(1, len(parts) + 1))
    assert {p.table_part[1] for p in parts} == {len(parts)}


def test_a_non_gfm_table_stays_whole_rather_than_being_mangled() -> None:
    chunker = StructureAwareChunker(
        ChunkerConfig(child_min_tokens=20, child_target_tokens=40, child_max_tokens=50, parent_max_tokens=100_000)
    )
    weird = "Item   Cost\n" + "\n".join(f"thing{i}   {i}" for i in range(100))
    result = chunker.chunk(doc(heading("T", 1), table(weird)))
    tables = [c for c in result.children if BlockKind.TABLE in c.block_kinds]
    assert len(tables) == 1


# --------------------------------------------------------------------------------------------
# Content preservation
# --------------------------------------------------------------------------------------------


def test_no_prose_is_silently_dropped(chunker: StructureAwareChunker) -> None:
    """The failure this guards: a chunk pipeline that quietly loses the last block."""
    markers = [f"Marker{i} is a distinctive token in paragraph {i}." for i in range(12)]
    result = chunker.chunk(doc(heading("S", 1), *[para(m) for m in markers]))
    combined = " ".join(child.text for child in result.children)
    for index in range(12):
        assert f"Marker{index}" in combined


def test_table_cells_survive_into_a_chunk(chunker: StructureAwareChunker) -> None:
    """Enterprise answers are often a number that appears only in a cell."""
    result = chunker.chunk(doc(heading("Rates", 1), table(SMALL_TABLE)))
    combined = " ".join(child.text for child in result.children)
    assert "120" in combined
    assert "Per diem" in combined


def test_page_ranges_propagate_for_citations(chunker: StructureAwareChunker) -> None:
    result = chunker.chunk(doc(heading("S", 1, page=3), para(words(100), page=3), para(words(100), page=4)))
    assert result.parents[0].page_from == 3
    assert result.parents[0].page_to == 4


def test_empty_document_produces_nothing(chunker: StructureAwareChunker) -> None:
    result = chunker.chunk(doc())
    assert result.parents == [] and result.children == []


def test_whitespace_only_blocks_are_ignored(chunker: StructureAwareChunker) -> None:
    result = chunker.chunk(doc(heading("S", 1), para("   \n\n  "), para(words(100))))
    assert len(result.children) == 1


def test_chunking_is_deterministic(chunker: StructureAwareChunker) -> None:
    """The CI ablation gate compares runs byte for byte."""
    document = doc(heading("A", 1), para(words(700)), table(SMALL_TABLE), heading("B", 2), para(words(300)))
    first = chunker.chunk(document)
    second = chunker.chunk(document)
    assert [c.text for c in first.children] == [c.text for c in second.children]
    assert [p.text for p in first.parents] == [p.text for p in second.parents]


def test_version_is_reported_so_the_fingerprint_can_include_it(chunker: StructureAwareChunker) -> None:
    assert chunker.chunk(doc(heading("A", 1), para(words(100)))).chunker_version == CHUNKER_VERSION


# --------------------------------------------------------------------------------------------
# Config validation
# --------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"child_min_tokens": 0},
        {"child_min_tokens": 400, "child_target_tokens": 300},
        {"child_target_tokens": 600, "child_max_tokens": 512},
        {"parent_max_tokens": 100},
    ],
)
def test_invalid_config_is_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        ChunkerConfig(**kwargs)


def test_block_rejects_an_impossible_heading_level() -> None:
    with pytest.raises(ValueError, match="heading level"):
        Block(kind=BlockKind.HEADING, text="x", level=9)


def test_non_heading_block_cannot_carry_a_level() -> None:
    with pytest.raises(ValueError, match="level 0"):
        Block(kind=BlockKind.PARAGRAPH, text="x", level=2)


def test_token_counts_are_recorded_consistently(chunker: StructureAwareChunker) -> None:
    result = chunker.chunk(doc(heading("S", 1), para(words(400))))
    for child in result.children:
        assert child.token_count == DEFAULT_TOKENIZER.count(child.text)
