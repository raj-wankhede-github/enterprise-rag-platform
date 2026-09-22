"""End-to-end ingestion, with no database and no cluster.

The tests that matter most here are the reuse ones. "Embeddings are never computed twice for
identical text" is a claim the product makes about cost, so it gets a spy on the embedder rather
than a comment asserting it.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

import pytest

from app.embeddings.hashing import HashingEmbedder
from app.ingestion.chunker import StructureAwareChunker
from app.ingestion.contextualize import DocumentContext, TemplateContextualizer, prefixed
from app.ingestion.loaders import default_registry
from app.ingestion.pipeline import IngestionPipeline, UnsupportedDocumentError, index_action
from app.ingestion.types import BlockKind
from app.ingestion.versioning import sha256_text
from app.search.dsl import PUBLIC_GROUP
from app.search.mappings import searchable_fields

TENANT = uuid.UUID("22222222-2222-2222-2222-222222222222")

DOC = """# Travel Policy

## Per Diem

Employees travelling on company business may claim a per diem allowance. The allowance depends
on the destination grade and is paid without receipts. Claims must be submitted within thirty
days of return, using the expense portal, and approved by a line manager before payment.

| Grade | Per diem |
| --- | --- |
| A | 120 |
| B | 90 |

## Receipts

Receipts are required for any single item over 50 EUR. The reference for this rule is SEC-4.2.1
and questions should be directed to the finance team.
"""


class SpyEmbedder(HashingEmbedder):
    """Counts how many texts were actually embedded."""

    def __init__(self, dimension: int = 64) -> None:
        super().__init__(dimension)
        self.embedded: list[str] = []

    async def embed_passages(self, texts: Sequence[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return await super().embed_passages(texts)


class SpyContextualizer(TemplateContextualizer):
    def __init__(self) -> None:
        self.calls: list[int] = []

    async def contextualize(self, document, chunks):  # type: ignore[no-untyped-def]
        self.calls.append(len(chunks))
        return await super().contextualize(document, chunks)


def build(embedder: HashingEmbedder | None = None, contextualizer=None) -> IngestionPipeline:  # type: ignore[no-untyped-def]
    return IngestionPipeline(
        registry=default_registry(),
        chunker=StructureAwareChunker(),
        contextualizer=contextualizer or TemplateContextualizer(),
        embedder=embedder or HashingEmbedder(64),
    )


@pytest.fixture
def context() -> DocumentContext:
    return DocumentContext(title="Travel Policy", doc_type="policy", source_system="sharepoint")


# --------------------------------------------------------------------------------------------
# Happy path
# --------------------------------------------------------------------------------------------


async def test_pipeline_produces_chunks_with_vectors(context: DocumentContext) -> None:
    result = await build().run(DOC.encode(), "travel.md", context=context)
    assert result.prepared
    assert all(item.embedding is not None for item in result.prepared)
    assert all(len(item.embedding) == 64 for item in result.prepared)


async def test_sections_become_parents(context: DocumentContext) -> None:
    result = await build().run(DOC.encode(), "travel.md", context=context)
    paths = {parent.heading_path for parent in result.parents}
    assert any("Per Diem" in path for path in paths)
    assert any("Receipts" in path for path in paths)


async def test_the_table_survives_into_a_chunk(context: DocumentContext) -> None:
    """The per-diem numbers exist only in cells; losing the table loses the answer."""
    result = await build().run(DOC.encode(), "travel.md", context=context)
    tables = [i for i in result.prepared if BlockKind.TABLE in i.chunk.block_kinds]
    assert len(tables) == 1
    assert "120" in tables[0].chunk.text


async def test_hashes_are_recorded_for_dedup(context: DocumentContext) -> None:
    result = await build().run(DOC.encode(), "travel.md", context=context)
    assert len(result.blob_sha256) == 32
    assert len(result.doc_text_sha256) == 32
    assert all(len(item.content_sha256) == 32 for item in result.prepared)


async def test_generation_fingerprint_is_stamped(context: DocumentContext) -> None:
    result = await build().run(DOC.encode(), "travel.md", context=context)
    assert len(result.generation_fingerprint) == 16


async def test_an_unsupported_format_raises_rather_than_indexing_nothing() -> None:
    """A silently-skipped document appears in the UI and never returns from a search."""
    with pytest.raises(UnsupportedDocumentError, match="PARSER_SERVICE_URL"):
        await build().run(b"%PDF-1.7 binary", "scan.pdf")


# --------------------------------------------------------------------------------------------
# Contextual retrieval
# --------------------------------------------------------------------------------------------


async def test_context_line_names_the_section_and_document(context: DocumentContext) -> None:
    result = await build().run(DOC.encode(), "travel.md", context=context)
    per_diem = next(i for i in result.prepared if "per diem allowance" in i.chunk.text)
    assert "Per Diem" in per_diem.context_line
    assert "Travel Policy" in per_diem.context_line
    assert "policy" in per_diem.context_line


async def test_indexed_text_is_the_context_line_plus_the_chunk(context: DocumentContext) -> None:
    """Both must be embedded, and BM25 must see the same string the vector describes."""
    result = await build().run(DOC.encode(), "travel.md", context=context)
    item = result.prepared[0]
    assert item.indexed_text == prefixed(item.context_line, item.chunk.text)
    assert item.chunk.text in item.indexed_text


async def test_context_line_is_stored_separately_from_content(context: DocumentContext) -> None:
    """Citations must quote source text, not a synthesized sentence."""
    result = await build().run(DOC.encode(), "travel.md", context=context)
    item = result.prepared[0]
    assert item.context_line not in item.chunk.text


async def test_cached_context_lines_are_not_regenerated(context: DocumentContext) -> None:
    spy = SpyContextualizer()
    first = await build(contextualizer=spy).run(DOC.encode(), "travel.md", context=context)
    cache = {i.content_sha256: i.context_line for i in first.prepared}

    spy.calls.clear()
    await build(contextualizer=spy).run(DOC.encode(), "travel.md", context=context, context_cache=cache)
    assert spy.calls == [], "every context line was cached; none should be generated"


# --------------------------------------------------------------------------------------------
# Reuse economics -- the claims the product makes about cost
# --------------------------------------------------------------------------------------------


async def test_identical_reingest_embeds_nothing(context: DocumentContext) -> None:
    spy = SpyEmbedder()
    first = await build(spy).run(DOC.encode(), "travel.md", context=context)
    cache = {i.content_sha256: i.embedding for i in first.prepared}

    spy.embedded.clear()
    second = await build(spy).run(DOC.encode(), "travel.md", context=context, vector_cache=cache)
    assert spy.embedded == [], "identical text must never be embedded twice"
    assert second.chunks_embedded == 0
    assert second.chunks_reused == len(second.prepared)


async def test_one_paragraph_edit_embeds_only_the_changed_chunk(context: DocumentContext) -> None:
    spy = SpyEmbedder()
    first = await build(spy).run(DOC.encode(), "travel.md", context=context)
    cache = {i.content_sha256: i.embedding for i in first.prepared}

    edited = DOC.replace("over 50 EUR", "over 75 EUR")
    spy.embedded.clear()
    second = await build(spy).run(edited.encode(), "travel.md", context=context, vector_cache=cache)

    assert second.chunks_embedded == 1
    assert len(spy.embedded) == 1
    assert second.reuse_ratio > 0.5


async def test_diff_reports_added_and_removed(context: DocumentContext) -> None:
    first = await build().run(DOC.encode(), "travel.md", context=context)
    previous = [i.content_sha256 for i in first.prepared]

    edited = DOC.replace("over 50 EUR", "over 75 EUR")
    second = await build().run(edited.encode(), "travel.md", context=context, previous_chunk_hashes=previous)

    assert second.diff is not None
    assert second.diff.embeddings_required == 1
    assert len(second.diff.removed) == 1
    assert second.diff.reuse_ratio > 0.5


async def test_a_shared_chunk_from_another_document_is_reused(context: DocumentContext) -> None:
    """Case 2 of the dedup design: the same content filed twice costs one embedding."""
    spy = SpyEmbedder()
    first = await build(spy).run(DOC.encode(), "policy-a.md", context=context)
    cache = {i.content_sha256: i.embedding for i in first.prepared}

    spy.embedded.clear()
    other = await build(spy).run(DOC.encode(), "policy-b.md", context=context, vector_cache=cache)
    assert spy.embedded == []
    assert other.chunks_reused == len(other.prepared)


async def test_reuse_is_keyed_on_text_not_on_position(context: DocumentContext) -> None:
    """Reordering sections must not re-embed anything."""
    spy = SpyEmbedder()
    first = await build(spy).run(DOC.encode(), "a.md", context=context)
    cache = {i.content_sha256: i.embedding for i in first.prepared}

    head, _, tail = DOC.partition("## Receipts")
    reordered = head.split("## Per Diem")[0] + "## Receipts" + tail + "## Per Diem" + head.split("## Per Diem")[1]
    spy.embedded.clear()
    await build(spy).run(reordered.encode(), "a.md", context=context, vector_cache=cache)
    assert spy.embedded == [], "same text in a different order must not be re-embedded"


# --------------------------------------------------------------------------------------------
# Index action
# --------------------------------------------------------------------------------------------


async def test_index_action_only_uses_mapped_fields(context: DocumentContext) -> None:
    """Under dynamic: strict an unmapped field fails the bulk write."""
    result = await build().run(DOC.encode(), "travel.md", context=context)
    action = index_action(
        result.prepared[0],
        tenant_id=TENANT,
        doc_id=uuid.uuid4(),
        doc_version_id=uuid.uuid4(),
        parent_id="p1",
        chunk_id="c1",
        generation_fingerprint=result.generation_fingerprint,
        title="Travel Policy",
        visibility_rank=10,
        access_groups=[],
    )
    unmapped = set(action) - searchable_fields()
    assert not unmapped, f"index_action writes unmapped fields: {sorted(unmapped)}"


async def test_index_action_defaults_to_the_public_group(context: DocumentContext) -> None:
    """Unrestricted must be a positive match, not the absence of a field."""
    result = await build().run(DOC.encode(), "travel.md", context=context)
    action = index_action(
        result.prepared[0],
        tenant_id=TENANT,
        doc_id=uuid.uuid4(),
        doc_version_id=uuid.uuid4(),
        parent_id="p1",
        chunk_id="c1",
        generation_fingerprint="x" * 16,
        title="T",
        visibility_rank=10,
        access_groups=[],
    )
    assert action["access_groups"] == [PUBLIC_GROUP]


async def test_content_field_is_verbatim_source_text(context: DocumentContext) -> None:
    result = await build().run(DOC.encode(), "travel.md", context=context)
    item = result.prepared[0]
    action = index_action(
        item,
        tenant_id=TENANT,
        doc_id=uuid.uuid4(),
        doc_version_id=uuid.uuid4(),
        parent_id="p",
        chunk_id="c",
        generation_fingerprint="x" * 16,
        title="T",
        visibility_rank=10,
        access_groups=[],
    )
    assert action["content"] == item.chunk.text
    assert action["context_line"] == item.context_line


# --------------------------------------------------------------------------------------------
# Determinism
# --------------------------------------------------------------------------------------------


async def test_ingestion_is_deterministic(context: DocumentContext) -> None:
    first = await build().run(DOC.encode(), "travel.md", context=context)
    second = await build().run(DOC.encode(), "travel.md", context=context)
    assert [i.content_sha256 for i in first.prepared] == [i.content_sha256 for i in second.prepared]
    assert [i.embedding for i in first.prepared] == [i.embedding for i in second.prepared]
    assert first.generation_fingerprint == second.generation_fingerprint


async def test_chunk_hash_is_unicode_normalized() -> None:
    assert sha256_text("résumé") == sha256_text("résumé")
