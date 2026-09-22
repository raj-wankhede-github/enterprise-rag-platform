"""Cursors, vector round-tripping and body shape for the Postgres backfill read.

The query itself needs a database and lives in the integration suite. What is testable here is
everything that decides whether a ten-hour rebuild survives an interruption, and whether the
documents it writes match the ones the ingest path writes -- which under ``dynamic: strict`` is
the difference between a working rebuild and a bulk rejection.
"""

from __future__ import annotations

import struct
import uuid
from typing import ClassVar

import pytest

from app.ingestion.types import BlockKind
from app.search.backfill_source import (
    _CHUNK_PHASE,
    _PARENT_PHASE,
    _format_cursor,
    _parse_cursor,
    chunk_body,
    decode_vector,
    encode_vector,
    parent_body,
)
from app.search.generations import chunk_document_id

TENANT = uuid.UUID("aaaa0000-0000-0000-0000-00000000000a")
VERSION = uuid.UUID("bbbb0000-0000-0000-0000-00000000000b")
DOCUMENT = uuid.UUID("cccc0000-0000-0000-0000-00000000000c")


# ------------------------------------------------------------------------------------------
# Cursors
# ------------------------------------------------------------------------------------------


def test_a_cursor_round_trips() -> None:
    cursor = _format_cursor(_CHUNK_PHASE, (TENANT, VERSION, 42))
    assert _parse_cursor(cursor) == (_CHUNK_PHASE, (TENANT, VERSION, 42))


def test_no_cursor_starts_at_the_chunk_phase() -> None:
    assert _parse_cursor(None) == (_CHUNK_PHASE, None)
    assert _parse_cursor("") == (_CHUNK_PHASE, None)


def test_a_parent_phase_cursor_skips_the_chunk_phase_entirely() -> None:
    """Otherwise a resume after the chunks finished re-walks the whole corpus to write nothing."""
    phase, key = _parse_cursor(_format_cursor(_PARENT_PHASE, (TENANT, VERSION, 7)))
    assert phase == _PARENT_PHASE
    assert key == (TENANT, VERSION, 7)


def test_cursors_order_the_same_way_the_keyset_query_does() -> None:
    """The cursor is compared as a tuple in SQL, so its parsed form must preserve ordinal order
    numerically -- string ordering would put ordinal 10 before ordinal 9."""
    keys = [_parse_cursor(_format_cursor(_CHUNK_PHASE, (TENANT, VERSION, n)))[1] for n in (9, 10, 100)]
    assert keys == sorted(keys)  # type: ignore[type-var]


# ------------------------------------------------------------------------------------------
# Vectors
# ------------------------------------------------------------------------------------------


def test_a_vector_round_trips_through_fp16() -> None:
    original = [0.125, -0.5, 0.0, 1.0]
    assert decode_vector(encode_vector(original)) == original


def test_fp16_precision_loss_stays_far_below_what_hnsw_already_costs() -> None:
    original = [0.1234567, -0.9876543, 0.5555555]
    recovered = decode_vector(encode_vector(original))
    assert recovered is not None
    assert all(abs(a - b) < 1e-3 for a, b in zip(original, recovered, strict=True))


def test_a_1024_dimensional_vector_occupies_two_kilobytes() -> None:
    """The economic claim that makes persisting every vector reasonable."""
    assert len(encode_vector([0.1] * 1024)) == 2048


def test_no_vector_decodes_to_none_rather_than_an_empty_list() -> None:
    """An empty list would index as a zero-length vector and be rejected by the mapping; None is
    the signal the caller checks before refusing to index the chunk at all."""
    assert decode_vector(None) is None
    assert decode_vector(b"") is None


def test_an_odd_trailing_byte_is_ignored_rather_than_raising() -> None:
    assert decode_vector(struct.pack("<2e", 1.0, 2.0) + b"\x00") == [1.0, 2.0]


# ------------------------------------------------------------------------------------------
# Document bodies
# ------------------------------------------------------------------------------------------


class FakeChunk:
    tenant_id = TENANT
    document_version_id = VERSION
    ordinal = 3
    text = "The per diem is 120 EUR."
    content_sha256 = b"\x01" * 32
    simhash64 = 12345
    token_count = 7
    page_from = 2
    page_to = 2
    block_kinds: ClassVar[list[str]] = ["paragraph"]
    id = uuid.UUID("dddd0000-0000-0000-0000-00000000000d")


class FakeParent:
    tenant_id = TENANT
    document_version_id = VERSION
    ordinal = 0
    heading_path = "Reimbursement > Per Diem"
    text = "Reimbursement section in full."
    token_count = 40
    page_from = 2
    page_to = 3


class FakeVersion:
    id = VERSION


class FakeDocument:
    id = DOCUMENT
    title = "Travel Policy"
    visibility_rank = 20
    allowed_groups: ClassVar[list[str]] = []
    denied_groups: ClassVar[list[str]] = []
    is_superseded = False
    source_system = "upload"
    doc_type = "policy"
    authority_rank = 0.5


def a_chunk_body(**overrides: object) -> dict[str, object]:
    body = chunk_body(
        chunk=FakeChunk(),  # type: ignore[arg-type]
        parent=FakeParent(),  # type: ignore[arg-type]
        version=FakeVersion(),  # type: ignore[arg-type]
        document=FakeDocument(),  # type: ignore[arg-type]
        chunk_id="abc123",
        embedding=[0.1] * 8,
        context_line="Clause 4.2 of the Travel Policy",
        fingerprint="fp0123456789abcd",
    )
    body.update(overrides)
    return body


def test_every_field_the_backfill_writes_is_declared_in_the_chunk_mapping() -> None:
    """The check that matters, and it has already caught two real bugs.

    ``dynamic: strict`` means an undeclared field is a rejected bulk item, and during a rebuild
    that means the whole generation fails reconciliation -- after the backfill has already run.
    Catching it here costs a millisecond instead of an hour.
    """
    from app.search.mappings import chunk_mapping

    declared = set(chunk_mapping(dimension=8)["properties"])
    written = set(a_chunk_body())
    assert written <= declared, f"undeclared fields: {sorted(written - declared)}"


def test_every_field_the_backfill_writes_to_parents_is_declared_too() -> None:
    """The parent mapping is not the chunk mapping minus a vector: it renames ``ordinal`` to
    ``parent_ordinal`` and drops the chunk-only fields."""
    from app.search.mappings import parent_mapping

    declared = set(parent_mapping()["properties"])
    written = set(parent_body(parent=FakeParent(), document=FakeDocument(), fingerprint="fp"))  # type: ignore[arg-type]
    assert written <= declared, f"undeclared fields: {sorted(written - declared)}"


def test_the_backfill_and_the_ingest_path_agree_on_every_shared_field() -> None:
    """Two writers into one index. Where they disagree, a rebuild silently changes what is
    indexed -- which is invisible until relevance moves and nobody knows why."""
    from app.ingestion.pipeline import index_action

    class FakePrepared:
        content_sha256 = b"" * 32
        context_line = "Clause 4.2 of the Travel Policy"
        embedding = [0.1] * 8
        simhash = 12345

        class chunk:  # noqa: N801 - mirrors the real attribute layout
            ordinal = 3
            text = "The per diem is 120 EUR."
            heading_path = "Reimbursement > Per Diem"
            # The ingest path holds BlockKind enums and projects `.value`; the backfill reads
            # them back from Postgres already as strings. Both must land as the same list.
            block_kinds: ClassVar[list[BlockKind]] = [BlockKind.PARAGRAPH]
            page_from = 2
            page_to = 2
            token_count = 7

    ingest = index_action(
        FakePrepared(),  # type: ignore[arg-type]
        tenant_id=TENANT,
        doc_id=DOCUMENT,
        doc_version_id=VERSION,
        parent_id=chunk_document_id(tenant_id=TENANT, doc_version_id=VERSION, ordinal=0),
        chunk_id="abc123",
        generation_fingerprint="fp0123456789abcd",
        title="Travel Policy",
        visibility_rank=20,
        access_groups=[],
    )
    backfill = a_chunk_body()
    shared = set(ingest) & set(backfill)
    differing = {key: (ingest[key], backfill[key]) for key in shared if ingest[key] != backfill[key]}
    assert not differing, f"the two writers disagree: {differing}"


def test_simhash_is_written_as_a_string_because_the_mapping_calls_it_a_keyword() -> None:
    """A bigint into a keyword field is coerced by OpenSearch rather than rejected, so this is
    exactly the kind of divergence that never surfaces as an error."""
    assert a_chunk_body()["simhash64"] == "12345"


def test_a_document_with_no_groups_indexes_the_wildcard() -> None:
    """An empty array would match no terms clause, making the document invisible to everyone."""
    assert a_chunk_body()["access_groups"] == ["*"]


def test_the_parent_id_is_the_parents_deterministic_id_not_its_row_id() -> None:
    """It is the key the parent leg projects through; a row id would link nothing."""
    expected = chunk_document_id(tenant_id=TENANT, doc_version_id=VERSION, ordinal=0)
    assert a_chunk_body()["parent_id"] == expected


def test_the_parent_body_carries_parent_id() -> None:
    """Dropped once already, and caught only by a dynamic: strict bulk rejection."""
    body = parent_body(parent=FakeParent(), document=FakeDocument(), fingerprint="fp")  # type: ignore[arg-type]
    assert body["parent_id"] == chunk_document_id(tenant_id=TENANT, doc_version_id=VERSION, ordinal=0)


def test_the_fingerprint_is_stamped_on_both_bodies() -> None:
    """Every query asserts it as a term filter, so a body without it is invisible."""
    assert a_chunk_body()["generation_fingerprint"] == "fp0123456789abcd"
    parent = parent_body(parent=FakeParent(), document=FakeDocument(), fingerprint="fp0123456789abcd")  # type: ignore[arg-type]
    assert parent["generation_fingerprint"] == "fp0123456789abcd"


def test_the_content_hash_is_hex_not_bytes() -> None:
    """JSON cannot carry bytes, and the serializer's fallback would produce a Python repr."""
    assert a_chunk_body()["content_sha256"] == "01" * 32


@pytest.mark.parametrize("field", ["tenant_id", "doc_id", "doc_version_id"])
def test_identifiers_are_strings_so_the_term_filter_matches(field: str) -> None:
    assert isinstance(a_chunk_body()[field], str)
