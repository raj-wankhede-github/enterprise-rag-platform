"""The four deduplication cases, plus the role interaction.

These tests are the executable specification of "what happens when the same document is uploaded
twice". They run with no database and no object store, because the decision is a pure function.
"""

from __future__ import annotations

import uuid

import pytest

from app.ingestion.versioning import (
    ContentMatch,
    Decision,
    ExistingVersion,
    UploadRequest,
    derive_external_id,
    diff_chunks,
    normalize_filename,
    plan_upload,
    resolve_visibility_rank,
    sha256_bytes,
    sha256_text,
)
from app.models.enums import UploadOutcome
from app.security.capabilities import RANK, Role

TENANT = uuid.UUID("00000000-0000-0000-0000-0000000000aa")
DOC = uuid.UUID("00000000-0000-0000-0000-0000000000d1")
VERSION = uuid.UUID("00000000-0000-0000-0000-0000000000e1")

BYTES_A = sha256_bytes(b"the original policy")
BYTES_B = sha256_bytes(b"the corrected policy")


def request(**kwargs: object) -> UploadRequest:
    base: dict[str, object] = {
        "tenant_id": TENANT,
        "blob_sha256": BYTES_A,
        "filename": "policy.pdf",
        "size_bytes": 1234,
    }
    base.update(kwargs)
    return UploadRequest(**base)  # type: ignore[arg-type]


def existing(blob: bytes = BYTES_A, *, rank: int = 10, version_no: int = 3) -> ExistingVersion:
    return ExistingVersion(
        document_id=DOC,
        version_id=VERSION,
        version_no=version_no,
        blob_sha256=blob,
        visibility_rank=rank,
    )


# --------------------------------------------------------------------------------------------
# Case 1 - identical bytes, same logical document
# --------------------------------------------------------------------------------------------


def test_case1_identical_reupload_creates_no_version() -> None:
    plan = plan_upload(request(), existing=existing(), content_match=None)
    assert plan.decision is Decision.REUSE_EXISTING_VERSION
    assert plan.outcome is UploadOutcome.DEDUPLICATED
    assert plan.version_id == VERSION
    assert plan.next_version_no == 3, "the version number must not advance"
    assert plan.needs_indexing is False
    assert plan.reuse_embeddings is True
    assert plan.creates_work is False


def test_case1_is_idempotent_across_repeated_attempts() -> None:
    """At-least-once job delivery and connector re-syncs depend on this."""
    plans = [plan_upload(request(), existing=existing(), content_match=None) for _ in range(5)]
    assert {p.decision for p in plans} == {Decision.REUSE_EXISTING_VERSION}
    assert {p.next_version_no for p in plans} == {3}


@pytest.mark.parametrize("role", [Role.ADMIN, Role.DEV])
def test_case1_behaves_identically_for_admin_and_dev(role: Role) -> None:
    plan = plan_upload(
        request(actor_rank=RANK[role], actor_may_set_any_visibility=role is Role.ADMIN),
        existing=existing(),
        content_match=None,
    )
    assert plan.decision is Decision.REUSE_EXISTING_VERSION


# --------------------------------------------------------------------------------------------
# Case 2 - identical bytes, different logical document
# --------------------------------------------------------------------------------------------


def test_case2_same_content_new_document_reuses_everything_but_still_indexes() -> None:
    match = ContentMatch(document_id=DOC, version_id=VERSION, visibility_rank=10)
    plan = plan_upload(request(actor_rank=30), existing=None, content_match=match)
    assert plan.decision is Decision.CREATE_DOCUMENT_REUSING_CONTENT
    assert plan.outcome is UploadOutcome.NEW_DOCUMENT_SHARED_CONTENT
    assert plan.reuse_parse is True
    assert plan.reuse_embeddings is True
    # Indexed again on purpose: different ACLs and metadata must be independently filterable.
    assert plan.needs_indexing is True
    assert plan.next_version_no == 1


def test_case2_applies_when_the_same_file_lands_in_a_different_collection() -> None:
    other_collection = uuid.uuid4()
    match = ContentMatch(document_id=DOC, version_id=VERSION, visibility_rank=10)
    plan = plan_upload(
        request(collection_id=other_collection, actor_rank=30),
        existing=None,
        content_match=match,
    )
    assert plan.decision is Decision.CREATE_DOCUMENT_REUSING_CONTENT


# --------------------------------------------------------------------------------------------
# Case 3 - same logical document, changed bytes
# --------------------------------------------------------------------------------------------


def test_case3_changed_bytes_create_the_next_version() -> None:
    plan = plan_upload(request(blob_sha256=BYTES_B), existing=existing(version_no=3), content_match=None)
    assert plan.decision is Decision.CREATE_NEW_VERSION
    assert plan.outcome is UploadOutcome.NEW_VERSION
    assert plan.document_id == DOC
    assert plan.next_version_no == 4
    assert plan.reuse_parse is False, "different bytes must be re-parsed"
    assert plan.reuse_embeddings is True, "the per-chunk diff still reuses most vectors"
    assert plan.needs_indexing is True


def test_case3_takes_priority_over_a_content_match_elsewhere() -> None:
    """Identity wins: this document changed, even if some other document holds the new bytes."""
    match = ContentMatch(document_id=uuid.uuid4(), version_id=uuid.uuid4(), visibility_rank=10)
    plan = plan_upload(request(blob_sha256=BYTES_B), existing=existing(), content_match=match)
    assert plan.decision is Decision.CREATE_NEW_VERSION


# --------------------------------------------------------------------------------------------
# Case 4 - content exists, uploader may not see it
# --------------------------------------------------------------------------------------------


def test_case4_dev_uploading_admin_only_content_is_rejected() -> None:
    match = ContentMatch(document_id=DOC, version_id=VERSION, visibility_rank=RANK[Role.ADMIN])
    plan = plan_upload(request(actor_rank=RANK[Role.DEV]), existing=None, content_match=match)
    assert plan.decision is Decision.REJECT_NOT_PERMITTED
    assert plan.outcome is UploadOutcome.REJECTED_NO_PERMISSION
    assert plan.is_rejection
    assert plan.needs_indexing is False


def test_case4_message_discloses_nothing() -> None:
    match = ContentMatch(document_id=DOC, version_id=VERSION, visibility_rank=RANK[Role.ADMIN])
    plan = plan_upload(request(actor_rank=RANK[Role.DEV]), existing=None, content_match=match)
    lowered = plan.reason.lower()
    assert str(DOC) not in plan.reason
    assert str(VERSION) not in plan.reason
    for leak in ("admin-only", "title", "owner", "rank 40"):
        assert leak not in lowered
    assert "do not have permission" in lowered


def test_case4_does_not_fire_for_an_admin() -> None:
    match = ContentMatch(document_id=DOC, version_id=VERSION, visibility_rank=RANK[Role.ADMIN])
    plan = plan_upload(request(actor_rank=RANK[Role.ADMIN]), existing=None, content_match=match)
    assert plan.decision is Decision.CREATE_DOCUMENT_REUSING_CONTENT


def test_case4_never_creates_a_shadow_duplicate() -> None:
    """The failure mode this guards: a copy visible at the uploader's lower rank."""
    match = ContentMatch(document_id=DOC, version_id=VERSION, visibility_rank=RANK[Role.ADMIN])
    for rank in (RANK[Role.PROD], RANK[Role.TEST], RANK[Role.DEV]):
        plan = plan_upload(request(actor_rank=rank), existing=None, content_match=match)
        assert plan.document_id is None
        assert plan.decision is Decision.REJECT_NOT_PERMITTED


# --------------------------------------------------------------------------------------------
# Genuinely new content
# --------------------------------------------------------------------------------------------


def test_new_content_creates_a_document() -> None:
    plan = plan_upload(request(), existing=None, content_match=None)
    assert plan.decision is Decision.CREATE_DOCUMENT
    assert plan.outcome is UploadOutcome.NEW_DOCUMENT
    assert plan.reuse_parse is False
    assert plan.next_version_no == 1


# --------------------------------------------------------------------------------------------
# Visibility ceiling
# --------------------------------------------------------------------------------------------


def test_dev_cannot_publish_above_their_own_rank() -> None:
    req = request(actor_rank=RANK[Role.DEV], requested_visibility_rank=RANK[Role.ADMIN])
    assert resolve_visibility_rank(req, tenant_default=10) == RANK[Role.DEV]


def test_admin_may_publish_at_any_rank() -> None:
    req = request(
        actor_rank=RANK[Role.ADMIN],
        actor_may_set_any_visibility=True,
        requested_visibility_rank=RANK[Role.ADMIN],
    )
    assert resolve_visibility_rank(req, tenant_default=10) == RANK[Role.ADMIN]


def test_visibility_falls_back_to_the_tenant_default() -> None:
    assert resolve_visibility_rank(request(actor_rank=30), tenant_default=20) == 20


def test_a_lower_request_is_honoured_even_by_admin() -> None:
    req = request(actor_rank=RANK[Role.ADMIN], actor_may_set_any_visibility=True, requested_visibility_rank=10)
    assert resolve_visibility_rank(req, tenant_default=40) == 10


# --------------------------------------------------------------------------------------------
# Identity derivation
# --------------------------------------------------------------------------------------------


def test_client_supplied_external_id_wins() -> None:
    assert derive_external_id(request(external_id="sharepoint:42")) == "sharepoint:42"


def test_same_filename_in_the_same_collection_is_the_same_document() -> None:
    collection = uuid.uuid4()
    a = derive_external_id(request(filename="Handbook.pdf", collection_id=collection))
    b = derive_external_id(request(filename="handbook.pdf ", collection_id=collection))
    assert a == b


def test_same_filename_in_different_collections_is_a_different_document() -> None:
    a = derive_external_id(request(filename="handbook.pdf", collection_id=uuid.uuid4()))
    b = derive_external_id(request(filename="handbook.pdf", collection_id=uuid.uuid4()))
    assert a != b


def test_filename_normalization_survives_unicode_differences() -> None:
    """macOS hands over NFD, Windows NFC. A re-upload must not fork the document."""
    nfc = "résumé.pdf"
    nfd = "résumé.pdf"
    assert normalize_filename(nfc) == normalize_filename(nfd)


# --------------------------------------------------------------------------------------------
# The incremental chunk diff
# --------------------------------------------------------------------------------------------


def hashes(*texts: str) -> list[bytes]:
    return [sha256_text(t) for t in texts]


def test_identical_chunks_are_all_reused() -> None:
    old = hashes("a", "b", "c")
    diff = diff_chunks(old, list(old))
    assert diff.added == frozenset()
    assert diff.removed == frozenset()
    assert diff.embeddings_required == 0
    assert diff.reuse_ratio == 1.0


def test_one_paragraph_edit_reembeds_only_that_paragraph() -> None:
    old = hashes(*[f"para {i}" for i in range(400)])
    new = hashes(*[f"para {i}" if i != 137 else "para 137 corrected" for i in range(400)])
    diff = diff_chunks(old, new)
    assert diff.embeddings_required == 1
    assert len(diff.removed) == 1
    assert len(diff.unchanged) == 399
    assert diff.reuse_ratio > 0.99


def test_reordering_chunks_requires_no_reembedding() -> None:
    """Positional metadata changes; the text does not, so vectors are kept."""
    old = hashes("a", "b", "c")
    diff = diff_chunks(old, list(reversed(old)))
    assert diff.embeddings_required == 0
    assert len(diff.unchanged) == 3


def test_appending_a_section_embeds_only_the_new_chunks() -> None:
    old = hashes("a", "b")
    diff = diff_chunks(old, hashes("a", "b", "c", "d"))
    assert diff.embeddings_required == 2
    assert diff.removed == frozenset()


def test_complete_rewrite_reuses_nothing() -> None:
    diff = diff_chunks(hashes("a", "b"), hashes("x", "y"))
    assert diff.embeddings_required == 2
    assert len(diff.removed) == 2
    assert diff.reuse_ratio == 0.0


def test_empty_previous_version() -> None:
    diff = diff_chunks([], hashes("a"))
    assert diff.embeddings_required == 1
    assert diff.reuse_ratio == 0.0


def test_empty_both_sides_is_full_reuse_not_a_division_error() -> None:
    assert diff_chunks([], []).reuse_ratio == 1.0


def test_duplicate_chunk_text_within_a_document_is_embedded_once() -> None:
    """Boilerplate repeated in a document collapses to one hash, so one embedding."""
    diff = diff_chunks([], hashes("footer", "body", "footer"))
    assert diff.embeddings_required == 2


def test_text_hash_is_unicode_normalized() -> None:
    assert sha256_text("résumé") == sha256_text("résumé")
