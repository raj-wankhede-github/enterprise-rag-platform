"""What happens when the same document is uploaded twice.

This module holds the decision, deliberately as pure functions over hashes and a small lookup
record, so the whole policy is unit-testable with no database, no object store and no OpenSearch.
The service layer supplies the lookups and applies the plan.

The four cases
--------------

1. **Identical bytes, same logical document** -- a retry, a double-click, a connector re-sync.
   No new version, no parse, no chunking, no embedding, no indexing. An upload event is recorded
   and the caller gets the existing version id with ``deduplicated: true``. Ingestion is
   idempotent by construction, which is what makes at-least-once job delivery safe.

2. **Identical bytes, different logical document** -- the same PDF filed in two collections.
   The blob is stored once and refcounted; the parse is reused; chunk texts are identical so
   their context lines and embeddings come from cache at zero cost. The chunks *are* indexed
   twice, because they carry different ACLs and metadata and must be independently filterable.
   Near-duplicate suppression at assembly time stops a reader ever seeing the passage twice.

3. **Same logical document, changed bytes** -- the normal "re-upload the corrected policy".
   A new version, then an incremental diff: unchanged chunks keep their embeddings and receive a
   metadata-only update, and only added chunks are embedded.

4. **Identical content the uploader may not see** -- a DEV re-uploading content that already
   exists as an ADMIN-only document. Creating a second copy would be an accidental ACL bypass,
   because the copy would be visible at the uploader's own rank. Rejected with 409 and a neutral
   message that names nothing.

The behaviour is identical for ADMIN and DEV. Versioning is a property of the data, not of the
uploader's role; the only role-sensitive part is case 4 and the visibility ceiling.
"""

from __future__ import annotations

import hashlib
import unicodedata
import uuid
from dataclasses import dataclass, field
from enum import StrEnum

from app.models.enums import UploadOutcome


class Decision(StrEnum):
    """What the ingest pipeline should do. Maps one-to-one onto the four cases."""

    REUSE_EXISTING_VERSION = "reuse_existing_version"  # case 1
    CREATE_DOCUMENT_REUSING_CONTENT = "create_document_reusing_content"  # case 2
    CREATE_NEW_VERSION = "create_new_version"  # case 3
    CREATE_DOCUMENT = "create_document"  # new content, new document
    REJECT_NOT_PERMITTED = "reject_not_permitted"  # case 4


@dataclass(frozen=True, slots=True)
class ExistingVersion:
    """The current state of a logical document, as far as dedup needs to know."""

    document_id: uuid.UUID
    version_id: uuid.UUID
    version_no: int
    blob_sha256: bytes
    visibility_rank: int


@dataclass(frozen=True, slots=True)
class ContentMatch:
    """Some document in this tenant already holds these exact bytes."""

    document_id: uuid.UUID
    version_id: uuid.UUID
    visibility_rank: int


@dataclass(frozen=True, slots=True)
class UploadRequest:
    tenant_id: uuid.UUID
    blob_sha256: bytes
    filename: str
    size_bytes: int
    source_system: str = "upload"
    external_id: str | None = None
    collection_id: uuid.UUID | None = None
    #: The uploader's rank. Used only for the case-4 check and the visibility ceiling.
    actor_rank: int = 30
    #: True when the actor holds ``doc:set_visibility_any`` (ADMIN).
    actor_may_set_any_visibility: bool = False
    requested_visibility_rank: int | None = None


@dataclass(frozen=True, slots=True)
class UploadPlan:
    decision: Decision
    outcome: UploadOutcome
    document_id: uuid.UUID | None = None
    version_id: uuid.UUID | None = None
    next_version_no: int = 1
    #: True when parse, chunk, contextualize and embed can all be skipped.
    reuse_parse: bool = False
    reuse_embeddings: bool = False
    needs_indexing: bool = True
    reason: str = ""

    @property
    def is_rejection(self) -> bool:
        return self.decision is Decision.REJECT_NOT_PERMITTED

    @property
    def creates_work(self) -> bool:
        return self.decision is not Decision.REUSE_EXISTING_VERSION


def normalize_filename(filename: str) -> str:
    """NFC-normalize, lowercase and strip a filename so identity is stable across clients.

    Windows, macOS and Linux disagree about Unicode normalization of filenames, and a re-upload
    from a different machine must not create a second document because of a combining accent.
    """
    return unicodedata.normalize("NFC", filename).strip().lower()


def derive_external_id(request: UploadRequest) -> str:
    """The stable identity key for an upload.

    A client-supplied ``external_id`` always wins. Otherwise identity is the normalized filename
    scoped to the collection -- so "handbook.pdf" in Legal and "handbook.pdf" in HR are two
    documents, which is what users expect, while re-uploading into the same collection updates.
    """
    if request.external_id:
        return request.external_id
    scope = str(request.collection_id) if request.collection_id else "-"
    digest = hashlib.sha256(f"{scope}/{normalize_filename(request.filename)}".encode()).hexdigest()
    return f"upload:{digest[:32]}"


def plan_upload(
    request: UploadRequest,
    *,
    existing: ExistingVersion | None,
    content_match: ContentMatch | None,
) -> UploadPlan:
    """Decide what an upload should do.

    ``existing`` is the active version of the document with this identity, if any.
    ``content_match`` is any document *in this tenant* already holding these bytes -- looked up
    tenant-wide and deliberately without applying the uploader's ACL, because the decision must
    account for documents the uploader cannot see. What the uploader is *told* is then filtered:
    case 4 returns a message that names nothing.
    """
    # --- Case 1: identical bytes, same logical document -------------------------------------
    if existing is not None and existing.blob_sha256 == request.blob_sha256:
        return UploadPlan(
            decision=Decision.REUSE_EXISTING_VERSION,
            outcome=UploadOutcome.DEDUPLICATED,
            document_id=existing.document_id,
            version_id=existing.version_id,
            next_version_no=existing.version_no,
            reuse_parse=True,
            reuse_embeddings=True,
            needs_indexing=False,
            reason="Identical content already stored as the active version.",
        )

    # --- Case 3: same logical document, changed bytes ---------------------------------------
    if existing is not None:
        return UploadPlan(
            decision=Decision.CREATE_NEW_VERSION,
            outcome=UploadOutcome.NEW_VERSION,
            document_id=existing.document_id,
            next_version_no=existing.version_no + 1,
            # The parse cannot be reused (different bytes), but per-chunk hashing will still
            # reuse most embeddings -- that diff happens after parsing, in ``diff_chunks``.
            reuse_parse=False,
            reuse_embeddings=True,
            reason="Content changed; creating a new version.",
        )

    # --- Case 4: content exists but is out of the uploader's reach --------------------------
    if content_match is not None and content_match.visibility_rank > request.actor_rank:
        return UploadPlan(
            decision=Decision.REJECT_NOT_PERMITTED,
            outcome=UploadOutcome.REJECTED_NO_PERMISSION,
            needs_indexing=False,
            reason=(
                "A document with this content already exists and you do not have permission "
                "to update it. Contact an administrator."
            ),
        )

    # --- Case 2: identical bytes, different logical document --------------------------------
    if content_match is not None:
        return UploadPlan(
            decision=Decision.CREATE_DOCUMENT_REUSING_CONTENT,
            outcome=UploadOutcome.NEW_DOCUMENT_SHARED_CONTENT,
            next_version_no=1,
            reuse_parse=True,
            reuse_embeddings=True,
            reason="New document; blob, parse and embeddings reused from existing content.",
        )

    return UploadPlan(
        decision=Decision.CREATE_DOCUMENT,
        outcome=UploadOutcome.NEW_DOCUMENT,
        next_version_no=1,
        reason="New content.",
    )


def resolve_visibility_rank(request: UploadRequest, *, tenant_default: int) -> int:
    """Clamp the requested visibility to what the uploader is allowed to publish.

    A DEV cannot publish above their own rank, because a document they cannot read is one they
    cannot verify. An ADMIN holds ``doc:set_visibility_any`` and is unconstrained.
    """
    requested = request.requested_visibility_rank
    if requested is None:
        requested = tenant_default
    if request.actor_may_set_any_visibility:
        return requested
    return min(requested, request.actor_rank)


@dataclass(frozen=True, slots=True)
class ChunkDiff:
    """The outcome of comparing a new version's chunks against the previous version's.

    ``unchanged`` chunks keep their vectors and context lines; in OpenSearch they receive a
    partial *update* of positional metadata rather than a re-index with a new vector, because
    ``ordinal``, ``page_from`` and ``parent_id`` shift even when the text does not.
    """

    unchanged: frozenset[bytes] = field(default_factory=frozenset)
    added: frozenset[bytes] = field(default_factory=frozenset)
    removed: frozenset[bytes] = field(default_factory=frozenset)

    @property
    def reuse_ratio(self) -> float:
        total = len(self.unchanged) + len(self.added)
        return 1.0 if total == 0 else len(self.unchanged) / total

    @property
    def embeddings_required(self) -> int:
        return len(self.added)


def diff_chunks(previous: list[bytes], current: list[bytes]) -> ChunkDiff:
    """Compare two versions by per-chunk content hash.

    A typical one-paragraph policy edit turns a 400-chunk reindex into a handful of embeddings
    and several hundred metadata updates.
    """
    old, new = set(previous), set(current)
    return ChunkDiff(
        unchanged=frozenset(old & new),
        added=frozenset(new - old),
        removed=frozenset(old - new),
    )


def sha256_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def sha256_text(text: str) -> bytes:
    """Hash chunk text.

    NFC-normalized so that the same visible text produced by two different parsers or editors
    hashes identically -- otherwise a harmless re-export would invalidate every cached embedding.
    """
    return hashlib.sha256(unicodedata.normalize("NFC", text).encode("utf-8")).digest()
