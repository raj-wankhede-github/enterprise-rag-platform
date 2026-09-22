"""Persisted enum values. Stored as text with CHECK constraints, not as Postgres ENUM types --
adding a value to a PG enum inside a transaction is awkward, and a CHECK is trivially altered.
"""

from __future__ import annotations

from enum import StrEnum


class TenantStatus(StrEnum):
    PROVISIONING = "PROVISIONING"
    ACTIVE = "ACTIVE"
    SUSPENDED = "SUSPENDED"
    DELETING = "DELETING"


class DocumentStatus(StrEnum):
    ACTIVE = "ACTIVE"
    INACTIVE = "INACTIVE"
    DELETED = "DELETED"


class AclMode(StrEnum):
    TENANT_PUBLIC = "tenant_public"
    RESTRICTED = "restricted"


class VersionStatus(StrEnum):
    UPLOADED = "UPLOADED"
    PROCESSING = "PROCESSING"
    READY = "READY"
    FAILED = "FAILED"
    SUPERSEDED = "SUPERSEDED"


class JobStatus(StrEnum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    DONE = "DONE"
    FAILED = "FAILED"
    DEAD = "DEAD"


class JobStage(StrEnum):
    """Checkpoints, so a crash at document 900/1000 resumes at 900 rather than at zero."""

    FETCHED = "FETCHED"
    PARSED = "PARSED"
    CHUNKED = "CHUNKED"
    CONTEXTUALIZED = "CONTEXTUALIZED"
    EMBEDDED = "EMBEDDED"
    INDEXED = "INDEXED"
    ACTIVATED = "ACTIVATED"


class UploadOutcome(StrEnum):
    """Which deduplication case an upload resolved to. Recorded for audit and for the metrics
    that tell an operator how much re-upload traffic they are absorbing for free.
    """

    #: Case 1 - identical bytes, same logical document. No version, no work.
    DEDUPLICATED = "DEDUPLICATED"
    #: Case 2 - identical bytes, different logical document. Blob and embeddings reused.
    NEW_DOCUMENT_SHARED_CONTENT = "NEW_DOCUMENT_SHARED_CONTENT"
    #: Case 3 - same logical document, changed bytes. New version, incremental re-embedding.
    NEW_VERSION = "NEW_VERSION"
    #: A genuinely new document with content never seen in this tenant.
    NEW_DOCUMENT = "NEW_DOCUMENT"
    #: Case 4 - content exists but the uploader may not see it. Rejected with 409.
    REJECTED_NO_PERMISSION = "REJECTED_NO_PERMISSION"
