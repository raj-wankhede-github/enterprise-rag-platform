"""Model registry.

Alembic's autogenerate and the architecture test both need every model imported, so this module
is the single import surface. A model that is not re-exported here will not appear in a
migration, which is exactly the kind of silent omission the test in
``tests/unit/test_model_registry.py`` exists to catch.
"""

from __future__ import annotations

from app.models.apikey import ApiKey, ApiKeyEvent
from app.models.chunk import Chunk, ChunkContext, ChunkVector, EmbeddingCache, Parent
from app.models.document import (
    Blob,
    Document,
    DocumentRelation,
    DocumentUploadEvent,
    DocumentVersion,
)
from app.models.identity import (
    IdpConfig,
    IdpKind,
    IdpProtocol,
    IdpState,
    LoginAttempt,
    Session,
    TenantDomain,
    UserIdentity,
)
from app.models.job import IndexGeneration, IngestJob, JobEvent
from app.models.tenant import Tenant, TenantIndexBinding
from app.models.user import Collection, RoleElevationGrant, User, UserCollectionScope

__all__ = [
    "ApiKey",
    "ApiKeyEvent",
    "Blob",
    "Chunk",
    "ChunkContext",
    "ChunkVector",
    "Collection",
    "Document",
    "DocumentRelation",
    "DocumentUploadEvent",
    "DocumentVersion",
    "EmbeddingCache",
    "IdpConfig",
    "IdpKind",
    "IdpProtocol",
    "IdpState",
    "IndexGeneration",
    "IngestJob",
    "JobEvent",
    "LoginAttempt",
    "Parent",
    "RoleElevationGrant",
    "Session",
    "Tenant",
    "TenantDomain",
    "TenantIndexBinding",
    "User",
    "UserCollectionScope",
    "UserIdentity",
]
