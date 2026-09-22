"""Roles, ranks and capabilities.

The product owner fixed the role model: a flat chain ``ADMIN > DEV > TEST > PROD``, one role per
user. This module is the single place that model is expressed, and the tests in
``tests/unit/test_capabilities.py`` enforce its two invariants:

1. every capability is granted to at least one role (no orphans);
2. the roles form a strict chain, so a higher rank's capability set is a proper superset of the
   one below it. A capability that breaks nesting is someone trying to smuggle in a fifth role.

Routes never mention a ``Role``. They declare capabilities, so the ranks can be renamed or
re-ordered here without touching the API surface.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class Role(StrEnum):
    ADMIN = "ADMIN"
    DEV = "DEV"
    TEST = "TEST"
    PROD = "PROD"


# Spaced by 10 so an intermediate level is a data change, not a migration of every stored
# integer. The rank -- not the name -- is what is stored on documents as ``visibility_rank``.
RANK: Final[dict[Role, int]] = {
    Role.PROD: 10,
    Role.TEST: 20,
    Role.DEV: 30,
    Role.ADMIN: 40,
}

RANK_TO_ROLE: Final[dict[int, Role]] = {rank: role for role, rank in RANK.items()}


class Capability(StrEnum):
    # --- read paths, everyone ---
    SEARCH = "search"
    ASK = "ask"
    DOC_READ = "doc:read"
    TRACE_VIEW_OWN = "trace:view_own"

    # --- TEST and above ---
    TRACE_VIEW_ALL = "trace:view_all"
    EVAL_RUN = "eval:run"
    EXPORT_EVAL = "export:eval"
    RETRIEVAL_READ_CONFIG = "retrieval:read_config"

    # --- DEV and above ---
    DOC_UPLOAD = "doc:upload"
    DOC_EDIT_METADATA = "doc:edit_metadata"
    DOC_REINDEX = "doc:reindex"
    DOC_DELETE_OWN = "doc:delete_own"
    DOC_SET_VISIBILITY = "doc:set_visibility"
    RETRIEVAL_TUNE = "retrieval:tune"
    EVAL_MANAGE_DATASET = "eval:manage_dataset"
    APIKEY_MANAGE_OWN = "apikey:manage_own"

    # --- ADMIN only ---
    DOC_DELETE_ANY = "doc:delete_any"
    DOC_SET_VISIBILITY_ANY = "doc:set_visibility_any"
    EXPORT_DOCUMENTS = "export:documents"
    USER_MANAGE = "user:manage"
    CONNECTOR_MANAGE = "connector:manage"
    SSO_CONFIGURE = "sso:configure"
    TENANT_SETTINGS = "tenant:settings"
    COLLECTION_MANAGE = "collection:manage"
    AUDIT_VIEW = "audit:view"
    APIKEY_MANAGE_ANY = "apikey:manage_any"


_PROD: Final[frozenset[Capability]] = frozenset(
    {
        Capability.SEARCH,
        Capability.ASK,
        Capability.DOC_READ,
        Capability.TRACE_VIEW_OWN,
    }
)

# TEST may run evals but not manage datasets: an eval *run* is its own audit trail and takes a
# read-only snapshot of the retrieval config, so it writes nothing a reader could influence.
# EXPORT_EVAL is metrics; EXPORT_DOCUMENTS (bulk content egress) stays ADMIN-only, because
# reading in the UI and exporting the corpus are different risk classes.
_TEST: Final[frozenset[Capability]] = _PROD | {
    Capability.TRACE_VIEW_ALL,
    Capability.EVAL_RUN,
    Capability.EXPORT_EVAL,
    Capability.RETRIEVAL_READ_CONFIG,
}

_DEV: Final[frozenset[Capability]] = _TEST | {
    Capability.DOC_UPLOAD,
    Capability.DOC_EDIT_METADATA,
    Capability.DOC_REINDEX,
    Capability.DOC_DELETE_OWN,
    Capability.DOC_SET_VISIBILITY,
    Capability.RETRIEVAL_TUNE,
    Capability.EVAL_MANAGE_DATASET,
    Capability.APIKEY_MANAGE_OWN,
}

_ADMIN: Final[frozenset[Capability]] = _DEV | {
    Capability.DOC_DELETE_ANY,
    Capability.DOC_SET_VISIBILITY_ANY,
    Capability.EXPORT_DOCUMENTS,
    Capability.USER_MANAGE,
    Capability.CONNECTOR_MANAGE,
    Capability.SSO_CONFIGURE,
    Capability.TENANT_SETTINGS,
    Capability.COLLECTION_MANAGE,
    Capability.AUDIT_VIEW,
    Capability.APIKEY_MANAGE_ANY,
}

ROLE_CAPABILITIES: Final[dict[Role, frozenset[Capability]]] = {
    Role.PROD: _PROD,
    Role.TEST: _TEST,
    Role.DEV: _DEV,
    Role.ADMIN: _ADMIN,
}

# Ascending rank order. Used by the nesting invariant test and by the admin UI.
ROLE_CHAIN: Final[tuple[Role, ...]] = (Role.PROD, Role.TEST, Role.DEV, Role.ADMIN)


def rank_of(role: Role) -> int:
    return RANK[role]


def capabilities_for(role: Role) -> frozenset[Capability]:
    return ROLE_CAPABILITIES[role]


def may_see(viewer_rank: int, visibility_rank: int) -> bool:
    """A user sees documents at or below their own rank.

    ``PROD``-level documents (rank 10) are the most widely visible; ``ADMIN``-level (40) the most
    restricted. This is the whole comparison -- not equality, not set membership.
    """
    return visibility_rank <= viewer_rank
