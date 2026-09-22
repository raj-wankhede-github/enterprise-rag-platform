"""The authenticated caller.

A ``Principal`` is immutable and always carries a ``tenant_id``. It is resolved once per request
by ``api/deps.py`` and bound to a ``ContextVar`` so the ORM guard and the background workers can
reach it without threading it through every signature.

Capabilities are materialized on the principal rather than derived from the role at each check,
because an API key or a platform support grant may hold a *narrowed* subset of its role's
capabilities. ``can()`` is therefore the only correct way to ask.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

from app.security.capabilities import RANK, Capability, Role

PrincipalKind = Literal["user", "api_key", "platform", "platform_impersonating"]
AuthMethod = Literal["password", "oidc", "saml", "api_key", "impersonation"]


@dataclass(frozen=True, slots=True)
class Principal:
    kind: PrincipalKind
    tenant_id: uuid.UUID
    role: Role
    capabilities: frozenset[Capability]
    email: str = ""
    display_name: str = ""
    user_id: uuid.UUID | None = None
    group_ids: frozenset[str] = field(default_factory=frozenset)
    collection_scope: frozenset[uuid.UUID] | None = None
    session_id: uuid.UUID | None = None
    api_key_id: uuid.UUID | None = None
    platform_operator_id: uuid.UUID | None = None
    support_grant_id: uuid.UUID | None = None
    elevated: bool = False
    auth_method: AuthMethod = "password"
    expires_at: datetime | None = None

    @property
    def rank(self) -> int:
        return RANK[self.role]

    def can(self, capability: Capability) -> bool:
        return capability in self.capabilities

    def may_see_rank(self, visibility_rank: int) -> bool:
        return visibility_rank <= self.rank

    def in_scope(self, collection_id: uuid.UUID | None) -> bool:
        """Collection scope is *subtractive*: ``None`` means tenant-wide.

        It can only ever reduce what a role grants, which is what keeps "one role per user,
        ADMIN > DEV > TEST > PROD" literally true while still expressing "Alice curates HR only".
        """
        if self.collection_scope is None:
            return True
        if collection_id is None:
            return False
        return collection_id in self.collection_scope
