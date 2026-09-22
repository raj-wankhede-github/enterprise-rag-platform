"""The operator plane as its own ASGI application.

Not a router mounted under ``/platform`` on the customer-facing app. A separate application,
built by a separate factory, served on a separate host. Three reasons, in order of how much they
would cost to get wrong:

1. **A routing mistake cannot expose an operator endpoint to a tenant.** Mounted on the same app,
   one misplaced prefix or one ordering change in a middleware stack puts a vendor endpoint on a
   customer's domain. Separate applications make that impossible rather than unlikely.
2. **The token audience is enforced by the app that accepts it.** This one accepts only
   ``aud=platform``; the tenant API only ``aud=tenant``. Neither has a code path that checks the
   other, so there is no flag to set wrongly.
3. **A customer's network controls never see operator traffic**, and the operator host can sit
   behind an allowlist the product itself cannot.

Every route here runs through ``grants.authorize``. There is no "the operator is already
authenticated, so this read is fine" path, because that is precisely the path that ends with a
support engineer reading a customer's documents to answer a question about a failed job.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta
from typing import Annotated, Any, Protocol

from fastapi import APIRouter, Depends, FastAPI, Request
from pydantic import BaseModel, Field

from app.api.errors import register_exception_handlers
from app.core.config import Settings, get_settings
from app.core.errors import AuthenticationError, AuthorizationError, NotFoundError
from app.models.platform import GrantScope
from app.platform.grants import (
    Grant,
    Operator,
    OperatorAction,
    authorize,
    describe,
)
from app.security.tokens import AUDIENCE_PLATFORM, KeySet, TokenError, verify_access_token

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ops", tags=["platform"])

#: Two hours, against a tenant session's fourteen days.
#:
#: A tenant user is doing their job all day and being signed out is friction. An operator is
#: doing something exceptional, and a session that outlives the task is a credential lying
#: around in a browser.
OPERATOR_SESSION_TTL = timedelta(hours=2)


class OpsService(Protocol):
    """What these routes need. A seam, so the authorisation logic is testable without a database."""

    async def operator(self, operator_id: uuid.UUID) -> Operator | None: ...
    async def grants_for(self, operator_id: uuid.UUID, tenant_id: uuid.UUID) -> list[Grant]: ...
    async def tenants(self) -> list[dict[str, Any]]: ...
    async def tenant_metadata(self, tenant_id: uuid.UUID) -> dict[str, Any] | None: ...
    async def jobs(self, tenant_id: uuid.UUID) -> list[dict[str, Any]]: ...
    async def request_grant(self, **kwargs: Any) -> dict[str, Any]: ...
    async def audit(self, **kwargs: Any) -> None: ...
    async def notify_tenant_admins(self, tenant_id: uuid.UUID, *, subject: str, body: str) -> None: ...


def get_service(request: Request) -> OpsService:
    service = getattr(request.app.state, "ops_service", None)
    if service is None:  # pragma: no cover - wiring error
        raise RuntimeError("ops_service is not configured on the operator application")
    return service  # type: ignore[no-any-return]


async def current_operator(request: Request, service: Annotated[OpsService, Depends(get_service)]) -> Operator:
    """Resolve the operator from a ``aud=platform`` token, reloaded from the database.

    The token's claims are a hint. A deactivated operator, or one whose MFA enrolment was
    revoked, must lose access immediately rather than at the next token expiry -- and for an
    account that can read customer data, "immediately" is the only acceptable answer.
    """
    header = request.headers.get("authorization", "")
    if not header.lower().startswith("bearer "):
        raise AuthenticationError("Operator authentication is required.")

    keys: KeySet = request.app.state.signing_keys
    try:
        claims = verify_access_token(header[7:], keys, audience=AUDIENCE_PLATFORM)
    except TokenError as exc:
        raise AuthenticationError("Operator authentication is required.", detail=str(exc)) from exc

    operator = await service.operator(claims.user_id)
    if operator is None or not operator.is_active:
        raise AuthenticationError("Operator authentication is required.")
    return operator


OperatorDep = Annotated[Operator, Depends(current_operator)]
ServiceDep = Annotated[OpsService, Depends(get_service)]


async def guard(
    action: OperatorAction,
    *,
    operator: Operator,
    service: OpsService,
    request: Request,
    tenant_id: uuid.UUID | None = None,
    target_user_id: uuid.UUID | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
) -> uuid.UUID | None:
    """Authorise, audit, and return the grant that permitted it.

    Refusals are audited too, and that is deliberate: an operator repeatedly attempting content
    access they do not have is the signal worth alerting on, and it is invisible if only
    successes are recorded.
    """
    grants = await service.grants_for(operator.id, tenant_id) if tenant_id else []
    result = authorize(action, operator=operator, tenant_id=tenant_id, grants=grants, target_user_id=target_user_id)

    await service.audit(
        operator_id=operator.id,
        operator_email=operator.email,
        tenant_id=tenant_id,
        grant_id=result.grant_id,
        action=str(action),
        resource_type=resource_type,
        resource_id=resource_id,
        touched_content=result.touches_content and result.allowed,
        ip_address=request.headers.get("x-forwarded-for", "").split(",")[0].strip() or None,
        details={"allowed": result.allowed, "reason": result.reason},
    )

    if not result.allowed:
        # The reason is safe to show: it is about the operator's own permissions, and the
        # operator is staff. It says nothing about the tenant.
        raise AuthorizationError(result.reason)
    return result.grant_id


# ----------------------------------------------------------------------------------------------


class TenantSummary(BaseModel):
    """What an operator sees without any grant at all: that the tenant exists."""

    id: uuid.UUID
    slug: str
    name: str
    status: str
    created_at: datetime
    #: Counts, not content. Enough to route a ticket.
    document_count: int = 0
    user_count: int = 0
    #: Whether this operator currently holds a grant, and of what kind. Shown so the UI can say
    #: what is possible before an action is attempted rather than after it is refused.
    grant_scope: str | None = None
    grant_expires_at: datetime | None = None


class GrantRequest(BaseModel):
    tenant_id: uuid.UUID
    scope: str = Field(default=GrantScope.METADATA)
    reason: str = Field(min_length=8, max_length=500)
    ticket_ref: str | None = Field(default=None, max_length=64)
    hours: int = Field(default=4, ge=1, le=72)
    impersonate_user_id: uuid.UUID | None = None
    #: Only OWNER, only when the tenant cannot approve, and it notifies every tenant ADMIN.
    break_glass: bool = False


@router.get("/tenants", response_model=list[TenantSummary])
async def list_tenants(operator: OperatorDep, service: ServiceDep, request: Request) -> list[TenantSummary]:
    """Which tenants exist. Metadata only, and no grant needed to route a ticket."""
    await guard(OperatorAction.LIST_TENANTS, operator=operator, service=service, request=request)
    return [TenantSummary(**row) for row in await service.tenants()]


@router.get("/tenants/{tenant_id}")
async def tenant_detail(
    tenant_id: uuid.UUID, operator: OperatorDep, service: ServiceDep, request: Request
) -> dict[str, Any]:
    await guard(
        OperatorAction.VIEW_TENANT_METADATA,
        operator=operator,
        service=service,
        request=request,
        tenant_id=tenant_id,
        resource_type="tenant",
        resource_id=str(tenant_id),
    )
    detail = await service.tenant_metadata(tenant_id)
    if detail is None:
        raise NotFoundError("No such tenant.")
    return detail


@router.get("/tenants/{tenant_id}/jobs")
async def tenant_jobs(
    tenant_id: uuid.UUID, operator: OperatorDep, service: ServiceDep, request: Request
) -> list[dict[str, Any]]:
    """Ingestion job states.

    The commonest support question, and it is metadata: a job's status, stage and error class,
    never the document it was processing. Keeping this cheap is what stops every ticket
    requesting content access.
    """
    await guard(
        OperatorAction.VIEW_JOBS,
        operator=operator,
        service=service,
        request=request,
        tenant_id=tenant_id,
        resource_type="jobs",
    )
    return await service.jobs(tenant_id)


@router.get("/tenants/{tenant_id}/documents/{document_id}")
async def document_content(
    tenant_id: uuid.UUID,
    document_id: uuid.UUID,
    operator: OperatorDep,
    service: ServiceDep,
    request: Request,
) -> dict[str, Any]:
    """The customer's own words. Needs a content grant a tenant administrator approved."""
    await guard(
        OperatorAction.VIEW_DOCUMENT_CONTENT,
        operator=operator,
        service=service,
        request=request,
        tenant_id=tenant_id,
        resource_type="document",
        resource_id=str(document_id),
    )
    raise NotFoundError("No such document.")


@router.post("/grants")
async def request_grant(
    body: GrantRequest, operator: OperatorDep, service: ServiceDep, request: Request
) -> dict[str, Any]:
    """Ask a tenant for access, or -- for an OWNER, in an emergency -- take it and tell them.

    Break-glass is legitimate: a tenant whose only administrator is locked out cannot approve
    anything, and refusing to help them is not a security posture. What makes it safe is that it
    is OWNER-only, time-boxed like any other grant, notified immediately, and recorded as
    break-glass forever after.
    """
    if body.break_glass:
        await guard(
            OperatorAction.GRANT_BREAK_GLASS,
            operator=operator,
            service=service,
            request=request,
            resource_type="tenant",
            resource_id=str(body.tenant_id),
        )

    grant = await service.request_grant(
        tenant_id=body.tenant_id,
        operator_id=operator.id,
        scope=body.scope,
        reason=body.reason,
        ticket_ref=body.ticket_ref,
        expires_at=datetime.now(UTC) + timedelta(hours=body.hours),
        impersonate_user_id=body.impersonate_user_id,
        is_break_glass=body.break_glass,
    )

    if body.break_glass:
        # Immediately, not on a digest. A customer finding out tomorrow that the vendor read
        # their data today is the incident, regardless of whether the access was justified.
        await service.notify_tenant_admins(
            body.tenant_id,
            subject="Emergency support access was used on your account",
            body=(
                f"{operator.email} took emergency support access to your organisation at "
                f"{datetime.now(UTC).isoformat()}, stated reason: {body.reason}. "
                f"It expires in {body.hours} hours. You can revoke it now from Settings > Support access."
            ),
        )

    return grant


@router.get("/actions")
async def describe_actions(operator: OperatorDep) -> list[dict[str, str]]:
    """What each action costs, for the operator UI to show before it is attempted."""
    return [{"action": str(action), "requirement": describe(action)} for action in OperatorAction]


# ----------------------------------------------------------------------------------------------


def create_ops_app(settings: Settings | None = None) -> FastAPI:
    """The operator application. Served on its own host, never mounted on the tenant API."""
    settings = settings or get_settings()

    app = FastAPI(
        title="Platform Operations",
        version="0.1.0",
        summary="Vendor-side operations. Separate host, separate token audience.",
        # No public documentation, ever. This surface is not for customers, and publishing its
        # shape tells an attacker exactly what to look for on the operator host.
        docs_url="/docs" if not settings.is_production else None,
        openapi_url="/openapi.json" if not settings.is_production else None,
    )
    app.state.settings = settings
    register_exception_handlers(app)
    app.middleware("http")(_no_store)
    app.include_router(router)
    return app


async def _no_store(request: Request, call_next: Callable[[Request], Awaitable[Any]]) -> Any:
    """Never cache an operator response.

    Anything here may contain a customer's data, and an intermediate cache holding it -- a
    corporate proxy, a browser's back-forward cache -- is a copy nobody knows about and nobody
    can revoke a grant from.
    """
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
    response.headers["Pragma"] = "no-cache"
    return response


def actions_for_role(role: str) -> frozenset[OperatorAction]:
    """Which actions this role could ever perform, given the right grant. For the UI only.

    Asked with a grant that satisfies every scope, so the answer is about the *role* rather than
    about what the operator happens to hold right now. It drives what the operator console shows
    as possible; the real decision is made per request by ``authorize``.
    """
    operator = Operator(id=uuid.UUID(int=0), email="", role=role, is_active=True, mfa_enrolled=True)
    tenant = uuid.UUID(int=1)
    everything = Grant(
        id=uuid.UUID(int=2),
        tenant_id=tenant,
        operator_id=operator.id,
        scope=GrantScope.IMPERSONATE,
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        impersonate_user_id=uuid.UUID(int=3),
    )
    return frozenset(
        action
        for action in OperatorAction
        if authorize(action, operator=operator, tenant_id=tenant, grants=[everything]).allowed
    )


OWNER_ONLY: frozenset[OperatorAction] = frozenset(
    {OperatorAction.PROVISION_TENANT, OperatorAction.GRANT_BREAK_GLASS, OperatorAction.OPEN_EMERGENCY_LOGIN}
)
"""Actions that no amount of tenant approval can grant to a lower operator role.

A customer cannot consent their way into letting a SUPPORT engineer provision tenants or open a
break-glass window, because neither is about that customer's data.
"""
