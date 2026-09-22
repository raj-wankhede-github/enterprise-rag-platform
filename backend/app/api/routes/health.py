"""Liveness and readiness.

``/health`` must answer without touching a datastore -- it is what the container orchestrator
polls, and a health check that fails when Postgres blips causes a restart storm that makes the
outage worse. ``/ready`` is the one that checks dependencies, and it is what a load balancer uses
to decide whether to send traffic.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response, status

router = APIRouter(tags=["health"])


@router.get("/health", summary="Liveness: the process is up")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready", summary="Readiness: dependencies are reachable")
async def ready(response: Response) -> dict[str, Any]:
    # Checks are added as each dependency is wired up (Postgres in step 1, OpenSearch in step 3).
    checks: dict[str, str] = {}
    ok = all(state == "ok" for state in checks.values())
    if not ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {"status": "ok" if ok else "degraded", "checks": checks}
