"""Exception handlers.

The contract: clients receive ``message`` and ``code``; ``detail`` is logged and never
serialized. That split is what lets an authorization failure say *why* in the logs without
telling the caller which capability they lacked -- and what lets a cross-tenant lookup return a
plain 404 while the log records that it was a tenancy violation.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.errors import AppError, RateLimitedError

logger = structlog.get_logger(__name__)


async def _app_error_handler(request: Request, exc: Exception) -> JSONResponse:
    assert isinstance(exc, AppError)
    log = logger.bind(code=exc.code, status=exc.status_code, path=request.url.path)
    if exc.status_code >= 500:
        log.error("request_failed", detail=exc.detail, exc_info=exc)
    else:
        log.info("request_rejected", detail=exc.detail)

    body: dict[str, Any] = {"code": exc.code, "message": exc.message}
    if exc.extra:
        body["extra"] = exc.extra

    headers: dict[str, str] = {}
    if isinstance(exc, RateLimitedError):
        headers["Retry-After"] = str(exc.retry_after_seconds)

    response = JSONResponse(status_code=exc.status_code, content=body, headers=headers)
    for name in exc.clear_cookies:
        response.delete_cookie(name, path="/")
    return response


async def _unhandled_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("unhandled_exception", path=request.url.path, exc_info=exc)
    return JSONResponse(status_code=500, content={"code": "internal_error", "message": "Something went wrong."})


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppError, _app_error_handler)
    app.add_exception_handler(Exception, _unhandled_handler)
