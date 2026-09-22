"""FastAPI application factory.

Deliberately a factory rather than a module-level ``app``: the API, the workers, the CLI and the
tests all build from the same settings object, and tests need to construct an app with injected
fakes without importing a half-configured global.

Migrations are never run here. They are an explicit step (``alembic upgrade head``, run by the
one-shot ``migrate`` service), because a process that migrates on startup will eventually run two
replicas that migrate concurrently.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.errors import register_exception_handlers
from app.api.routes import auth, health, oidc
from app.core.config import Settings, get_settings


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    yield


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="Enterprise RAG Platform",
        version="0.1.0",
        summary="Multi-tenant enterprise knowledge search with grounded, cited answers.",
        lifespan=_lifespan,
        docs_url="/docs" if not settings.is_production else None,
        redoc_url=None,
    )
    app.state.settings = settings

    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    register_exception_handlers(app)
    app.include_router(health.router)
    app.include_router(auth.router)
    app.include_router(oidc.router)
    return app
