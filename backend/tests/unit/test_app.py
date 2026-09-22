from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.errors import AuthorizationError, NotFoundError
from app.main import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(Settings(environment="ci")))


def test_health_does_not_touch_a_datastore(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_reports_checks(client: TestClient) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    assert "checks" in response.json()


def test_openapi_schema_builds(client: TestClient) -> None:
    schema = client.get("/openapi.json").json()
    assert "/health" in schema["paths"]


def test_app_errors_serialize_message_and_code_but_never_detail() -> None:
    app = create_app(Settings(environment="ci"))

    @app.get("/boom-404")
    async def boom_404() -> None:
        raise NotFoundError(detail="tenant B tried to read tenant A's document 1234")

    @app.get("/boom-403")
    async def boom_403() -> None:
        raise AuthorizationError(detail="missing capability doc:upload")

    client = TestClient(app, raise_server_exceptions=False)

    r404 = client.get("/boom-404")
    assert r404.status_code == 404
    assert r404.json() == {"code": "not_found", "message": "Not found."}
    assert "1234" not in r404.text, "detail must never reach the client"

    r403 = client.get("/boom-403")
    assert r403.status_code == 403
    assert "doc:upload" not in r403.text


def test_docs_are_disabled_in_production() -> None:
    settings = Settings(
        environment="production",
        jwt_secret="x" * 40,
        cookie_secure=True,
        opensearch_url="https://opensearch.internal:9200",
        redis_url="redis://cache:6379/0",
    )
    app = create_app(settings)
    assert app.docs_url is None


def test_production_config_rejects_weak_settings() -> None:
    with pytest.raises(ValueError, match="JWT_SECRET"):
        Settings(
            environment="production",
            jwt_secret="short",
            opensearch_url="https://x:9200",
            redis_url="redis://x:6379/0",
        )


def test_production_config_rejects_wildcard_cors() -> None:
    with pytest.raises(ValueError, match="wildcard"):
        Settings(
            environment="production",
            jwt_secret="x" * 40,
            cors_origins=["*"],
            opensearch_url="https://x:9200",
            redis_url="redis://x:6379/0",
        )


def test_production_config_requires_redis_for_cross_replica_limits() -> None:
    with pytest.raises(ValueError, match="REDIS_URL"):
        Settings(environment="production", jwt_secret="x" * 40, opensearch_url="https://x:9200")
