"""OpenSearch client.

A thin wrapper rather than a repository: the query bodies are built by the retrieval legs and the
filters by ``search/dsl.py``, so this layer only has to move bytes and turn transport errors into
our own error taxonomy.

``msearch`` is the important method. Running the legs as one multi-search means the four queries
are dispatched together and execute in parallel server-side, so hybrid retrieval costs one round
trip rather than four -- which is the whole reason application-side fusion is not slower than
OpenSearch's in-engine hybrid query.
"""

from __future__ import annotations

import json
from typing import Any

from opensearchpy import AsyncOpenSearch
from opensearchpy.exceptions import OpenSearchException

from app.core.config import Settings
from app.core.errors import AppError


class SearchUnavailableError(AppError):
    """The cluster could not be reached or refused the request."""

    status_code = 503
    code = "search_unavailable"
    message = "Search is temporarily unavailable."


def build_client(settings: Settings) -> AsyncOpenSearch:
    auth = None
    if settings.opensearch_user and settings.opensearch_password:
        auth = (settings.opensearch_user, settings.opensearch_password.get_secret_value())
    return AsyncOpenSearch(
        hosts=[settings.opensearch_url],
        http_auth=auth,
        verify_certs=settings.opensearch_verify_certs,
        ssl_show_warn=settings.opensearch_verify_certs,
        timeout=settings.opensearch_request_timeout_s,
        # Retries are safe here because every request this client issues is a read or an
        # idempotent write (deterministic document ids), so a duplicate delivery overwrites
        # rather than duplicating.
        max_retries=2,
        retry_on_timeout=True,
    )


class SearchClient:
    def __init__(self, client: AsyncOpenSearch) -> None:
        self._client = client

    async def ping(self) -> bool:
        try:
            result: bool = await self._client.ping()
            return result
        except OpenSearchException:
            return False

    async def search(self, *, index: str, body: dict[str, Any], routing: str | None = None) -> dict[str, Any]:
        try:
            params: dict[str, Any] = {"index": index, "body": body}
            if routing:
                params["routing"] = routing
            result: dict[str, Any] = await self._client.search(**params)
            return result
        except OpenSearchException as exc:
            raise SearchUnavailableError(detail=f"search on {index} failed: {exc}") from exc

    async def msearch(
        self, items: list[tuple[dict[str, Any], dict[str, Any]]], *, routing: str | None = None
    ) -> list[dict[str, Any]]:
        """Run several searches in one round trip.

        Responses come back positionally, so the caller pairs them with the legs it sent. A leg
        that errored yields a response carrying ``error`` rather than ``hits``; the caller decides
        whether that is fatal. Degrading one leg is better than failing the whole search -- losing
        the dense leg costs recall, losing the request costs the answer.
        """
        if not items:
            return []
        lines: list[str] = []
        for header, body in items:
            head = dict(header)
            if routing:
                head["routing"] = routing
            lines.append(json.dumps(head))
            lines.append(json.dumps(body))
        payload = "\n".join(lines) + "\n"
        try:
            response = await self._client.msearch(body=payload)
        except OpenSearchException as exc:
            raise SearchUnavailableError(detail=f"msearch failed: {exc}") from exc
        responses: list[dict[str, Any]] = response.get("responses", [])
        return responses

    async def mget(self, *, index: str, ids: list[str], routing: str | None = None) -> list[dict[str, Any]]:
        if not ids:
            return []
        try:
            params: dict[str, Any] = {"index": index, "body": {"ids": ids}}
            if routing:
                params["routing"] = routing
            response = await self._client.mget(**params)
        except OpenSearchException as exc:
            raise SearchUnavailableError(detail=f"mget on {index} failed: {exc}") from exc
        docs: list[dict[str, Any]] = response.get("docs", [])
        return [doc for doc in docs if doc.get("found")]

    async def close(self) -> None:
        await self._client.close()
