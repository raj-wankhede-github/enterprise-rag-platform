"""Index and alias administration.

Every read and write in the product goes through an alias, never a concrete index name. That
single rule is what makes a generation swap, a tenant promotion to a dedicated index, and a
rollback all the same operation: move an alias. Code that names an index directly would have to
be found and changed for each of those.
"""

from __future__ import annotations

import logging
from typing import Any

from app.search.generations import (
    AliasAction,
    alias_add_actions,
    chunk_index_name,
    parent_index_name,
    swap_actions,
)
from app.search.mappings import chunk_index_body, parent_index_body

logger = logging.getLogger(__name__)


class IndexAdmin:
    def __init__(self, client: Any, *, shards: int = 3, replicas: int = 1) -> None:
        self.client = client
        self.shards = shards
        self.replicas = replicas

    async def create_generation(
        self,
        *,
        generation: int,
        pool: int,
        dimension: int,
        quantize: bool = False,
        for_backfill: bool = True,
    ) -> tuple[str, str]:
        """Create the concrete indices for one generation of one pool.

        A backfill target is created with no replicas and refresh disabled. Both are restored
        before it goes live. Indexing into a replicated, refreshing index costs roughly twice as
        much for a copy nobody is reading yet -- and a backfill is the one moment where that
        matters, because it touches every document the tenant owns.
        """
        chunks = chunk_index_name(generation=generation, pool=pool)
        parents = parent_index_name(generation=generation, pool=pool)

        chunk_body = chunk_index_body(
            dimension=dimension,
            shards=self.shards,
            replicas=0 if for_backfill else self.replicas,
            quantize=quantize,
            refresh_interval="-1" if for_backfill else "30s",
        )
        parent_body = parent_index_body(
            shards=self.shards,
            replicas=0 if for_backfill else self.replicas,
            refresh_interval="-1" if for_backfill else "30s",
        )

        await self.client.indices.create(index=chunks, body=chunk_body)
        await self.client.indices.create(index=parents, body=parent_body)
        return chunks, parents

    async def finalise_for_serving(self, *, generation: int, pool: int) -> None:
        """Restore replicas and refresh, then merge, before a generation can serve.

        Called at BACKFILLED. Force-merging a read-only index into one segment measurably
        improves query latency, and it is only safe to do here -- once writing has stopped.
        """
        for index in (
            chunk_index_name(generation=generation, pool=pool),
            parent_index_name(generation=generation, pool=pool),
        ):
            await self.client.indices.put_settings(
                index=index,
                body={"index": {"number_of_replicas": self.replicas, "refresh_interval": "30s"}},
            )
            await self.client.indices.refresh(index=index)
            try:
                await self.client.indices.forcemerge(index=index, max_num_segments=1)
            except Exception as exc:  # merging is an optimisation, never a gate
                logger.warning("forcemerge skipped for %s: %s", index, exc)

    async def point_aliases(self, *, generation: int, pools: list[int]) -> None:
        """Point every alias at one generation. Used when bootstrapping, not when swapping."""
        actions: list[AliasAction] = []
        for pool in pools:
            actions.extend(alias_add_actions(generation=generation, pool=pool))
        await self.client.indices.update_aliases(body={"actions": actions})

    async def swap(self, *, from_generation: int, to_generation: int, pools: list[int]) -> None:
        """Promote a generation across every pool in ONE request.

        The atomicity is the entire point. A per-pool loop leaves the cluster serving a mixture
        of generations for as long as the loop runs, and a failure halfway through leaves it
        that way permanently -- with no error that names the problem, because each individual
        call succeeded.
        """
        actions: list[AliasAction] = []
        for pool in pools:
            actions.extend(swap_actions(from_generation=from_generation, to_generation=to_generation, pool=pool))
        await self.client.indices.update_aliases(body={"actions": actions})
        logger.info("alias swap complete", extra={"from": from_generation, "to": to_generation, "pools": len(pools)})

    async def drop_generation(self, *, generation: int, pools: list[int]) -> None:
        """Delete a retired generation. Only ever called after the drain window."""
        for pool in pools:
            for index in (
                chunk_index_name(generation=generation, pool=pool),
                parent_index_name(generation=generation, pool=pool),
            ):
                await self.client.indices.delete(index=index, ignore=[404])

    async def counts(self, *, generation: int, pools: list[int]) -> dict[str, int]:
        """Document counts per index, for the reconciliation check against Postgres."""
        result: dict[str, int] = {}
        for pool in pools:
            for index in (
                chunk_index_name(generation=generation, pool=pool),
                parent_index_name(generation=generation, pool=pool),
            ):
                try:
                    response = await self.client.count(index=index)
                    result[index] = int(response.get("count", 0))
                except Exception:
                    result[index] = 0
        return result
