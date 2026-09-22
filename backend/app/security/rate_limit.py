"""Rate limits and concurrency caps.

**Two different controls, and only having one of them is the usual mistake.**

A *rate* limit bounds requests per window. It is what stops a runaway script and what makes cost
predictable. It does nothing about fifty concurrent `ask` requests arriving in the same second:
every one is within the per-minute budget, and every one holds an LLM call open. The service
falls over while the rate limiter reports that nothing is wrong.

A *concurrency* cap bounds requests in flight. It is what actually protects the expensive,
slow resources -- the LLM, the reranker, the embedder -- because those are limited by
simultaneous work rather than by work per minute.

**Two scopes, and only having one of those is the other usual mistake.** A per-user limit does
not stop one tenant's two hundred users from saturating shared infrastructure; a per-tenant limit
alone lets one user inside a tenant consume the whole allowance. Both are checked, and the
tighter one wins.

**The Redis window is a single Lua script**, so read-decide-write is atomic. The same logic
across two round trips is a race: under real concurrency, every caller reads the count before any
of them writes, and they all conclude there is room. That failure appears only under the load the
limiter exists to survive.

The in-memory implementation behind the same Protocol is the test and single-node fallback. It is
honest about what it is: correct for one process, wrong the moment there are two, which is why
production configuration requires Redis and the settings validator says so.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: Sliding window in one atomic step.
#:
#: ZSET of request timestamps per key: drop what has aged out, count what is left, and add the
#: current request only if there is room. Returning the oldest surviving score lets the caller
#: compute an exact Retry-After rather than guessing a whole window.
SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local cost = tonumber(ARGV[4])
local record = tonumber(ARGV[5])

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local used = redis.call('ZCARD', key)

if used + cost > limit then
  local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
  local retry = window
  if oldest[2] then retry = (tonumber(oldest[2]) + window) - now end
  return {0, used, retry}
end

if record == 0 then
  return {1, used, 0}
end

for i = 1, cost do
  -- The member must be unique or identical timestamps collapse into one ZSET entry and the
  -- window silently under-counts. now plus a counter is unique within a request.
  redis.call('ZADD', key, now, now .. ':' .. i .. ':' .. redis.call('INCR', key .. ':seq'))
end
redis.call('EXPIRE', key, math.ceil(window) + 1)
redis.call('EXPIRE', key .. ':seq', math.ceil(window) + 1)

return {1, used + cost, 0}
"""


@dataclass(frozen=True, slots=True)
class Limit:
    requests: int
    window_seconds: float

    def __post_init__(self) -> None:
        if self.requests < 1 or self.window_seconds <= 0:
            raise ValueError("a limit must allow at least one request in a positive window")


@dataclass(frozen=True, slots=True)
class Decision:
    allowed: bool
    used: int
    limit: int
    retry_after_seconds: float = 0.0

    @property
    def remaining(self) -> int:
        return max(0, self.limit - self.used)

    def headers(self) -> dict[str, str]:
        """``RateLimit-*`` as in the IETF draft, plus ``Retry-After`` when refused.

        Worth sending even on success: a client that can see it is at 950 of 1000 can slow down,
        and one that cannot will keep going until it is refused.
        """
        headers = {
            "RateLimit-Limit": str(self.limit),
            "RateLimit-Remaining": str(self.remaining),
        }
        if not self.allowed:
            headers["Retry-After"] = str(max(1, math.ceil(self.retry_after_seconds)))
        return headers


class RateLimiter(Protocol):
    async def check(self, key: str, limit: Limit, *, cost: int = 1, record: bool = True) -> Decision:
        """Decide, and by default record the hit.

        ``record=False`` answers "would this be allowed" without consuming budget. It exists so
        that a request refused at one scope does not charge the others -- see ``enforce``.
        """
        ...

    async def reset(self, key: str) -> None: ...


class InMemoryRateLimiter:
    """Correct for one process. Wrong the moment there are two.

    The fallback for tests and single-node development, and it is named so nobody mistakes it for
    something else. Two API replicas each enforce the full limit, so the effective limit is
    double -- which is exactly the kind of quietly-wrong that a production settings check should
    refuse, and does.
    """

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)
        self._lock = asyncio.Lock()

    async def check(self, key: str, limit: Limit, *, cost: int = 1, record: bool = True) -> Decision:
        async with self._lock:
            now = time.monotonic()
            hits = self._hits[key]
            cutoff = now - limit.window_seconds
            while hits and hits[0] <= cutoff:
                hits.popleft()

            if len(hits) + cost > limit.requests:
                retry = (hits[0] + limit.window_seconds) - now if hits else limit.window_seconds
                return Decision(False, len(hits), limit.requests, max(0.0, retry))

            if not record:
                return Decision(True, len(hits), limit.requests)

            hits.extend([now] * cost)
            return Decision(True, len(hits), limit.requests)

    async def reset(self, key: str) -> None:
        async with self._lock:
            self._hits.pop(key, None)


class RedisRateLimiter:
    """The production limiter. One atomic Lua call per check."""

    def __init__(self, redis: Any, *, namespace: str = "rl") -> None:
        self.redis = redis
        self.namespace = namespace
        self._script: Any = None

    async def check(self, key: str, limit: Limit, *, cost: int = 1, record: bool = True) -> Decision:
        if self._script is None:
            self._script = self.redis.register_script(SLIDING_WINDOW_LUA)

        try:
            allowed, used, retry = await self._script(
                keys=[f"{self.namespace}:{key}"],
                args=[time.time(), limit.window_seconds, limit.requests, cost, int(record)],
            )
        except Exception as exc:
            # Fail OPEN, and say so loudly.
            #
            # A limiter outage must not become an availability outage for every customer. The
            # exposure is bounded -- Redis being down is already an alert -- and the alternative
            # is that a cache failure takes the whole product offline, which is a far larger
            # incident than a few minutes of unlimited requests.
            logger.error("rate_limit.unavailable", extra={"error": str(exc), "key": key})
            return Decision(True, 0, limit.requests)

        return Decision(bool(allowed), int(used), limit.requests, float(retry))

    async def reset(self, key: str) -> None:
        await self.redis.delete(f"{self.namespace}:{key}", f"{self.namespace}:{key}:seq")


# ----------------------------------------------------------------------------------------------
# Policy
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LimitPolicy:
    """What a given route costs, at each scope.

    Per-user and per-tenant are both checked and the tighter wins. Per-IP applies only to
    unauthenticated routes, because behind corporate NAT a whole customer shares one address --
    IP-limiting an authenticated route would throttle an entire office because of one script.
    """

    per_user: Limit | None = None
    per_tenant: Limit | None = None
    per_ip: Limit | None = None
    cost: int = 1


#: Defaults per route group. Generous enough not to interfere with a person using the product,
#: tight enough to bound a runaway client.
DEFAULT_POLICIES: dict[str, LimitPolicy] = {
    # The expensive one: an LLM call per request.
    "ask": LimitPolicy(per_user=Limit(30, 60), per_tenant=Limit(300, 60)),
    # Retrieval only. Cheap enough to be generous, because a person typing into a search box
    # legitimately produces bursts.
    "search": LimitPolicy(per_user=Limit(120, 60), per_tenant=Limit(1200, 60)),
    "upload": LimitPolicy(per_user=Limit(60, 60), per_tenant=Limit(600, 60)),
    # Unauthenticated and the cheapest place to probe which domains are customers, so this one
    # is per-IP and deliberately tight.
    "discover": LimitPolicy(per_ip=Limit(20, 60)),
    # Password login. Tight per-IP, and the account lockout is a separate control on top.
    "login": LimitPolicy(per_ip=Limit(10, 60)),
    "default": LimitPolicy(per_user=Limit(600, 60), per_tenant=Limit(6000, 60)),
}


async def enforce(
    limiter: RateLimiter,
    policy: LimitPolicy,
    *,
    route: str,
    user_id: str | None,
    tenant_id: str | None,
    ip: str | None,
) -> Decision:
    """Check every applicable scope, then charge them all -- or none.

    Two passes, because charging as we go is wrong in a way that is easy to miss. Checking the
    user scope first records a hit there; if the *tenant* scope then refuses, the user has been
    charged for a request that never ran. Under a tenant-level throttle every user also drains
    their own allowance doing nothing, so when the tenant limit clears they are individually
    throttled as well -- an outage that outlasts its cause and has no obvious explanation.

    The gap between deciding and recording admits at most one extra request per scope under
    concurrency, which is a far smaller error than the one it removes.
    """
    checks: list[tuple[str, Limit]] = []
    if policy.per_user and user_id:
        checks.append((f"u:{user_id}:{route}", policy.per_user))
    if policy.per_tenant and tenant_id:
        checks.append((f"t:{tenant_id}:{route}", policy.per_tenant))
    if policy.per_ip and ip:
        checks.append((f"ip:{ip}:{route}", policy.per_ip))

    for key, limit in checks:
        decision = await limiter.check(key, limit, cost=policy.cost, record=False)
        if not decision.allowed:
            return decision

    tightest = Decision(True, 0, 0)
    for key, limit in checks:
        decision = await limiter.check(key, limit, cost=policy.cost)
        if tightest.limit == 0 or decision.remaining < tightest.remaining:
            tightest = decision
    return tightest


# ----------------------------------------------------------------------------------------------
# Concurrency
# ----------------------------------------------------------------------------------------------


class ConcurrencyExceededError(Exception):
    """Too many requests of this kind are already in flight for this tenant."""


class ConcurrencyLimiter:
    """A semaphore per tenant, bounding *in-flight* work rather than work per minute.

    This is the control a per-minute limit cannot provide. Fifty simultaneous ``ask`` requests
    are all within a 300-per-minute budget and all hold an LLM call open at once; the service
    degrades for everyone while the rate limiter reports that nothing is wrong.

    Acquisition does not queue indefinitely. A caller that waits behind forty others has already
    exceeded any useful latency budget, and telling them to retry is both more honest and
    cheaper than holding a connection open to eventually time out.
    """

    def __init__(self, *, per_tenant: int = 8, acquire_timeout_s: float = 2.0) -> None:
        self.per_tenant = per_tenant
        self.acquire_timeout_s = acquire_timeout_s
        self._semaphores: dict[str, asyncio.Semaphore] = {}
        self._lock = asyncio.Lock()

    async def _semaphore(self, key: str) -> asyncio.Semaphore:
        async with self._lock:
            if key not in self._semaphores:
                self._semaphores[key] = asyncio.Semaphore(self.per_tenant)
            return self._semaphores[key]

    async def acquire(self, key: str) -> None:
        semaphore = await self._semaphore(key)
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=self.acquire_timeout_s)
        except TimeoutError as exc:
            raise ConcurrencyExceededError(
                "Too many requests are being processed for your organisation at once. Please try again in a moment."
            ) from exc

    def release(self, key: str) -> None:
        semaphore = self._semaphores.get(key)
        if semaphore is not None:
            semaphore.release()

    def slot(self, key: str) -> _Slot:
        return _Slot(self, key)

    def in_flight(self, key: str) -> int:
        semaphore = self._semaphores.get(key)
        # ``_value`` is the remaining permits; the difference is what is held.
        return self.per_tenant - semaphore._value if semaphore else 0


class _Slot:
    """``async with limiter.slot(key):`` -- released even when the handler raises."""

    def __init__(self, limiter: ConcurrencyLimiter, key: str) -> None:
        self.limiter = limiter
        self.key = key

    async def __aenter__(self) -> None:
        await self.limiter.acquire(self.key)

    async def __aexit__(self, *exc_info: object) -> None:
        self.limiter.release(self.key)


def build_limiter(settings: Any) -> RateLimiter:
    """Redis when configured, in-memory otherwise.

    The production settings validator requires a Redis URL, so the in-memory path cannot be
    reached in production by omission -- which is the only way anyone would reach it, since
    nobody chooses a limiter that is wrong across replicas on purpose.
    """
    url = getattr(settings, "redis_url", None)
    if not url:
        logger.warning("rate_limit.in_memory", extra={"reason": "no redis_url configured"})
        return InMemoryRateLimiter()

    from redis.asyncio import Redis

    return RedisRateLimiter(Redis.from_url(str(url), decode_responses=True))
