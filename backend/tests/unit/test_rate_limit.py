"""Rate limits, the two scopes, and the concurrency cap a rate limit cannot provide.

The test worth reading is ``test_a_rate_limit_alone_does_not_stop_concurrent_expensive_work``.
It is the reason both controls exist: fifty simultaneous requests are all inside a 300-per-minute
budget and all hold an LLM call open at once, so the service degrades while the rate limiter
reports that nothing is wrong.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from app.security.rate_limit import (
    DEFAULT_POLICIES,
    ConcurrencyExceededError,
    ConcurrencyLimiter,
    Decision,
    InMemoryRateLimiter,
    Limit,
    LimitPolicy,
    RedisRateLimiter,
    build_limiter,
    enforce,
)


@pytest.fixture
def limiter() -> InMemoryRateLimiter:
    return InMemoryRateLimiter()


# ------------------------------------------------------------------------------------------
# The window
# ------------------------------------------------------------------------------------------


async def test_requests_inside_the_limit_are_allowed(limiter: InMemoryRateLimiter) -> None:
    limit = Limit(3, 60)
    for _ in range(3):
        assert (await limiter.check("k", limit)).allowed


async def test_the_request_past_the_limit_is_refused(limiter: InMemoryRateLimiter) -> None:
    limit = Limit(2, 60)
    await limiter.check("k", limit)
    await limiter.check("k", limit)

    decision = await limiter.check("k", limit)
    assert not decision.allowed
    assert decision.remaining == 0


async def test_a_refusal_does_not_consume_more_of_the_budget(limiter: InMemoryRateLimiter) -> None:
    """Otherwise a client that keeps retrying can never recover: each refusal pushes the window
    forward and the limit is permanent rather than temporary."""
    limit = Limit(1, 60)
    await limiter.check("k", limit)
    first = await limiter.check("k", limit)
    second = await limiter.check("k", limit)

    assert not first.allowed and not second.allowed
    assert second.used == first.used


async def test_the_window_slides_rather_than_resetting_on_a_boundary(limiter: InMemoryRateLimiter) -> None:
    """A fixed window lets a client send the full budget at the end of one and again at the
    start of the next -- twice the intended rate, reliably, for anyone who notices."""
    limit = Limit(2, 0.15)
    await limiter.check("k", limit)
    await limiter.check("k", limit)
    assert not (await limiter.check("k", limit)).allowed

    await asyncio.sleep(0.2)
    assert (await limiter.check("k", limit)).allowed


async def test_keys_are_independent(limiter: InMemoryRateLimiter) -> None:
    limit = Limit(1, 60)
    assert (await limiter.check("a", limit)).allowed
    assert (await limiter.check("b", limit)).allowed


async def test_a_reset_clears_one_key(limiter: InMemoryRateLimiter) -> None:
    limit = Limit(1, 60)
    await limiter.check("k", limit)
    await limiter.reset("k")
    assert (await limiter.check("k", limit)).allowed


async def test_a_costly_request_can_consume_several_units(limiter: InMemoryRateLimiter) -> None:
    limit = Limit(10, 60)
    assert (await limiter.check("k", limit, cost=8)).allowed
    assert not (await limiter.check("k", limit, cost=5)).allowed


def test_a_limit_must_allow_something() -> None:
    with pytest.raises(ValueError, match="at least one request"):
        Limit(0, 60)
    with pytest.raises(ValueError):
        Limit(10, 0)


# ------------------------------------------------------------------------------------------
# Headers
# ------------------------------------------------------------------------------------------


def test_headers_are_sent_on_success_too() -> None:
    """A client that can see it is at 950 of 1000 can slow down. One that cannot will keep
    going until it is refused."""
    headers = Decision(True, 950, 1000).headers()
    assert headers["RateLimit-Limit"] == "1000"
    assert headers["RateLimit-Remaining"] == "50"
    assert "Retry-After" not in headers


def test_a_refusal_carries_a_retry_after_of_at_least_one_second() -> None:
    """Retry-After: 0 is an invitation to retry immediately, which is the opposite of the point."""
    assert Decision(False, 10, 10, 0.2).headers()["Retry-After"] == "1"


def test_retry_after_is_rounded_up_not_down() -> None:
    assert Decision(False, 10, 10, 4.2).headers()["Retry-After"] == "5"


def test_remaining_never_goes_negative() -> None:
    assert Decision(False, 15, 10).remaining == 0


# ------------------------------------------------------------------------------------------
# The two scopes
# ------------------------------------------------------------------------------------------


async def test_a_users_own_limit_refuses_them(limiter: InMemoryRateLimiter) -> None:
    policy = LimitPolicy(per_user=Limit(1, 60), per_tenant=Limit(100, 60))
    await enforce(limiter, policy, route="ask", user_id="u1", tenant_id="t1", ip=None)

    decision = await enforce(limiter, policy, route="ask", user_id="u1", tenant_id="t1", ip=None)
    assert not decision.allowed


async def test_a_tenant_limit_catches_what_a_per_user_limit_cannot(limiter: InMemoryRateLimiter) -> None:
    """Two hundred users each well inside their own allowance still saturate shared
    infrastructure. A per-user limit alone never sees it."""
    policy = LimitPolicy(per_user=Limit(100, 60), per_tenant=Limit(3, 60))
    for user in range(3):
        assert (await enforce(limiter, policy, route="ask", user_id=f"u{user}", tenant_id="t1", ip=None)).allowed

    decision = await enforce(limiter, policy, route="ask", user_id="u99", tenant_id="t1", ip=None)
    assert not decision.allowed


async def test_one_tenants_exhaustion_does_not_affect_another(limiter: InMemoryRateLimiter) -> None:
    policy = LimitPolicy(per_tenant=Limit(1, 60))
    await enforce(limiter, policy, route="ask", user_id="u1", tenant_id="t1", ip=None)

    assert not (await enforce(limiter, policy, route="ask", user_id="u1", tenant_id="t1", ip=None)).allowed
    assert (await enforce(limiter, policy, route="ask", user_id="u2", tenant_id="t2", ip=None)).allowed


async def test_a_tenant_refusal_does_not_also_burn_the_users_allowance(limiter: InMemoryRateLimiter) -> None:
    """Charging for a request that never ran means a user throttled at the tenant level also
    exhausts their personal budget doing nothing."""
    policy = LimitPolicy(per_user=Limit(10, 60), per_tenant=Limit(1, 60))
    await enforce(limiter, policy, route="ask", user_id="u1", tenant_id="t1", ip=None)
    await enforce(limiter, policy, route="ask", user_id="u2", tenant_id="t1", ip=None)

    # u2 was refused at the tenant scope; their own counter must not have moved.
    assert (await limiter.check("u:u2:ask", Limit(10, 60))).used == 1


async def test_routes_have_independent_budgets(limiter: InMemoryRateLimiter) -> None:
    """Searching a lot must not exhaust the ability to ask a question."""
    policy = LimitPolicy(per_user=Limit(1, 60))
    await enforce(limiter, policy, route="search", user_id="u1", tenant_id="t1", ip=None)
    assert (await enforce(limiter, policy, route="ask", user_id="u1", tenant_id="t1", ip=None)).allowed


async def test_an_unauthenticated_route_is_limited_by_address(limiter: InMemoryRateLimiter) -> None:
    policy = LimitPolicy(per_ip=Limit(1, 60))
    await enforce(limiter, policy, route="discover", user_id=None, tenant_id=None, ip="203.0.113.5")
    decision = await enforce(limiter, policy, route="discover", user_id=None, tenant_id=None, ip="203.0.113.5")
    assert not decision.allowed


def test_authenticated_routes_are_not_limited_by_address() -> None:
    """Behind corporate NAT a whole customer shares one address, so IP-limiting an authenticated
    route throttles an entire office because of one script."""
    for route in ("ask", "search", "upload"):
        assert DEFAULT_POLICIES[route].per_ip is None


def test_the_unauthenticated_routes_are_the_ones_limited_by_address() -> None:
    assert DEFAULT_POLICIES["discover"].per_ip is not None
    assert DEFAULT_POLICIES["login"].per_ip is not None


def test_asking_is_budgeted_more_tightly_than_searching() -> None:
    """One is an LLM call; the other is a retrieval. A person typing into a search box
    legitimately produces bursts that would be alarming on /ask."""
    assert DEFAULT_POLICIES["ask"].per_user is not None
    assert DEFAULT_POLICIES["search"].per_user is not None
    assert DEFAULT_POLICIES["ask"].per_user.requests < DEFAULT_POLICIES["search"].per_user.requests


# ------------------------------------------------------------------------------------------
# Redis
# ------------------------------------------------------------------------------------------


class FakeRedis:
    def __init__(self, result: list[Any] | Exception) -> None:
        self.result = result
        self.calls: list[dict[str, Any]] = []
        self.deleted: list[tuple[str, ...]] = []

    def register_script(self, source: str) -> Any:
        async def run(*, keys: list[str], args: list[Any]) -> Any:
            self.calls.append({"keys": keys, "args": args})
            if isinstance(self.result, Exception):
                raise self.result
            return self.result

        return run

    async def delete(self, *keys: str) -> None:
        self.deleted.append(keys)


async def test_the_redis_check_is_one_atomic_call() -> None:
    """Read-decide-write across two round trips is a race: under real concurrency every caller
    reads the count before any of them writes, and they all conclude there is room."""
    redis = FakeRedis([1, 3, 0])
    decision = await RedisRateLimiter(redis).check("u:1:ask", Limit(10, 60))

    assert decision.allowed and decision.used == 3
    assert len(redis.calls) == 1


async def test_redis_returns_an_exact_retry_after_rather_than_a_whole_window() -> None:
    redis = FakeRedis([0, 10, 17.5])
    decision = await RedisRateLimiter(redis).check("u:1:ask", Limit(10, 60))
    assert not decision.allowed
    assert decision.retry_after_seconds == 17.5


async def test_a_peek_does_not_consume_budget() -> None:
    """What makes the two-pass enforce possible: a scope can be asked without being charged."""
    limiter = InMemoryRateLimiter()
    limit = Limit(1, 60)

    assert (await limiter.check("k", limit, record=False)).allowed
    assert (await limiter.check("k", limit, record=False)).allowed
    assert (await limiter.check("k", limit)).allowed, "neither peek should have consumed the one unit"


async def test_a_redis_peek_skips_the_write() -> None:
    redis = FakeRedis([1, 0, 0])
    await RedisRateLimiter(redis).check("u:1:ask", Limit(10, 60), record=False)
    assert redis.calls[0]["args"][4] == 0


async def test_a_redis_outage_fails_open() -> None:
    """A limiter outage must not become an availability outage for every customer. Redis being
    down is already an alert; the alternative is a cache failure taking the product offline."""
    redis = FakeRedis(ConnectionError("redis unreachable"))
    assert (await RedisRateLimiter(redis).check("u:1:ask", Limit(10, 60))).allowed


async def test_a_reset_clears_the_sequence_counter_too() -> None:
    """The counter is what keeps ZSET members unique. Leaving it behind makes the next window
    start from a stale sequence, which is harmless but confusing in an incident."""
    redis = FakeRedis([1, 0, 0])
    await RedisRateLimiter(redis).reset("u:1:ask")
    assert any("seq" in key for key in redis.deleted[0])


def test_the_lua_script_gives_each_hit_a_unique_member() -> None:
    """Identical timestamps would collapse into one ZSET entry and the window would silently
    under-count -- allowing several times the intended rate under exactly the burst the limiter
    exists to catch."""
    from app.security.rate_limit import SLIDING_WINDOW_LUA

    assert "INCR" in SLIDING_WINDOW_LUA
    assert "ZADD" in SLIDING_WINDOW_LUA


def test_the_lua_script_expires_its_keys() -> None:
    """Without this, every key a tenant ever used stays in Redis for good."""
    from app.security.rate_limit import SLIDING_WINDOW_LUA

    assert SLIDING_WINDOW_LUA.count("EXPIRE") >= 2


def test_without_redis_the_fallback_is_the_in_memory_one() -> None:
    class Settings:
        redis_url = None

    assert isinstance(build_limiter(Settings()), InMemoryRateLimiter)


# ------------------------------------------------------------------------------------------
# Concurrency
# ------------------------------------------------------------------------------------------


async def test_a_rate_limit_alone_does_not_stop_concurrent_expensive_work() -> None:
    """The reason both controls exist.

    Fifty simultaneous requests are every one of them inside a 300-per-minute budget, and every
    one holds an LLM call open. The service degrades for everyone while the rate limiter reports
    that nothing is wrong.
    """
    limiter = InMemoryRateLimiter()
    limit = Limit(300, 60)

    decisions = await asyncio.gather(*(limiter.check("t:1:ask", limit) for _ in range(50)))
    assert all(decision.allowed for decision in decisions)  # the rate limiter sees no problem

    concurrency = ConcurrencyLimiter(per_tenant=8, acquire_timeout_s=0.05)

    async def hold() -> bool:
        try:
            async with concurrency.slot("t:1"):
                await asyncio.sleep(0.2)
            return True
        except ConcurrencyExceededError:
            return False

    outcomes = await asyncio.gather(*(hold() for _ in range(50)))
    assert sum(outcomes) <= 8, "the concurrency cap is what actually bounds in-flight work"
    assert sum(outcomes) >= 1


async def test_a_slot_is_released_when_the_handler_raises() -> None:
    """A leaked permit is permanent: the tenant's concurrency drops by one for the life of the
    process, and nothing reports it."""
    limiter = ConcurrencyLimiter(per_tenant=1, acquire_timeout_s=0.05)

    with pytest.raises(RuntimeError):
        async with limiter.slot("t:1"):
            raise RuntimeError("handler failed")

    async with limiter.slot("t:1"):
        pass  # reacquired, so the permit came back


async def test_waiting_forever_is_not_an_option() -> None:
    """A caller queued behind forty others has already blown any useful latency budget. Telling
    them to retry is more honest and cheaper than holding a connection to eventually time out."""
    limiter = ConcurrencyLimiter(per_tenant=1, acquire_timeout_s=0.05)

    async with limiter.slot("t:1"):
        with pytest.raises(ConcurrencyExceededError, match="try again"):
            await limiter.acquire("t:1")


async def test_tenants_do_not_share_a_concurrency_budget() -> None:
    limiter = ConcurrencyLimiter(per_tenant=1, acquire_timeout_s=0.05)
    async with limiter.slot("t:1"), limiter.slot("t:2"):
        pass


async def test_in_flight_is_reported_for_monitoring() -> None:
    limiter = ConcurrencyLimiter(per_tenant=4)
    assert limiter.in_flight("t:1") == 0
    async with limiter.slot("t:1"):
        assert limiter.in_flight("t:1") == 1
    assert limiter.in_flight("t:1") == 0
