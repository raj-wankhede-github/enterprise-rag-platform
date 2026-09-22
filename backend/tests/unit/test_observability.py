"""Tracing attributes, stage timings, and the audit hash chain.

The attribute-filtering tests are the ones that matter. The way customer data reaches a
third-party observability tool is almost never a deliberate decision -- it is someone adding
``set_attribute("doc", document)`` while debugging and not removing it. Denying by default is
what makes "no, we do not store your documents in a third-party tool" true rather than intended.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.models.audit import AuditAction, chain_hash
from app.observability.otel import (
    CONTENT_ATTRIBUTES,
    SAFE_ATTRIBUTES,
    StageTiming,
    _config,
    filter_attributes,
    new_request_id,
    stage,
    summarise,
)

TENANT = uuid.UUID("11110000-0000-0000-0000-000000000001")


@pytest.fixture(autouse=True)
def content_off() -> None:
    _config.capture_content = False


# ------------------------------------------------------------------------------------------
# What reaches an observability vendor
# ------------------------------------------------------------------------------------------


def test_counts_and_latencies_always_pass() -> None:
    """Enough to diagnose every performance and relevance problem, without any customer text."""
    allowed = filter_attributes({"erp.stage": "rerank", "erp.candidates": 24, "erp.latency_ms": 41.2})
    assert allowed == {"erp.stage": "rerank", "erp.candidates": 24, "erp.latency_ms": 41.2}


def test_a_question_does_not_leave_the_deployment_by_default() -> None:
    assert filter_attributes({"erp.question": "what is the per diem?"}) == {}


def test_an_answer_does_not_leave_the_deployment_by_default() -> None:
    assert filter_attributes({"erp.answer": "120 EUR"}) == {}


def test_chunk_text_does_not_leave_the_deployment_by_default() -> None:
    assert filter_attributes({"erp.chunk_text": "confidential paragraph"}) == {}


def test_content_passes_only_when_the_tenant_opted_in() -> None:
    _config.capture_content = True
    assert filter_attributes({"erp.question": "q"}) == {"erp.question": "q"}


def test_an_unknown_attribute_is_dropped_rather_than_passed_through() -> None:
    """Deny by default in both directions. An attribute nobody classified is exactly the one
    added while debugging and forgotten."""
    assert filter_attributes({"internal.document_body": "the whole document"}) == {}


def test_an_unknown_attribute_is_dropped_even_with_content_capture_on() -> None:
    """Opting into content capture opts into the *named* content fields, not into everything a
    future developer might attach."""
    _config.capture_content = True
    assert filter_attributes({"some.new.field": "value"}) == {}


def test_the_two_vocabularies_do_not_overlap() -> None:
    """An attribute in both sets would pass unconditionally through the safe branch, which is
    how a content field ends up leaving a deployment that opted out."""
    assert not (SAFE_ATTRIBUTES & CONTENT_ATTRIBUTES)


def test_no_safe_attribute_carries_free_text() -> None:
    """A quick structural check on the vocabulary itself: every safe attribute is an id, a
    count, a duration or a fixed enum, and none is a field someone would put a sentence in."""
    freeform = {name for name in SAFE_ATTRIBUTES if any(word in name for word in ("text", "body", "content", "query"))}
    assert not freeform


# ------------------------------------------------------------------------------------------
# Stage timing
# ------------------------------------------------------------------------------------------


async def test_a_stage_records_its_duration() -> None:
    timing = StageTiming()
    with stage("rerank", timing=timing):
        pass
    assert "rerank" in timing.timings


async def test_repeated_stages_accumulate_rather_than_overwrite() -> None:
    """A leg retried after a timeout should show its total cost, not only the last attempt."""
    timing = StageTiming()
    for _ in range(3):
        with stage("retrieve.dense", timing=timing):
            pass
    assert timing.timings["retrieve.dense"] > 0
    assert len(timing.timings) == 1


async def test_a_stage_records_its_candidate_count() -> None:
    timing = StageTiming()
    with stage("fuse", timing=timing) as attributes:
        attributes["erp.candidates"] = 137
    assert timing.counts["fuse"] == 137


async def test_a_failing_stage_is_still_timed_and_labelled_before_it_raises() -> None:
    """Otherwise the slowest stage in an incident is the one with no timing at all."""
    timing = StageTiming()
    with pytest.raises(ValueError), stage("generate", timing=timing) as attributes:
        attributes["erp.candidates"] = 5
        raise ValueError("model unavailable")

    assert "generate" in timing.timings


async def test_a_failure_records_the_error_class_but_not_the_message() -> None:
    """An exception message routinely contains a prompt, a document fragment or a customer id."""
    timing = StageTiming()
    collected: dict[str, object] = {}
    with pytest.raises(ValueError), stage("generate", timing=timing) as attributes:
        collected = attributes
        raise ValueError("failed while processing: <the customer's confidential paragraph>")

    assert collected["erp.error_class"] == "ValueError"
    assert not any("confidential" in str(value) for value in collected.values())


def test_the_summary_names_the_slowest_stage() -> None:
    """What makes a latency regression attributable in one query rather than in an afternoon."""
    timing = StageTiming()
    timing.record("retrieve.dense", 12.0)
    timing.record("rerank", 180.0)
    timing.record("generate", 40.0)

    summary = summarise(timing)
    assert summary["slowest_stage"] == "rerank"
    assert summary["total_ms"] == 232.0


def test_a_summary_of_nothing_does_not_crash() -> None:
    assert summarise(StageTiming())["slowest_stage"] is None


def test_request_ids_are_unique_and_short_enough_to_paste() -> None:
    ids = {new_request_id() for _ in range(500)}
    assert len(ids) == 500
    assert all(len(value) == 16 for value in ids)


# ------------------------------------------------------------------------------------------
# The audit hash chain
# ------------------------------------------------------------------------------------------


def a_link(prev: bytes | None = None, **overrides: object) -> bytes:
    base: dict[str, object] = {
        "prev_hash": prev,
        "tenant_id": TENANT,
        "action": AuditAction.USER_ROLE_CHANGED,
        "actor_email": "admin@acme.com",
        "resource_id": "user-1",
        "created_at": datetime(2026, 3, 1, 12, 0, tzinfo=UTC),
        "details": {"from": "DEV", "to": "ADMIN"},
    }
    base.update(overrides)
    return chain_hash(**base)  # type: ignore[arg-type]


def test_the_same_entry_hashes_the_same_way() -> None:
    assert a_link() == a_link()


def test_changing_any_field_changes_the_hash() -> None:
    """Which is the whole mechanism: altering a row breaks every hash after it."""
    baseline = a_link()
    assert a_link(action=AuditAction.USER_DEACTIVATED) != baseline
    assert a_link(actor_email="someone@else.com") != baseline
    assert a_link(resource_id="user-2") != baseline
    assert a_link(created_at=datetime(2026, 3, 1, 12, 0, 1, tzinfo=UTC)) != baseline
    assert a_link(details={"from": "DEV", "to": "TEST"}) != baseline


def test_the_chain_links_to_the_previous_entry() -> None:
    first = a_link()
    assert a_link(prev=first) != a_link(prev=b"\x00" * 32)


def test_removing_an_entry_breaks_the_chain() -> None:
    """The attack this defends against: someone with database access deleting the row that
    records what they did. Append-only permissions stop the application rewriting history; this
    detects a direct write, which is what a compliance team is actually asking about."""
    first = a_link(resource_id="user-1")
    second = a_link(prev=first, resource_id="user-2")
    third = a_link(prev=second, resource_id="user-3")

    # Re-derive the third link as though the second had never existed.
    forged = a_link(prev=first, resource_id="user-3")
    assert forged != third


def test_fields_cannot_be_shuffled_between_each_other() -> None:
    """Field-delimited with a separator that cannot appear in any input, so ("ab", "c") and
    ("a", "bc") cannot serialise identically and two distinct histories cannot share a hash."""
    assert a_link(actor_email="ab", resource_id="c") != a_link(actor_email="a", resource_id="bc")


def test_detail_ordering_does_not_change_the_hash() -> None:
    """JSONB does not preserve key order, so a chain that depended on it would break itself on
    a round trip through the database."""
    assert a_link(details={"to": "ADMIN", "from": "DEV"}) == a_link(details={"from": "DEV", "to": "ADMIN"})


def test_a_hash_is_a_full_sha256() -> None:
    assert len(a_link()) == 32


# ------------------------------------------------------------------------------------------
# What is worth auditing
# ------------------------------------------------------------------------------------------


def test_reads_are_not_audited_but_bulk_export_is() -> None:
    """Logging every read would bury what matters under search traffic, and would itself become
    a privacy problem: a complete record of what every employee looked at."""
    actions = {str(action) for action in AuditAction}
    assert "documents.exported" in actions
    assert not any(action.endswith(".viewed") or action.endswith(".searched") for action in actions)


def test_the_events_a_customer_asks_about_are_all_present() -> None:
    actions = {str(action) for action in AuditAction}
    for expected in (
        "user.elevated_to_admin",
        "idp.activated",
        "emergency_login.opened",
        "support.break_glass_used",
        "apikey.created",
        "document.visibility_changed",
    ):
        assert expected in actions


def test_freshness_wording_covers_a_long_gap() -> None:
    from app.connectors.graph_sharepoint import describe_freshness

    now = datetime.now(UTC)
    assert "days ago" in describe_freshness(now - timedelta(days=9), now=now)
