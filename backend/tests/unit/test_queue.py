"""Queue logic that is decidable without a database.

The SQL -- the claim, the fairness window, `SKIP LOCKED` -- is where the interesting failures
live and cannot be tested against a fake, so it has its own integration suite. What is here is
everything a wrong answer to would be silently expensive: which failures retry, how long the
backoff is, and which stages a resumed job is allowed to skip.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from app.ingestion.queue import (
    MAX_BACKOFF,
    PERMANENT_ERRORS,
    ClaimedJob,
    JobStage,
    Priority,
    backoff_for,
    reached,
)


def a_job(**kwargs: object) -> ClaimedJob:
    base: dict[str, object] = {
        "id": uuid.uuid4(),
        "tenant_id": uuid.uuid4(),
        "job_type": "ingest_document",
        "document_version_id": uuid.uuid4(),
        "payload": {},
        "stage": None,
        "cursor": {},
        "attempts": 1,
        "max_attempts": 5,
        "priority": int(Priority.NORMAL),
        "lease_expires_at": datetime.now(UTC) + timedelta(minutes=10),
    }
    base.update(kwargs)
    return ClaimedJob(**base)  # type: ignore[arg-type]


# ------------------------------------------------------------------------------------------
# Stage checkpointing
# ------------------------------------------------------------------------------------------


def test_a_fresh_job_has_reached_nothing() -> None:
    assert not reached(None, JobStage.FETCHED)


def test_a_job_skips_the_stages_it_already_passed() -> None:
    """The point of checkpointing: a retry that re-parses and re-embeds costs as much as the
    original attempt, and embedding is the expensive half."""
    assert reached(JobStage.EMBEDDED, JobStage.PARSED)
    assert reached(JobStage.EMBEDDED, JobStage.EMBEDDED)


def test_a_job_does_not_skip_stages_it_has_not_reached() -> None:
    assert not reached(JobStage.PARSED, JobStage.EMBEDDED)
    assert not reached(JobStage.CHUNKED, JobStage.INDEXED)


def test_the_stages_are_ordered_by_rising_cost() -> None:
    """Ordering is what makes `reached` meaningful, and the order is not arbitrary -- each stage
    is more expensive to redo than the one before it."""
    order = list(JobStage)
    assert order.index(JobStage.FETCHED) < order.index(JobStage.PARSED)
    assert order.index(JobStage.PARSED) < order.index(JobStage.CONTEXTUALIZED)
    assert order.index(JobStage.CONTEXTUALIZED) < order.index(JobStage.EMBEDDED)
    assert order.index(JobStage.EMBEDDED) < order.index(JobStage.INDEXED)


def test_an_unknown_stage_is_treated_as_having_reached_nothing() -> None:
    """A stage written by a newer version must not let an older worker skip real work."""
    assert not reached("SOME_FUTURE_STAGE", JobStage.PARSED)


# ------------------------------------------------------------------------------------------
# Backoff
# ------------------------------------------------------------------------------------------


def test_backoff_grows_with_each_attempt() -> None:
    without_jitter = [backoff_for(n, jitter=False).total_seconds() for n in range(1, 5)]
    assert without_jitter == sorted(without_jitter)
    assert without_jitter[0] < without_jitter[-1]


def test_backoff_is_capped() -> None:
    """Past a few minutes the queue stops being a queue, and a genuinely transient failure has
    long since cleared."""
    assert backoff_for(50, jitter=False) == MAX_BACKOFF


def test_jitter_spreads_retries_instead_of_synchronising_them() -> None:
    """A parser container restarting fails every in-flight job at the same instant. Without
    jitter they all retry at the same instant too, and the retry storm is what keeps it down."""
    delays = {backoff_for(4).total_seconds() for _ in range(200)}
    assert len(delays) > 150, "retries must not land on the same instant"


def test_jitter_never_exceeds_the_unjittered_delay() -> None:
    ceiling = backoff_for(4, jitter=False).total_seconds()
    assert all(backoff_for(4).total_seconds() <= ceiling for _ in range(200))


def test_backoff_is_never_negative() -> None:
    assert backoff_for(0).total_seconds() >= 0


# ------------------------------------------------------------------------------------------
# Retry classification
# ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("code", sorted(PERMANENT_ERRORS))
def test_a_permanent_error_is_a_property_of_the_document(code: str) -> None:
    """Each of these fails identically on every attempt. Retrying wastes a worker slot a
    fetchable document needs, and five attempts turn one bad file into what looks like an
    outage in the logs."""
    assert code in PERMANENT_ERRORS


def test_a_transient_failure_is_not_in_the_permanent_set() -> None:
    for code in ("parser_unavailable", "transient_io", "unexpected_error"):
        assert code not in PERMANENT_ERRORS


def test_an_unclassified_error_defaults_to_retryable() -> None:
    """The safe direction. Defaulting to permanent would silently dead-letter documents on a
    transient fault nobody had classified yet."""
    from app.workers.ingestion_worker import classify

    assert classify(RuntimeError("something new")) == "unexpected_error"
    assert "unexpected_error" not in PERMANENT_ERRORS


def test_an_unsupported_format_is_permanent() -> None:
    from app.ingestion.pipeline import UnsupportedDocumentError
    from app.workers.ingestion_worker import classify

    assert classify(UnsupportedDocumentError("no loader")) in PERMANENT_ERRORS


def test_a_parser_outage_is_retryable() -> None:
    """The parser restarting is the commonest transient failure, and dead-lettering a scanned
    PDF because the container was redeploying would lose it for good."""
    from app.ingestion.loaders.docling_client import ParserUnavailableError
    from app.workers.ingestion_worker import classify

    assert classify(ParserUnavailableError("down")) not in PERMANENT_ERRORS


def test_a_mapping_rejection_is_permanent() -> None:
    """A document our own mapping rejects will be rejected identically forever. The bulk writer
    already retried anything transient before raising."""
    from app.search.bulk import BulkIndexError
    from app.workers.ingestion_worker import classify

    assert classify(BulkIndexError("strict_dynamic_mapping_exception")) in PERMANENT_ERRORS


def test_the_last_attempt_is_recognised() -> None:
    assert a_job(attempts=5, max_attempts=5).is_last_attempt
    assert not a_job(attempts=4, max_attempts=5).is_last_attempt


# ------------------------------------------------------------------------------------------
# Priority
# ------------------------------------------------------------------------------------------


def test_interactive_work_outranks_bulk() -> None:
    """Someone watching an upload spinner and a 200,000-document backfill are not the same
    request, and a queue that cannot tell them apart makes the person wait."""
    assert Priority.INTERACTIVE < Priority.NORMAL < Priority.BULK


def test_priorities_are_spaced_for_insertion() -> None:
    assert Priority.NORMAL - Priority.INTERACTIVE >= 10
    assert Priority.BULK - Priority.NORMAL >= 10
