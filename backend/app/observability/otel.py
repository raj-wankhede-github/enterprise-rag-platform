"""Tracing, instrumented once.

**OpenTelemetry SDK in business code, never a vendor SDK.** Spans go to a Collector over OTLP,
and the Collector fans out -- Tempo or Jaeger for infrastructure, Langfuse for LLM spans, Phoenix
in a developer's profile for retrieval tuning. A BYOC customer drops Langfuse and loses nothing
structural, because nothing in ``app/`` ever imported it.

**Content capture is off by default, and that is a product decision rather than a setting.** It
is what lets "does the vendor store our documents in a third-party observability tool" be
answered with "no". The attributes recorded without it -- counts, latencies, fingerprints, leg
names -- are enough to diagnose every performance and relevance problem; the text is only needed
when someone is debugging a specific answer, and then it should be a deliberate, time-boxed
choice.

**Every stage is a span with the same attribute vocabulary.** ``erp.stage``, ``erp.tenant_id``,
``erp.candidates``, ``erp.latency_ms``. A consistent vocabulary is what makes "p95 of the rerank
stage, grouped by tenant" a query rather than a project.

The SDK is optional. If it is not installed -- a BYOC deployment that does not want it, or CI --
every function here degrades to a no-op that costs a dictionary lookup, and nothing else changes.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

#: The stages a request passes through. Fixed, because an ad-hoc stage name makes every
#: cross-request comparison silently incomplete.
STAGES = (
    "understand",
    "retrieve.bm25",
    "retrieve.exact",
    "retrieve.dense",
    "retrieve.parent",
    "fuse",
    "rerank",
    "expand",
    "assemble",
    "generate",
    "verify",
)

#: Attributes safe to record whatever the tenant's content setting. Everything else is gated.
SAFE_ATTRIBUTES = frozenset(
    {
        "erp.stage",
        "erp.tenant_id",
        "erp.user_role",
        "erp.candidates",
        "erp.latency_ms",
        "erp.generation_fingerprint",
        "erp.answerable",
        "erp.abstention_reason",
        "erp.fast_path",
        "erp.leg",
        "erp.rerank_status",
        "erp.input_tokens",
        "erp.output_tokens",
        "erp.cached_input_tokens",
        "erp.cost_usd",
        "erp.cited_count",
        "erp.error_class",
    }
)

#: Attributes that carry, or could carry, what a customer wrote.
CONTENT_ATTRIBUTES = frozenset({"erp.question", "erp.answer", "erp.chunk_text", "erp.claim", "erp.document_title"})


@dataclass(slots=True)
class TracingConfig:
    enabled: bool = False
    endpoint: str | None = None
    service_name: str = "enterprise-rag-platform"
    #: Off by default. See the module docstring -- this is the answer to a procurement question,
    #: not a debugging convenience.
    capture_content: bool = False
    #: Head sampling. 1.0 in development, lower under load; the error path is always sampled.
    sample_ratio: float = 1.0


_config = TracingConfig()
_tracer: Any = None


def configure(settings: Any) -> None:
    """Wire the SDK once at startup. A no-op when tracing is off or the SDK is absent."""
    global _tracer

    _config.enabled = bool(getattr(settings, "otel_enabled", False))
    _config.endpoint = getattr(settings, "otel_endpoint", None)
    _config.capture_content = bool(getattr(settings, "trace_content_capture", False))
    _config.sample_ratio = float(getattr(settings, "otel_sample_ratio", 1.0))

    if not _config.enabled:
        return

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
    except ImportError:
        # A deployment that does not want the SDK is a supported configuration, not an error.
        logger.info("otel.sdk_absent", extra={"detail": "tracing requested but the SDK is not installed"})
        _config.enabled = False
        return

    provider = TracerProvider(
        resource=Resource.create({"service.name": _config.service_name}),
        sampler=ParentBased(TraceIdRatioBased(_config.sample_ratio)),
    )
    if _config.endpoint:
        provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter(endpoint=_config.endpoint)))
    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(_config.service_name)
    logger.info("otel.configured", extra={"endpoint": _config.endpoint, "content": _config.capture_content})


def capture_content_enabled() -> bool:
    return _config.capture_content


@dataclass(slots=True)
class StageTiming:
    """Per-stage milliseconds, accumulated across one request.

    Recorded regardless of whether the OTel SDK is present, because it feeds ``answer_traces`` --
    the product's own customer-facing record, which must not depend on an ops tool being
    installed.
    """

    timings: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    def record(self, stage: str, elapsed_ms: float, *, count: int | None = None) -> None:
        # Added, not replaced: the four retrieval legs run under one logical stage, and a leg
        # retried after a timeout should show its total cost rather than only the last attempt.
        self.timings[stage] = self.timings.get(stage, 0.0) + elapsed_ms
        if count is not None:
            self.counts[stage] = count

    @property
    def total_ms(self) -> float:
        return sum(self.timings.values())

    def slowest(self) -> tuple[str, float] | None:
        return max(self.timings.items(), key=lambda item: item[1]) if self.timings else None


@contextmanager
def stage(
    name: str,
    *,
    timing: StageTiming | None = None,
    attributes: Mapping[str, Any] | None = None,
) -> Iterator[dict[str, Any]]:
    """Time one stage, and open a span for it when tracing is on.

    Yields a mutable dict the caller adds attributes to as it learns them -- a leg does not know
    how many candidates it found until it has finished. Attributes are filtered on the way out,
    so a caller cannot accidentally put a chunk's text into a span on a tenant that has content
    capture off.
    """
    started = time.perf_counter()
    collected: dict[str, Any] = dict(attributes or {})

    span_cm = _tracer.start_as_current_span(f"erp.{name}") if _tracer is not None else None
    span = span_cm.__enter__() if span_cm is not None else None

    try:
        yield collected
    except Exception as exc:
        collected["erp.error_class"] = type(exc).__name__
        if span is not None:
            span.record_exception(exc)
        raise
    finally:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        collected.setdefault("erp.stage", name)
        collected["erp.latency_ms"] = round(elapsed_ms, 2)

        if timing is not None:
            timing.record(name, elapsed_ms, count=collected.get("erp.candidates"))

        if span is not None:
            for key, value in filter_attributes(collected).items():
                span.set_attribute(key, value)
        if span_cm is not None:
            span_cm.__exit__(None, None, None)


def filter_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    """Drop content attributes unless the tenant opted in, and anything not on the vocabulary.

    Deny by default in both directions. An unknown attribute is dropped rather than passed
    through, because the way customer data reaches an observability vendor is almost never a
    deliberate decision -- it is someone adding ``span.set_attribute("doc", document)`` while
    debugging and not removing it.
    """
    allowed: dict[str, Any] = {}
    for key, value in attributes.items():
        if key in SAFE_ATTRIBUTES or (key in CONTENT_ATTRIBUTES and _config.capture_content):
            allowed[key] = value
        elif key not in CONTENT_ATTRIBUTES:
            logger.debug("otel.attribute_dropped", extra={"key": key})
    return allowed


def new_request_id() -> str:
    """Correlates a log line, an audit row, an answer trace and a span."""
    return uuid.uuid4().hex[:16]


def current_trace_id() -> str | None:
    """The active OTel trace id, so ``answer_traces`` can point at the span.

    Stored on our own row rather than relying on the ops tool for correlation: an engineer should
    be able to move from a customer's complaint to a span, and that path must not break when the
    trace has aged out of the backend's retention.
    """
    if _tracer is None:
        return None
    try:
        from opentelemetry import trace

        context = trace.get_current_span().get_span_context()
        return format(context.trace_id, "032x") if context.is_valid else None
    except Exception:
        return None


def summarise(timing: StageTiming) -> dict[str, Any]:
    """What goes onto the answer trace and into the response's diagnostics."""
    slowest = timing.slowest()
    return {
        "total_ms": round(timing.total_ms, 2),
        "stages": {name: round(value, 2) for name, value in timing.timings.items()},
        "counts": dict(timing.counts),
        "slowest_stage": slowest[0] if slowest else None,
    }
