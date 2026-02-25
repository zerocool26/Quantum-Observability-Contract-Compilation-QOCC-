"""Span data model for QOCC tracing.

Inspired by OpenTelemetry's span model but simplified for quantum
workflow instrumentation.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class SpanEvent:
    """An event attached to a span (e.g. warning, exception)."""

    name: str
    timestamp: str = field(default_factory=_now)
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "timestamp": self.timestamp,
            "attributes": self.attributes,
        }


@dataclass
class SpanLink:
    """A link to another span (e.g. across candidate pipelines)."""

    trace_id: str
    span_id: str
    attributes: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "attributes": self.attributes,
        }


@dataclass
class Span:
    """A single trace span.

    Attributes:
        trace_id: Unique identifier for the overall trace.
        span_id: Unique identifier for this span.
        parent_span_id: ID of the parent span (None for root).
        name: Human-readable span name (e.g. ``"compile/qiskit"``).
        start_time: ISO-8601 start timestamp.
        end_time: ISO-8601 end timestamp (set when span ends).
        attributes: Key-value metadata.
        events: Recorded events during this span.
        links: Links to other spans.
        status: ``"OK"``, ``"ERROR"``, or ``"UNSET"``.
    """

    trace_id: str
    name: str
    span_id: str = field(default_factory=_new_id)
    parent_span_id: str | None = None
    start_time: str = field(default_factory=_now)
    end_time: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    links: list[SpanLink] = field(default_factory=list)
    status: str = "UNSET"

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def end(self, status: str = "OK") -> None:
        """Mark the span as ended."""
        self.end_time = _now()
        self.status = status

    def add_event(self, name: str, **attrs: Any) -> None:
        self.events.append(SpanEvent(name=name, attributes=attrs))

    def add_link(self, trace_id: str, span_id: str, **attrs: Any) -> None:
        self.links.append(SpanLink(trace_id=trace_id, span_id=span_id, attributes=attrs))

    def set_attribute(self, key: str, value: Any) -> None:
        self.attributes[key] = value

    def record_exception(self, exc: BaseException) -> None:
        self.add_event(
            "exception",
            type=type(exc).__name__,
            message=str(exc),
        )
        self.status = "ERROR"

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "attributes": self.attributes,
            "events": [e.to_dict() for e in self.events],
            "links": [l.to_dict() for l in self.links],
            "status": self.status,
        }
