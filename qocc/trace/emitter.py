"""Trace emitter — creates and manages spans for a single trace.

Usage::

    emitter = TraceEmitter()
    with emitter.span("compile") as s:
        s.set_attribute("adapter", "qiskit")
        ...
    spans = emitter.finished_spans()
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from typing import Any, Generator

from qocc.trace.span import Span


class TraceEmitter:
    """Creates spans and collects them for export.

    Attributes:
        trace_id: The trace-level identifier (shared by all spans in a run).
    """

    def __init__(self, trace_id: str | None = None) -> None:
        self.trace_id: str = trace_id or uuid.uuid4().hex
        self._spans: list[Span] = []
        self._active_span: Span | None = None

    # ------------------------------------------------------------------
    # Span creation
    # ------------------------------------------------------------------

    def start_span(
        self,
        name: str,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Start a new span, optionally as a child of *parent*."""
        parent_id = parent.span_id if parent else (
            self._active_span.span_id if self._active_span else None
        )
        s = Span(
            trace_id=self.trace_id,
            name=name,
            parent_span_id=parent_id,
            attributes=attributes or {},
        )
        return s

    def finish_span(self, span: Span, status: str = "OK") -> None:
        """End *span* and add it to the collector."""
        if span.end_time is None:
            span.end(status=status)
        self._spans.append(span)

    @contextmanager
    def span(
        self,
        name: str,
        parent: Span | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """Context-manager that starts and finishes a span automatically."""
        s = self.start_span(name, parent=parent, attributes=attributes)
        prev = self._active_span
        self._active_span = s
        try:
            yield s
            self.finish_span(s, status="OK")
        except Exception as exc:
            s.record_exception(exc)
            self.finish_span(s, status="ERROR")
            raise
        finally:
            self._active_span = prev

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def finished_spans(self) -> list[Span]:
        """Return all finished spans in order."""
        return list(self._spans)

    def to_dicts(self) -> list[dict[str, Any]]:
        """Serialised spans ready for ``trace.jsonl``."""
        return [s.to_dict() for s in self._spans]
