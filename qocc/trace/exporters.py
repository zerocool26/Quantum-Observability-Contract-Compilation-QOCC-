"""Trace exporters — convert collected spans to various formats.

Supports:
  - JSON Lines (native)
  - OpenTelemetry-compatible OTLP JSON (for Jaeger / Grafana / Datadog)
  - OpenTelemetry SDK bridge (when ``opentelemetry-sdk`` is installed)
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from qocc.trace.span import Span


def export_jsonl(spans: list[Span], path: str | Path) -> Path:
    """Write spans as JSON Lines."""
    p = Path(path)
    with p.open("w", encoding="utf-8") as f:
        for s in spans:
            f.write(json.dumps(s.to_dict(), default=str) + "\n")
    return p


def spans_to_dicts(spans: list[Span]) -> list[dict[str, Any]]:
    """Convert spans to list of dicts."""
    return [s.to_dict() for s in spans]


# ======================================================================
# OpenTelemetry OTLP JSON export
# ======================================================================


def _iso_to_unix_nano(iso: str | None) -> int:
    """Convert ISO-8601 timestamp to Unix nanoseconds."""
    if iso is None:
        return 0
    try:
        dt = datetime.fromisoformat(iso)
        return int(dt.timestamp() * 1_000_000_000)
    except (ValueError, TypeError):
        return 0


def _span_to_otlp(span: Span, service_name: str = "qocc") -> dict[str, Any]:
    """Convert a single QOCC Span to OTLP JSON format."""
    # Convert attributes to OTLP key-value pairs
    otlp_attrs: list[dict[str, Any]] = []
    for k, v in span.attributes.items():
        otlp_attrs.append(_attr_to_otlp(k, v))

    # Convert events to OTLP events
    otlp_events: list[dict[str, Any]] = []
    for ev in span.events:
        otlp_events.append({
            "timeUnixNano": _iso_to_unix_nano(ev.timestamp),
            "name": ev.name,
            "attributes": [_attr_to_otlp(k, v) for k, v in ev.attributes.items()],
        })

    # Convert links
    otlp_links: list[dict[str, Any]] = []
    for lnk in span.links:
        otlp_links.append({
            "traceId": lnk.trace_id,
            "spanId": lnk.span_id,
            "attributes": [_attr_to_otlp(k, v) for k, v in lnk.attributes.items()],
        })

    # Map status
    status_code = 1  # OK
    if span.status == "ERROR":
        status_code = 2
    elif span.status == "UNSET":
        status_code = 0

    return {
        "traceId": span.trace_id,
        "spanId": span.span_id,
        "parentSpanId": span.parent_span_id or "",
        "name": span.name,
        "kind": 1,  # SPAN_KIND_INTERNAL
        "startTimeUnixNano": _iso_to_unix_nano(span.start_time),
        "endTimeUnixNano": _iso_to_unix_nano(span.end_time),
        "attributes": otlp_attrs,
        "events": otlp_events,
        "links": otlp_links,
        "status": {"code": status_code, "message": span.status},
    }


def _attr_to_otlp(key: str, value: Any) -> dict[str, Any]:
    """Convert a key-value pair to OTLP attribute format."""
    if isinstance(value, bool):
        return {"key": key, "value": {"boolValue": value}}
    elif isinstance(value, int):
        return {"key": key, "value": {"intValue": str(value)}}
    elif isinstance(value, float):
        return {"key": key, "value": {"doubleValue": value}}
    elif isinstance(value, (list, tuple)):
        return {"key": key, "value": {"stringValue": json.dumps(value, default=str)}}
    elif isinstance(value, dict):
        return {"key": key, "value": {"stringValue": json.dumps(value, default=str)}}
    else:
        return {"key": key, "value": {"stringValue": str(value)}}


def export_otlp_json(
    spans: list[Span],
    path: str | Path,
    service_name: str = "qocc",
    service_version: str = "0.1.0",
) -> Path:
    """Export spans as OTLP-compatible JSON for ingestion by collectors.

    The output follows the OpenTelemetry Protocol JSON encoding:
    https://opentelemetry.io/docs/specs/otlp/#json-protobuf-encoding

    Parameters:
        spans: List of QOCC Span objects.
        path: Output file path.
        service_name: Service name for the resource.
        service_version: Service version for the resource.

    Returns:
        Path to the written file.
    """
    p = Path(path)

    # Group spans by trace_id for proper OTLP structure
    traces: dict[str, list[Span]] = {}
    for s in spans:
        traces.setdefault(s.trace_id, []).append(s)

    scope_spans = []
    for trace_id, trace_spans in traces.items():
        otlp_spans = [_span_to_otlp(s, service_name) for s in trace_spans]
        scope_spans.append({
            "scope": {
                "name": "qocc.trace",
                "version": service_version,
            },
            "spans": otlp_spans,
        })

    otlp_payload = {
        "resourceSpans": [
            {
                "resource": {
                    "attributes": [
                        {"key": "service.name", "value": {"stringValue": service_name}},
                        {"key": "service.version", "value": {"stringValue": service_version}},
                        {"key": "telemetry.sdk.name", "value": {"stringValue": "qocc"}},
                        {"key": "telemetry.sdk.language", "value": {"stringValue": "python"}},
                    ],
                },
                "scopeSpans": scope_spans,
            }
        ]
    }

    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(otlp_payload, indent=2, default=str) + "\n", encoding="utf-8")
    return p


def export_to_otel_sdk(spans: list[Span]) -> bool:
    """Bridge QOCC spans into the OpenTelemetry Python SDK.

    Requires ``opentelemetry-sdk`` to be installed.  Creates real
    OTel spans that can be exported by any configured OTel exporter
    (OTLP/gRPC, Jaeger, Zipkin, etc.).

    Returns:
        True if export succeeded, False if OTel SDK is not available.
    """
    try:
        from opentelemetry import trace as otel_trace  # type: ignore
        from opentelemetry.trace import StatusCode  # type: ignore
    except ImportError:
        return False

    tracer = otel_trace.get_tracer("qocc.trace", "0.1.0")

    # Build a map of QOCC spans by span_id for parent resolution
    id_map: dict[str, Any] = {}

    # Process in order, creating OTel spans
    for s in spans:
        ctx = None
        if s.parent_span_id and s.parent_span_id in id_map:
            ctx = otel_trace.set_span_in_context(id_map[s.parent_span_id])

        otel_span = tracer.start_span(
            name=s.name,
            context=ctx,
            attributes={k: _otel_safe_value(v) for k, v in s.attributes.items()},
        )

        # Add events
        for ev in s.events:
            otel_span.add_event(
                ev.name,
                attributes={k: _otel_safe_value(v) for k, v in ev.attributes.items()},
            )

        # Set status
        if s.status == "ERROR":
            otel_span.set_status(StatusCode.ERROR)
        elif s.status == "OK":
            otel_span.set_status(StatusCode.OK)

        otel_span.end()
        id_map[s.span_id] = otel_span

    return True


def _otel_safe_value(v: Any) -> str | int | float | bool:
    """Coerce a value to an OTel-safe attribute type."""
    if isinstance(v, (str, int, float, bool)):
        return v
    return str(v)
