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
        # Sort trace_spans by start_time so parents come before children
        trace_spans = sorted(trace_spans, key=lambda s: s.start_time or "")
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

    # Sort spans by start time so parents are processed before children
    sorted_spans = sorted(spans, key=lambda s: s.start_time or "")

    # Process in order, creating OTel spans
    for s in sorted_spans:
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


def export_otlp_grpc(
    spans: list[Span],
    endpoint: str,
    headers: dict[str, str] | None = None,
    service_name: str = "qocc-grpc",
) -> bool:
    """Stream spans to an OTLP-compatible collector via gRPC.

    Requires ``opentelemetry-exporter-otlp-proto-grpc`` to be installed.
    """
    try:
        from opentelemetry import trace as otel_trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.trace import StatusCode
    except ImportError:
        return False

    resource = Resource.create(
        {
            "service.name": service_name,
            "telemetry.sdk.name": "qocc",
            "telemetry.sdk.language": "python",
        }
    )

    provider = TracerProvider(resource=resource)
    
    exporter_kwargs: dict[str, Any] = {"endpoint": endpoint, "insecure": endpoint.startswith("http://") or ":80" in endpoint or ":4317" in endpoint}
    if headers:
        exporter_kwargs["headers"] = headers
        
    exporter = OTLPSpanExporter(**exporter_kwargs)
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)

    tracer = provider.get_tracer("qocc.trace", "0.1.0")
    id_map: dict[str, Any] = {}

    sorted_spans = sorted(spans, key=lambda s: s.start_time or "")

    for s in sorted_spans:
        ctx = None
        if s.parent_span_id and s.parent_span_id in id_map:
            ctx = otel_trace.set_span_in_context(id_map[s.parent_span_id])

        # Semantic conventions for QOCC
        attrs = {k: _otel_safe_value(v) for k, v in s.attributes.items()}
        # Ensure we bubble up standard attributes if present in the top-level span but maybe missing from conventions
        for k in ["adapter", "circuit_hash", "n_qubits"]:
            if k in s.attributes and f"quantum.{k}" not in attrs:
                attrs[f"quantum.{k}"] = _otel_safe_value(s.attributes[k])

        otel_span = tracer.start_span(
            name=s.name,
            context=ctx,
            attributes=attrs,
            start_time=_iso_to_unix_nano(s.start_time)
        )

        for ev in s.events:
            otel_span.add_event(
                ev.name,
                attributes={k: _otel_safe_value(v) for k, v in ev.attributes.items()},
                timestamp=_iso_to_unix_nano(ev.timestamp)
            )

        if s.status == "ERROR":
            otel_span.set_status(StatusCode.ERROR)
        elif s.status == "OK":
            otel_span.set_status(StatusCode.OK)

        otel_span.end(end_time=_iso_to_unix_nano(s.end_time))
        id_map[s.span_id] = otel_span

    provider.force_flush()
    return True


class OTLPLiveExporter:
    """Real-time OpenTelemetry Exporter using gRPC."""

    def __init__(self, endpoint: str, headers: dict[str, str] | None = None, service_name: str = "qocc-grpc"):
        from opentelemetry import trace as otel_trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor

        resource = Resource.create({
            "service.name": service_name,
            "telemetry.sdk.name": "qocc",
            "telemetry.sdk.language": "python",
        })
        self.provider = TracerProvider(resource=resource)
        
        exporter_kwargs: dict[str, Any] = {"endpoint": endpoint, "insecure": endpoint.startswith("http://") or ":80" in endpoint or ":4317" in endpoint}
        if headers:
            exporter_kwargs["headers"] = headers
            
        exporter = OTLPSpanExporter(**exporter_kwargs)
        self.provider.add_span_processor(SimpleSpanProcessor(exporter))
        self.tracer = self.provider.get_tracer("qocc.trace", "0.1.0")
        
        self.otel_trace = otel_trace
        self.id_map: dict[str, Any] = {}

    def on_span_started(self, span: Span) -> None:
        ctx = None
        if span.parent_span_id and span.parent_span_id in self.id_map:
            ctx = self.otel_trace.set_span_in_context(self.id_map[span.parent_span_id])
            
        attrs = {k: _otel_safe_value(v) for k, v in span.attributes.items()}
        for k in ["adapter", "circuit_hash", "n_qubits"]:
            if k in span.attributes and f"quantum.{k}" not in attrs:
                attrs[f"quantum.{k}"] = _otel_safe_value(span.attributes[k])
                
        otel_span = self.tracer.start_span(
            name=span.name,
            context=ctx,
            attributes=attrs,
            start_time=_iso_to_unix_nano(span.start_time)
        )
        self.id_map[span.span_id] = otel_span

    def on_span_finished(self, span: Span) -> None:
        from opentelemetry.trace import StatusCode
        
        otel_span = self.id_map.get(span.span_id)
        if not otel_span:
            return
            
        # Update attributes which might have been added after start
        for k, v in span.attributes.items():
            otel_span.set_attribute(k, _otel_safe_value(v))
            
        for k in ["adapter", "circuit_hash", "n_qubits"]:
            if k in span.attributes:
                otel_span.set_attribute(f"quantum.{k}", _otel_safe_value(span.attributes[k]))

        for ev in span.events:
            otel_span.add_event(
                ev.name,
                attributes={k: _otel_safe_value(v) for k, v in ev.attributes.items()},
                timestamp=_iso_to_unix_nano(ev.timestamp)
            )

        if span.status == "ERROR":
            otel_span.set_status(StatusCode.ERROR)
        elif span.status == "OK":
            otel_span.set_status(StatusCode.OK)

        otel_span.end(end_time=_iso_to_unix_nano(span.end_time))
        # Remove from id_map to avoid memory leak? Wait, child spans might still need it to set context.
        # Don't pop it immediately.

    def flush(self) -> None:
        self.provider.force_flush()


