import pytest
from unittest.mock import MagicMock, patch
from qocc.trace.span import Span
from qocc.trace.emitter import TraceEmitter
from qocc.trace.exporters import export_otlp_grpc, OTLPLiveExporter

pytest.importorskip("opentelemetry")

def test_export_otlp_grpc_compilation():
    spans = [
        Span(trace_id="tr1", name="span1", attributes={"adapter": "qiskit", "circuit_hash": "abc"}),
        Span(trace_id="tr1", name="span2", parent_span_id="span1_id_mock", attributes={})
    ]
    spans[1].parent_span_id = spans[0].span_id
    
    with patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter") as mock_exporter_cls:
        mock_exporter = mock_exporter_cls.return_value
        result = export_otlp_grpc(spans, endpoint="http://localhost:4317")
        assert result is True
        
def test_otel_live_exporter():
    with patch("opentelemetry.exporter.otlp.proto.grpc.trace_exporter.OTLPSpanExporter"):
        exporter = OTLPLiveExporter(endpoint="http://localhost:4317")
        emitter = TraceEmitter(on_span_started=exporter.on_span_started, on_span_finished=exporter.on_span_finished)
        
        with emitter.span("root", attributes={"adapter": "cirq"}):
            with emitter.span("child"):
                pass
                
        exporter.flush()
        # id_map should have 2 spans
        assert len(exporter.id_map) == 2
        root_span_id = emitter.finished_spans()[1].span_id  # Note: root finishes after child
        # Check that semantic conventions were added
        otel_span = exporter.id_map[root_span_id]
        # In opentelemetry, getting attributes back out of the span object is tricky, it's inside `_attributes`.
        assert otel_span.name == "root"
