"""Tests for metrics computation."""

from __future__ import annotations

import pytest

from qocc.adapters.base import MetricsSnapshot
from qocc.core.circuit_handle import CircuitHandle
from qocc.metrics.compute import _duration_estimate, _proxy_error


def test_duration_estimate_with_model():
    counts = {"h": 3, "cx": 2, "x": 1}
    model = {"h": 25.0, "cx": 300.0, "x": 25.0, "default": 50.0}
    result = _duration_estimate(counts, model)
    expected = 3 * 25.0 + 2 * 300.0 + 1 * 25.0
    assert result == expected


def test_duration_estimate_none_without_model():
    assert _duration_estimate({"h": 3}, None) is None


def test_proxy_error_with_model():
    counts = {"h": 3, "cx": 2}
    model = {"h": 0.001, "cx": 0.01, "default": 0.001}
    depth = 5
    result = _proxy_error(counts, depth, model, decoherence_weight=0.001)
    expected = 3 * 0.001 + 2 * 0.01 + 5 * 0.001
    assert abs(result - expected) < 1e-10


def test_proxy_error_none_without_model():
    assert _proxy_error({"h": 3}, 5, None) is None


def test_metrics_snapshot_immutable_access():
    data = {"width": 3, "depth": 5, "total_gates": 10}
    snap = MetricsSnapshot(data)
    assert snap["width"] == 3
    assert snap.get("depth") == 5
    assert snap.get("missing", -1) == -1
    d = snap.to_dict()
    assert d["total_gates"] == 10


class TestMetricsFromQASM:
    """Test metrics from a generic (non-native) circuit handle."""

    def test_generic_fallback(self):
        from qocc.metrics.compute import compute_metrics

        handle = CircuitHandle(
            name="test",
            num_qubits=4,
            native_circuit=None,
            source_format="unknown",
        )
        m = compute_metrics(handle)
        assert m["width"] == 4
        assert m["total_gates"] is None
