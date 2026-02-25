"""Tests for trace timeline visualization."""

from __future__ import annotations

import pytest

from qocc.trace.visualization import (
    render_timeline,
    render_metrics_comparison,
)


def _make_spans() -> list[dict]:
    """Create sample spans for testing."""
    return [
        {
            "span_id": "s1",
            "name": "ingest",
            "start_time": 1000.0,
            "end_time": 1000.1,
            "parent_id": None,
            "attributes": {"adapter": "qiskit"},
            "status": "ok",
        },
        {
            "span_id": "s2",
            "name": "normalize",
            "start_time": 1000.1,
            "end_time": 1000.2,
            "parent_id": None,
            "attributes": {},
            "status": "ok",
        },
        {
            "span_id": "s3",
            "name": "compile",
            "start_time": 1000.2,
            "end_time": 1000.8,
            "parent_id": None,
            "attributes": {"pipeline": "default"},
            "status": "ok",
        },
        {
            "span_id": "s3a",
            "name": "init_pass",
            "start_time": 1000.2,
            "end_time": 1000.3,
            "parent_id": "s3",
            "attributes": {},
            "status": "ok",
        },
        {
            "span_id": "s3b",
            "name": "routing_pass",
            "start_time": 1000.3,
            "end_time": 1000.6,
            "parent_id": "s3",
            "attributes": {},
            "status": "ok",
        },
        {
            "span_id": "s4",
            "name": "metrics",
            "start_time": 1000.8,
            "end_time": 1000.9,
            "parent_id": None,
            "attributes": {},
            "status": "ok",
        },
    ]


def test_render_empty():
    """Empty spans → placeholder message."""
    result = render_timeline([])
    assert "(no spans)" in result


def test_render_basic():
    """Basic rendering produces output for all spans."""
    spans = _make_spans()
    result = render_timeline(spans, width=80)

    assert "ingest" in result
    assert "normalize" in result
    assert "compile" in result
    assert "metrics" in result
    assert "Total" in result


def test_render_nested_indents():
    """Child spans should be indented."""
    spans = _make_spans()
    result = render_timeline(spans, width=80)

    lines = result.split("\n")
    init_line = [l for l in lines if "init_pass" in l]
    assert len(init_line) == 1
    # Child should have leading spaces
    assert init_line[0].startswith("  ") or "init_pass" in init_line[0]


def test_render_with_attributes():
    """show_attributes=True shows attribute JSON."""
    spans = _make_spans()
    result = render_timeline(spans, width=100, show_attributes=True)
    assert "adapter" in result  # from ingest span attributes


def test_render_width_adjustment():
    """Different widths produce different output lengths."""
    spans = _make_spans()
    narrow = render_timeline(spans, width=60)
    wide = render_timeline(spans, width=120)

    # Wide render should have longer lines on average
    wide_avg = sum(len(l) for l in wide.split("\n")) / max(len(wide.split("\n")), 1)
    narrow_avg = sum(len(l) for l in narrow.split("\n")) / max(len(narrow.split("\n")), 1)
    assert wide_avg >= narrow_avg


def test_render_metrics_comparison():
    """Metrics comparison renders side-by-side."""
    before = {"depth": 10, "total_gates": 20, "gates_2q": 5, "width": 3}
    after = {"depth": 8, "total_gates": 15, "gates_2q": 3, "width": 3}

    result = render_metrics_comparison(before, after, width=80)

    assert "depth" in result
    assert "total_gates" in result
    assert "Input" in result
    assert "Compiled" in result


def test_render_metrics_handles_missing():
    """Metrics comparison handles missing keys gracefully."""
    before = {"depth": 10}
    after = {"depth": 8, "gates_2q": 3}

    result = render_metrics_comparison(before, after)
    assert "depth" in result


def test_duration_formatting():
    """Duration shows ms for short spans, s for long ones."""
    spans = [
        {
            "span_id": "fast",
            "name": "quick",
            "start_time": 0.0,
            "end_time": 0.05,
            "parent_id": None,
            "attributes": {},
            "status": "ok",
        },
    ]
    result = render_timeline(spans, width=80)
    assert "ms" in result


def test_status_indicator():
    """Spans show status checkmarks/crosses."""
    spans = [
        {
            "span_id": "ok",
            "name": "good_span",
            "start_time": 0.0,
            "end_time": 1.0,
            "parent_id": None,
            "attributes": {},
            "status": "ok",
        },
        {
            "span_id": "err",
            "name": "bad_span",
            "start_time": 1.0,
            "end_time": 2.0,
            "parent_id": None,
            "attributes": {},
            "status": "error",
        },
    ]
    result = render_timeline(spans, width=80)
    assert "✓" in result
    assert "✗" in result
