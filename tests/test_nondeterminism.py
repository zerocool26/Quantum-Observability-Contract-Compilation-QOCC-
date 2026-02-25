"""Tests for the nondeterminism detection module."""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from qocc.core.nondeterminism import (
    NondeterminismReport,
    detect_nondeterminism,
    compare_run_hashes,
)


class FakeMetrics:
    def to_dict(self):
        return {"depth": 5, "total_gates": 10}


class FakeCompileResult:
    def __init__(self, h: str):
        self.circuit = MagicMock()
        self.circuit.stable_hash.return_value = h


class FakeAdapter:
    def __init__(self, hashes: list[str]):
        self._hashes = iter(hashes)

    def compile(self, circuit, pipeline):
        h = next(self._hashes)
        return FakeCompileResult(h)

    def get_metrics(self, circuit):
        return FakeMetrics()


def test_reproducible_runs():
    """All runs produce same hash → reproducible."""
    adapter = FakeAdapter(["abc123"] * 5)
    circuit = MagicMock()
    pipeline = MagicMock()

    report = detect_nondeterminism(adapter, circuit, pipeline, num_runs=5)

    assert report.reproducible is True
    assert report.unique_hashes == 1
    assert report.num_runs == 5
    assert report.confidence > 0.8


def test_nondeterministic_runs():
    """Different hashes → not reproducible."""
    adapter = FakeAdapter(["a", "b", "a", "c", "a"])
    circuit = MagicMock()
    pipeline = MagicMock()

    report = detect_nondeterminism(adapter, circuit, pipeline, num_runs=5)

    assert report.reproducible is False
    assert report.unique_hashes == 3
    assert report.hash_counts == {"a": 3, "b": 1, "c": 1}
    assert report.confidence < 1.0


def test_nondeterminism_report_serialization():
    """Report can be serialized to dict."""
    report = NondeterminismReport(
        reproducible=True,
        num_runs=3,
        unique_hashes=1,
        hash_counts={"abc": 3},
        confidence=0.75,
        details={"timings_ms": [10, 12, 11]},
    )
    d = report.to_dict()
    assert d["reproducible"] is True
    assert d["num_runs"] == 3
    assert d["confidence"] == 0.75


def test_compare_run_hashes_identical():
    """Cross-comparison of identical single-hash sets."""
    result = compare_run_hashes(["abc"], ["abc"])
    assert result["identical"] is True
    assert result["overlap_fraction"] == 1.0


def test_compare_run_hashes_disjoint():
    """Cross-comparison with no overlap."""
    result = compare_run_hashes(["a", "b"], ["c", "d"])
    assert result["identical"] is False
    assert result["overlap_fraction"] == 0.0
    assert set(result["only_a"]) == {"a", "b"}
    assert set(result["only_b"]) == {"c", "d"}


def test_compare_run_hashes_partial():
    """Cross-comparison with partial overlap."""
    result = compare_run_hashes(["a", "b", "c"], ["b", "c", "d"])
    assert result["identical"] is False
    assert set(result["common"]) == {"b", "c"}
    assert len(result["only_a"]) == 1
    assert len(result["only_b"]) == 1


def test_two_runs_detection():
    """Even with only 2 runs, nondeterminism can be detected."""
    adapter = FakeAdapter(["x", "y"])
    report = detect_nondeterminism(adapter, MagicMock(), MagicMock(), num_runs=2)
    assert report.reproducible is False
    assert report.unique_hashes == 2
