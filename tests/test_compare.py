"""Tests for bundle comparison."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from qocc.core.artifacts import ArtifactStore
from qocc.api import compare_bundles


def _make_bundle(tmpdir: str, run_id: str, width: int, depth: int) -> Path:
    """Helper to create a minimal bundle."""
    bdir = Path(tmpdir) / run_id
    store = ArtifactStore(bdir)
    store.write_manifest(run_id)
    store.write_env()
    store.write_seeds({"global_seed": 42, "rng_algorithm": "MT19937", "stage_seeds": {}})
    store.write_metrics({
        "input": {"width": width, "depth": depth, "total_gates": width * depth},
        "compiled": {"width": width, "depth": depth - 1, "total_gates": width * (depth - 1)},
    })
    store.write_contracts([])
    store.write_contract_results([])
    store.write_trace([
        {
            "trace_id": "t1", "span_id": "s1", "name": "test",
            "start_time": "2025-01-01T00:00:00+00:00",
            "parent_span_id": None,
            "attributes": {}, "events": [], "links": [], "status": "OK",
        },
    ])

    zip_path = bdir.with_suffix(".zip")
    store.export_zip(zip_path)
    return zip_path


def test_compare_identical_bundles():
    """Comparing a bundle to itself should report no differences."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle = _make_bundle(tmpdir, "same", width=3, depth=5)
        report = compare_bundles(str(bundle), str(bundle))
        metrics_diff = report["diffs"]["metrics"]
        assert metrics_diff == {}


def test_compare_different_bundles():
    """Different bundles should produce diffs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        a = _make_bundle(tmpdir, "bundle_a", width=3, depth=5)
        b = _make_bundle(tmpdir, "bundle_b", width=3, depth=10)
        report = compare_bundles(str(a), str(b))
        metrics_diff = report["diffs"]["metrics"]
        assert len(metrics_diff) > 0


def test_compare_produces_markdown():
    """Compare should generate a markdown report."""
    with tempfile.TemporaryDirectory() as tmpdir:
        a = _make_bundle(tmpdir, "md_a", width=2, depth=3)
        b = _make_bundle(tmpdir, "md_b", width=2, depth=6)
        report_dir = Path(tmpdir) / "reports"
        report = compare_bundles(str(a), str(b), report_dir=str(report_dir))
        assert "markdown" in report
        assert (report_dir / "comparison.md").exists()
        assert (report_dir / "comparison.json").exists()
