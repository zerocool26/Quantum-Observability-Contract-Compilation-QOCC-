"""Tests for Trace Bundle roundtrip — create, zip, load, validate."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from qocc.core.artifacts import ArtifactStore
from qocc.core.schemas import validate_bundle


def test_bundle_roundtrip_creates_all_required_files():
    """A bundle written by ArtifactStore should contain all required files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArtifactStore(tmpdir)

        store.write_manifest("test-run-001")
        store.write_env()
        store.write_seeds({"global_seed": 42, "rng_algorithm": "MT19937", "stage_seeds": {}})
        store.write_metrics({
            "width": 3, "total_gates": 10, "gates_1q": 6, "gates_2q": 4,
            "depth": 5, "depth_2q": 3, "gate_histogram": {"h": 3, "cx": 4, "x": 3},
            "topology_violations": None, "duration_estimate": None, "proxy_error_score": None,
        })
        store.write_contracts([
            {"name": "dist_check", "type": "distribution", "tolerances": {"tvd": 0.1}},
        ])
        store.write_contract_results([
            {"name": "dist_check", "passed": True, "details": {}},
        ])
        store.write_trace([
            {
                "trace_id": "abc123",
                "span_id": "span001",
                "parent_span_id": None,
                "name": "compile",
                "start_time": "2025-01-01T00:00:00+00:00",
                "end_time": "2025-01-01T00:00:01+00:00",
                "attributes": {},
                "events": [],
                "links": [],
                "status": "OK",
            },
        ])
        store.write_summary_report("# Test Summary\n")

        # Verify files exist
        root = Path(tmpdir)
        required = [
            "manifest.json", "env.json", "seeds.json", "metrics.json",
            "contracts.json", "contract_results.json", "trace.jsonl",
            "reports/summary.md",
        ]
        for fname in required:
            assert (root / fname).exists(), f"Missing: {fname}"


def test_bundle_zip_roundtrip():
    """Bundle should survive zip → extract → load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_dir = Path(tmpdir) / "my_bundle"
        store = ArtifactStore(bundle_dir)

        store.write_manifest("roundtrip-test")
        store.write_env()
        store.write_seeds({"global_seed": 99, "rng_algorithm": "MT19937", "stage_seeds": {}})
        store.write_metrics({"width": 2, "total_gates": 5, "gates_1q": 3, "gates_2q": 2,
                             "depth": 3, "depth_2q": 1, "gate_histogram": {"h": 2, "cx": 2, "x": 1}})
        store.write_contracts([])
        store.write_contract_results([])
        store.write_trace([
            {
                "trace_id": "t1",
                "span_id": "s1",
                "parent_span_id": None,
                "name": "test",
                "start_time": "2025-01-01T00:00:00+00:00",
                "attributes": {},
                "events": [],
                "links": [],
                "status": "OK",
            },
        ])

        zip_path = Path(tmpdir) / "test_bundle.zip"
        store.export_zip(zip_path)
        assert zip_path.exists()

        # Load back
        loaded = ArtifactStore.load_bundle(zip_path)
        assert loaded["manifest"]["run_id"] == "roundtrip-test"
        assert loaded["seeds"]["global_seed"] == 99
        assert loaded["metrics"]["width"] == 2
        assert len(loaded["trace"]) == 1


def test_bundle_schema_validation():
    """All written files should pass schema validation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        store = ArtifactStore(tmpdir)

        store.write_manifest("validate-test")
        store.write_env()
        store.write_seeds({"global_seed": 42, "rng_algorithm": "MT19937"})
        store.write_metrics({"width": 3, "total_gates": 10, "gates_1q": 6, "gates_2q": 4,
                             "depth": 5, "gate_histogram": {"h": 3, "cx": 4}})
        store.write_contracts([
            {"name": "test_contract", "type": "observable"},
        ])
        store.write_contract_results([
            {"name": "test_contract", "passed": True},
        ])
        store.write_trace([
            {
                "trace_id": "abc",
                "span_id": "s1",
                "name": "test_span",
                "start_time": "2025-01-01T00:00:00+00:00",
                "status": "OK",
                "attributes": {},
                "events": [],
                "links": [],
            },
        ])

        results = validate_bundle(tmpdir)
        for fname, errors in results.items():
            assert errors == [], f"{fname} validation errors: {errors}"
