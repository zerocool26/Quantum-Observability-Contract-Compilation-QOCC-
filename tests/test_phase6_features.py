"""Tests for Phase 6 features.

Covers:
- Thread-safe TraceEmitter
- Structured logging / trace events
- CLI JSON compare output
- Idle penalty duration model
- CLI JSON schema validation
- Span links across candidates
- Cache key with seeds extra
- RNG algorithm in seeds
"""

from __future__ import annotations

import json
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

# ======================================================================
# Thread-safe TraceEmitter
# ======================================================================

class TestThreadSafeEmitter:
    """Verify TraceEmitter is safe under concurrent access."""

    def test_per_thread_active_span_isolation(self):
        """Each thread should have its own active span stack."""
        from qocc.trace.emitter import TraceEmitter

        emitter = TraceEmitter()
        results: dict[str, str | None] = {}

        def worker(name: str) -> None:
            with emitter.span(name) as s:
                time.sleep(0.01)
                results[name] = emitter._active_span.name if emitter._active_span else None

        threads = [threading.Thread(target=worker, args=(f"span_{i}",)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Each thread should see only its own span
        for i in range(5):
            assert results[f"span_{i}"] == f"span_{i}"

    def test_concurrent_finish_no_data_loss(self):
        """Finishing spans concurrently should not lose any."""
        from qocc.trace.emitter import TraceEmitter

        emitter = TraceEmitter()
        n = 50

        def worker(idx: int) -> None:
            with emitter.span(f"s{idx}"):
                pass

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        spans = emitter.finished_spans()
        assert len(spans) == n

    def test_main_thread_active_span_unaffected(self):
        """Spawning child thread should not corrupt main thread's active span."""
        from qocc.trace.emitter import TraceEmitter

        emitter = TraceEmitter()

        with emitter.span("main_span"):
            assert emitter._active_span is not None
            assert emitter._active_span.name == "main_span"

            def child() -> None:
                # Child should NOT see main thread's span
                assert emitter._active_span is None

            t = threading.Thread(target=child)
            t.start()
            t.join()

            # Main thread span should still be intact
            assert emitter._active_span.name == "main_span"


# ======================================================================
# Idle penalty duration model
# ======================================================================

class TestParallelDuration:
    """Test parallel-aware duration computation with idle penalties."""

    def test_parallel_duration_cirq_basic(self):
        """Verify critical_path and idle_penalty for a simple Cirq circuit."""
        pytest.importorskip("cirq")
        import cirq

        from qocc.metrics.compute import _parallel_duration_cirq

        q0, q1 = cirq.LineQubit.range(2)
        circuit = cirq.Circuit([
            cirq.Moment([cirq.X(q0)]),              # Only q0 active, q1 idle
            cirq.Moment([cirq.CNOT(q0, q1)]),        # Both active
        ])

        model = {"X": 25.0, "CNOT": 300.0, "default": 50.0}
        result = _parallel_duration_cirq(circuit, model, idle_penalty_rate=0.01)

        # Layer 1: dur=25, 1 idle qubit → penalty 0.25
        # Layer 2: dur=300, 0 idle qubits → penalty 0
        assert result["critical_path"] == pytest.approx(325.0)
        assert result["idle_penalty"] == pytest.approx(0.25)
        assert result["total"] == pytest.approx(325.25)

    def test_parallel_duration_returns_dict_keys(self):
        """Result always has the expected keys."""
        pytest.importorskip("cirq")
        import cirq

        from qocc.metrics.compute import _parallel_duration_cirq

        circuit = cirq.Circuit()
        result = _parallel_duration_cirq(circuit, {"default": 10.0})
        assert set(result.keys()) == {"critical_path", "idle_penalty", "total"}

    def test_metrics_include_duration_parallel(self):
        """compute_metrics() should include 'duration_parallel' key."""
        pytest.importorskip("cirq")
        import cirq

        from qocc.core.circuit_handle import CircuitHandle
        from qocc.metrics.compute import compute_metrics

        q = cirq.LineQubit.range(2)
        c = cirq.Circuit([cirq.X(q[0]), cirq.CNOT(q[0], q[1])])
        handle = CircuitHandle(native_circuit=c, source_format="cirq")

        m = compute_metrics(handle, duration_model={"X": 25.0, "CNOT": 300.0, "default": 50.0})
        assert "duration_parallel" in m
        assert m["duration_parallel"] is not None
        assert m["duration_parallel"]["critical_path"] > 0

    def test_duration_parallel_none_without_model(self):
        """Without a duration model, duration_parallel should be None."""
        pytest.importorskip("cirq")
        import cirq

        from qocc.core.circuit_handle import CircuitHandle
        from qocc.metrics.compute import compute_metrics

        q = cirq.LineQubit.range(1)
        c = cirq.Circuit([cirq.X(q[0])])
        handle = CircuitHandle(native_circuit=c, source_format="cirq")

        m = compute_metrics(handle)
        assert m["duration_parallel"] is None


# ======================================================================
# CLI JSON compare output
# ======================================================================

class TestCompareJsonOutput:
    """Test --format json flag on compare CLI."""

    def _make_bundle(self, tmpdir: str, run_id: str, depth: int) -> Path:
        from qocc.core.artifacts import ArtifactStore
        bdir = Path(tmpdir) / run_id
        store = ArtifactStore(bdir)
        store.write_manifest(run_id)
        store.write_env()
        store.write_seeds({"global_seed": 42, "rng_algorithm": "MT19937", "stage_seeds": {}})
        store.write_metrics({
            "input": {"width": 2, "depth": depth, "total_gates": 2 * depth},
            "compiled": {"width": 2, "depth": depth - 1, "total_gates": 2 * (depth - 1)},
        })
        store.write_contracts([])
        store.write_contract_results([])
        store.write_trace([{
            "trace_id": "t1", "span_id": "s1", "name": "test",
            "start_time": "2025-01-01T00:00:00+00:00",
            "parent_span_id": None,
            "attributes": {}, "events": [], "links": [], "status": "OK",
        }])
        zip_path = bdir.with_suffix(".zip")
        store.export_zip(zip_path)
        return zip_path

    def test_json_output_to_stdout(self):
        """--format json should output valid JSON."""
        from click.testing import CliRunner
        from qocc.cli.commands_compare import compare

        with tempfile.TemporaryDirectory() as tmpdir:
            a = self._make_bundle(tmpdir, "a", 5)
            b = self._make_bundle(tmpdir, "b", 10)
            runner = CliRunner()
            result = runner.invoke(compare, [str(a), str(b), "--format", "json"])
            assert result.exit_code == 0
            # The JSON is emitted via click.echo; Rich header goes to console
            # Find the first '{' to start of JSON
            out = result.output
            start = out.find("{")
            assert start >= 0, f"No JSON object in output: {out!r}"
            data = json.loads(out[start:])
            assert "diffs" in data

    def test_json_output_to_file(self):
        """--format json with --report should write comparison.json."""
        from click.testing import CliRunner
        from qocc.cli.commands_compare import compare

        with tempfile.TemporaryDirectory() as tmpdir:
            a = self._make_bundle(tmpdir, "a", 5)
            b = self._make_bundle(tmpdir, "b", 10)
            report_dir = str(Path(tmpdir) / "reports")
            runner = CliRunner()
            result = runner.invoke(compare, [str(a), str(b), "--format", "json", "-r", report_dir])
            assert result.exit_code == 0
            jf = Path(report_dir) / "comparison.json"
            assert jf.exists()
            data = json.loads(jf.read_text())
            assert "diffs" in data


# ======================================================================
# CLI JSON schema validation
# ======================================================================

class TestCliValidation:
    """Test the qocc.cli.validation module."""

    def test_validate_valid_contracts(self, tmp_path: Path):
        from qocc.cli.validation import validate_json_file

        contracts = [{"name": "depth_check", "type": "observable", "spec": {}}]
        fp = tmp_path / "contracts.json"
        fp.write_text(json.dumps(contracts))

        result = validate_json_file(str(fp), "contracts")
        assert isinstance(result, list)
        assert result[0]["name"] == "depth_check"

    def test_validate_invalid_contracts_strict(self, tmp_path: Path):
        """Missing required field should raise ClickException in strict mode."""
        import click
        from qocc.cli.validation import validate_json_file

        contracts = [{"name": "bad"}]  # missing "type"
        fp = tmp_path / "bad.json"
        fp.write_text(json.dumps(contracts))

        with pytest.raises(click.ClickException, match="type"):
            validate_json_file(str(fp), "contracts", strict=True)

    def test_validate_invalid_contracts_lenient(self, tmp_path: Path):
        """Lenient mode returns data even on validation errors."""
        from qocc.cli.validation import validate_json_file

        contracts = [{"name": "bad"}]  # missing "type"
        fp = tmp_path / "bad.json"
        fp.write_text(json.dumps(contracts))

        result = validate_json_file(str(fp), "contracts", strict=False)
        assert result == contracts

    def test_validate_malformed_json(self, tmp_path: Path):
        """Malformed JSON should always raise."""
        import click
        from qocc.cli.validation import validate_json_file

        fp = tmp_path / "bad.json"
        fp.write_text("{not valid json}")

        with pytest.raises(click.ClickException, match="Invalid JSON"):
            validate_json_file(str(fp), "contracts")

    def test_validate_unknown_schema_skips(self, tmp_path: Path):
        """Unknown schema name should skip validation and return data."""
        from qocc.cli.validation import validate_json_file

        fp = tmp_path / "data.json"
        fp.write_text('{"any": "data"}')

        result = validate_json_file(str(fp), "nonexistent_schema_name")
        assert result == {"any": "data"}


# ======================================================================
# Span links across candidates
# ======================================================================

class TestSpanLinks:
    """Verify span links are created on per-candidate spans."""

    def test_span_add_link(self):
        from qocc.trace.span import Span

        s = Span(trace_id="t1", name="child")
        s.add_link("t1", "parent_span_id", relationship="child_of_batch")
        assert len(s.links) == 1
        assert s.links[0].span_id == "parent_span_id"
        assert s.links[0].attributes["relationship"] == "child_of_batch"

    def test_span_link_serialization(self):
        from qocc.trace.span import Span

        s = Span(trace_id="t1", name="test")
        s.add_link("t1", "linked_span", reason="batch")
        d = s.to_dict()
        assert len(d["links"]) == 1
        assert d["links"][0]["span_id"] == "linked_span"


# ======================================================================
# Cache key with seeds
# ======================================================================

class TestCacheKeySeeds:
    """Verify seeds in extra affect cache key."""

    def test_different_seeds_different_key(self):
        from qocc.core.cache import CompilationCache

        k1 = CompilationCache.cache_key("h1", {"opt": 2}, "v1", extra={"seed": 42})
        k2 = CompilationCache.cache_key("h1", {"opt": 2}, "v1", extra={"seed": 99})
        assert k1 != k2

    def test_same_seeds_same_key(self):
        from qocc.core.cache import CompilationCache

        k1 = CompilationCache.cache_key("h1", {"opt": 2}, "v1", extra={"seed": 42})
        k2 = CompilationCache.cache_key("h1", {"opt": 2}, "v1", extra={"seed": 42})
        assert k1 == k2

    def test_no_extra_backward_compat(self):
        """Omitting extra should still work."""
        from qocc.core.cache import CompilationCache

        k1 = CompilationCache.cache_key("h1", {"opt": 2}, "v1")
        k2 = CompilationCache.cache_key("h1", {"opt": 2}, "v1", extra=None)
        assert k1 == k2


# ======================================================================
# RNG algorithm field
# ======================================================================

class TestRngAlgorithm:
    """Verify seeds include rng_algorithm in CLI and API."""

    def test_trace_run_cli_seeds_include_rng(self):
        """The trace run CLI should source rng_algorithm from shared constants."""
        # Read the source file directly to verify the seeds dict
        from pathlib import Path
        src_file = Path(__file__).resolve().parent.parent / "qocc" / "cli" / "commands_trace.py"
        src = src_file.read_text(encoding="utf-8")
        assert "rng_algorithm" in src
        assert "DEFAULT_RNG_ALGORITHM" in src

    def test_artifact_store_seeds_roundtrip(self, tmp_path: Path):
        """Seeds written and read back should preserve rng_algorithm."""
        from qocc.core.artifacts import ArtifactStore

        seeds = {"global_seed": 42, "rng_algorithm": "MT19937", "stage_seeds": {}}
        store = ArtifactStore(tmp_path)
        store.write_seeds(seeds)

        loaded = json.loads((tmp_path / "seeds.json").read_text())
        assert loaded["rng_algorithm"] == "MT19937"


# ======================================================================
# Zip bundle round-trip
# ======================================================================

class TestZipRoundTrip:
    """Test full bundle zip/unzip round-trip."""

    def test_export_and_load_bundle(self, tmp_path: Path):
        from qocc.core.artifacts import ArtifactStore

        bdir = tmp_path / "bundle"
        store = ArtifactStore(bdir)
        store.write_manifest("test-run-001")
        store.write_env()
        store.write_seeds({"global_seed": 42, "rng_algorithm": "MT19937", "stage_seeds": {}})
        store.write_metrics({
            "input": {"width": 2, "depth": 5},
            "compiled": {"width": 2, "depth": 3},
        })
        store.write_contracts([])
        store.write_contract_results([])
        store.write_trace([{
            "trace_id": "t1", "span_id": "s1", "name": "root",
            "start_time": "2025-01-01T00:00:00+00:00",
            "parent_span_id": None,
            "attributes": {}, "events": [], "links": [], "status": "OK",
        }])

        zip_path = bdir.with_suffix(".zip")
        store.export_zip(zip_path)
        assert zip_path.exists()

        # Load back
        loaded = ArtifactStore.load_bundle(str(zip_path))
        assert loaded["manifest"]["run_id"] == "test-run-001"
        assert loaded["seeds"]["global_seed"] == 42
        assert loaded["metrics"]["input"]["width"] == 2

    def test_bundle_contains_all_artifacts(self, tmp_path: Path):
        """Zip should contain expected files."""
        import zipfile
        from qocc.core.artifacts import ArtifactStore

        bdir = tmp_path / "b"
        store = ArtifactStore(bdir)
        store.write_manifest("run1")
        store.write_env()
        store.write_seeds({"global_seed": 1, "rng_algorithm": "MT19937", "stage_seeds": {}})
        store.write_metrics({"input": {}, "compiled": {}})
        store.write_contracts([])
        store.write_contract_results([])
        store.write_trace([])

        zp = bdir.with_suffix(".zip")
        store.export_zip(zp)

        with zipfile.ZipFile(zp) as zf:
            names = zf.namelist()
            assert any("manifest.json" in n for n in names)
            assert any("seeds.json" in n for n in names)
            assert any("metrics.json" in n for n in names)


# ======================================================================
# CLI trace run basic
# ======================================================================

class TestCliTraceRun:
    """Smoke-test the trace run CLI."""

    def test_trace_run_help(self):
        from click.testing import CliRunner
        from qocc.cli.commands_trace import trace

        runner = CliRunner()
        result = runner.invoke(trace, ["run", "--help"])
        assert result.exit_code == 0
        assert "--adapter" in result.output
        assert "--seed" in result.output

    def test_compare_help(self):
        from click.testing import CliRunner
        from qocc.cli.commands_compare import compare

        runner = CliRunner()
        result = runner.invoke(compare, ["--help"])
        assert result.exit_code == 0
        assert "--format" in result.output

    def test_contract_check_help(self):
        from click.testing import CliRunner
        from qocc.cli.commands_contract import contract

        runner = CliRunner()
        result = runner.invoke(contract, ["check", "--help"])
        assert result.exit_code == 0
        assert "--contracts" in result.output

    def test_compile_search_help(self):
        from click.testing import CliRunner
        from qocc.cli.commands_search import compile_group

        runner = CliRunner()
        result = runner.invoke(compile_group, ["search", "--help"])
        assert result.exit_code == 0
        assert "--strategy" in result.output
