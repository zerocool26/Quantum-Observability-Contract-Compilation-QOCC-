"""Tests for Phase 8 features — audit fixes and hardening.

Covers:
- validate CLI with zip input (was broken by import bug)
- parent_span_id key consistency in visualization
- tracemalloc scope with empty specs
- _counts_to_observable_values import dedup
- ContractSpec type validation warning
- __all__ exports in subpackages
- DEFAULT_SEED constant
- Input validation (repeat, top_k)
- Replay module basics
- Topology module basics
- Visualization with Span.to_dict() output
"""

from __future__ import annotations

import json
import tempfile
import warnings
import zipfile
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner


# ======================================================================
# 1. validate CLI with zip bundles (was CRITICAL bug)
# ======================================================================

class TestValidateZipBundle:
    """Verify ``qocc validate`` works on .zip bundles."""

    def test_validate_zip_bundle(self, tmp_path: Path) -> None:
        """Passing a zip file should extract and validate."""
        from qocc.cli.commands_validate import validate

        manifest = {
            "schema_version": "0.1.0",
            "created_at": "2025-01-01T00:00:00Z",
            "run_id": "test-zip",
            "qocc_version": "0.1.0",
        }
        zip_path = tmp_path / "bundle.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))

        runner = CliRunner()
        result = runner.invoke(validate, [str(zip_path), "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        manifest_result = next(r for r in data["results"] if r["file"] == "manifest.json")
        assert manifest_result["status"] == "valid"

    def test_validate_zip_no_manifest(self, tmp_path: Path) -> None:
        """Zip with no known files should skip all."""
        from qocc.cli.commands_validate import validate

        zip_path = tmp_path / "empty.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("readme.txt", "hello")

        runner = CliRunner()
        result = runner.invoke(validate, [str(zip_path), "--format", "json"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert all(r["status"] == "skipped" for r in data["results"])


# ======================================================================
# 2. parent_span_id key consistency in visualization
# ======================================================================

class TestVisualizationParentKey:
    """Ensure render_timeline works with both old and new parent key."""

    def test_render_with_parent_span_id_key(self) -> None:
        """Spans using 'parent_span_id' should nest correctly."""
        from qocc.trace.visualization import render_timeline

        spans = [
            {"name": "root", "span_id": "a", "start_time": 0, "end_time": 1.0, "status": "ok"},
            {"name": "child", "span_id": "b", "parent_span_id": "a",
             "start_time": 0.1, "end_time": 0.9, "status": "ok"},
        ]
        output = render_timeline(spans, width=80)
        # Child should be indented (has more leading spaces than root)
        lines = output.strip().split("\n")
        root_line = next(l for l in lines if "root" in l)
        child_line = next(l for l in lines if "child" in l)
        assert len(child_line) - len(child_line.lstrip()) >= len(root_line) - len(root_line.lstrip())

    def test_render_with_span_to_dict_output(self) -> None:
        """Spans produced by Span.to_dict() (using 'parent_span_id') should work."""
        from qocc.trace.emitter import TraceEmitter
        from qocc.trace.visualization import render_timeline

        emitter = TraceEmitter()
        with emitter.span("parent_task") as parent:
            child = emitter.start_span("child_task", parent=parent)
            emitter.finish_span(child)

        span_dicts = [s.to_dict() for s in emitter.finished_spans()]
        output = render_timeline(span_dicts, width=80)
        assert "parent_task" in output
        assert "child_task" in output


# ======================================================================
# 3. tracemalloc scope with empty specs
# ======================================================================

class TestTraceMallocEmptySpecs:
    """check_contract with empty specs + max_memory_mb should not crash."""

    def test_empty_specs_no_error(self, tmp_path: Path) -> None:
        from qocc.api import check_contract

        bundle = tmp_path / "b"
        bundle.mkdir()
        (bundle / "manifest.json").write_text(
            json.dumps({"schema_version": "0.1.0", "run_id": "t"})
        )

        results = check_contract(
            str(bundle),
            contract_spec=[],
            max_memory_mb=100.0,
        )
        assert results == []


# ======================================================================
# 4. _counts_to_observable_values dedup
# ======================================================================

class TestCountsToObservable:
    """Verify only one copy of _counts_to_observable_values exists."""

    def test_import_from_eval_sampling(self) -> None:
        from qocc.contracts.eval_sampling import _counts_to_observable_values

        counts = {"00": 10, "11": 10}
        values = _counts_to_observable_values(counts)
        assert len(values) == 20
        # 00 → parity 0 → +1, 11 → parity 0 → +1
        assert all(v == 1.0 for v in values)

    def test_no_duplicate_in_api(self) -> None:
        """api.py should not define its own _counts_to_observable_values."""
        import inspect
        import qocc.api as api_mod

        source = inspect.getsource(api_mod)
        # Should contain the import but NOT a def
        assert "from qocc.contracts.eval_sampling import" in source
        # Count 'def _counts_to_observable_values' occurrences
        assert source.count("def _counts_to_observable_values") == 0


# ======================================================================
# 5. ContractSpec type validation warning
# ======================================================================

class TestContractSpecValidation:
    """ContractSpec should warn on unknown types."""

    def test_valid_type_no_warning(self) -> None:
        from qocc.contracts.spec import ContractSpec

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = ContractSpec(name="test", type="distribution")
            assert len(w) == 0
            assert spec._type_valid is True

    def test_invalid_type_emits_warning(self) -> None:
        from qocc.contracts.spec import ContractSpec

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = ContractSpec(name="test", type="bogus_type")
            assert len(w) == 1
            assert "Unknown contract type" in str(w[0].message)
            assert spec._type_valid is False

    def test_custom_evaluator_no_warning(self) -> None:
        from qocc.contracts.spec import ContractSpec

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            spec = ContractSpec(name="test", type="custom", evaluator="my_eval")
            assert len(w) == 0
            assert spec._type_valid is True


# ======================================================================
# 6. __all__ exports in subpackages
# ======================================================================

class TestSubpackageExports:
    """All subpackages should have __all__ and proper exports."""

    @pytest.mark.parametrize("pkg_path,expected", [
        ("qocc.core", ["ArtifactStore", "CircuitHandle", "CompilationCache", "canonicalize_qasm3"]),
        ("qocc.adapters", ["get_adapter", "AdapterBase"]),
        ("qocc.contracts", ["ContractSpec", "ContractResult", "ContractType"]),
        ("qocc.trace", ["Span", "TraceEmitter", "render_timeline"]),
        ("qocc.search", ["generate_candidates", "surrogate_score", "select_best"]),
        ("qocc.metrics", ["compute_metrics", "check_topology"]),
        ("qocc.cli", ["cli"]),
    ])
    def test_all_defined(self, pkg_path: str, expected: list[str]) -> None:
        import importlib

        mod = importlib.import_module(pkg_path)
        assert hasattr(mod, "__all__"), f"{pkg_path} missing __all__"
        for name in expected:
            assert name in mod.__all__, f"{name} not in {pkg_path}.__all__"
            assert hasattr(mod, name), f"{name} not importable from {pkg_path}"


# ======================================================================
# 7. DEFAULT_SEED constant
# ======================================================================

class TestDefaultSeed:
    """Verify DEFAULT_SEED is defined and accessible."""

    def test_default_seed_exists(self) -> None:
        from qocc import DEFAULT_SEED

        assert DEFAULT_SEED == 42
        assert isinstance(DEFAULT_SEED, int)

    def test_default_seed_in_all(self) -> None:
        import qocc

        assert "DEFAULT_SEED" in qocc.__all__


# ======================================================================
# 8. Input validation
# ======================================================================

class TestInputValidation:
    """Verify parameter validation in public API functions."""

    def test_run_trace_repeat_zero_raises(self) -> None:
        from qocc.api import run_trace

        with pytest.raises(ValueError, match="repeat must be >= 1"):
            run_trace(adapter_name="qiskit", input_source="OPENQASM 3;", repeat=0)

    def test_run_trace_repeat_negative_raises(self) -> None:
        from qocc.api import run_trace

        with pytest.raises(ValueError, match="repeat must be >= 1"):
            run_trace(adapter_name="qiskit", input_source="OPENQASM 3;", repeat=-5)

    def test_search_compile_topk_zero_raises(self) -> None:
        from qocc.api import search_compile

        with pytest.raises(ValueError, match="top_k must be >= 1"):
            search_compile(
                adapter_name="qiskit",
                input_source="OPENQASM 3;",
                top_k=0,
            )


# ======================================================================
# 9. Replay module
# ======================================================================

class TestReplayModule:
    """Basic tests for qocc.core.replay."""

    def test_replay_result_dataclass(self) -> None:
        from qocc.core.replay import ReplayResult

        r = ReplayResult(
            original_run_id="abc",
            input_hash_match=True,
            compiled_hash_match=True,
            metrics_match=True,
        )
        assert r.bit_exact is True

        d = r.to_dict()
        assert d["original_run_id"] == "abc"
        assert d["input_hash_match"] is True

    def test_replay_result_not_bit_exact(self) -> None:
        from qocc.core.replay import ReplayResult

        r = ReplayResult(
            original_run_id="def",
            input_hash_match=True,
            compiled_hash_match=False,
            metrics_match=True,
        )
        assert r.bit_exact is False

    def test_replay_missing_circuit_returns_error(self, tmp_path: Path) -> None:
        """Replaying a bundle with no circuits should return an error diff."""
        from qocc.core.replay import replay_bundle

        # Create a minimal bundle directory with no circuits
        bundle = tmp_path / "bundle"
        bundle.mkdir()
        (bundle / "manifest.json").write_text(
            json.dumps({"schema_version": "0.1.0", "run_id": "r", "adapter": "qiskit"})
        )
        (bundle / "seeds.json").write_text(json.dumps({"global_seed": 42}))
        (bundle / "metrics.json").write_text(json.dumps({}))

        result = replay_bundle(str(bundle))
        assert "error" in result.diff

    def test_replay_unknown_compiled_hash_is_explicit(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Missing selected.qasm should be reported as unknown and fail-closed."""
        from qocc.core.circuit_handle import CircuitHandle
        from qocc.core.replay import replay_bundle

        bundle = tmp_path / "bundle"
        (bundle / "circuits").mkdir(parents=True)
        (bundle / "manifest.json").write_text(
            json.dumps({"schema_version": "0.1.0", "run_id": "r1", "adapter": "fake"}),
            encoding="utf-8",
        )
        (bundle / "seeds.json").write_text(
            json.dumps({"global_seed": 42, "rng_algorithm": "PCG64", "stage_seeds": {}}),
            encoding="utf-8",
        )
        (bundle / "metrics.json").write_text(
            json.dumps({"compiled": {"depth": 1}}),
            encoding="utf-8",
        )
        input_qasm = "OPENQASM 3.0; qubit[1] q;"
        (bundle / "circuits" / "input.qasm").write_text(input_qasm, encoding="utf-8")

        class _FakeAdapter:
            def ingest(self, source: str) -> CircuitHandle:
                text = Path(source).read_text(encoding="utf-8") if Path(source).exists() else str(source)
                return CircuitHandle(
                    name="x",
                    num_qubits=1,
                    native_circuit={"qasm": text},
                    source_format="qasm3",
                    qasm3=text,
                )

        expected_input_hash = _FakeAdapter().ingest(str(bundle / "circuits" / "input.qasm")).stable_hash()

        def _fake_run_trace(**_: object) -> dict[str, object]:
            return {
                "bundle_zip": str(tmp_path / "replay.zip"),
                "input_hash": expected_input_hash,
                "compiled_hash": "compiled-replay-hash",
                "metrics_after": {"depth": 1},
            }

        monkeypatch.setattr("qocc.adapters.base.get_adapter", lambda _: _FakeAdapter())
        monkeypatch.setattr("qocc.api.run_trace", _fake_run_trace)

        result = replay_bundle(str(bundle))
        assert result.input_hash_status == "matched"
        assert result.compiled_hash_status == "unknown"
        assert result.compiled_hash_match is False
        assert result.bit_exact is False
        assert result.diff.get("_verification", {}).get("compiled_hash") == "unknown"


# ======================================================================
# 10. Topology module
# ======================================================================

class TestTopologyModule:
    """Basic tests for qocc.metrics.topology."""

    def test_predefined_coupling_maps_exist(self) -> None:
        from qocc.metrics.topology import LINEAR_5, GRID_2x3, HEAVY_HEX_7

        assert len(LINEAR_5) == 4
        assert len(GRID_2x3) == 7
        assert len(HEAVY_HEX_7) == 9

    def test_linear_5_edges(self) -> None:
        from qocc.metrics.topology import LINEAR_5

        # Should connect 0-1, 1-2, 2-3, 3-4
        assert (0, 1) in LINEAR_5
        assert (3, 4) in LINEAR_5


# ======================================================================
# 11. pydantic removed from dependencies
# ======================================================================

class TestDependencies:
    """Verify pydantic is not in required dependencies."""

    def test_pydantic_not_required(self) -> None:
        import tomllib
        toml_path = Path(__file__).resolve().parent.parent / "pyproject.toml"
        with open(toml_path, "rb") as f:
            cfg = tomllib.load(f)
        deps = cfg["project"]["dependencies"]
        assert not any("pydantic" in d for d in deps), f"pydantic still in deps: {deps}"


# ======================================================================
# 12. Dead code removal verification
# ======================================================================

class TestDeadCodeRemoved:
    """Verify dead code was removed."""

    def test_compare_params_removed(self) -> None:
        """_compare_params list should no longer exist."""
        import qocc.cli.commands_compare as mod

        assert not hasattr(mod, "_compare_params")
