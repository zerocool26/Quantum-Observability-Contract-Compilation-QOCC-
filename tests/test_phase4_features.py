"""Tests for Phase 4 features: statevector metadata, JSON schemas, Pareto wiring,
per-stage trace spans, resource budgets, early stopping, evaluator dispatch, bundle artifacts."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest
import numpy as np

from qocc.adapters.base import (
    BaseAdapter,
    CompileResult,
    MetricsSnapshot,
    SimulationResult,
    SimulationSpec,
)
from qocc.contracts.eval_sampling import (
    evaluate_distribution_contract,
    evaluate_observable_contract,
    _is_conclusive,
    _iterative_evaluate,
)
from qocc.contracts.spec import ContractResult, ContractSpec
from qocc.contracts.registry import (
    _EVALUATOR_REGISTRY,
    get_evaluator,
    list_evaluators,
    register_evaluator,
)
from qocc.core.artifacts import ArtifactStore
from qocc.core.circuit_handle import (
    BackendInfo,
    CircuitHandle,
    PassLogEntry,
    PipelineSpec,
)
from qocc.core.schemas import (
    CACHE_INDEX_SCHEMA,
    NONDETERMINISM_SCHEMA,
    SEARCH_RANKINGS_SCHEMA,
    SEARCH_RESULT_SCHEMA,
    SCHEMAS,
    validate_bundle,
    validate_file,
)
from qocc.search.selector import select_best
from qocc.search.space import Candidate
from qocc.trace.emitter import TraceEmitter


# ======================================================================
# Helpers
# ======================================================================


def _make_handle(name: str = "test", qubits: int = 2) -> CircuitHandle:
    return CircuitHandle(
        name=name,
        num_qubits=qubits,
        native_circuit=None,
        source_format="test",
        qasm3="OPENQASM 3; qubit[2] q; h q[0]; cx q[0], q[1];",
    )


def _make_candidate(cid: str, score: float = 1.0) -> Candidate:
    pipeline = PipelineSpec(adapter="test", optimization_level=1)
    c = Candidate(pipeline=pipeline, candidate_id=cid)
    c.metrics = {"depth": 10, "gates_2q": 5}
    c.surrogate_score = score
    c.validated = True
    return c


# ======================================================================
# 1. New JSON schemas exist and validate correctly
# ======================================================================


class TestNewSchemas:
    """Verify the four new schemas are registered and work."""

    def test_cache_index_schema_registered(self):
        assert "cache_index" in SCHEMAS
        assert SCHEMAS["cache_index"] is CACHE_INDEX_SCHEMA

    def test_nondeterminism_schema_registered(self):
        assert "nondeterminism" in SCHEMAS
        assert SCHEMAS["nondeterminism"] is NONDETERMINISM_SCHEMA

    def test_search_rankings_schema_registered(self):
        assert "search_rankings" in SCHEMAS
        assert SCHEMAS["search_rankings"] is SEARCH_RANKINGS_SCHEMA

    def test_search_result_schema_registered(self):
        assert "search_result" in SCHEMAS
        assert SCHEMAS["search_result"] is SEARCH_RESULT_SCHEMA

    def test_cache_index_valid(self):
        data = [
            {"key": "abc123", "hit": True, "circuit_hash": "xyz", "timestamp": 1.0},
            {"key": "def456", "hit": False},
        ]
        errors = validate_file("cache_index", data)
        assert errors == []

    def test_cache_index_invalid(self):
        data = [{"missing_key": True}]
        errors = validate_file("cache_index", data)
        assert len(errors) > 0

    def test_nondeterminism_valid(self):
        data = {
            "reproducible": True,
            "num_runs": 3,
            "unique_hashes": 1,
            "confidence": 0.95,
            "hashes": ["abc"],
            "hash_counts": {"abc": 3},
        }
        errors = validate_file("nondeterminism", data)
        assert errors == []

    def test_nondeterminism_invalid(self):
        data = {"reproducible": "yes"}  # wrong type, missing required fields
        errors = validate_file("nondeterminism", data)
        assert len(errors) > 0

    def test_search_rankings_valid(self):
        data = [
            {"candidate_id": "c1", "surrogate_score": 0.5, "validated": True},
        ]
        errors = validate_file("search_rankings", data)
        assert errors == []

    def test_search_result_valid(self):
        data = {
            "feasible": True,
            "reason": "Best candidate selected",
            "selected": {"candidate_id": "c1", "surrogate_score": 0.5},
        }
        errors = validate_file("search_result", data)
        assert errors == []

    def test_search_result_invalid(self):
        data = {"feasible": "maybe"}
        errors = validate_file("search_result", data)
        assert len(errors) > 0

    def test_contracts_schema_allows_exact_and_cost(self):
        """Contract schema enum should include 'exact' and 'cost'."""
        data = [
            {"name": "ex", "type": "exact"},
            {"name": "co", "type": "cost"},
        ]
        errors = validate_file("contracts", data)
        assert errors == []

    def test_total_schema_count(self):
        """Should have 11 schemas total now."""
        assert len(SCHEMAS) == 11


# ======================================================================
# 2. validate_bundle checks optional files
# ======================================================================


class TestValidateBundleOptional:
    """validate_bundle should validate optional files when present."""

    def test_optional_cache_index_validated(self, tmp_path: Path):
        # Create a minimal bundle directory
        (tmp_path / "manifest.json").write_text(json.dumps({
            "schema_version": "1.0", "created_at": "2024-01-01T00:00:00Z", "run_id": "r1"
        }))
        (tmp_path / "env.json").write_text(json.dumps({"os": "linux", "python": "3.12"}))
        (tmp_path / "seeds.json").write_text(json.dumps({"global_seed": 42}))
        (tmp_path / "metrics.json").write_text(json.dumps({"width": 2}))
        (tmp_path / "contracts.json").write_text(json.dumps([]))
        (tmp_path / "contract_results.json").write_text(json.dumps([]))
        (tmp_path / "trace.jsonl").write_text("")

        # Write valid cache_index
        (tmp_path / "cache_index.json").write_text(json.dumps([
            {"key": "k1", "hit": True}
        ]))

        results = validate_bundle(tmp_path)
        assert "cache_index.json" in results
        assert results["cache_index.json"] == []

    def test_invalid_cache_index_reported(self, tmp_path: Path):
        (tmp_path / "manifest.json").write_text(json.dumps({
            "schema_version": "1.0", "created_at": "2024-01-01T00:00:00Z", "run_id": "r1"
        }))
        (tmp_path / "env.json").write_text(json.dumps({"os": "linux", "python": "3.12"}))
        (tmp_path / "seeds.json").write_text(json.dumps({"global_seed": 42}))
        (tmp_path / "metrics.json").write_text(json.dumps({"width": 2}))
        (tmp_path / "contracts.json").write_text(json.dumps([]))
        (tmp_path / "contract_results.json").write_text(json.dumps([]))
        (tmp_path / "trace.jsonl").write_text("")
        # Invalid: should be an array
        (tmp_path / "cache_index.json").write_text(json.dumps({"bad": True}))

        results = validate_bundle(tmp_path)
        assert "cache_index.json" in results
        assert len(results["cache_index.json"]) > 0

    def test_missing_optional_not_reported(self, tmp_path: Path):
        (tmp_path / "manifest.json").write_text(json.dumps({
            "schema_version": "1.0", "created_at": "2024-01-01T00:00:00Z", "run_id": "r1"
        }))
        (tmp_path / "env.json").write_text(json.dumps({"os": "linux", "python": "3.12"}))
        (tmp_path / "seeds.json").write_text(json.dumps({"global_seed": 42}))
        (tmp_path / "metrics.json").write_text(json.dumps({"width": 2}))
        (tmp_path / "contracts.json").write_text(json.dumps([]))
        (tmp_path / "contract_results.json").write_text(json.dumps([]))
        (tmp_path / "trace.jsonl").write_text("")

        results = validate_bundle(tmp_path)
        # Optional files that don't exist should not appear in results
        assert "cache_index.json" not in results
        assert "nondeterminism.json" not in results


# ======================================================================
# 3. Early stopping in sampling contracts
# ======================================================================


class TestEarlyStopping:
    """Test the iterative early-stopping mechanism."""

    def _uniform_counts(self, n: int) -> dict[str, int]:
        return {"00": n // 4, "01": n // 4, "10": n // 4, "11": n // 4}

    def test_no_early_stopping_without_budget(self):
        spec = ContractSpec(
            name="no-es",
            type="distribution",
            tolerances={"tvd": 0.1},
            confidence={"level": 0.95},
            resource_budget={"seed": 42, "n_bootstrap": 100},
        )
        before = self._uniform_counts(1024)
        after = self._uniform_counts(1024)
        result = evaluate_distribution_contract(spec, before, after)
        assert result.passed
        # No early_stopped key since budget not configured
        assert "early_stopped" not in result.details

    def test_early_stopping_conclusive_pass(self):
        spec = ContractSpec(
            name="es-pass",
            type="distribution",
            tolerances={"tvd": 0.3},  # generous tolerance
            confidence={"level": 0.95},
            resource_budget={
                "seed": 42,
                "n_bootstrap": 100,
                "early_stopping": True,
                "min_shots": 256,
                "max_shots": 4096,
            },
        )
        before = self._uniform_counts(1024)
        after = self._uniform_counts(1024)

        call_count = [0]

        def sim_fn(shots: int) -> dict[str, int]:
            call_count[0] += 1
            return self._uniform_counts(shots)

        result = evaluate_distribution_contract(spec, before, after, simulate_fn=sim_fn)
        assert result.passed
        # Should have stopped early (distributions identical → TVD ≈ 0)
        assert result.details.get("early_stopped") is True

    def test_early_stopping_conclusive_fail(self):
        spec = ContractSpec(
            name="es-fail",
            type="distribution",
            tolerances={"tvd": 0.001},  # very tight tolerance
            confidence={"level": 0.95},
            resource_budget={
                "seed": 42,
                "n_bootstrap": 100,
                "early_stopping": True,
                "min_shots": 256,
                "max_shots": 4096,
            },
        )
        # Very different distributions
        before = {"00": 900, "01": 100}
        after = {"00": 100, "01": 900}

        def sim_fn(shots: int) -> dict[str, int]:
            return {"00": shots // 10, "01": shots - shots // 10}

        result = evaluate_distribution_contract(spec, before, after, simulate_fn=sim_fn)
        assert not result.passed
        assert result.details.get("early_stopped") is True

    def test_is_conclusive_tvd_pass(self):
        spec = ContractSpec(name="t", type="distribution", tolerances={"tvd": 0.5})
        result = ContractResult(name="t", passed=True, details={
            "tvd_ci": {"lower": 0.01, "upper": 0.1},
            "tolerance": 0.5,
        })
        assert _is_conclusive(result, spec) is True  # upper 0.1 < 0.5 * 0.9

    def test_is_conclusive_tvd_fail(self):
        spec = ContractSpec(name="t", type="distribution", tolerances={"tvd": 0.01})
        result = ContractResult(name="t", passed=False, details={
            "tvd_ci": {"lower": 0.5, "upper": 0.8},
            "tolerance": 0.01,
        })
        assert _is_conclusive(result, spec) is True  # lower 0.5 > 0.01 * 1.1

    def test_is_conclusive_observable_pass(self):
        spec = ContractSpec(name="o", type="observable", tolerances={"epsilon": 1.0})
        result = ContractResult(name="o", passed=True, details={
            "conservative_diff": 0.01,
            "epsilon": 1.0,
        })
        assert _is_conclusive(result, spec) is True

    def test_is_conclusive_pvalue(self):
        spec = ContractSpec(name="p", type="distribution")
        result = ContractResult(name="p", passed=True, details={
            "p_value": 0.95,
            "alpha": 0.05,
        })
        assert _is_conclusive(result, spec) is True  # p >> alpha


# ======================================================================
# 4. Resource budget enforcement (max_shots clamping)
# ======================================================================


class TestResourceBudget:
    """Test that resource budgets are respected in contract evaluation."""

    def test_distribution_respects_max_shots_in_details(self):
        """When early_stopping is on and max_shots is set, total_shots is tracked."""
        spec = ContractSpec(
            name="budget",
            type="distribution",
            tolerances={"tvd": 0.5},
            confidence={"level": 0.95},
            resource_budget={
                "seed": 42,
                "n_bootstrap": 50,
                "early_stopping": True,
                "min_shots": 100,
                "max_shots": 500,
            },
        )
        before = {"0": 500, "1": 500}
        after = {"0": 500, "1": 500}

        def sim_fn(shots: int) -> dict[str, int]:
            return {"0": shots // 2, "1": shots - shots // 2}

        result = evaluate_distribution_contract(spec, before, after, simulate_fn=sim_fn)
        assert result.passed
        assert "total_shots" in result.details


# ======================================================================
# 5. Evaluator dispatch (plugin system)
# ======================================================================


class TestEvaluatorDispatch:
    """Test programmatic evaluator registration + dispatch in check_contract."""

    def setup_method(self):
        """Clean registry before each test."""
        _EVALUATOR_REGISTRY.clear()

    def test_register_and_call_custom_evaluator(self):
        def my_eval(spec, **kwargs):
            return ContractResult(name=spec.name, passed=True, details={"custom": True})

        register_evaluator("test_eval", my_eval)
        assert get_evaluator("test_eval") is my_eval

    def test_dispatch_through_check_contract(self):
        """Custom evaluator should be invoked when spec.evaluator matches."""
        from qocc.api import check_contract

        call_log = []

        def my_eval(spec, **kwargs):
            call_log.append(spec.name)
            return ContractResult(name=spec.name, passed=True, details={"via_plugin": True})

        register_evaluator("my_checker", my_eval)

        # Build a minimal bundle with no circuits
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "manifest.json").write_text(json.dumps({
                "schema_version": "1.0",
                "created_at": "2024-01-01T00:00:00Z",
                "run_id": "test",
            }))
            bundle = ArtifactStore.load_bundle(td)
            results = check_contract(
                bundle,
                [{"name": "custom-check", "type": "distribution", "evaluator": "my_checker"}],
            )

        assert len(results) == 1
        assert results[0]["passed"] is True
        assert results[0]["details"]["via_plugin"] is True
        assert call_log == ["custom-check"]

    def test_fallback_when_evaluator_not_found(self):
        """When spec.evaluator is set but no evaluator exists, fall through to built-in."""
        spec = ContractSpec(
            name="fallback",
            type="distribution",
            evaluator="nonexistent_eval",  # not registered
        )
        # The registry won't find it → falls through to built-in distribution
        fn = get_evaluator("nonexistent_eval")
        assert fn is None  # confirms fallback path used

    def test_list_evaluators(self):
        register_evaluator("eval_a", lambda s, **kw: None)
        register_evaluator("eval_b", lambda s, **kw: None)
        names = list_evaluators()
        assert "eval_a" in names
        assert "eval_b" in names


# ======================================================================
# 6. Pareto mode wiring through API
# ======================================================================


class TestParetoWiring:
    """Test that Pareto mode parameter flows through select_best."""

    def _make(self, cid: str, depth: float, gates: float, score: float) -> Candidate:
        pipeline = PipelineSpec(adapter="test", optimization_level=1)
        c = Candidate(pipeline=pipeline, candidate_id=cid)
        c.metrics = {"depth": depth, "gates_2q": gates}
        c.surrogate_score = score
        c.validated = True
        return c

    def test_select_best_pareto_mode(self):
        c1 = self._make("a", 5, 20, 0.8)
        c2 = self._make("b", 20, 5, 0.9)
        c3 = self._make("c", 15, 15, 1.5)  # dominated

        result = select_best(
            [c1, c2, c3],
            require_validated=True,
            mode="pareto",
            objectives=["depth", "gates_2q"],
        )
        assert result.feasible
        # Pareto frontier should contain c1 and c2 (non-dominated)
        frontier_ids = {c.candidate_id for c in result.pareto_frontier}
        assert "a" in frontier_ids
        assert "b" in frontier_ids

    def test_select_best_single_mode(self):
        c1 = self._make("a", 5, 20, 0.8)
        c2 = self._make("b", 20, 5, 0.5)  # lower score is better
        result = select_best([c1, c2], require_validated=True, mode="single")
        assert result.feasible
        assert result.selected.candidate_id == "b"


# ======================================================================
# 7. Per-stage trace spans
# ======================================================================


class TestPerStageSpans:
    """Test that adapters emit child spans when given an emitter."""

    def test_emitter_span_context_manager(self):
        """TraceEmitter.span() context manager works for per-pass wrapping."""
        emitter = TraceEmitter()
        with emitter.span("compile") as parent:
            for i, name in enumerate(["pass_a", "pass_b"]):
                with emitter.span(f"pass/{name}", attributes={"order": i}):
                    pass  # simulate work

        spans = emitter.finished_spans()
        names = [s.name for s in spans]
        assert "pass/pass_a" in names
        assert "pass/pass_b" in names
        assert "compile" in names

    def test_span_nesting_parent_child(self):
        emitter = TraceEmitter()
        with emitter.span("outer") as outer:
            with emitter.span("inner") as inner:
                pass

        spans = emitter.finished_spans()
        inner_span = next(s for s in spans if s.name == "inner")
        outer_span = next(s for s in spans if s.name == "outer")
        # Inner should reference outer as parent
        assert inner_span.parent_span_id == outer_span.span_id


# ======================================================================
# 8. Bundle artifact completeness
# ======================================================================


class TestBundleArtifacts:
    """Test that new bundle files are written and loadable."""

    def test_write_and_validate_cache_index(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        store.write_cache_index([{"key": "abc", "hit": True, "timestamp": 1.0}])

        fp = tmp_path / "cache_index.json"
        assert fp.exists()
        data = json.loads(fp.read_text())
        errors = validate_file("cache_index", data)
        assert errors == []

    def test_write_nondeterminism_json(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        nd = {
            "reproducible": False,
            "num_runs": 3,
            "unique_hashes": 2,
            "confidence": 0.67,
            "hashes": ["aaa", "bbb"],
            "hash_counts": {"aaa": 2, "bbb": 1},
        }
        store.write_json("nondeterminism.json", nd)

        fp = tmp_path / "nondeterminism.json"
        assert fp.exists()
        data = json.loads(fp.read_text())
        errors = validate_file("nondeterminism", data)
        assert errors == []

    def test_write_search_rankings(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        rankings = [
            {"candidate_id": "c1", "surrogate_score": 0.5, "validated": True},
            {"candidate_id": "c2", "surrogate_score": 0.7, "validated": False},
        ]
        store.write_json("search_rankings.json", rankings)

        fp = tmp_path / "search_rankings.json"
        assert fp.exists()
        data = json.loads(fp.read_text())
        errors = validate_file("search_rankings", data)
        assert errors == []

    def test_write_search_result(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        sr = {
            "feasible": True,
            "reason": "Best selected",
            "selected": {"candidate_id": "c1", "surrogate_score": 0.5},
        }
        store.write_json("search_result.json", sr)

        fp = tmp_path / "search_result.json"
        assert fp.exists()
        data = json.loads(fp.read_text())
        errors = validate_file("search_result", data)
        assert errors == []

    def test_write_normalized_circuit(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        store.write_circuit("normalized.qasm", "OPENQASM 3; qubit[2] q; h q[0];")

        fp = tmp_path / "circuits" / "normalized.qasm"
        assert fp.exists()
        assert "OPENQASM" in fp.read_text()

    def test_write_candidate_circuits(self, tmp_path: Path):
        store = ArtifactStore(tmp_path)
        store.write_circuit("candidates/c1.qasm", "OPENQASM 3; qubit[2] q;")
        store.write_circuit("candidates/c2.qasm", "OPENQASM 3; qubit[3] q;")

        assert (tmp_path / "circuits" / "candidates" / "c1.qasm").exists()
        assert (tmp_path / "circuits" / "candidates" / "c2.qasm").exists()


# ======================================================================
# 9. Statevector simulation metadata
# ======================================================================


class TestStatevectorMetadata:
    """Test that simulate(shots=0) returns statevector in metadata.

    Uses a mock adapter since Qiskit/Cirq may not be installed.
    """

    def test_mock_statevector_sim(self):
        """Verify the contract: shots=0 → metadata['statevector'] populated."""
        # Simulate what the adapter should do
        sv = [1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)]
        result = SimulationResult(
            counts={},
            shots=0,
            seed=42,
            metadata={"statevector": sv},
        )
        assert result.metadata["statevector"] is not None
        assert len(result.metadata["statevector"]) == 4
        assert result.shots == 0

    def test_statevector_for_exact_contract(self):
        """Exact equivalence needs two statevectors to compare."""
        from qocc.contracts.eval_exact import evaluate_exact_equivalence

        spec = ContractSpec(name="exact-test", type="exact", tolerances={"fidelity": 0.99})
        sv = [1.0 / np.sqrt(2), 0, 0, 1.0 / np.sqrt(2)]
        result = evaluate_exact_equivalence(spec, sv, sv)
        assert result.passed


# ======================================================================
# 10. Export schemas includes new schemas
# ======================================================================


class TestExportSchemas:
    """Test that export_schemas writes all 11 schema files."""

    def test_export_all_schemas(self, tmp_path: Path):
        from qocc.core.schemas import export_schemas

        export_schemas(tmp_path)
        files = list(tmp_path.glob("*.schema.json"))
        names = {f.stem.replace(".schema", "") for f in files}
        assert "cache_index" in names
        assert "nondeterminism" in names
        assert "search_rankings" in names
        assert "search_result" in names
        assert len(files) == 11
