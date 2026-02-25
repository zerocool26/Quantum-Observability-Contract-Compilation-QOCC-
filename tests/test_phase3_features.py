"""Tests for Phase 3 features: Pareto selection, G-test, plugin system, enhanced diffs."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from qocc.contracts.eval_sampling import evaluate_distribution_contract
from qocc.contracts.registry import (
    _EVALUATOR_REGISTRY,
    get_evaluator,
    list_evaluators,
    register_evaluator,
)
from qocc.contracts.spec import ContractResult, ContractSpec
from qocc.contracts.stats import g_test, chi_square_test
from qocc.core.artifacts import ArtifactStore
from qocc.core.cache import CompilationCache
from qocc.search.selector import (
    SelectionResult,
    compute_pareto_frontier,
    select_best,
)
from qocc.search.space import Candidate
from qocc.core.circuit_handle import PipelineSpec


# ======================================================================
# Pareto selection
# ======================================================================


class TestParetoSelection:
    """Test multi-objective Pareto frontier computation."""

    def _make_candidate(self, cid: str, depth: float, gates_2q: float, error: float, score: float = 1.0) -> Candidate:
        pipeline = PipelineSpec(adapter="qiskit", optimization_level=1)
        c = Candidate(pipeline=pipeline, candidate_id=cid)
        c.metrics = {"depth": depth, "gates_2q": gates_2q, "proxy_error_score": error}
        c.surrogate_score = score
        c.validated = True
        return c

    def test_single_candidate_is_frontier(self):
        c = self._make_candidate("a", 10, 5, 0.01)
        frontier = compute_pareto_frontier([c])
        assert len(frontier) == 1
        assert frontier[0].candidate_id == "a"

    def test_dominated_candidate_excluded(self):
        a = self._make_candidate("a", 10, 5, 0.01)  # dominates b on all
        b = self._make_candidate("b", 20, 10, 0.05)
        frontier = compute_pareto_frontier([a, b])
        assert len(frontier) == 1
        assert frontier[0].candidate_id == "a"

    def test_non_dominated_candidates_kept(self):
        # a is better on depth, b is better on gates_2q
        a = self._make_candidate("a", 5, 10, 0.05)
        b = self._make_candidate("b", 10, 3, 0.05)
        frontier = compute_pareto_frontier([a, b])
        assert len(frontier) == 2

    def test_three_candidates_mixed(self):
        a = self._make_candidate("a", 5, 10, 0.05, score=2.0)
        b = self._make_candidate("b", 10, 3, 0.05, score=1.0)
        c = self._make_candidate("c", 20, 20, 0.10, score=3.0)  # dominated by a and b
        frontier = compute_pareto_frontier([a, b, c])
        ids = {f.candidate_id for f in frontier}
        assert "a" in ids
        assert "b" in ids
        assert "c" not in ids

    def test_custom_objectives(self):
        a = self._make_candidate("a", 5, 10, 0.05)
        b = self._make_candidate("b", 10, 3, 0.05)
        # Only considering depth → a dominates b
        frontier = compute_pareto_frontier([a, b], objectives=["depth"])
        assert len(frontier) == 1
        assert frontier[0].candidate_id == "a"

    def test_select_best_pareto_mode(self):
        a = self._make_candidate("a", 5, 10, 0.05, score=2.0)
        b = self._make_candidate("b", 10, 3, 0.05, score=1.0)
        c = self._make_candidate("c", 20, 20, 0.10, score=3.0)
        result = select_best([a, b, c], require_validated=True, mode="pareto")
        assert result.feasible
        assert result.selected is not None
        # b has lowest score on the frontier
        assert result.selected.candidate_id == "b"
        assert len(result.pareto_frontier) == 2
        assert "Pareto" in result.reason

    def test_select_best_pareto_result_serialization(self):
        a = self._make_candidate("a", 5, 10, 0.05, score=1.0)
        result = select_best([a], require_validated=True, mode="pareto")
        d = result.to_dict()
        assert "pareto_frontier" in d
        assert len(d["pareto_frontier"]) == 1


# ======================================================================
# G-test
# ======================================================================


class TestGTest:
    """Test G-test (log-likelihood ratio) implementation."""

    def test_identical_distributions_pass(self):
        counts = {"00": 250, "01": 250, "10": 250, "11": 250}
        result = g_test(counts, counts)
        assert result["passed"] is True
        assert result["statistic"] == pytest.approx(0.0, abs=1e-10)

    def test_very_different_distributions_fail(self):
        a = {"00": 900, "01": 50, "10": 30, "11": 20}
        b = {"00": 100, "01": 300, "10": 300, "11": 300}
        result = g_test(a, b)
        assert result["passed"] is False
        assert result["statistic"] > 0

    def test_has_williams_correction(self):
        a = {"00": 50, "01": 50}
        b = {"00": 40, "01": 60}
        result = g_test(a, b)
        assert "williams_correction" in result
        assert result["williams_correction"] > 1.0  # correction > 1

    def test_has_df(self):
        a = {"00": 100, "01": 100, "10": 100}
        b = {"00": 100, "01": 100, "10": 100}
        result = g_test(a, b)
        assert result["df"] == 2  # 3 categories - 1

    def test_single_bin(self):
        result = g_test({"00": 100}, {"00": 100})
        assert result["passed"] is True


class TestChiSquareFromDistributionContract:
    """Test chi-square dispatch via evaluate_distribution_contract."""

    def test_chi_square_via_spec(self):
        pytest.importorskip("scipy")
        spec = ContractSpec(
            name="chi_test",
            type="distribution",
            spec={"test": "chi_square"},
            confidence={"level": 0.95},
        )
        counts = {"00": 250, "01": 250, "10": 250, "11": 250}
        result = evaluate_distribution_contract(spec, counts, counts)
        assert result.passed is True
        assert result.details["method"] == "chi_square"

    def test_g_test_via_spec(self):
        spec = ContractSpec(
            name="g_test",
            type="distribution",
            spec={"test": "g_test"},
            confidence={"level": 0.95},
        )
        counts = {"00": 250, "01": 250, "10": 250, "11": 250}
        result = evaluate_distribution_contract(spec, counts, counts)
        assert result.passed is True
        assert result.details["method"] == "g_test"

    def test_default_uses_tvd(self):
        spec = ContractSpec(
            name="tvd_test",
            type="distribution",
            tolerances={"tvd": 0.5},
            confidence={"level": 0.95},
        )
        a = {"00": 260, "01": 240, "10": 250, "11": 250}
        b = {"00": 250, "01": 250, "10": 250, "11": 250}
        result = evaluate_distribution_contract(spec, a, b)
        assert result.details.get("method", "tvd") == "tvd"


# ======================================================================
# Plugin / evaluator registry
# ======================================================================


class TestEvaluatorRegistry:
    """Test custom evaluator registration and lookup."""

    def setup_method(self):
        # Clear registry between tests
        _EVALUATOR_REGISTRY.clear()

    def test_register_and_get(self):
        def my_eval(spec, **kwargs):
            return ContractResult(name=spec.name, passed=True, details={"custom": True})

        register_evaluator("my_eval", my_eval)
        fn = get_evaluator("my_eval")
        assert fn is not None
        result = fn(ContractSpec(name="test", type="custom"))
        assert result.passed is True
        assert result.details["custom"] is True

    def test_get_missing_returns_none(self):
        assert get_evaluator("nonexistent") is None

    def test_list_evaluators_empty(self):
        assert list_evaluators() == []

    def test_list_evaluators_after_register(self):
        register_evaluator("alpha", lambda s, **kw: ContractResult(name="", passed=True))
        register_evaluator("beta", lambda s, **kw: ContractResult(name="", passed=True))
        names = list_evaluators()
        assert "alpha" in names
        assert "beta" in names


# ======================================================================
# Enhanced comparison diffs
# ======================================================================


class TestEnhancedComparison:
    """Test pass-log, circuit-hash, and seeds diff in compare_bundles."""

    def test_pass_log_diff(self):
        from qocc.api import compare_bundles

        a = {
            "manifest": {"run_id": "a"},
            "metrics": {"input": {}, "compiled": {}, "pass_log": [
                {"pass_name": "init"},
                {"pass_name": "layout"},
                {"pass_name": "routing"},
            ]},
            "env": {},
        }
        b = {
            "manifest": {"run_id": "b"},
            "metrics": {"input": {}, "compiled": {}, "pass_log": [
                {"pass_name": "init"},
                {"pass_name": "optimization"},
                {"pass_name": "routing"},
            ]},
            "env": {},
        }
        report = compare_bundles(a, b)
        pass_diff = report["diffs"]["pass_log"]
        assert pass_diff  # non-empty means diff detected
        assert "layout" in pass_diff.get("removed", [])
        assert "optimization" in pass_diff.get("added", [])

    def test_seeds_diff(self):
        from qocc.api import compare_bundles

        a = {"manifest": {"run_id": "a"}, "metrics": {}, "env": {},
             "seeds": {"global_seed": 42}}
        b = {"manifest": {"run_id": "b"}, "metrics": {}, "env": {},
             "seeds": {"global_seed": 99}}
        report = compare_bundles(a, b)
        assert report["diffs"]["seeds"]["global_seed"]["a"] == 42
        assert report["diffs"]["seeds"]["global_seed"]["b"] == 99

    def test_circuit_hash_diff_empty_when_no_root(self):
        from qocc.api import compare_bundles

        a = {"manifest": {"run_id": "a"}, "metrics": {}, "env": {}}
        b = {"manifest": {"run_id": "b"}, "metrics": {}, "env": {}}
        report = compare_bundles(a, b)
        # No roots → no hash differences
        assert report["diffs"]["circuit_hashes"] == {} or report["diffs"]["circuit_hashes"] is not None

    def test_comparison_md_includes_new_sections(self):
        from qocc.api import compare_bundles

        a = {
            "manifest": {"run_id": "a"},
            "metrics": {"input": {"depth": 5}, "compiled": {"depth": 10},
                        "pass_log": [{"pass_name": "init"}]},
            "env": {},
            "seeds": {"global_seed": 42},
        }
        b = {
            "manifest": {"run_id": "b"},
            "metrics": {"input": {"depth": 5}, "compiled": {"depth": 15},
                        "pass_log": [{"pass_name": "opt"}]},
            "env": {},
            "seeds": {"global_seed": 99},
        }
        report = compare_bundles(a, b)
        md = report["markdown"]
        assert "Pass-Log Differences" in md
        assert "Seed Differences" in md


# ======================================================================
# Cache index in artifacts
# ======================================================================


class TestCacheIndex:
    """Test that ArtifactStore can write cache_index.json."""

    def test_write_cache_index(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            store = ArtifactStore(tmpdir)
            entries = [
                {"key": "abc123", "hit": True, "timestamp": 12345},
                {"key": "def456", "hit": False, "timestamp": 12346},
            ]
            p = store.write_cache_index(entries)
            assert p.exists()
            data = json.loads(p.read_text(encoding="utf-8"))
            assert len(data) == 2
            assert data[0]["hit"] is True
            assert data[1]["hit"] is False


# ======================================================================
# Enhanced summary reporting
# ======================================================================


class TestEnhancedSummary:
    """Test _generate_summary with new parameters."""

    def test_summary_includes_environment(self):
        from qocc.api import _generate_summary
        from qocc.core.circuit_handle import CircuitHandle

        handle = CircuitHandle(name="test", num_qubits=2, native_circuit=None, source_format="qasm3", qasm3="OPENQASM 3.0;")
        pipeline = PipelineSpec(adapter="qiskit")
        summary = _generate_summary(
            run_id="test123",
            adapter_name="qiskit",
            input_handle=handle,
            compiled_handle=handle,
            metrics_before={"depth": 3},
            metrics_after={"depth": 5},
            pipeline_spec=pipeline,
            pass_log=[],
        )
        assert "## Environment" in summary
        assert "OS:" in summary
        assert "Python:" in summary

    def test_summary_includes_nondet_warning(self):
        from qocc.api import _generate_summary
        from qocc.core.circuit_handle import CircuitHandle

        handle = CircuitHandle(name="test", num_qubits=2, native_circuit=None, source_format="qasm3", qasm3="OPENQASM 3.0;")
        pipeline = PipelineSpec(adapter="qiskit")
        nondet = {
            "reproducible": False,
            "unique_hashes": 3,
            "num_runs": 5,
            "confidence": 0.95,
        }
        summary = _generate_summary(
            run_id="test123",
            adapter_name="qiskit",
            input_handle=handle,
            compiled_handle=handle,
            metrics_before={},
            metrics_after={},
            pipeline_spec=pipeline,
            pass_log=[],
            nondet_report=nondet,
        )
        assert "Nondeterminism detected" in summary
        assert "3 unique hashes" in summary

    def test_summary_includes_contract_results(self):
        from qocc.api import _generate_summary
        from qocc.core.circuit_handle import CircuitHandle

        handle = CircuitHandle(name="test", num_qubits=2, native_circuit=None, source_format="qasm3", qasm3="OPENQASM 3.0;")
        pipeline = PipelineSpec(adapter="qiskit")
        contracts = [
            {"name": "tvd_check", "passed": True, "details": {"type": "distribution"}},
            {"name": "depth_limit", "passed": False, "details": {"type": "cost"}},
        ]
        summary = _generate_summary(
            run_id="test123",
            adapter_name="qiskit",
            input_handle=handle,
            compiled_handle=handle,
            metrics_before={},
            metrics_after={},
            pipeline_spec=pipeline,
            pass_log=[],
            contract_results=contracts,
        )
        assert "## Contract Results" in summary
        assert "tvd_check" in summary
        assert "depth_limit" in summary


class TestSearchSummary:
    """Test _generate_search_summary."""

    def test_search_summary_has_candidate_table(self):
        from qocc.api import _generate_search_summary

        c1 = Candidate(
            pipeline=PipelineSpec(adapter="qiskit", optimization_level=1),
            candidate_id="cand1",
        )
        c1.surrogate_score = 0.5
        c1.metrics = {"depth": 10, "gates_2q": 5}
        c1.validated = True

        selection = SelectionResult(
            selected=c1,
            all_candidates=[c1],
            feasible=True,
            reason="Selected",
        )

        summary = _generate_search_summary(
            run_id="s123",
            adapter_name="qiskit",
            num_candidates=1,
            top_k=1,
            selection=selection,
            ranked=[c1],
            contract_specs=[],
            all_contract_results=[],
        )
        assert "## Candidate Rankings" in summary
        assert "cand1" in summary
        assert "0.5000" in summary


# ======================================================================
# Clifford fallback to distribution
# ======================================================================


class TestCliffordFallback:
    """Test that Clifford evaluator falls back to distribution when non-Clifford."""

    def test_fallback_with_counts(self):
        from qocc.contracts.eval_clifford import evaluate_clifford_contract
        from qocc.core.circuit_handle import CircuitHandle

        # Non-Clifford handles (no native circuits → is_clifford is False)
        h_before = CircuitHandle(name="before", num_qubits=2, native_circuit=None, source_format="qasm3", qasm3="OPENQASM 3.0;")
        h_after = CircuitHandle(name="after", num_qubits=2, native_circuit=None, source_format="qasm3", qasm3="OPENQASM 3.0;")
        spec = ContractSpec(
            name="cliff_test",
            type="clifford",
            tolerances={"tvd": 0.5},
            confidence={"level": 0.95},
        )
        counts = {"00": 250, "01": 250, "10": 250, "11": 250}
        result = evaluate_clifford_contract(spec, h_before, h_after,
                                            counts_before=counts, counts_after=counts)
        assert result.details["method"] == "clifford_fallback_to_distribution"
        assert result.passed is True

    def test_no_fallback_without_counts(self):
        from qocc.contracts.eval_clifford import evaluate_clifford_contract
        from qocc.core.circuit_handle import CircuitHandle

        h_before = CircuitHandle(name="before", num_qubits=2, native_circuit=None, source_format="qasm3", qasm3="OPENQASM 3.0;")
        h_after = CircuitHandle(name="after", num_qubits=2, native_circuit=None, source_format="qasm3", qasm3="OPENQASM 3.0;")
        spec = ContractSpec(name="cliff_test", type="clifford")
        result = evaluate_clifford_contract(spec, h_before, h_after)
        assert result.passed is False
        assert "no simulation counts" in result.details["note"].lower()
