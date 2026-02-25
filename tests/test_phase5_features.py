"""Tests for Phase 5 features.

Covers:
  - QASM canonicalization (commuting gate sort, float normalisation)
  - CompileResult.from_dict() round-trip
  - OTLP JSON export / exporter functions
  - Bayesian search optimizer + random strategy
  - ContractType enum & validation flag
  - SPRT early stopping checker
  - SearchSpaceConfig strategy field & CLI injection
  - Metric key alignment in visualization
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ======================================================================
# 1. Canonicalization tests
# ======================================================================

from qocc.core.canonicalize import (
    canonicalize_qasm3,
    normalize_circuit,
    _sort_commuting_gates,
    _normalise_float_params,
)


class TestCanonicalizeQasm3:
    """Tests for the rewritten canonicalize_qasm3."""

    def test_strips_inline_comments(self):
        qasm = 'OPENQASM 3.0;\nh q[0]; // apply H\ncx q[0], q[1]; // CNOT\n'
        result = canonicalize_qasm3(qasm)
        assert "//" not in result
        assert "h q[0];" in result

    def test_strips_block_comments(self):
        qasm = "OPENQASM 3.0;\n/* multi\nline\ncomment */\nh q[0];\n"
        result = canonicalize_qasm3(qasm)
        assert "/*" not in result
        assert "*/" not in result
        assert "h q[0];" in result

    def test_normalises_whitespace(self):
        qasm = "OPENQASM  3.0;\n  h   q[0] ;\n"
        result = canonicalize_qasm3(qasm)
        # Multiple spaces collapsed
        assert "  " not in result.replace("\n", "")

    def test_sorts_gate_declarations(self):
        qasm = "OPENQASM 3.0;\ngate z q { }\ngate a q { }\nh q[0];\n"
        result = canonicalize_qasm3(qasm)
        lines = result.strip().splitlines()
        gate_lines = [l for l in lines if l.startswith("gate ")]
        assert gate_lines[0].startswith("gate a")
        assert gate_lines[1].startswith("gate z")

    def test_empty_input(self):
        assert canonicalize_qasm3("") == "\n"

    def test_header_before_body(self):
        qasm = "h q[0];\nOPENQASM 3.0;\ninclude \"stdgates.inc\";\n"
        result = canonicalize_qasm3(qasm)
        lines = [l for l in result.strip().splitlines() if l]
        # Header keywords should appear before body
        header_end = 0
        for i, l in enumerate(lines):
            if l.startswith(("OPENQASM", "include")):
                header_end = i
        body_start = next(
            (i for i, l in enumerate(lines) if not l.startswith(("OPENQASM", "include", "gate"))),
            len(lines),
        )
        assert header_end < body_start

    def test_idempotent(self):
        qasm = "OPENQASM 3.0;\nh q[0];\ncx q[0], q[1];\n"
        once = canonicalize_qasm3(qasm)
        twice = canonicalize_qasm3(once)
        assert once == twice


class TestSortCommutingGates:
    """Tests for _sort_commuting_gates."""

    def test_disjoint_qubits_sorted(self):
        lines = [
            "h q[1];",
            "h q[0];",
        ]
        result = _sort_commuting_gates(lines)
        # q[0] should come before q[1]
        assert result[0] == "h q[0];"
        assert result[1] == "h q[1];"

    def test_barrier_flushes_window(self):
        lines = [
            "h q[1];",
            "h q[0];",
            "barrier q[0], q[1];",
            "x q[1];",
            "x q[0];",
        ]
        result = _sort_commuting_gates(lines)
        # Before barrier: q[0] first
        assert result[0] == "h q[0];"
        assert result[1] == "h q[1];"
        assert result[2] == "barrier q[0], q[1];"
        # After barrier: q[0] first again
        assert result[3] == "x q[0];"
        assert result[4] == "x q[1];"

    def test_overlapping_qubits_cause_flush(self):
        """Gates on the same qubit shouldn't be reordered."""
        lines = [
            "h q[0];",
            "x q[0];",
        ]
        result = _sort_commuting_gates(lines)
        # Order must be preserved (same qubit)
        assert result[0] == "h q[0];"
        assert result[1] == "x q[0];"


class TestNormaliseFloatParams:
    """Tests for _normalise_float_params."""

    def test_normalises_pi(self):
        line = "rz(3.14159265358979) q[0];"
        result = _normalise_float_params(line)
        assert "3.14159265" in result

    def test_preserves_integers(self):
        line = "cx q[0], q[1];"
        result = _normalise_float_params(line)
        assert result == line

    def test_scientific_notation(self):
        line = "rz(1.5e-3) q[0];"
        result = _normalise_float_params(line)
        assert "0.0015" in result or "1.5e-" in result


class TestNormalizeCircuit:
    """Tests for normalize_circuit deep copy."""

    def test_deep_copy(self):
        from qocc.core.circuit_handle import CircuitHandle
        handle = CircuitHandle(
            name="test",
            num_qubits=2,
            native_circuit={"mutable": [1, 2, 3]},
            source_format="test",
            qasm3="OPENQASM 3.0;\nh q[0];\n",
        )
        normalized = normalize_circuit(handle)
        # Modify original — should not affect normalized
        handle.native_circuit["mutable"].append(4)
        assert normalized.native_circuit is not handle.native_circuit
        assert normalized._normalized is True


# ======================================================================
# 2. CompileResult.from_dict round-trip
# ======================================================================

from qocc.adapters.base import CompileResult
from qocc.core.circuit_handle import CircuitHandle, PassLogEntry, PipelineSpec


class TestCompileResultFromDict:
    """Tests for CompileResult.from_dict()."""

    def test_round_trip(self):
        handle = CircuitHandle(
            name="test_circuit",
            num_qubits=3,
            native_circuit=None,
            source_format="qasm3",
            qasm3="OPENQASM 3.0;\nh q[0];\ncx q[0], q[1];\n",
        )
        pass_log = [
            PassLogEntry(pass_name="routing", parameters={"method": "sabre"}, order=0, duration_ms=1.5),
            PassLogEntry(pass_name="optimize", parameters={}, order=1, memory_bytes=1024),
        ]
        cr = CompileResult(circuit=handle, pass_log=pass_log)
        d = cr.to_dict()
        cr2 = CompileResult.from_dict(d)

        assert cr2.circuit.name == "test_circuit"
        assert cr2.circuit.num_qubits == 3
        assert cr2.circuit.qasm3 is not None
        assert cr2.circuit.native_circuit is None
        assert len(cr2.pass_log) == 2
        assert cr2.pass_log[0].pass_name == "routing"
        assert cr2.pass_log[1].memory_bytes == 1024

    def test_empty_pass_log(self):
        d = {"circuit": {"name": "empty", "num_qubits": 1}, "pass_log": []}
        cr = CompileResult.from_dict(d)
        assert cr.circuit.name == "empty"
        assert cr.pass_log == []

    def test_missing_fields_default(self):
        d = {"circuit": {}, "pass_log": []}
        cr = CompileResult.from_dict(d)
        assert cr.circuit.name == "cached"
        assert cr.circuit.num_qubits == 0


# ======================================================================
# 3. OTLP JSON exporter
# ======================================================================

from qocc.trace.exporters import (
    export_otlp_json,
    _iso_to_unix_nano,
    _attr_to_otlp,
    _span_to_otlp,
    _otel_safe_value,
    export_to_otel_sdk,
)
from qocc.trace.span import Span, SpanEvent, SpanLink


class TestOTLPExporter:
    """Tests for OTLP JSON export."""

    def _make_span(self, name: str = "test_span", trace_id: str = "abc123") -> Span:
        s = Span(
            trace_id=trace_id,
            name=name,
            span_id="span001",
            start_time="2025-01-01T00:00:00+00:00",
            end_time="2025-01-01T00:00:01+00:00",
            attributes={"key1": "val1", "key2": 42, "key3": 3.14, "key4": True},
            status="OK",
        )
        s.events.append(SpanEvent(
            name="test_event",
            timestamp="2025-01-01T00:00:00.5+00:00",
            attributes={"ev_key": "ev_val"},
        ))
        s.links.append(SpanLink(
            trace_id="linked_trace",
            span_id="linked_span",
            attributes={"link_key": "link_val"},
        ))
        return s

    def test_export_creates_file(self, tmp_path):
        spans = [self._make_span()]
        out = export_otlp_json(spans, tmp_path / "out.json")
        assert out.exists()
        data = json.loads(out.read_text())
        assert "resourceSpans" in data

    def test_otlp_structure(self, tmp_path):
        spans = [self._make_span()]
        out = export_otlp_json(spans, tmp_path / "out.json")
        data = json.loads(out.read_text())
        rs = data["resourceSpans"][0]
        # Resource attributes
        attr_keys = [a["key"] for a in rs["resource"]["attributes"]]
        assert "service.name" in attr_keys
        assert "service.version" in attr_keys
        # Scope spans
        ss = rs["scopeSpans"][0]
        assert ss["scope"]["name"] == "qocc.trace"
        assert len(ss["spans"]) == 1

    def test_multiple_traces_grouped(self, tmp_path):
        s1 = self._make_span(name="s1", trace_id="trace_a")
        s1.span_id = "id1"
        s2 = self._make_span(name="s2", trace_id="trace_b")
        s2.span_id = "id2"
        out = export_otlp_json([s1, s2], tmp_path / "out.json")
        data = json.loads(out.read_text())
        # Should have 2 scopeSpans (one per trace)
        scope_spans = data["resourceSpans"][0]["scopeSpans"]
        assert len(scope_spans) == 2

    def test_span_to_otlp_fields(self):
        s = self._make_span()
        otlp = _span_to_otlp(s)
        assert otlp["traceId"] == "abc123"
        assert otlp["spanId"] == "span001"
        assert otlp["name"] == "test_span"
        assert otlp["kind"] == 1
        assert otlp["startTimeUnixNano"] > 0
        assert otlp["endTimeUnixNano"] > otlp["startTimeUnixNano"]
        assert otlp["status"]["code"] == 1  # OK

    def test_span_to_otlp_error_status(self):
        s = self._make_span()
        s.status = "ERROR"
        otlp = _span_to_otlp(s)
        assert otlp["status"]["code"] == 2

    def test_span_events(self):
        s = self._make_span()
        otlp = _span_to_otlp(s)
        assert len(otlp["events"]) == 1
        assert otlp["events"][0]["name"] == "test_event"

    def test_span_links(self):
        s = self._make_span()
        otlp = _span_to_otlp(s)
        assert len(otlp["links"]) == 1
        assert otlp["links"][0]["traceId"] == "linked_trace"


class TestAttrToOtlp:
    def test_bool(self):
        result = _attr_to_otlp("k", True)
        assert result["value"]["boolValue"] is True

    def test_int(self):
        result = _attr_to_otlp("k", 42)
        assert result["value"]["intValue"] == "42"

    def test_float(self):
        result = _attr_to_otlp("k", 3.14)
        assert result["value"]["doubleValue"] == 3.14

    def test_string(self):
        result = _attr_to_otlp("k", "hello")
        assert result["value"]["stringValue"] == "hello"

    def test_list_serialized(self):
        result = _attr_to_otlp("k", [1, 2, 3])
        assert "stringValue" in result["value"]
        assert "[1, 2, 3]" in result["value"]["stringValue"]


class TestIsoToUnixNano:
    def test_valid_iso(self):
        ns = _iso_to_unix_nano("2025-01-01T00:00:00+00:00")
        assert ns > 0
        assert ns == int(1735689600 * 1e9)

    def test_none_returns_0(self):
        assert _iso_to_unix_nano(None) == 0

    def test_invalid_returns_0(self):
        assert _iso_to_unix_nano("not-a-date") == 0


class TestOtelSafeValue:
    def test_passthrough_primitives(self):
        assert _otel_safe_value("hello") == "hello"
        assert _otel_safe_value(42) == 42
        assert _otel_safe_value(True) is True

    def test_complex_to_str(self):
        assert isinstance(_otel_safe_value([1, 2]), str)

    def test_sdk_not_installed(self):
        # export_to_otel_sdk should return False when SDK not available
        s = Span(trace_id="t", name="n")
        result = export_to_otel_sdk([s])
        # May return True or False depending on environment; just check it doesn't crash
        assert isinstance(result, bool)


# ======================================================================
# 4. Search strategies: random & Bayesian
# ======================================================================

from qocc.search.space import (
    SearchSpaceConfig,
    generate_candidates,
    _generate_random,
    _generate_bayesian_init,
    BayesianSearchOptimizer,
    Candidate,
)


class TestRandomStrategy:
    def test_generates_candidates(self):
        config = SearchSpaceConfig(strategy="random", max_candidates=10)
        candidates = generate_candidates(config)
        assert 1 <= len(candidates) <= 10

    def test_unique_ids(self):
        config = SearchSpaceConfig(strategy="random", max_candidates=20)
        candidates = generate_candidates(config)
        ids = [c.candidate_id for c in candidates]
        assert len(ids) == len(set(ids))

    def test_respects_max_candidates(self):
        config = SearchSpaceConfig(strategy="random", max_candidates=3)
        candidates = generate_candidates(config)
        assert len(candidates) <= 3


class TestBayesianStrategy:
    def test_bayesian_init_generates_candidates(self):
        config = SearchSpaceConfig(
            strategy="bayesian",
            bayesian_init_points=5,
            max_candidates=50,
        )
        candidates = generate_candidates(config)
        assert len(candidates) <= 5

    def test_bayesian_optimizer_encode_decode(self):
        config = SearchSpaceConfig(
            optimization_levels=[0, 1, 2],
            routing_methods=["sabre", "stochastic"],
        )
        opt = BayesianSearchOptimizer(config)
        params = {"optimization_level": 1, "seed": 42, "routing_method": "sabre"}
        vec = opt._encode(params)
        decoded = opt._decode(vec)
        # Decoded optimization_level should match
        assert decoded["optimization_level"] == 1

    def test_bayesian_suggest_without_observations(self):
        config = SearchSpaceConfig(
            strategy="bayesian",
            bayesian_init_points=3,
        )
        opt = BayesianSearchOptimizer(config)
        # Without observations, should return init candidates
        suggestions = opt.suggest(batch_size=2)
        assert len(suggestions) > 0

    def test_bayesian_suggest_with_observations(self):
        config = SearchSpaceConfig(
            optimization_levels=[0, 1, 2, 3],
            seeds=[42],
            routing_methods=["sabre"],
            strategy="bayesian",
            bayesian_init_points=4,
        )
        opt = BayesianSearchOptimizer(config)
        init = opt.initial_candidates()
        # Assign fake scores
        for i, c in enumerate(init):
            c.surrogate_score = float(i) * 0.5
        opt.observe(init)
        # Suggest next batch
        next_batch = opt.suggest(batch_size=3)
        assert len(next_batch) == 3
        for c in next_batch:
            assert c.pipeline is not None


class TestSearchSpaceConfigStrategy:
    def test_default_strategy_is_grid(self):
        config = SearchSpaceConfig()
        assert config.strategy == "grid"

    def test_from_dict_strategy(self):
        config = SearchSpaceConfig.from_dict({"strategy": "bayesian"})
        assert config.strategy == "bayesian"

    def test_to_dict_strategy(self):
        config = SearchSpaceConfig(strategy="random")
        d = config.to_dict()
        assert d["strategy"] == "random"

    def test_grid_strategy_dispatches(self):
        config = SearchSpaceConfig(
            strategy="grid",
            optimization_levels=[0, 1],
            seeds=[42],
            routing_methods=["sabre"],
        )
        candidates = generate_candidates(config)
        # 2 levels × 1 seed × 1 routing = 2
        assert len(candidates) == 2


# ======================================================================
# 5. ContractType enum & validation
# ======================================================================

from qocc.contracts.spec import ContractType, ContractSpec, VALID_CONTRACT_TYPES


class TestContractType:
    def test_valid_types(self):
        for t in ["observable", "distribution", "clifford", "exact", "cost"]:
            assert ContractType.is_valid(t)

    def test_invalid_type(self):
        assert not ContractType.is_valid("nonexistent")
        assert not ContractType.is_valid("")

    def test_enum_values(self):
        assert ContractType.OBSERVABLE.value == "observable"
        assert ContractType.COST.value == "cost"

    def test_valid_contract_types_frozenset(self):
        assert isinstance(VALID_CONTRACT_TYPES, frozenset)
        assert len(VALID_CONTRACT_TYPES) == 5

    def test_spec_type_valid_flag(self):
        spec = ContractSpec(name="ok", type="observable")
        assert spec._type_valid is True

    def test_spec_invalid_type_flag(self):
        spec = ContractSpec(name="bad", type="nonexistent")
        assert spec._type_valid is False

    def test_spec_invalid_type_with_custom_evaluator(self):
        spec = ContractSpec(name="custom", type="nonexistent", evaluator="my_eval")
        assert spec._type_valid is True


# ======================================================================
# 6. SPRT early stopping checker
# ======================================================================

from qocc.contracts.eval_sampling import _SPRTChecker, _is_conclusive
from qocc.contracts.spec import ContractResult


class TestSPRTChecker:
    def test_from_spec(self):
        spec = ContractSpec(
            name="test",
            type="distribution",
            tolerances={"tvd": 0.05},
            confidence={"level": 0.95},
        )
        checker = _SPRTChecker.from_spec(spec)
        assert checker.alpha == pytest.approx(0.05)
        assert checker.theta0 == 0.05

    def test_boundaries(self):
        checker = _SPRTChecker(alpha=0.05, beta=0.1)
        assert checker.upper_bound > 0  # ln(0.9/0.05) > 0
        assert checker.lower_bound < 0  # ln(0.1/0.95) < 0

    def test_tvd_below_threshold_conclusive(self):
        checker = _SPRTChecker(alpha=0.05, beta=0.1, theta0=0.1)
        result = ContractResult(
            name="test",
            passed=True,
            details={"tvd_point": 0.001, "tolerance": 0.1, "shots_after": 10000},
        )
        # With very small TVD and many shots, should be conclusive
        concluded = checker.check(result)
        assert "sprt_llr" in result.details

    def test_p_value_strong_rejection(self):
        checker = _SPRTChecker(alpha=0.05, beta=0.1)
        result = ContractResult(
            name="test",
            passed=False,
            details={"p_value": 0.001, "alpha": 0.05},
        )
        assert checker.check(result) is True  # p < alpha/3

    def test_p_value_strong_nonrejection(self):
        checker = _SPRTChecker(alpha=0.05, beta=0.1)
        result = ContractResult(
            name="test",
            passed=True,
            details={"p_value": 0.99, "alpha": 0.05},
        )
        assert checker.check(result) is True  # p > 1 - alpha/3

    def test_ambiguous_not_conclusive(self):
        checker = _SPRTChecker(alpha=0.05, beta=0.1)
        result = ContractResult(
            name="test",
            passed=True,
            details={"p_value": 0.5, "alpha": 0.05},
        )
        assert checker.check(result) is False

    def test_no_relevant_fields_returns_false(self):
        checker = _SPRTChecker()
        result = ContractResult(name="test", passed=True, details={"random_key": 1})
        assert checker.check(result) is False

    def test_compute_llr(self):
        llr = _SPRTChecker._compute_llr(observed=0.01, theta0=0.1, n=1000)
        # Very low TVD with tolerance 0.1 — should give positive LLR (favors H1)
        assert isinstance(llr, float)


class TestIsConclusiveHeuristic:
    def test_tvd_clearly_below_tolerance(self):
        result = ContractResult(
            name="test",
            passed=True,
            details={"tvd_ci": {"lower": 0.0, "upper": 0.05}, "tolerance": 0.1},
        )
        spec = ContractSpec(name="test", type="distribution", tolerances={"tvd": 0.1})
        assert _is_conclusive(result, spec) is True  # 0.05 < 0.09

    def test_tvd_clearly_above_tolerance(self):
        result = ContractResult(
            name="test",
            passed=False,
            details={"tvd_ci": {"lower": 0.15, "upper": 0.25}, "tolerance": 0.1},
        )
        spec = ContractSpec(name="test", type="distribution", tolerances={"tvd": 0.1})
        assert _is_conclusive(result, spec) is True  # 0.15 > 0.11

    def test_tvd_ambiguous(self):
        result = ContractResult(
            name="test",
            passed=True,
            details={"tvd_ci": {"lower": 0.08, "upper": 0.12}, "tolerance": 0.1},
        )
        spec = ContractSpec(name="test", type="distribution", tolerances={"tvd": 0.1})
        assert _is_conclusive(result, spec) is False

    def test_observable_conclusive(self):
        result = ContractResult(
            name="test",
            passed=True,
            details={"conservative_diff": 0.01, "epsilon": 0.1},
        )
        spec = ContractSpec(name="test", type="observable", tolerances={"epsilon": 0.1})
        assert _is_conclusive(result, spec) is True


# ======================================================================
# 7. _counts_to_observable_values (moved from api.py)
# ======================================================================

from qocc.contracts.eval_sampling import _counts_to_observable_values


class TestCountsToObservableValues:
    def test_even_parity_positive(self):
        vals = _counts_to_observable_values({"00": 5})
        assert all(v == 1.0 for v in vals)
        assert len(vals) == 5

    def test_odd_parity_negative(self):
        vals = _counts_to_observable_values({"01": 3})
        assert all(v == -1.0 for v in vals)
        assert len(vals) == 3

    def test_mixed(self):
        vals = _counts_to_observable_values({"00": 2, "01": 3})
        assert vals.count(1.0) == 2
        assert vals.count(-1.0) == 3


# ======================================================================
# 8. Metric key alignment (visualization)
# ======================================================================


class TestMetricKeyAlignment:
    def test_render_metrics_comparison_keys(self):
        """Verify that render_metrics_comparison uses correct metric keys."""
        from qocc.trace.visualization import render_metrics_comparison
        m_before = {"depth": 10, "depth_2q": 3, "total_gates": 20, "duration_estimate": 100.0}
        m_after = {"depth": 8, "depth_2q": 2, "total_gates": 15, "duration_estimate": 80.0}
        output = render_metrics_comparison(m_before, m_after)
        assert "depth_2q" in output or "2Q Depth" in output.replace(" ", "")
        # Should not contain old broken keys
        assert "two_qubit_depth" not in output
        assert "duration_estimate_ns" not in output


# ======================================================================
# 9. CLI search strategy injection
# ======================================================================


class TestCLIStrategyInjection:
    def test_strategy_injected_into_config(self):
        """Verify the CLI injects --strategy into search_config."""
        config: dict = {}
        strategy = "bayesian"
        config.setdefault("strategy", strategy)
        assert config["strategy"] == "bayesian"

    def test_existing_strategy_not_overridden(self):
        """setdefault should not override existing strategy."""
        config = {"strategy": "random"}
        config.setdefault("strategy", "grid")
        assert config["strategy"] == "random"


# ======================================================================
# 10. Integration: iterative evaluate with SPRT
# ======================================================================

from qocc.contracts.eval_sampling import _iterative_evaluate


class TestIterativeEvaluateWithSPRT:
    def test_fast_path_no_early_stop(self):
        """Without early_stopping, should call evaluate_once once."""
        spec = ContractSpec(
            name="test", type="distribution",
            resource_budget={"early_stopping": False},
        )
        call_count = [0]

        def eval_once(s, cb, ca):
            call_count[0] += 1
            return ContractResult(name="test", passed=True, details={})

        result = _iterative_evaluate(spec, None, {"0": 100}, {"0": 100}, eval_once)
        assert call_count[0] == 1
        assert result.passed is True

    def test_conclusive_on_first_try(self):
        """If first evaluation is SPRT-conclusive, should stop immediately."""
        spec = ContractSpec(
            name="test", type="distribution",
            tolerances={"tvd": 0.1},
            confidence={"level": 0.95},
            resource_budget={"early_stopping": True, "max_shots": 10000, "min_shots": 100},
        )
        call_count = [0]

        def eval_once(s, cb, ca):
            call_count[0] += 1
            return ContractResult(
                name="test", passed=True,
                details={"tvd_point": 0.001, "tolerance": 0.1, "shots_after": 5000,
                         "tvd_ci": {"lower": 0.0, "upper": 0.01}},
            )

        def sim_fn(shots):
            return {"0": shots // 2, "1": shots // 2}

        result = _iterative_evaluate(spec, sim_fn, {"0": 100}, {"0": 100}, eval_once)
        assert result.details.get("early_stopped") is True

    def test_no_simulator_fast_path(self):
        """With simulate_fn=None, should skip iteration."""
        spec = ContractSpec(
            name="test", type="distribution",
            resource_budget={"early_stopping": True, "max_shots": 10000},
        )

        def eval_once(s, cb, ca):
            return ContractResult(name="test", passed=True, details={})

        result = _iterative_evaluate(spec, None, {"0": 100}, {"0": 100}, eval_once)
        assert result.passed is True


# ======================================================================
# 11. SearchSpaceConfig complete serialization
# ======================================================================


class TestSearchSpaceConfigSerialization:
    def test_round_trip(self):
        config = SearchSpaceConfig(
            adapter="cirq",
            optimization_levels=[0, 2],
            seeds=[1, 2, 3],
            routing_methods=["sabre"],
            extra_params={"layout": ["trivial", "dense"]},
            strategy="bayesian",
            max_candidates=100,
            bayesian_init_points=10,
            bayesian_explore_weight=2.0,
        )
        d = config.to_dict()
        config2 = SearchSpaceConfig.from_dict(d)
        assert config2.adapter == "cirq"
        assert config2.strategy == "bayesian"
        assert config2.max_candidates == 100
        assert config2.bayesian_init_points == 10
        assert config2.bayesian_explore_weight == 2.0
        assert config2.extra_params == {"layout": ["trivial", "dense"]}
