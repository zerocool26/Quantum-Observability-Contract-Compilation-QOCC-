"""Tests for the search_compile API and search components."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qocc.search.space import SearchSpaceConfig, Candidate, generate_candidates
from qocc.search.scorer import surrogate_score, rank_candidates
from qocc.search.selector import select_best, SelectionResult


def test_generate_candidates_default():
    """Default config generates 4 opt_levels × 1 seed × 2 routing = 8 candidates."""
    config = SearchSpaceConfig()
    candidates = generate_candidates(config)
    assert len(candidates) == 8


def test_generate_candidates_custom():
    """Custom config generates correct Cartesian product."""
    config = SearchSpaceConfig(
        adapter="qiskit",
        optimization_levels=[1, 2],
        seeds=[42, 99],
        routing_methods=["sabre"],
    )
    candidates = generate_candidates(config)
    assert len(candidates) == 4  # 2 levels × 2 seeds × 1 routing


def test_candidate_ids_unique():
    """Each candidate has a unique ID."""
    config = SearchSpaceConfig()
    candidates = generate_candidates(config)
    ids = [c.candidate_id for c in candidates]
    assert len(ids) == len(set(ids))


def test_surrogate_score_basic():
    """Score is computed from weighted metrics."""
    metrics = {"depth": 10, "gates_2q": 5, "duration": 1000, "proxy_error": 0.1}
    result = surrogate_score(metrics)
    assert result["score"] > 0
    assert "breakdown" in result


def test_surrogate_score_custom_weights():
    """Custom weights change the score."""
    metrics = {"depth": 10, "gates_2q": 5}
    w1 = {"depth": 1.0, "gates_2q": 0.0}
    w2 = {"depth": 0.0, "gates_2q": 1.0}

    r1 = surrogate_score(metrics, weights=w1)
    r2 = surrogate_score(metrics, weights=w2)

    assert r1["score"] == 10.0
    assert r2["score"] == 5.0


def test_rank_candidates():
    """Candidates are sorted by score ascending."""
    from qocc.core.circuit_handle import PipelineSpec

    c1 = Candidate(
        pipeline=PipelineSpec(adapter="test", optimization_level=1),
        metrics={"depth": 100},
    )
    c2 = Candidate(
        pipeline=PipelineSpec(adapter="test", optimization_level=2),
        metrics={"depth": 10},
    )

    ranked = rank_candidates([c1, c2], weights={"depth": 1.0})
    assert ranked[0].surrogate_score < ranked[1].surrogate_score


def test_select_best_with_passing():
    """Selection picks the candidate with lowest score among those passing contracts."""
    from qocc.core.circuit_handle import PipelineSpec

    c1 = Candidate(
        pipeline=PipelineSpec(adapter="test", optimization_level=1),
        candidate_id="c1",
        surrogate_score=50.0,
        validated=True,
        contract_results=[{"name": "test", "passed": True}],
    )
    c2 = Candidate(
        pipeline=PipelineSpec(adapter="test", optimization_level=2),
        candidate_id="c2",
        surrogate_score=10.0,
        validated=True,
        contract_results=[{"name": "test", "passed": True}],
    )

    result = select_best([c1, c2])
    assert result.selected is not None
    assert result.selected.candidate_id == "c2"
    assert result.feasible is True


def test_select_best_infeasible():
    """Infeasible selection when all contract results fail."""
    from qocc.core.circuit_handle import PipelineSpec

    c1 = Candidate(
        pipeline=PipelineSpec(adapter="test"),
        candidate_id="c1",
        surrogate_score=10.0,
        validated=True,
        contract_results=[{"name": "test", "passed": False}],
    )

    result = select_best([c1])
    assert result.feasible is False
    assert result.selected is not None  # best-effort


def test_select_best_no_candidates():
    """No candidates → infeasible."""
    result = select_best([])
    assert result.feasible is False
    assert result.selected is None


def test_selection_result_serialization():
    """SelectionResult serializes to dict."""
    result = SelectionResult(feasible=True, reason="test")
    d = result.to_dict()
    assert d["feasible"] is True
    assert d["reason"] == "test"


def test_search_space_config_roundtrip():
    """Config serializes and deserializes correctly."""
    config = SearchSpaceConfig(
        adapter="cirq",
        optimization_levels=[0, 1, 2],
        seeds=[42, 100],
        routing_methods=["greedy"],
    )
    d = config.to_dict()
    config2 = SearchSpaceConfig.from_dict(d)
    assert config2.adapter == "cirq"
    assert config2.optimization_levels == [0, 1, 2]
    assert config2.seeds == [42, 100]


def test_search_compile_distribution_contract_uses_candidate_simulation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """search_compile should evaluate distribution contracts from real candidate simulations."""
    from qocc.adapters.base import (
        BaseAdapter,
        CompileResult,
        MetricsSnapshot,
        SimulationResult,
    )
    from qocc.api import search_compile
    from qocc.core.circuit_handle import BackendInfo, CircuitHandle, PipelineSpec

    class _FakeAdapter(BaseAdapter):
        def name(self) -> str:
            return "fake"

        def ingest(self, source: str | object) -> CircuitHandle:
            text = str(source)
            if text.startswith("compiled-"):
                qasm = text
            elif text == "normalized":
                qasm = "normalized"
            else:
                qasm = "input"
            return CircuitHandle(
                name=qasm,
                num_qubits=2,
                native_circuit={"qasm": qasm},
                source_format="qasm3",
                qasm3=qasm,
            )

        def normalize(self, circuit: CircuitHandle) -> CircuitHandle:
            return CircuitHandle(
                name="normalized",
                num_qubits=2,
                native_circuit={"qasm": "normalized"},
                source_format="qasm3",
                qasm3="normalized",
            )

        def export(self, circuit: CircuitHandle, fmt: str = "qasm3") -> str:
            return circuit.qasm3 or ""

        def compile(
            self,
            circuit: CircuitHandle,
            pipeline: PipelineSpec,
            emitter: object | None = None,
        ) -> CompileResult:
            seed = pipeline.parameters.get("seed", 0)
            qasm = f"compiled-{seed}"
            handle = CircuitHandle(
                name=qasm,
                num_qubits=2,
                native_circuit={"qasm": qasm},
                source_format="qasm3",
                qasm3=qasm,
            )
            return CompileResult(circuit=handle, pass_log=[])

        def simulate(self, circuit: CircuitHandle, spec: object) -> SimulationResult:
            shots = getattr(spec, "shots", 1024)
            seed = getattr(spec, "seed", None)
            if circuit.qasm3 == "normalized":
                counts = {"00": shots}
            else:
                counts = {"11": shots}
            return SimulationResult(
                counts=counts,
                shots=shots,
                seed=seed,
                metadata={"statevector": [1.0, 0.0, 0.0, 0.0]},
            )

        def get_metrics(self, circuit: CircuitHandle) -> MetricsSnapshot:
            return MetricsSnapshot({"depth": 1, "gates_2q": 0, "duration": 0.0, "proxy_error": 0.0})

        def hash(self, circuit: CircuitHandle) -> str:
            return circuit.stable_hash()

        def describe_backend(self) -> BackendInfo:
            return BackendInfo(name="fake", version="candidate-contract-test")

    monkeypatch.setattr("qocc.api.get_adapter", lambda _: _FakeAdapter())

    result = search_compile(
        adapter_name="fake",
        input_source="OPENQASM 3.0;",
        search_config={
            "adapter": "fake",
            "optimization_levels": [1],
            "seeds": [7],
            "routing_methods": ["fake"],
        },
        contracts=[
            {"name": "dist", "type": "distribution", "tolerances": {"tvd": 0.1}},
        ],
        output=str(tmp_path / "search.zip"),
        top_k=1,
        simulation_shots=32,
    )

    contract = result["top_rankings"][0]["contract_results"][0]
    assert contract["passed"] is False
    assert "tvd_point" in contract["details"]
    assert "error" not in contract["details"]
