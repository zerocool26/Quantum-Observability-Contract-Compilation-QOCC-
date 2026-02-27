"""Phase 14 tests: contract composition operators."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_bundle(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "circuits").mkdir(parents=True, exist_ok=True)

    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "created_at": "2026-02-26T00:00:00Z",
                "run_id": "compose",
                "adapter": "qiskit",
            }
        ),
        encoding="utf-8",
    )
    (root / "env.json").write_text(json.dumps({"os": "x", "python": "3.11"}), encoding="utf-8")
    (root / "seeds.json").write_text(json.dumps({"global_seed": 1, "rng_algorithm": "PCG64"}), encoding="utf-8")
    (root / "metrics.json").write_text(
        json.dumps({"input": {"depth": 12}, "compiled": {"depth": 8, "gates_2q": 4, "total_gates": 20}}),
        encoding="utf-8",
    )
    (root / "contracts.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "contract_results.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "trace.jsonl").write_text(
        json.dumps({"trace_id": "t", "span_id": "s", "name": "n", "start_time": "x", "status": "OK"}) + "\n",
        encoding="utf-8",
    )
    return root


def test_check_contract_all_of_any_of_best_effort(tmp_path: Path) -> None:
    from qocc.api import check_contract

    bundle = _write_bundle(tmp_path / "bundleA")
    contracts = [
        {
            "name": "all_gate_budgets",
            "op": "all_of",
            "contracts": [
                {"name": "depth_ok", "type": "cost", "resource_budget": {"max_depth": 10}},
                {"name": "g2q_fail", "type": "cost", "resource_budget": {"max_gates_2q": 3}},
            ],
        },
        {
            "name": "any_gate_budget",
            "op": "any_of",
            "contracts": [
                {"name": "too_strict", "type": "cost", "resource_budget": {"max_depth": 5}},
                {"name": "lenient", "type": "cost", "resource_budget": {"max_depth": 10}},
            ],
        },
        {
            "name": "best_effort_check",
            "op": "best_effort",
            "contract": {"name": "fail_inner", "type": "cost", "resource_budget": {"max_depth": 1}},
        },
    ]

    results = check_contract(str(bundle), contracts)
    by_name = {r["name"]: r for r in results}

    assert by_name["all_gate_budgets"]["passed"] is False
    assert by_name["any_gate_budget"]["passed"] is True
    assert by_name["best_effort_check"]["passed"] is True
    assert by_name["best_effort_check"]["details"]["effective_passed"] is False


def test_check_contract_with_fallback_uses_fallback_on_not_implemented(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from qocc.api import check_contract
    from qocc.contracts.registry import _EVALUATOR_REGISTRY

    bundle = _write_bundle(tmp_path / "bundleB")

    def _primary_not_impl(*args, **kwargs):
        raise NotImplementedError("primary not supported")

    monkeypatch.setitem(_EVALUATOR_REGISTRY, "primary_not_impl", _primary_not_impl)

    contracts = [
        {
            "name": "fallback_envelope",
            "op": "with_fallback",
            "primary": {"name": "primary", "type": "cost", "evaluator": "primary_not_impl"},
            "fallback": {"name": "fallback", "type": "cost", "resource_budget": {"max_depth": 10}},
        }
    ]

    results = check_contract(str(bundle), contracts)
    assert len(results) == 1
    r = results[0]
    assert r["passed"] is True
    assert r["details"]["used_fallback"] is True


def test_search_compile_supports_contract_composition(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from qocc.adapters.base import BaseAdapter, CompileResult, MetricsSnapshot, SimulationResult
    from qocc.api import search_compile
    from qocc.core.circuit_handle import BackendInfo, CircuitHandle, PipelineSpec

    class _FakeAdapter(BaseAdapter):
        def name(self) -> str:
            return "fake"

        def ingest(self, source: str | object) -> CircuitHandle:
            qasm = str(source)
            return CircuitHandle(
                name="fake",
                num_qubits=2,
                native_circuit={"qasm": qasm},
                source_format="qasm3",
                qasm3=qasm,
            )

        def normalize(self, circuit: CircuitHandle) -> CircuitHandle:
            return circuit

        def export(self, circuit: CircuitHandle, fmt: str = "qasm3") -> str:
            return circuit.qasm3 or ""

        def compile(self, circuit: CircuitHandle, pipeline: PipelineSpec, emitter: object | None = None) -> CompileResult:
            return CompileResult(circuit=circuit, pass_log=[])

        def simulate(self, circuit: CircuitHandle, spec: object) -> SimulationResult:
            return SimulationResult(counts={"00": getattr(spec, "shots", 64)}, shots=getattr(spec, "shots", 64), seed=getattr(spec, "seed", None), metadata={})

        def get_metrics(self, circuit: CircuitHandle) -> MetricsSnapshot:
            return MetricsSnapshot({"depth": 8, "gates_1q": 1, "gates_2q": 2, "duration": 100, "total_gates": 3})

        def hash(self, circuit: CircuitHandle) -> str:
            return circuit.stable_hash()

        def describe_backend(self) -> BackendInfo:
            return BackendInfo(name="fake", version="1.0")

    monkeypatch.setattr("qocc.api.get_adapter", lambda _: _FakeAdapter())

    contracts = [
        {
            "name": "composed",
            "op": "all_of",
            "contracts": [
                {"name": "depth", "type": "cost", "resource_budget": {"max_depth": 10}},
                {"name": "g2q", "type": "cost", "resource_budget": {"max_gates_2q": 3}},
            ],
        }
    ]

    result = search_compile(
        adapter_name="fake",
        input_source="OPENQASM 3.0;",
        search_config={
            "adapter": "fake",
            "optimization_levels": [1],
            "seeds": [7],
            "routing_methods": ["fake"],
        },
        contracts=contracts,
        output=str(tmp_path / "search.zip"),
        top_k=1,
        simulation_shots=64,
    )

    assert result["num_candidates"] == 1
    top = result["top_rankings"][0]
    assert top["contract_results"]
    assert top["contract_results"][0]["details"].get("op") == "all_of"
