"""Phase 15.3 tests for mitigation pipeline stage integration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from qocc.adapters.base import BaseAdapter, CompileResult, MetricsSnapshot, SimulationResult
from qocc.core.artifacts import ArtifactStore
from qocc.core.circuit_handle import BackendInfo, CircuitHandle, MitigationSpec, PipelineSpec


class _MitigationFakeAdapter(BaseAdapter):
    def name(self) -> str:
        return "mitigation-fake"

    def ingest(self, source: str | object) -> CircuitHandle:
        return CircuitHandle(
            name="input",
            num_qubits=2,
            native_circuit={"raw": str(source)},
            source_format="qasm3",
            qasm3="OPENQASM 3.0; // input",
        )

    def normalize(self, circuit: CircuitHandle) -> CircuitHandle:
        return circuit

    def export(self, circuit: CircuitHandle, fmt: str = "qasm3") -> str:
        return circuit.qasm3 or ""

    def compile(self, circuit: CircuitHandle, pipeline: PipelineSpec, emitter: object | None = None) -> CompileResult:
        _ = emitter
        compiled = CircuitHandle(
            name="compiled",
            num_qubits=2,
            native_circuit={"compiled": True, "pipeline": pipeline.to_dict()},
            source_format="qasm3",
            qasm3="OPENQASM 3.0; // compiled",
        )
        return CompileResult(circuit=compiled, pass_log=[])

    def simulate(self, circuit: CircuitHandle, spec: Any) -> SimulationResult:
        shots = int(getattr(spec, "shots", 200))
        ones = max(1, shots // 4)
        return SimulationResult(
            counts={"0": shots - ones, "1": ones},
            shots=shots,
            seed=getattr(spec, "seed", None),
            metadata={},
        )

    def get_metrics(self, circuit: CircuitHandle) -> MetricsSnapshot:
        return MetricsSnapshot({"depth": 7, "gates_2q": 2, "duration_estimate": 11.0})

    def hash(self, circuit: CircuitHandle) -> str:
        return circuit.stable_hash()

    def describe_backend(self) -> BackendInfo:
        return BackendInfo(
            name="mitigation-fake",
            version="1",
            extra={"supported_mitigation_methods": ["twirling", "zne", "m3_readout"]},
        )


def test_mitigation_spec_serialization_roundtrip() -> None:
    spec = MitigationSpec(
        method="twirling",
        params={"shot_multiplier": 2.0, "runtime_multiplier": 1.5},
        overhead_budget={"max_runtime_multiplier": 2.0},
    )

    payload = spec.to_dict()
    rebuilt = MitigationSpec.from_dict(payload)

    assert rebuilt.method == "twirling"
    assert rebuilt.params["shot_multiplier"] == 2.0
    assert rebuilt.overhead_budget["max_runtime_multiplier"] == 2.0


def test_run_trace_mitigation_span_and_metrics(monkeypatch: Any, tmp_path: Path) -> None:
    from qocc.api import run_trace

    monkeypatch.setattr("qocc.api.get_adapter", lambda _name: _MitigationFakeAdapter())

    result = run_trace(
        adapter_name="mitigation-fake",
        input_source="OPENQASM 3.0; // input",
        pipeline={
            "adapter": "mitigation-fake",
            "optimization_level": 1,
            "mitigation": {
                "method": "twirling",
                "params": {"shot_multiplier": 2.0, "runtime_multiplier": 1.25},
                "overhead_budget": {"max_runtime_multiplier": 2.0},
            },
        },
        output=str(tmp_path / "mitig_trace.zip"),
    )

    bundle = ArtifactStore.load_bundle(result["bundle_zip"])
    compiled_metrics = bundle["metrics"]["compiled"]
    assert compiled_metrics["mitigation"]["method"] == "twirling"
    assert compiled_metrics["mitigation_shot_multiplier"] == 2.0
    assert compiled_metrics["mitigation_runtime_multiplier"] == 1.25
    assert compiled_metrics["mitigation_overhead_factor"] == 2.5

    mitigation_spans = [s for s in bundle["trace"] if s.get("name") == "mitigation"]
    assert len(mitigation_spans) >= 1
    assert mitigation_spans[0]["attributes"]["method"] == "twirling"


def test_search_compile_mitigation_metrics_and_span(monkeypatch: Any, tmp_path: Path) -> None:
    from qocc.api import search_compile

    monkeypatch.setattr("qocc.api.get_adapter", lambda _name: _MitigationFakeAdapter())

    result = search_compile(
        adapter_name="mitigation-fake",
        input_source="OPENQASM 3.0; // input",
        search_config={
            "adapter": "mitigation-fake",
            "optimization_levels": [1],
            "seeds": [7],
            "routing_methods": ["sabre"],
            "extra_params": {
                "mitigation": [{
                    "method": "zne",
                    "params": {"shot_multiplier": 1.5, "runtime_multiplier": 1.2},
                }],
            },
        },
        contracts=[],
        output=str(tmp_path / "mitig_search.zip"),
        top_k=1,
        simulation_shots=200,
        simulation_seed=123,
    )

    top = result["top_rankings"][0]
    assert top["metrics"]["mitigation"]["method"] == "zne"
    assert top["metrics"]["mitigation_overhead_factor"] == pytest.approx(1.8)

    bundle = ArtifactStore.load_bundle(result["bundle_zip"])
    mitigation_spans = [s for s in bundle.get("trace", []) if s.get("name") == "mitigation"]
    assert len(mitigation_spans) >= 1
    assert mitigation_spans[0]["attributes"]["method"] == "zne"
