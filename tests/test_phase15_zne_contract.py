"""Phase 15.2 tests for ZNE contract evaluation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qocc.adapters.base import BaseAdapter, CompileResult, MetricsSnapshot, SimulationResult
from qocc.core.circuit_handle import BackendInfo, CircuitHandle, PipelineSpec


class _ZNEFakeAdapter(BaseAdapter):
    def name(self) -> str:
        return "zne-fake"

    def ingest(self, source: str | object) -> CircuitHandle:
        text = str(source)
        role = "input" if "input" in text.lower() else "compiled"
        return CircuitHandle(
            name=role,
            num_qubits=2,
            native_circuit={"role": role},
            source_format="qasm3",
            qasm3=f"OPENQASM 3.0; // {role}",
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
            native_circuit={"role": "compiled"},
            source_format="qasm3",
            qasm3="OPENQASM 3.0; // compiled",
        )
        return CompileResult(circuit=compiled, pass_log=[])

    def simulate(self, circuit: CircuitHandle, spec: object) -> SimulationResult:
        shots = int(getattr(spec, "shots", 200))
        noise_scale = float(getattr(spec, "extra", {}).get("noise_scale", 1.0))
        role = (circuit.native_circuit or {}).get("role", "compiled")

        if role == "input":
            p1 = 0.1
        else:
            # degraded with scale factor to make extrapolation useful
            p1 = min(0.5, 0.1 + 0.08 * max(0.0, noise_scale - 1.0))

        ones = int(round(shots * p1))
        zeros = shots - ones
        return SimulationResult(
            counts={"0": zeros, "1": ones},
            shots=shots,
            seed=getattr(spec, "seed", None),
            metadata={},
        )

    def get_metrics(self, circuit: CircuitHandle) -> MetricsSnapshot:
        return MetricsSnapshot({"depth": 10, "gates_2q": 2})

    def hash(self, circuit: CircuitHandle) -> str:
        return circuit.stable_hash()

    def describe_backend(self) -> BackendInfo:
        return BackendInfo(name="zne-fake", version="1")


def test_check_contract_zne_pass(monkeypatch: Any, tmp_path: Path) -> None:
    from qocc.api import check_contract

    monkeypatch.setattr("qocc.api.get_adapter", lambda _name: _ZNEFakeAdapter())

    root = tmp_path / "bundle"
    (root / "circuits").mkdir(parents=True, exist_ok=True)
    (root / "circuits" / "input.qasm").write_text("OPENQASM 3.0; // input", encoding="utf-8")
    (root / "circuits" / "selected.qasm").write_text("OPENQASM 3.0; // compiled", encoding="utf-8")

    bundle = {
        "manifest": {"adapter": "zne-fake"},
        "metrics": {"input": {}, "compiled": {}},
        "_root": str(root),
    }

    contracts = [
        {
            "name": "zne_ok",
            "type": "zne",
            "spec": {"noise_scale_factors": [1.0, 1.5, 2.0, 2.5]},
            "tolerances": {"abs_error": 0.3},
        }
    ]

    out = check_contract(bundle, contracts, adapter_name="zne-fake", simulation_shots=300)
    assert out[0]["details"]["type"] == "zne"
    assert "extrapolation_coefficients" in out[0]["details"]
    assert len(out[0]["details"]["per_level"]) == 4
    assert out[0]["passed"] is True


def test_search_compile_zne_emits_noise_level_spans(monkeypatch: Any, tmp_path: Path) -> None:
    from qocc.api import search_compile
    from qocc.core.artifacts import ArtifactStore

    monkeypatch.setattr("qocc.api.get_adapter", lambda _name: _ZNEFakeAdapter())

    unique_seed = abs(hash(str(tmp_path))) % 1_000_000 + 1000
    contract_name = f"zne-{unique_seed}"

    result = search_compile(
        adapter_name="zne-fake",
        input_source="OPENQASM 3.0; // input",
        search_config={
            "adapter": "zne-fake",
            "optimization_levels": [1],
            "seeds": [42],
            "routing_methods": ["sabre"],
        },
        contracts=[
            {
                "name": contract_name,
                "type": "zne",
                "spec": {"noise_scale_factors": [1.0, 1.5, 2.0]},
                "tolerances": {"abs_error": 0.4},
            }
        ],
        output=str(tmp_path / "zne_search.zip"),
        top_k=1,
        simulation_shots=200,
        simulation_seed=unique_seed,
    )

    assert result["top_rankings"][0]["contract_results"][0]["details"]["type"] == "zne"

    bundle = ArtifactStore.load_bundle(result["bundle_zip"])
    spans = bundle.get("trace", [])
    zne_spans = [s for s in spans if s.get("name") == "zne/noise_level"]
    assert len(zne_spans) >= 3
