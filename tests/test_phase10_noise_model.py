"""Phase 10 tests: noise model registry and scorer integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qocc.metrics.noise_model import NoiseModel, NoiseModelRegistry
from qocc.search.scorer import rank_candidates, surrogate_score
from qocc.search.space import Candidate, SearchSpaceConfig


class TestNoiseModel:
    def test_stable_hash_order_independent(self) -> None:
        a = NoiseModel.from_dict(
            {
                "single_qubit_error": 0.001,
                "two_qubit_error": 0.01,
                "readout_error": 0.02,
                "t1": 100000,
                "t2": 80000,
            }
        )
        b = NoiseModel.from_dict(
            {
                "t2": 80000,
                "readout_error": 0.02,
                "single_qubit_error": 0.001,
                "t1": 100000,
                "two_qubit_error": 0.01,
            }
        )
        assert a.stable_hash() == b.stable_hash()

    def test_registry_builtins_available(self) -> None:
        reg = NoiseModelRegistry()
        builtins = reg.list_builtins()
        assert "uniform_depolarizing" in builtins
        assert "thermal_relaxation" in builtins
        assert "readout_error" in builtins

    def test_registry_load_from_file_validates(self, tmp_path: Path) -> None:
        reg = NoiseModelRegistry()
        fp = tmp_path / "noise.json"
        fp.write_text(
            json.dumps(
                {
                    "single_qubit_error": 0.001,
                    "two_qubit_error": 0.01,
                    "readout_error": 0.02,
                    "t1": 100000,
                    "t2": 80000,
                }
            ),
            encoding="utf-8",
        )
        model = reg.load_from_file(fp)
        assert isinstance(model, NoiseModel)
        assert model.two_qubit_error == 0.01

    def test_registry_validation_rejects_invalid(self) -> None:
        reg = NoiseModelRegistry()
        with pytest.raises(Exception):
            reg.validate({"single_qubit_error": -0.5, "two_qubit_error": 0.01, "readout_error": 0.01})


class TestNoiseAwareScoring:
    def test_surrogate_score_includes_noise_score_and_hash(self) -> None:
        metrics = {"depth": 10, "gates_1q": 20, "gates_2q": 5, "duration": 1000}
        nm = {
            "single_qubit_error": 0.001,
            "two_qubit_error": 0.01,
            "readout_error": 0.02,
            "t1": 100000,
            "t2": 80000,
        }
        r = surrogate_score(metrics, noise_model=nm)
        assert "noise_score" in r["breakdown"]
        assert r["breakdown"]["noise_score"] > 0.0
        assert "noise_model_hash" in r

    def test_rank_candidates_uses_noise_penalty(self) -> None:
        from qocc.core.circuit_handle import PipelineSpec

        c_fast = Candidate(
            pipeline=PipelineSpec(adapter="test", optimization_level=1),
            metrics={"depth": 5, "gates_1q": 5, "gates_2q": 1, "duration": 1000},
        )
        c_slow = Candidate(
            pipeline=PipelineSpec(adapter="test", optimization_level=2),
            metrics={"depth": 5, "gates_1q": 5, "gates_2q": 1, "duration": 400000},
        )
        nm = {
            "single_qubit_error": 0.0,
            "two_qubit_error": 0.0,
            "readout_error": 0.0,
            "t1": 1000,
            "t2": 1000,
        }
        ranked = rank_candidates([c_slow, c_fast], weights={"depth": 0.0, "noise_score": 1.0}, noise_model=nm)
        assert ranked[0].candidate_id == c_fast.candidate_id


class TestNoiseModelWiring:
    def test_search_space_roundtrip_noise_model(self) -> None:
        cfg = SearchSpaceConfig(noise_model={"single_qubit_error": 0.001, "two_qubit_error": 0.01, "readout_error": 0.02})
        out = SearchSpaceConfig.from_dict(cfg.to_dict())
        assert out.noise_model is not None
        assert out.noise_model["single_qubit_error"] == 0.001

    def test_cli_validation_noise_model_schema(self, tmp_path: Path) -> None:
        from qocc.cli.validation import validate_json_file

        fp = tmp_path / "nm.json"
        fp.write_text(
            json.dumps(
                {
                    "single_qubit_error": 0.001,
                    "two_qubit_error": 0.01,
                    "readout_error": 0.02,
                }
            ),
            encoding="utf-8",
        )
        data = validate_json_file(str(fp), "noise_model", strict=True)
        assert isinstance(data, dict)

    def test_search_compile_cache_key_includes_noise_hash(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        from qocc.adapters.base import BaseAdapter, CompileResult, MetricsSnapshot, SimulationResult
        from qocc.api import search_compile
        from qocc.core.cache import CompilationCache
        from qocc.core.circuit_handle import BackendInfo, CircuitHandle, PipelineSpec

        captured: dict[str, object] = {}
        original_cache_key = CompilationCache.cache_key

        def wrapped_cache_key(circuit_hash: str, pipeline_dict: dict, backend_version: str = "", extra: dict | None = None):
            captured["extra"] = dict(extra or {})
            return original_cache_key(circuit_hash, pipeline_dict, backend_version, extra)

        monkeypatch.setattr(CompilationCache, "cache_key", staticmethod(wrapped_cache_key))

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
                return MetricsSnapshot({"depth": 1, "gates_1q": 1, "gates_2q": 0, "duration": 100})

            def hash(self, circuit: CircuitHandle) -> str:
                return circuit.stable_hash()

            def describe_backend(self) -> BackendInfo:
                return BackendInfo(name="fake", version="1.0")

        monkeypatch.setattr("qocc.api.get_adapter", lambda _: _FakeAdapter())

        noise_payload = {
            "single_qubit_error": 0.001,
            "two_qubit_error": 0.01,
            "readout_error": 0.02,
        }

        result = search_compile(
            adapter_name="fake",
            input_source="OPENQASM 3.0;",
            search_config={
                "adapter": "fake",
                "optimization_levels": [1],
                "seeds": [7],
                "routing_methods": ["fake"],
                "noise_model": noise_payload,
            },
            output=str(tmp_path / "bundle.zip"),
            top_k=1,
            simulation_shots=64,
        )

        assert result["num_candidates"] == 1
        extra = captured.get("extra")
        assert isinstance(extra, dict)
        assert extra.get("noise_model_hash") == NoiseModel.from_dict(noise_payload).stable_hash()
