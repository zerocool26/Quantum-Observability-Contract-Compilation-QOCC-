"""Phase 12.3 tests for Bayesian transfer-learning prior support."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from qocc.adapters.base import BaseAdapter, CompileResult, MetricsSnapshot, SimulationResult
from qocc.core.artifacts import ArtifactStore
from qocc.core.circuit_handle import BackendInfo, CircuitHandle, PipelineSpec
from qocc.search.space import BayesianSearchOptimizer, Candidate, SearchSpaceConfig


class _FakeAdapter(BaseAdapter):
    def name(self) -> str:
        return "fake"

    def ingest(self, source: str | object) -> CircuitHandle:
        return CircuitHandle(
            name="input",
            num_qubits=2,
            native_circuit={"src": str(source)},
            source_format="qasm3",
            qasm3="OPENQASM 3.0;",
        )

    def normalize(self, circuit: CircuitHandle) -> CircuitHandle:
        return CircuitHandle(
            name="normalized",
            num_qubits=2,
            native_circuit={"normalized": True},
            source_format="qasm3",
            qasm3="OPENQASM 3.0; // norm",
            _normalized=True,
        )

    def export(self, circuit: CircuitHandle, fmt: str = "qasm3") -> str:
        return circuit.qasm3 or ""

    def compile(self, circuit: CircuitHandle, pipeline: PipelineSpec, emitter: object | None = None) -> CompileResult:
        _ = emitter
        opt = int(pipeline.optimization_level)
        seed = int(pipeline.parameters.get("seed", 0))
        qasm = f"OPENQASM 3.0; // o{opt}-s{seed}"
        return CompileResult(
            circuit=CircuitHandle(
                name=f"c-{opt}-{seed}",
                num_qubits=2,
                native_circuit={"opt": opt, "seed": seed},
                source_format="qasm3",
                qasm3=qasm,
            ),
            pass_log=[],
        )

    def simulate(self, circuit: CircuitHandle, spec: object) -> SimulationResult:
        shots = int(getattr(spec, "shots", 64))
        return SimulationResult(counts={"00": shots}, shots=shots, seed=getattr(spec, "seed", None), metadata={})

    def get_metrics(self, circuit: CircuitHandle) -> MetricsSnapshot:
        src = circuit.native_circuit if isinstance(circuit.native_circuit, dict) else {}
        opt = int(src.get("opt", 1))
        seed = int(src.get("seed", 0))
        depth = 120 - (20 * opt) + (seed % 4)
        g2 = 40 - (6 * opt)
        return MetricsSnapshot(
            {
                "depth": float(depth),
                "gates_2q": float(g2),
                "duration": float(depth * 10),
                "proxy_error": float(g2) * 0.01,
                "proxy_error_score": float(g2) * 0.01,
            }
        )

    def hash(self, circuit: CircuitHandle) -> str:
        return circuit.stable_hash()

    def describe_backend(self) -> BackendInfo:
        return BackendInfo(name="fake", version="v-prior")


def test_bayesian_optimizer_load_prior_weighted(tmp_path: Path) -> None:
    history = tmp_path / "search_history.json"
    now = time.time()
    payload = [
        {
            "adapter": "fake",
            "backend_version": "v1",
            "params": {"optimization_level": 1, "seed": 42, "routing_method": "sabre"},
            "score": 10.0,
            "timestamp": now - (2 * 86400),
        },
        {
            "adapter": "fake",
            "backend_version": "v1",
            "params": {"optimization_level": 2, "seed": 42, "routing_method": "stochastic"},
            "score": 12.0,
            "timestamp": now - (90 * 86400),
        },
        {
            "adapter": "other",
            "backend_version": "v1",
            "params": {"optimization_level": 0, "seed": 42, "routing_method": "sabre"},
            "score": 100.0,
            "timestamp": now,
        },
    ]
    history.write_text(json.dumps(payload), encoding="utf-8")

    cfg = SearchSpaceConfig(adapter="fake", seeds=[42], routing_methods=["sabre", "stochastic"])
    opt = BayesianSearchOptimizer(cfg, history_path=history)
    loaded = opt.load_prior("v1", half_life_days=30.0)

    assert loaded == 2
    assert len(opt._observed_Y) == 2
    assert len(opt._observed_W) == 2
    assert opt._observed_W[0] > opt._observed_W[1]


def test_bayesian_optimizer_persist_history(tmp_path: Path) -> None:
    history = tmp_path / "search_history.json"
    cfg = SearchSpaceConfig(adapter="fake", seeds=[42])
    opt = BayesianSearchOptimizer(cfg, history_path=history)

    c1 = Candidate(PipelineSpec(adapter="fake", optimization_level=1, parameters={"seed": 42}))
    c1.surrogate_score = 5.0
    c2 = Candidate(PipelineSpec(adapter="fake", optimization_level=2, parameters={"seed": 42}))
    c2.surrogate_score = 3.0
    opt.observe([c1, c2])

    appended = opt.persist_history("v-test")
    assert appended == 2
    data = json.loads(history.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert all(d["backend_version"] == "v-test" for d in data)


def test_search_compile_bayesian_prior_span_and_history(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from qocc.api import search_compile

    monkeypatch.setattr("qocc.api.get_adapter", lambda _name: _FakeAdapter())

    history = tmp_path / "history.json"
    prior = [
        {
            "adapter": "fake",
            "backend_version": "v-prior",
            "params": {"optimization_level": 3, "seed": 12, "routing_method": "sabre"},
            "score": 8.0,
            "timestamp": time.time() - 86400,
        }
    ]
    history.write_text(json.dumps(prior), encoding="utf-8")

    result = search_compile(
        adapter_name="fake",
        input_source="OPENQASM 3.0;",
        search_config={
            "adapter": "fake",
            "strategy": "bayesian",
            "optimization_levels": [0, 1, 2, 3],
            "seeds": [11, 12, 13],
            "routing_methods": ["sabre", "stochastic"],
            "max_candidates": 9,
            "bayesian_init_points": 3,
            "bayesian_prior_half_life_days": 30.0,
            "bayesian_history_path": str(history),
        },
        output=str(tmp_path / "bayes.zip"),
        top_k=3,
    )

    assert result["num_candidates"] >= 1
    bundle = ArtifactStore.load_bundle(result["bundle_zip"])
    spans = bundle.get("trace", [])
    bo = [s for s in spans if s.get("name") == "bayesian_optimizer"]
    assert bo
    attrs = bo[0].get("attributes", {})
    assert attrs.get("prior_loaded") is True
    assert int(attrs.get("prior_size", 0)) >= 1

    stored = json.loads(history.read_text(encoding="utf-8"))
    assert len(stored) >= 2
    assert any(x.get("backend_version") == "v-prior" for x in stored)
