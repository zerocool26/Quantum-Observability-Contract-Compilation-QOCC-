"""Phase 12.1 tests for evolutionary search strategy."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qocc.adapters.base import BaseAdapter, CompileResult, MetricsSnapshot, SimulationResult
from qocc.core.artifacts import ArtifactStore
from qocc.core.circuit_handle import BackendInfo, CircuitHandle, PipelineSpec
from qocc.search.evolutionary import EvolutionaryOptimizer, population_diversity
from qocc.search.space import SearchSpaceConfig, generate_candidates


class _FakeAdapter(BaseAdapter):
    def name(self) -> str:
        return "fake"

    def ingest(self, source: str | object) -> CircuitHandle:
        return CircuitHandle(
            name="input",
            num_qubits=2,
            native_circuit={"source": str(source)},
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
        seed = int(pipeline.parameters.get("seed", 0))
        opt = int(pipeline.optimization_level)
        routing = str(pipeline.parameters.get("routing_method", "none"))
        qasm = f"OPENQASM 3.0; // o{opt}-s{seed}-r{routing}"
        handle = CircuitHandle(
            name=f"c-{opt}-{seed}-{routing}",
            num_qubits=2,
            native_circuit={"opt": opt, "seed": seed, "routing": routing},
            source_format="qasm3",
            qasm3=qasm,
        )
        return CompileResult(circuit=handle, pass_log=[])

    def simulate(self, circuit: CircuitHandle, spec: object) -> SimulationResult:
        shots = int(getattr(spec, "shots", 32))
        return SimulationResult(counts={"00": shots}, shots=shots, seed=getattr(spec, "seed", None), metadata={})

    def get_metrics(self, circuit: CircuitHandle) -> MetricsSnapshot:
        src = circuit.native_circuit if isinstance(circuit.native_circuit, dict) else {}
        opt = int(src.get("opt", 1))
        seed = int(src.get("seed", 0))
        routing = str(src.get("routing", "stochastic"))
        routing_penalty = 3 if routing == "stochastic" else 1
        depth = 100 - (20 * opt) + routing_penalty + (seed % 3)
        gates_2q = 30 - (5 * opt) + routing_penalty
        return MetricsSnapshot(
            {
                "depth": float(depth),
                "gates_2q": float(gates_2q),
                "duration": float(depth * 10),
                "proxy_error": float(gates_2q) * 0.01,
                "proxy_error_score": float(gates_2q) * 0.01,
            }
        )

    def hash(self, circuit: CircuitHandle) -> str:
        return circuit.stable_hash()

    def describe_backend(self) -> BackendInfo:
        return BackendInfo(name="fake", version="phase12")


def test_generate_candidates_evolutionary_initial_population() -> None:
    cfg = SearchSpaceConfig(strategy="evolutionary", evolutionary_population_size=7)
    candidates = generate_candidates(cfg)
    assert 1 <= len(candidates) <= 7


def test_evolutionary_optimizer_next_generation() -> None:
    cfg = SearchSpaceConfig(
        strategy="evolutionary",
        optimization_levels=[0, 1, 2, 3],
        seeds=[7, 8],
        routing_methods=["sabre", "stochastic"],
        evolutionary_population_size=8,
    )
    opt = EvolutionaryOptimizer(cfg)
    pop = opt.initial_population()
    for i, c in enumerate(pop):
        c.surrogate_score = float(i)
    nxt = opt.next_generation(pop)
    assert len(nxt) == 8
    assert 0.0 <= population_diversity(nxt) <= 1.0


def test_search_compile_evolutionary_emits_generation_spans(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from qocc.api import search_compile

    monkeypatch.setattr("qocc.api.get_adapter", lambda _name: _FakeAdapter())

    result = search_compile(
        adapter_name="fake",
        input_source="OPENQASM 3.0;",
        search_config={
            "adapter": "fake",
            "strategy": "evolutionary",
            "optimization_levels": [0, 1, 2, 3],
            "seeds": [11, 12, 13],
            "routing_methods": ["sabre", "stochastic"],
            "evolutionary_population_size": 6,
            "evolutionary_max_generations": 4,
            "evolutionary_elitism": 2,
            "evolutionary_tournament_size": 3,
            "evolutionary_mutation_rate": 0.4,
            "evolutionary_crossover_rate": 0.8,
            "evolutionary_convergence_std": 0.0,
        },
        output=str(tmp_path / "evo.zip"),
        top_k=3,
        simulation_shots=64,
    )

    assert result["num_candidates"] >= 1
    assert result["num_validated"] >= 1
    assert result["selected"] is not None

    bundle = ArtifactStore.load_bundle(result["bundle_zip"])
    spans = bundle.get("trace", [])
    gens = [s for s in spans if s.get("name") == "evolutionary_generation"]
    assert gens
    assert all("generation" in s.get("attributes", {}) for s in gens)
    assert all("best_score" in s.get("attributes", {}) for s in gens)
    assert all("population_diversity" in s.get("attributes", {}) for s in gens)

    evo = [s for s in spans if s.get("name") == "evolutionary_search"]
    assert evo
    assert "termination_reason" in evo[0].get("attributes", {})


def test_search_compile_evolutionary_wall_clock_termination(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    from qocc.api import search_compile

    monkeypatch.setattr("qocc.api.get_adapter", lambda _name: _FakeAdapter())

    result = search_compile(
        adapter_name="fake",
        input_source="OPENQASM 3.0;",
        search_config={
            "adapter": "fake",
            "strategy": "evolutionary",
            "optimization_levels": [0, 1, 2],
            "seeds": [1, 2, 3],
            "routing_methods": ["sabre", "stochastic"],
            "evolutionary_population_size": 5,
            "evolutionary_max_generations": 20,
            "evolutionary_wall_clock_s": 0.0001,
        },
        output=str(tmp_path / "evo2.zip"),
        top_k=2,
    )

    bundle = ArtifactStore.load_bundle(result["bundle_zip"])
    evo = [s for s in bundle.get("trace", []) if s.get("name") == "evolutionary_search"]
    assert evo
    reason = str(evo[0].get("attributes", {}).get("termination_reason", ""))
    assert reason in {"wall_clock_budget", "convergence", "max_generations", "no_population"}
