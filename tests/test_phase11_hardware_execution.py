from __future__ import annotations

from typing import Any

import pytest

import qocc.api as api
from qocc.adapters.base import (
    BaseAdapter,
    CompileResult,
    ExecutionResult,
    MetricsSnapshot,
    SimulationResult,
    SimulationSpec,
)
from qocc.core.circuit_handle import BackendInfo, CircuitHandle, PipelineSpec
from qocc.trace.emitter import TraceEmitter


class _DummyAdapter(BaseAdapter):
    def name(self) -> str:
        return "dummy"

    def ingest(self, source: str | Any) -> CircuitHandle:
        qasm = source if isinstance(source, str) else "OPENQASM 3.0;"
        return CircuitHandle(
            name="dummy",
            num_qubits=1,
            native_circuit=None,
            source_format="dummy",
            qasm3=qasm,
        )

    def normalize(self, circuit: CircuitHandle) -> CircuitHandle:
        return circuit

    def export(self, circuit: CircuitHandle, fmt: str = "qasm3") -> str:
        return circuit.qasm3 or ""

    def compile(
        self,
        circuit: CircuitHandle,
        pipeline: PipelineSpec,
        emitter: Any | None = None,
    ) -> CompileResult:
        return CompileResult(circuit=circuit, pass_log=[])

    def simulate(self, circuit: CircuitHandle, spec: SimulationSpec) -> SimulationResult:
        return SimulationResult(counts={"0": spec.shots}, shots=spec.shots, seed=spec.seed)

    def get_metrics(self, circuit: CircuitHandle) -> MetricsSnapshot:
        return MetricsSnapshot({"depth": 1, "gates_2q": 0, "total_gates": 1})

    def hash(self, circuit: CircuitHandle) -> str:
        return circuit.stable_hash()

    def describe_backend(self) -> BackendInfo:
        return BackendInfo(name="dummy", version="0")


class _HardwareAdapter(_DummyAdapter):
    def execute(
        self,
        circuit: CircuitHandle,
        backend_spec: dict[str, Any],
        shots: int = 1024,
        emitter: TraceEmitter | None = None,
    ) -> ExecutionResult:
        if emitter is not None:
            with emitter.span("job_submit", attributes={"provider": "mock", "job_id": "job-123"}):
                pass
            with emitter.span("queue_wait", attributes={"backend_version": "v1"}) as wait_span:
                wait_span.add_event("job_polling", interval_s=backend_spec.get("poll_interval_s", 1.0))
            with emitter.span("job_complete", attributes={"basis_gates": ["cx", "rz"]}):
                pass
            with emitter.span("result_fetch", attributes={"coupling_map_hash": "abc"}):
                pass

        return ExecutionResult(
            job_id="job-123",
            backend_name="mock_backend",
            shots=shots,
            counts={"0": shots},
            metadata={"provider": "mock", "backend_version": "v1"},
            queue_time_s=0.25,
            run_time_s=0.75,
            error_mitigation_applied=False,
        )


def test_execution_result_dataclass_roundtrip() -> None:
    result = ExecutionResult(
        job_id="abc",
        backend_name="backend",
        shots=2048,
        counts={"00": 2000, "11": 48},
        metadata={"provider": "mock"},
        queue_time_s=1.2,
        run_time_s=2.3,
        error_mitigation_applied=True,
    )

    payload = result.to_dict()
    assert payload["job_id"] == "abc"
    assert payload["shots"] == 2048
    assert payload["error_mitigation_applied"] is True


def test_base_adapter_execute_default_not_implemented() -> None:
    adapter = _DummyAdapter()
    handle = adapter.ingest("OPENQASM 3.0; qubit[1] q;")

    with pytest.raises(NotImplementedError):
        adapter.execute(handle, backend_spec={"provider": "mock"})


def test_execute_emits_required_hardware_spans() -> None:
    adapter = _HardwareAdapter()
    handle = adapter.ingest("OPENQASM 3.0; qubit[1] q;")
    emitter = TraceEmitter()

    result = adapter.execute(handle, backend_spec={"poll_interval_s": 0.5}, shots=32, emitter=emitter)
    span_names = [s.name for s in emitter.finished_spans()]

    assert result.job_id == "job-123"
    assert span_names == ["job_submit", "queue_wait", "job_complete", "result_fetch"]
    queue_wait = next(s for s in emitter.finished_spans() if s.name == "queue_wait")
    assert any(ev.name == "job_polling" for ev in queue_wait.events)


def test_extract_hardware_counts_parses_supported_shapes() -> None:
    bundle = {
        "hardware": {
            "input_counts": {"00": 20},
            "result": {"counts": {"00": 18, "11": 2}},
        }
    }

    before, after = api._extract_hardware_counts(bundle)
    assert before == {"00": 20}
    assert after == {"00": 18, "11": 2}


def test_check_contract_uses_hardware_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    from qocc.contracts import eval_sampling
    from qocc.contracts.spec import ContractResult

    def _fake_distribution(spec: Any, counts_before: dict[str, int], counts_after: dict[str, int]) -> ContractResult:
        captured["before"] = counts_before
        captured["after"] = counts_after
        return ContractResult(name=spec.name, passed=True, details={"type": "distribution"})

    monkeypatch.setattr(eval_sampling, "evaluate_distribution_contract", _fake_distribution)

    bundle = {
        "metrics": {"input": {}, "compiled": {}},
        "hardware": {
            "input_counts": {"00": 12, "11": 4},
            "counts": {"00": 10, "11": 6},
        },
    }
    contracts = [{"name": "dist", "type": "distribution", "tolerances": {"tvd": 0.2}}]

    results = api.check_contract(bundle, contracts, adapter_name=None)

    assert results[0]["passed"] is True
    assert captured["before"] == {"00": 12, "11": 4}
    assert captured["after"] == {"00": 10, "11": 6}
