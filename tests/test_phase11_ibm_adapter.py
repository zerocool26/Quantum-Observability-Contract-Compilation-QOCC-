"""Phase 11.2 tests for IBM Quantum Runtime adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class _FakeConfig:
    basis_gates: list[str]
    coupling_map: list[list[int]]


class _FakeBackend:
    backend_version = "v-test"

    def configuration(self) -> _FakeConfig:
        return _FakeConfig(basis_gates=["cx", "rz"], coupling_map=[[0, 1], [1, 2]])


class _FakeCircuit:
    def __init__(self, depth_v: int = 3, size_v: int = 7) -> None:
        self._depth = depth_v
        self._size = size_v

    def depth(self) -> int:
        return self._depth

    def size(self) -> int:
        return self._size


class _FakeResult:
    def __init__(self, counts: dict[str, int] | None = None, quasi: dict[int, float] | None = None) -> None:
        self._counts = counts
        self.quasi_dists = [quasi] if quasi is not None else None

    def get_counts(self) -> dict[str, int]:
        if self._counts is None:
            raise RuntimeError("no counts")
        return self._counts

    def to_dict(self) -> dict[str, Any]:
        return {"counts": self._counts, "quasi_dists": self.quasi_dists}


class _FakeJob:
    def __init__(self, statuses: list[str], result: _FakeResult) -> None:
        self._statuses = list(statuses)
        self._result = result
        self._idx = 0

    def job_id(self) -> str:
        return "job-ibm-123"

    def status(self) -> str:
        if self._idx >= len(self._statuses):
            return self._statuses[-1]
        value = self._statuses[self._idx]
        self._idx += 1
        return value

    def result(self) -> _FakeResult:
        return self._result


class _FakeSampler:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def run(self, circuits: list[Any], shots: int | None = None) -> _FakeJob:
        _ = circuits
        return _FakeJob(["RUNNING", "DONE"], _FakeResult(counts={"00": int(shots or 0)}))


class _FakeEstimator:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def run(self, payload: list[Any], shots: int | None = None) -> _FakeJob:
        _ = payload
        return _FakeJob(["QUEUED", "RUNNING", "COMPLETED"], _FakeResult(quasi={0: 0.8, 1: 0.2}))


class _FakeService:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    def backend(self, name: str) -> _FakeBackend:
        assert name == "ibm_fake"
        return _FakeBackend()


def _runtime_dict() -> dict[str, Any]:
    return {
        "module": object(),
        "QiskitRuntimeService": _FakeService,
        "SamplerV2": _FakeSampler,
        "EstimatorV2": _FakeEstimator,
        "Sampler": None,
        "Estimator": None,
    }


def _qiskit_dict() -> dict[str, Any]:
    class _QC:
        pass

    def _transpile(circuit: Any, **kwargs: Any) -> _FakeCircuit:
        _ = kwargs
        if isinstance(circuit, _FakeCircuit):
            return _FakeCircuit(depth_v=max(1, circuit.depth() - 1), size_v=max(1, circuit.size() - 2))
        return _FakeCircuit(depth_v=2, size_v=4)

    return {"QuantumCircuit": _QC, "transpile": _transpile}


def _make_adapter(monkeypatch: pytest.MonkeyPatch):
    from qocc.adapters import ibm_adapter as ia

    monkeypatch.setattr(ia, "_import_qiskit", lambda: _qiskit_dict())
    monkeypatch.setattr(ia, "_import_ibm_runtime", lambda: _runtime_dict())
    return ia.IBMAdapter()


def test_get_adapter_ibm_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from qocc.adapters import ibm_adapter as ia
    from qocc.adapters.base import get_adapter

    monkeypatch.setattr(ia, "_import_qiskit", lambda: _qiskit_dict())
    monkeypatch.setattr(ia, "_import_ibm_runtime", lambda: _runtime_dict())

    adapter = get_adapter("ibm")
    assert adapter.name() == "ibm"


def test_execute_sampler_emits_required_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    from qocc.core.circuit_handle import CircuitHandle
    from qocc.trace.emitter import TraceEmitter

    adapter = _make_adapter(monkeypatch)
    emitter = TraceEmitter()

    handle = CircuitHandle(
        name="c",
        num_qubits=2,
        native_circuit=_FakeCircuit(depth_v=6, size_v=12),
        source_format="qiskit",
        qasm3="OPENQASM 3.0;",
    )

    result = adapter.execute(
        handle,
        backend_spec={"backend_name": "ibm_fake", "poll_interval_s": 0.001},
        shots=64,
        emitter=emitter,
    )

    names = [s.name for s in emitter.finished_spans()]
    assert "compile/transpile_hardware" in names
    assert "job_submit" in names
    assert "queue_wait" in names
    assert "job_complete" in names
    assert "result_fetch" in names

    queue_span = next(s for s in emitter.finished_spans() if s.name == "queue_wait")
    assert any(ev.name == "job_polling" for ev in queue_span.events)

    submit_span = next(s for s in emitter.finished_spans() if s.name == "job_submit")
    assert submit_span.attributes.get("provider") == "ibm"
    assert submit_span.attributes.get("job_id") == "job-ibm-123"
    assert "basis_gates" in submit_span.attributes
    assert "coupling_map_hash" in submit_span.attributes

    assert result.job_id == "job-ibm-123"
    assert result.backend_name == "ibm_fake"
    assert result.counts["00"] == 64
    assert result.metadata["primitive"] == "sampler"
    assert isinstance(result.metadata["raw_result"], dict)


def test_execute_estimator_path_supports_quasi_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    from qocc.core.circuit_handle import CircuitHandle

    adapter = _make_adapter(monkeypatch)
    handle = CircuitHandle(
        name="c",
        num_qubits=1,
        native_circuit=_FakeCircuit(depth_v=2, size_v=3),
        source_format="qiskit",
        qasm3="OPENQASM 3.0;",
    )

    result = adapter.execute(
        handle,
        backend_spec={
            "backend_name": "ibm_fake",
            "primitive": "estimator",
            "poll_interval_s": 0.001,
        },
        shots=100,
        emitter=None,
    )

    assert result.metadata["primitive"] == "estimator"
    assert result.counts == {"0": 80, "1": 20}


def test_execute_requires_backend_name(monkeypatch: pytest.MonkeyPatch) -> None:
    from qocc.core.circuit_handle import CircuitHandle

    adapter = _make_adapter(monkeypatch)
    handle = CircuitHandle(
        name="c",
        num_qubits=1,
        native_circuit=_FakeCircuit(),
        source_format="qiskit",
        qasm3="OPENQASM 3.0;",
    )

    with pytest.raises(ValueError):
        adapter.execute(handle, backend_spec={})
