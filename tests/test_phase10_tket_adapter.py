"""Phase 10 tests: production-grade pytket adapter.

These tests mock pytket-like objects so CI does not require optional tket deps.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


class _FakeOp:
    def __init__(self, name: str) -> None:
        self.name = name

    def __str__(self) -> str:
        return self.name


class _FakeCmd:
    def __init__(self, op: str, n_qubits: int = 1) -> None:
        self.op = _FakeOp(op)
        self.qubits = list(range(n_qubits))


class _FakeCircuit:
    def __init__(
        self,
        name: str = "fake",
        n_qubits: int = 2,
        commands: list[_FakeCmd] | None = None,
    ) -> None:
        self.name = name
        self.n_qubits = n_qubits
        self._commands = list(commands or [_FakeCmd("H", 1), _FakeCmd("CX", 2)])

    @classmethod
    def from_qasm(cls, qasm: str) -> "_FakeCircuit":
        return cls(name="from_qasm", commands=[_FakeCmd("QASM", 1)])

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_FakeCircuit":
        cmds = [_FakeCmd(c["op"], c.get("n_qubits", 1)) for c in data.get("commands", [])]
        return cls(name=data.get("name", "from_dict"), n_qubits=data.get("n_qubits", 2), commands=cmds)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_qubits": self.n_qubits,
            "commands": [{"op": str(c.op), "n_qubits": len(c.qubits)} for c in self._commands],
        }

    def copy(self) -> "_FakeCircuit":
        return _FakeCircuit.from_dict(self.to_dict())

    def to_qasm(self) -> str:
        return f"OPENQASM 2.0; // {self.name}"

    def get_commands(self) -> list[_FakeCmd]:
        return list(self._commands)

    def depth(self) -> int:
        return len(self._commands)

    def depth_2q(self) -> int:
        return sum(1 for c in self._commands if len(c.qubits) >= 2)


class _RemoveRedundancies:
    def apply(self, circuit: _FakeCircuit) -> None:
        if circuit._commands:
            circuit._commands = circuit._commands[:-1]


class _CommuteThroughMultis:
    def apply(self, circuit: _FakeCircuit) -> None:
        circuit._commands = list(circuit._commands)


class _SequencePass:
    def __init__(self, passes: list[Any]) -> None:
        self._passes = passes

    def apply(self, circuit: _FakeCircuit) -> None:
        for p in self._passes:
            p.apply(circuit)


def _fake_tket_dict() -> dict[str, Any]:
    return {
        "module": object(),
        "version": "1.99.0",
        "Circuit": _FakeCircuit,
        "SequencePass": _SequencePass,
        "RemoveRedundancies": _RemoveRedundancies,
        "CommuteThroughMultis": _CommuteThroughMultis,
    }


def _make_adapter(monkeypatch: pytest.MonkeyPatch):
    from qocc.adapters import tket_adapter as ta

    monkeypatch.setattr(ta, "_import_tket", lambda: _fake_tket_dict())
    return ta.TketAdapter()


def test_ingest_native_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    c = _FakeCircuit(name="native")
    h = adapter.ingest(c)
    assert h.source_format == "tket"
    assert h.num_qubits == 2


def test_ingest_qasm_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adapter = _make_adapter(monkeypatch)
    p = tmp_path / "x.qasm"
    p.write_text("OPENQASM 2.0;", encoding="utf-8")
    h = adapter.ingest(str(p))
    assert h.qasm3 is not None


def test_ingest_json_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adapter = _make_adapter(monkeypatch)
    p = tmp_path / "x.json"
    p.write_text(json.dumps(_FakeCircuit(name="j").to_dict()), encoding="utf-8")
    h = adapter.ingest(str(p))
    assert h.name in ("j", "tket_circuit", "from_dict")


def test_ingest_raw_qasm(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    h = adapter.ingest("OPENQASM 2.0;")
    assert h.qasm3 is not None


def test_ingest_raw_json(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    payload = json.dumps(_FakeCircuit(name="raw_json").to_dict())
    h = adapter.ingest(payload)
    assert h.num_qubits == 2


def test_ingest_unsupported_type(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    with pytest.raises(TypeError):
        adapter.ingest(12345)


def test_normalize_sets_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    h = adapter.ingest(_FakeCircuit())
    n = adapter.normalize(h)
    assert n._normalized is True
    assert n.metadata["normalized"] is True


@pytest.mark.parametrize("fmt", ["qasm", "qasm2", "qasm3"])
def test_export_qasm_formats(monkeypatch: pytest.MonkeyPatch, fmt: str) -> None:
    adapter = _make_adapter(monkeypatch)
    h = adapter.ingest(_FakeCircuit())
    out = adapter.export(h, fmt=fmt)
    assert "OPENQASM" in out


def test_export_json(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    h = adapter.ingest(_FakeCircuit())
    out = adapter.export(h, fmt="json")
    d = json.loads(out)
    assert "commands" in d


def test_export_invalid_format(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    h = adapter.ingest(_FakeCircuit())
    with pytest.raises(ValueError):
        adapter.export(h, fmt="invalid")


def test_compile_default_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    from qocc.core.circuit_handle import PipelineSpec

    h = adapter.ingest(_FakeCircuit())
    result = adapter.compile(h, PipelineSpec(adapter="tket"))
    assert len(result.pass_log) == 2
    assert result.pass_log[0].pass_name.startswith("tket.")


def test_compile_emits_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    from qocc.core.circuit_handle import PipelineSpec
    from qocc.trace.emitter import TraceEmitter

    h = adapter.ingest(_FakeCircuit())
    emitter = TraceEmitter()
    _ = adapter.compile(h, PipelineSpec(adapter="tket"), emitter=emitter)
    names = [s.name for s in emitter.finished_spans()]
    assert any(n.startswith("pass/tket.") for n in names)


def test_compile_fallback_sequence_when_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    from qocc.core.circuit_handle import PipelineSpec

    h = adapter.ingest(_FakeCircuit())
    spec = PipelineSpec(adapter="tket", parameters={"pass_sequence": ["DoesNotExist"]})
    result = adapter.compile(h, spec)
    assert len(result.pass_log) == 2


def test_compile_records_error(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    from qocc.adapters import tket_adapter as ta
    from qocc.core.circuit_handle import PipelineSpec

    class _BoomPass:
        def apply(self, circuit: _FakeCircuit) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(ta, "_resolve_pass_sequence", lambda _tk, _p: [_BoomPass()])
    h = adapter.ingest(_FakeCircuit())
    result = adapter.compile(h, PipelineSpec(adapter="tket"))
    assert result.pass_log[0].errors


def test_simulate_missing_backend_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    from qocc.adapters import tket_adapter as ta
    from qocc.adapters.base import SimulationSpec

    monkeypatch.setattr(ta, "_select_sim_backend", lambda _m: (None, "none"))
    h = adapter.ingest(_FakeCircuit())
    with pytest.raises(ImportError):
        adapter.simulate(h, SimulationSpec(shots=10))


def test_simulate_statevector(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    from qocc.adapters import tket_adapter as ta
    from qocc.adapters.base import SimulationSpec

    class _StateBackend:
        def get_state(self, circuit: _FakeCircuit) -> list[float]:
            return [1.0, 0.0]

    monkeypatch.setattr(ta, "_select_sim_backend", lambda _m: (_StateBackend(), "qulacs"))
    h = adapter.ingest(_FakeCircuit())
    res = adapter.simulate(h, SimulationSpec(shots=0))
    assert res.metadata["backend"] == "qulacs"
    assert "statevector" in res.metadata


def test_simulate_counts_from_get_counts(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    from qocc.adapters import tket_adapter as ta
    from qocc.adapters.base import SimulationSpec

    class _CountsBackend:
        def get_counts(self, circuit: _FakeCircuit, n_shots: int, seed: int | None = None) -> dict[Any, int]:
            return {(0, 1): n_shots}

    monkeypatch.setattr(ta, "_select_sim_backend", lambda _m: (_CountsBackend(), "qulacs"))
    h = adapter.ingest(_FakeCircuit())
    res = adapter.simulate(h, SimulationSpec(shots=7, seed=1))
    assert res.counts["01"] == 7


def test_simulate_counts_from_run_circuit(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    from qocc.adapters import tket_adapter as ta
    from qocc.adapters.base import SimulationSpec

    class _RunBackend:
        def run_circuit(self, circuit: _FakeCircuit, n_shots: int, seed: int | None = None) -> dict[str, int]:
            return {"11": n_shots}

    monkeypatch.setattr(ta, "_select_sim_backend", lambda _m: (_RunBackend(), "projectq"))
    h = adapter.ingest(_FakeCircuit())
    res = adapter.simulate(h, SimulationSpec(shots=5))
    assert res.counts["11"] == 5


def test_get_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    h = adapter.ingest(_FakeCircuit())
    m = adapter.get_metrics(h).to_dict()
    assert m["width"] == 2
    assert "gate_histogram" in m


def test_hash_is_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    h1 = adapter.ingest(_FakeCircuit(name="a"))
    h2 = adapter.ingest(_FakeCircuit(name="a"))
    assert adapter.hash(h1) == adapter.hash(h2)


def test_describe_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_adapter(monkeypatch)
    info = adapter.describe_backend().to_dict()
    assert info["name"] == "tket"
    assert "active_extension" in info["extra"]
    assert "pass_set_hash" in info["extra"]


def test_get_adapter_tket_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from qocc.adapters import tket_adapter as ta
    from qocc.adapters.base import get_adapter

    monkeypatch.setattr(ta, "_import_tket", lambda: _fake_tket_dict())
    adapter = get_adapter("tket")
    assert adapter.name() == "tket"


def test_helpers_metrics_delta() -> None:
    from qocc.adapters.tket_adapter import _metrics_delta

    delta = _metrics_delta({"depth": 5, "total_gates": 10}, {"depth": 3, "total_gates": 9})
    assert delta["depth"] == -2
    assert delta["total_gates"] == -1


def test_helpers_active_extension_none(monkeypatch: pytest.MonkeyPatch) -> None:
    from qocc.adapters import tket_adapter as ta

    monkeypatch.setattr(ta, "_select_sim_backend", lambda _m: (None, "none"))
    assert ta._active_extension_name() == "none"
