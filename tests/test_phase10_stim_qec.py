"""Phase 10 tests: Stim adapter + QEC contracts + QEC schema wiring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest


class _FakeInstruction:
    def __init__(self, name: str, targets: list[int]) -> None:
        self.name = name
        self._targets = targets

    def targets_copy(self) -> list[int]:
        return list(self._targets)

    def __str__(self) -> str:
        return f"{self.name} {' '.join(str(t) for t in self._targets)}"


class _FakeDem:
    def __str__(self) -> str:
        return "error(0.001) D0"


class _FakeSampler:
    def sample(self, shots: int) -> list[list[int]]:
        return [[0, 0], [1, 0], [1, 1], [0, 0]][:shots] if shots <= 4 else [[0, 0]] * shots


class _FakeStimCircuit:
    def __init__(self, text: str = "H 0\nCX 0 1") -> None:
        self.text = text
        self.num_qubits = 2
        self._instructions = [_FakeInstruction("H", [0]), _FakeInstruction("CX", [0, 1])]

    def __str__(self) -> str:
        return self.text

    def __iter__(self):
        return iter(self._instructions)

    def copy(self) -> "_FakeStimCircuit":
        out = _FakeStimCircuit(self.text)
        out._instructions = list(self._instructions)
        return out

    def detector_error_model(self) -> _FakeDem:
        return _FakeDem()

    def compile_sampler(self) -> _FakeSampler:
        return _FakeSampler()


class _FakeTableauSimulator:
    pass


def _make_stim_adapter(monkeypatch: pytest.MonkeyPatch):
    from qocc.adapters import stim_adapter as sa

    monkeypatch.setattr(
        sa,
        "_import_stim",
        lambda: {
            "module": object(),
            "version": "1.15.0",
            "Circuit": _FakeStimCircuit,
            "TableauSimulator": _FakeTableauSimulator,
        },
    )
    return sa.StimAdapter()


def test_stim_ingest_native(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_stim_adapter(monkeypatch)
    h = adapter.ingest(_FakeStimCircuit())
    assert h.source_format == "stim"
    assert h.metadata["framework"] == "stim"


def test_stim_ingest_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    adapter = _make_stim_adapter(monkeypatch)
    p = tmp_path / "x.stim"
    p.write_text("H 0", encoding="utf-8")
    h = adapter.ingest(str(p))
    assert "stim_text" in h.metadata


def test_stim_ingest_raw(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_stim_adapter(monkeypatch)
    h = adapter.ingest("H 0")
    assert h.num_qubits >= 0


def test_stim_ingest_bad_type(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_stim_adapter(monkeypatch)
    with pytest.raises(TypeError):
        adapter.ingest(123)


def test_stim_normalize_sets_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_stim_adapter(monkeypatch)
    h = adapter.ingest(_FakeStimCircuit())
    n = adapter.normalize(h)
    assert n._normalized is True


def test_stim_export_text(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_stim_adapter(monkeypatch)
    h = adapter.ingest(_FakeStimCircuit("M 0"))
    out = adapter.export(h, "stim")
    assert "M 0" in out


def test_stim_export_bad_format(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_stim_adapter(monkeypatch)
    h = adapter.ingest(_FakeStimCircuit())
    with pytest.raises(ValueError):
        adapter.export(h, "json")


def test_stim_compile_adds_dem(monkeypatch: pytest.MonkeyPatch) -> None:
    from qocc.core.circuit_handle import PipelineSpec

    adapter = _make_stim_adapter(monkeypatch)
    h = adapter.ingest(_FakeStimCircuit())
    result = adapter.compile(h, PipelineSpec(adapter="stim"))
    assert "dem" in result.circuit.metadata
    assert len(result.pass_log) >= 2


def test_stim_compile_emits_spans(monkeypatch: pytest.MonkeyPatch) -> None:
    from qocc.core.circuit_handle import PipelineSpec
    from qocc.trace.emitter import TraceEmitter

    adapter = _make_stim_adapter(monkeypatch)
    h = adapter.ingest(_FakeStimCircuit())
    emitter = TraceEmitter()
    _ = adapter.compile(h, PipelineSpec(adapter="stim"), emitter=emitter)
    names = [s.name for s in emitter.finished_spans()]
    assert "pass/stim.detector_error_model" in names


def test_stim_simulate_shots(monkeypatch: pytest.MonkeyPatch) -> None:
    from qocc.adapters.base import SimulationSpec

    adapter = _make_stim_adapter(monkeypatch)
    h = adapter.ingest(_FakeStimCircuit())
    res = adapter.simulate(h, SimulationSpec(shots=4))
    assert res.shots == 4
    assert "logical_error_rate" in res.metadata
    assert "syndrome_weight_distribution" in res.metadata


def test_stim_simulate_statevector_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    from qocc.adapters.base import SimulationSpec

    adapter = _make_stim_adapter(monkeypatch)
    h = adapter.ingest(_FakeStimCircuit())
    res = adapter.simulate(h, SimulationSpec(shots=0))
    assert res.shots == 0
    assert res.metadata["simulator"] == "tableau"


def test_stim_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_stim_adapter(monkeypatch)
    h = adapter.ingest(_FakeStimCircuit())
    m = adapter.get_metrics(h).to_dict()
    assert m["total_gates"] == 2
    assert m["gates_2q"] == 1


def test_stim_hash_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_stim_adapter(monkeypatch)
    h1 = adapter.ingest(_FakeStimCircuit("H 0"))
    h2 = adapter.ingest(_FakeStimCircuit("H 0"))
    assert adapter.hash(h1) == adapter.hash(h2)


def test_stim_describe_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    adapter = _make_stim_adapter(monkeypatch)
    info = adapter.describe_backend().to_dict()
    assert info["name"] == "stim"
    assert "pymatching" in info["extra"]


def test_qec_eval_logical_error_rate_pass() -> None:
    from qocc.contracts.eval_qec import evaluate_qec_contract
    from qocc.contracts.spec import ContractSpec

    spec = ContractSpec(name="qec", type="qec", tolerances={"logical_error_rate_threshold": 0.2})
    r = evaluate_qec_contract(spec, simulation_metadata={"logical_error_rate": 0.1})
    assert r.passed is True


def test_qec_eval_logical_error_rate_fail() -> None:
    from qocc.contracts.eval_qec import evaluate_qec_contract
    from qocc.contracts.spec import ContractSpec

    spec = ContractSpec(name="qec", type="qec", tolerances={"logical_error_rate_threshold": 0.05})
    r = evaluate_qec_contract(spec, simulation_metadata={"logical_error_rate": 0.1})
    assert r.passed is False


def test_qec_eval_code_distance_pass() -> None:
    from qocc.contracts.eval_qec import evaluate_qec_contract
    from qocc.contracts.spec import ContractSpec

    spec = ContractSpec(name="qec", type="qec", tolerances={"code_distance": 5})
    r = evaluate_qec_contract(spec, simulation_metadata={"code_distance": 7})
    assert r.passed is True


def test_qec_eval_syndrome_budget_from_distribution() -> None:
    from qocc.contracts.eval_qec import evaluate_qec_contract
    from qocc.contracts.spec import ContractSpec

    spec = ContractSpec(name="qec", type="qec", tolerances={"syndrome_weight_budget": 1.0})
    r = evaluate_qec_contract(
        spec,
        simulation_metadata={"syndrome_weight_distribution": {"0": 5, "1": 5}},
    )
    assert r.passed is True


def test_qec_eval_no_thresholds_errors() -> None:
    from qocc.contracts.eval_qec import evaluate_qec_contract
    from qocc.contracts.spec import ContractSpec

    spec = ContractSpec(name="qec", type="qec")
    r = evaluate_qec_contract(spec)
    assert r.passed is False
    assert "error" in r.details


def test_contract_type_includes_qec() -> None:
    from qocc.contracts.spec import ContractType

    assert ContractType.is_valid("qec") is True


def test_contracts_schema_allows_qec(tmp_path: Path) -> None:
    from qocc.cli.validation import validate_json_file

    p = tmp_path / "contracts.json"
    p.write_text(json.dumps([{"name": "q", "type": "qec"}]), encoding="utf-8")
    data = validate_json_file(str(p), "contracts", strict=True)
    assert isinstance(data, list)


def test_validate_bundle_accepts_qec_optional_files(tmp_path: Path) -> None:
    from qocc.core.schemas import validate_bundle

    (tmp_path / "manifest.json").write_text(json.dumps({"schema_version": "0.1.0", "created_at": "2025-01-01T00:00:00Z", "run_id": "x"}), encoding="utf-8")
    (tmp_path / "env.json").write_text(json.dumps({"os": "x", "python": "3.11"}), encoding="utf-8")
    (tmp_path / "seeds.json").write_text(json.dumps({"global_seed": 1, "rng_algorithm": "PCG64"}), encoding="utf-8")
    (tmp_path / "metrics.json").write_text(json.dumps({}), encoding="utf-8")
    (tmp_path / "contracts.json").write_text(json.dumps([]), encoding="utf-8")
    (tmp_path / "contract_results.json").write_text(json.dumps([]), encoding="utf-8")
    (tmp_path / "trace.jsonl").write_text(json.dumps({"trace_id": "t", "span_id": "s", "name": "n", "start_time": "x", "status": "OK"}) + "\n", encoding="utf-8")

    (tmp_path / "dem.json").write_text(json.dumps({"dem": "error(0.001) D0"}), encoding="utf-8")
    (tmp_path / "logical_error_rates.json").write_text(json.dumps({"logical_error_rate": 0.1}), encoding="utf-8")
    (tmp_path / "decoder_stats.json").write_text(json.dumps({"decoder_rounds": 1}), encoding="utf-8")

    results = validate_bundle(tmp_path)
    assert results["dem.json"] == []
    assert results["logical_error_rates.json"] == []
    assert results["decoder_stats.json"] == []


def test_stim_helper_rows_to_counts() -> None:
    from qocc.adapters.stim_adapter import _rows_to_counts

    counts = _rows_to_counts([[0, 1], [0, 1], [1, 1]])
    assert counts["01"] == 2
    assert counts["11"] == 1


def test_stim_helper_syndrome_distribution() -> None:
    from qocc.adapters.stim_adapter import _syndrome_weight_distribution

    d = _syndrome_weight_distribution([[0, 0], [1, 0], [1, 1]])
    assert d["0"] == 1
    assert d["1"] == 1
    assert d["2"] == 1


def test_stim_get_adapter_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    from qocc.adapters import stim_adapter as sa
    from qocc.adapters.base import get_adapter

    monkeypatch.setattr(
        sa,
        "_import_stim",
        lambda: {
            "module": object(),
            "version": "1.15.0",
            "Circuit": _FakeStimCircuit,
            "TableauSimulator": _FakeTableauSimulator,
        },
    )
    adapter = get_adapter("stim")
    assert adapter.name() == "stim"


def test_check_contract_qec_dispatch() -> None:
    from qocc.api import check_contract

    bundle = {
        "manifest": {"adapter": "stim"},
        "metrics": {"compiled": {}},
        "logical_error_rates": {"logical_error_rate": 0.02},
        "decoder_stats": {"code_distance": 5},
    }
    results = check_contract(bundle, [{"name": "qec", "type": "qec", "tolerances": {"logical_error_rate_threshold": 0.05, "code_distance": 3}}])
    assert results[0]["passed"] is True
