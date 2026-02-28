"""Phase 14 tests: contract result caching behavior."""

from __future__ import annotations

import json
import time
from pathlib import Path


def _write_bundle(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "circuits").mkdir(parents=True, exist_ok=True)

    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "created_at": "2026-02-26T00:00:00Z",
                "run_id": "cache",
                "adapter": "qiskit",
            }
        ),
        encoding="utf-8",
    )
    (root / "env.json").write_text(json.dumps({"os": "x", "python": "3.11"}), encoding="utf-8")
    (root / "seeds.json").write_text(json.dumps({"global_seed": 1, "rng_algorithm": "PCG64"}), encoding="utf-8")
    (root / "metrics.json").write_text(
        json.dumps({"input": {"depth": 12}, "compiled": {"depth": 8, "gates_2q": 4}}),
        encoding="utf-8",
    )
    (root / "contracts.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "contract_results.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "trace.jsonl").write_text(
        json.dumps({"trace_id": "t", "span_id": "s", "name": "n", "start_time": "x", "status": "OK"}) + "\n",
        encoding="utf-8",
    )
    (root / "circuits" / "selected.qasm").write_text("OPENQASM 3.0; qubit[1] q; h q[0];", encoding="utf-8")
    return root


def test_contract_cache_hit_reuses_result(tmp_path: Path, monkeypatch) -> None:
    import qocc.api as api
    from qocc.core.cache import CompilationCache

    bundle = _write_bundle(tmp_path / "bundleA")

    class _TmpCache(CompilationCache):
        def __init__(self, cache_dir=None):
            super().__init__(tmp_path / "cache")

    monkeypatch.setattr(api, "CompilationCache", _TmpCache)

    calls = {"n": 0}
    original = api._evaluate_cost_contract

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(api, "_evaluate_cost_contract", _counting)

    contracts = [{"name": "depth", "type": "cost", "resource_budget": {"max_depth": 10}}]

    r1 = api.check_contract(str(bundle), contracts, simulation_shots=111, simulation_seed=7)
    r2 = api.check_contract(str(bundle), contracts, simulation_shots=111, simulation_seed=7)

    assert r1[0]["passed"] is True
    assert r2[0]["passed"] is True
    assert calls["n"] == 1


def test_contract_cache_key_includes_shots_and_seed(tmp_path: Path, monkeypatch) -> None:
    import qocc.api as api
    from qocc.core.cache import CompilationCache

    bundle = _write_bundle(tmp_path / "bundleB")

    class _TmpCache(CompilationCache):
        def __init__(self, cache_dir=None):
            super().__init__(tmp_path / "cache")

    monkeypatch.setattr(api, "CompilationCache", _TmpCache)

    calls = {"n": 0}
    original = api._evaluate_cost_contract

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(api, "_evaluate_cost_contract", _counting)

    contracts = [{"name": "depth", "type": "cost", "resource_budget": {"max_depth": 10}}]

    api.check_contract(str(bundle), contracts, simulation_shots=100, simulation_seed=1)
    api.check_contract(str(bundle), contracts, simulation_shots=101, simulation_seed=1)
    api.check_contract(str(bundle), contracts, simulation_shots=101, simulation_seed=2)

    assert calls["n"] == 3


def test_contract_cache_respects_max_cache_age(tmp_path: Path, monkeypatch) -> None:
    import qocc.api as api
    from qocc.core.cache import CompilationCache

    bundle = _write_bundle(tmp_path / "bundleC")

    class _TmpCache(CompilationCache):
        def __init__(self, cache_dir=None):
            super().__init__(tmp_path / "cache")

    monkeypatch.setattr(api, "CompilationCache", _TmpCache)

    calls = {"n": 0}
    original = api._evaluate_cost_contract

    def _counting(*args, **kwargs):
        calls["n"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(api, "_evaluate_cost_contract", _counting)

    contracts = [{"name": "depth", "type": "cost", "resource_budget": {"max_depth": 10}}]

    api.check_contract(str(bundle), contracts, simulation_shots=111, simulation_seed=7, max_cache_age_days=365)
    assert calls["n"] == 1

    # Force cache entry stale by rewriting meta timestamps
    cache_root = tmp_path / "cache"
    for entry in cache_root.iterdir():
        meta = entry / "meta.json"
        if not meta.exists():
            continue
        payload = json.loads(meta.read_text(encoding="utf-8"))
        payload["created"] = time.time() - (10 * 86400)
        meta.write_text(json.dumps(payload), encoding="utf-8")

    api.check_contract(str(bundle), contracts, simulation_shots=111, simulation_seed=7, max_cache_age_days=1)
    assert calls["n"] == 2
