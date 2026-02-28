"""Phase 12.4 tests for batch search compile mode."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from click.testing import CliRunner


def test_batch_search_compile_api_writes_batch_bundle(monkeypatch, tmp_path: Path) -> None:
    from qocc.api import batch_search_compile
    from qocc.core.artifacts import ArtifactStore

    calls: list[dict[str, Any]] = []

    def _fake_search_compile(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        cid = Path(str(kwargs["input_source"])).stem
        return {
            "run_id": f"r-{cid}",
            "bundle_dir": str(tmp_path / f"{cid}_dir"),
            "bundle_zip": str(tmp_path / f"{cid}.zip"),
            "num_candidates": 5,
            "num_validated": 3,
            "cache_hits": 2,
            "cache_misses": 3,
            "feasible": True,
            "selected": {
                "candidate_id": f"sel-{cid}",
                "surrogate_score": 1.23,
                "metrics": {"depth": 10, "gates_2q": 4},
                "pipeline": {"optimization_level": 2},
            },
            "selection_reason": "ok",
            "top_rankings": [],
        }

    monkeypatch.setattr("qocc.api.search_compile", _fake_search_compile)

    manifest = {
        "defaults": {"adapter": "qiskit", "top_k": 2, "simulation_shots": 64},
        "circuits": [
            {"id": "a", "input": str(tmp_path / "a.qasm")},
            {"id": "b", "input": str(tmp_path / "b.qasm"), "adapter": "cirq"},
        ],
    }

    result = batch_search_compile(manifest=manifest, output=str(tmp_path / "batch.zip"), workers=2)

    assert result["n_circuits"] == 2
    assert len(calls) == 2
    assert all(c["top_k"] == 2 for c in calls)

    bundle = ArtifactStore.load_bundle(result["bundle_zip"])
    root = Path(bundle["_root"])

    batch_results = json.loads((root / "batch_results.json").read_text(encoding="utf-8"))
    cross = json.loads((root / "cross_circuit_metrics.json").read_text(encoding="utf-8"))
    assert len(batch_results) == 2
    assert len(cross["rows"]) == 2

    spans = bundle.get("trace", [])
    batch_spans = [s for s in spans if s.get("name") == "batch_search"]
    assert batch_spans
    attrs = batch_spans[0].get("attributes", {})
    assert attrs.get("n_circuits") == 2
    assert attrs.get("n_cache_hits") == 4
    assert attrs.get("total_candidates_evaluated") == 10


def test_compile_batch_cli_invokes_api(monkeypatch, tmp_path: Path) -> None:
    from qocc.cli.commands_search import compile_group

    manifest = {
        "circuits": [{"id": "x", "input": str(tmp_path / "x.qasm")}],
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    called: dict[str, Any] = {}

    def _fake_batch_search_compile(manifest: str | dict[str, Any], output: str | None = None, workers: int | None = None) -> dict[str, Any]:
        called["manifest"] = manifest
        called["output"] = output
        called["workers"] = workers
        return {
            "bundle_zip": str(tmp_path / "batch.zip"),
            "cross_circuit_metrics": {
                "rows": [
                    {
                        "id": "x",
                        "status": "ok",
                        "num_candidates": 3,
                        "num_validated": 1,
                        "feasible": True,
                    }
                ]
            },
        }

    monkeypatch.setattr("qocc.api.batch_search_compile", _fake_batch_search_compile)

    runner = CliRunner()
    res = runner.invoke(
        compile_group,
        [
            "batch",
            "--manifest",
            str(manifest_path),
            "--workers",
            "3",
            "--out",
            str(tmp_path / "out.zip"),
        ],
    )

    assert res.exit_code == 0
    assert called["manifest"] == str(manifest_path)
    assert called["workers"] == 3
