"""Phase 10 tests: regression database and CLI integration."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner


def _write_minimal_bundle(root: Path, run_id: str = "run1", adapter: str = "qiskit") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "circuits").mkdir(parents=True, exist_ok=True)

    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "created_at": "2026-02-26T00:00:00Z",
                "run_id": run_id,
                "adapter": adapter,
            }
        ),
        encoding="utf-8",
    )
    (root / "env.json").write_text(json.dumps({"os": "x", "python": "3.11"}), encoding="utf-8")
    (root / "seeds.json").write_text(json.dumps({"global_seed": 1, "rng_algorithm": "PCG64"}), encoding="utf-8")
    (root / "metrics.json").write_text(
        json.dumps({"input": {"depth": 8}, "compiled": {"depth": 10, "gates_2q": 6}}),
        encoding="utf-8",
    )
    (root / "contracts.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "contract_results.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "trace.jsonl").write_text(
        json.dumps({"trace_id": "t", "span_id": "s", "name": "n", "start_time": "x", "status": "OK"}) + "\n",
        encoding="utf-8",
    )
    (root / "circuits" / "input.qasm").write_text("OPENQASM 3.0; qubit[1] q; h q[0];", encoding="utf-8")
    return root


def test_regression_db_ingest_and_query(tmp_path: Path) -> None:
    from qocc.core.regression_db import RegressionDatabase

    bundle = _write_minimal_bundle(tmp_path / "bundleA", run_id="rA")
    db = RegressionDatabase(tmp_path / "regression.db")

    ingest = db.ingest(bundle)
    assert ingest["rows_ingested"] == 1
    assert ingest["run_id"] == "rA"

    rows = db.query(adapter="qiskit")
    assert len(rows) == 1
    assert rows[0]["run_id"] == "rA"
    assert rows[0]["metrics"]["depth"] == 10


def test_regression_db_ingest_search_rankings_rows(tmp_path: Path) -> None:
    from qocc.core.regression_db import RegressionDatabase

    bundle = _write_minimal_bundle(tmp_path / "bundleB", run_id="rB")
    (bundle / "search_rankings.json").write_text(
        json.dumps(
            [
                {"candidate_id": "c1", "surrogate_score": 1.23, "metrics": {"depth": 10}, "contract_results": []},
                {"candidate_id": "c2", "surrogate_score": 0.95, "metrics": {"depth": 9}, "contract_results": []},
            ]
        ),
        encoding="utf-8",
    )

    db = RegressionDatabase(tmp_path / "regression.db")
    ingest = db.ingest(bundle)
    assert ingest["rows_ingested"] == 2

    rows = db.query()
    assert {r["candidate_id"] for r in rows} == {"c1", "c2"}


def test_regression_db_tag_and_detect_regression(tmp_path: Path) -> None:
    from qocc.core.regression_db import RegressionDatabase

    baseline = _write_minimal_bundle(tmp_path / "baseline", run_id="baseline")
    current = _write_minimal_bundle(tmp_path / "current", run_id="current")
    (current / "metrics.json").write_text(
        json.dumps({"input": {"depth": 8}, "compiled": {"depth": 20, "gates_2q": 12}}),
        encoding="utf-8",
    )

    db = RegressionDatabase(tmp_path / "regression.db")
    db.ingest(baseline)
    db.tag(baseline, "baseline")

    regressions = db.detect_regressions(current, baseline_tag="baseline", delta_threshold=0.1)
    assert regressions["has_regressions"] is True
    assert regressions["regressions"]


def test_db_cli_commands(tmp_path: Path) -> None:
    from qocc.cli.main import cli

    bundle = _write_minimal_bundle(tmp_path / "bundleC", run_id="rC")
    db_path = tmp_path / "qocc.db"
    runner = CliRunner()

    result_ingest = runner.invoke(cli, ["db", "ingest", str(bundle), "--db-path", str(db_path)])
    assert result_ingest.exit_code == 0
    assert "Ingested bundle" in result_ingest.output

    result_tag = runner.invoke(cli, ["db", "tag", str(bundle), "--tag", "baseline", "--db-path", str(db_path)])
    assert result_tag.exit_code == 0
    assert "Tagged run" in result_tag.output

    result_query = runner.invoke(cli, ["db", "query", "--adapter", "qiskit", "--db-path", str(db_path)])
    assert result_query.exit_code == 0
    assert "Regression Query" in result_query.output


def test_trace_run_db_ingest_flag(tmp_path: Path, monkeypatch) -> None:
    from qocc.cli.commands_trace import trace_run

    db_path = tmp_path / "trace_regression.db"
    out_zip = tmp_path / "out.zip"
    input_qasm = tmp_path / "in.qasm"
    input_qasm.write_text("OPENQASM 3.0;", encoding="utf-8")

    fake_bundle_root = _write_minimal_bundle(tmp_path / "fake_bundle", run_id="trace1")

    def _fake_run_trace(*args, **kwargs):
        return {
            "bundle_zip": str(fake_bundle_root),
            "run_id": "trace1",
            "input_hash": "a" * 64,
            "compiled_hash": "b" * 64,
            "num_spans": 4,
            "metrics_before": {"depth": 1},
            "metrics_after": {"depth": 2},
        }

    monkeypatch.setattr("qocc.api.run_trace", _fake_run_trace)

    runner = CliRunner()
    result = runner.invoke(
        trace_run,
        [
            "--adapter",
            "qiskit",
            "--input",
            str(input_qasm),
            "--out",
            str(out_zip),
            "--db",
            "--db-path",
            str(db_path),
        ],
    )
    assert result.exit_code == 0
    assert "Ingested into regression DB" in result.output
