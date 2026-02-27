"""Phase 13 tests: interactive HTML trace report export."""

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
        json.dumps(
            {
                "input": {"depth": 8, "gate_histogram": {"h": 1, "cx": 1}},
                "compiled": {"depth": 6, "gates_2q": 2, "gate_histogram": {"h": 1, "cx": 2}},
            }
        ),
        encoding="utf-8",
    )
    (root / "contracts.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "contract_results.json").write_text(
        json.dumps([
            {"name": "dist", "passed": True, "details": {"confidence_interval": [0.01, 0.03]}}
        ]),
        encoding="utf-8",
    )
    (root / "trace.jsonl").write_text(
        json.dumps(
            {
                "trace_id": "t",
                "span_id": "s",
                "name": "adapter/compile",
                "start_time": "2026-02-26T00:00:00Z",
                "end_time": "2026-02-26T00:00:01Z",
                "parent_span_id": None,
                "attributes": {"k": "v"},
                "events": [],
                "links": [],
                "status": "OK",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return root


def test_export_html_report_creates_sections(tmp_path: Path) -> None:
    from qocc.trace.html_report import export_html_report

    bundle = _write_minimal_bundle(tmp_path / "bundleA", run_id="rA")
    out = tmp_path / "report.html"

    written = export_html_report(str(bundle), str(out))
    assert written.exists()

    html = written.read_text(encoding="utf-8")
    assert "Flame Chart" in html
    assert "Metric Dashboard" in html
    assert "Contract Results" in html
    assert "Circuit Diff" in html


def test_trace_html_cli_command(tmp_path: Path) -> None:
    from qocc.cli.main import cli

    bundle = _write_minimal_bundle(tmp_path / "bundleB", run_id="rB")
    out = tmp_path / "cli_report.html"

    runner = CliRunner()
    result = runner.invoke(
        cli,
        ["trace", "html", "--bundle", str(bundle), "--out", str(out)],
    )

    assert result.exit_code == 0
    assert "HTML report created" in result.output
    assert out.exists()


def test_trace_run_html_flag(tmp_path: Path, monkeypatch) -> None:
    from qocc.cli.commands_trace import trace_run

    out_zip = tmp_path / "out.zip"
    in_qasm = tmp_path / "input.qasm"
    in_qasm.write_text("OPENQASM 3.0;", encoding="utf-8")
    fake_bundle = _write_minimal_bundle(tmp_path / "bundleC", run_id="rC")

    def _fake_run_trace(*args, **kwargs):
        return {
            "bundle_zip": str(fake_bundle),
            "run_id": "rC",
            "input_hash": "a" * 64,
            "compiled_hash": "b" * 64,
            "num_spans": 1,
            "metrics_before": {"depth": 8},
            "metrics_after": {"depth": 6},
        }

    monkeypatch.setattr("qocc.api.run_trace", _fake_run_trace)

    called = {"count": 0}

    def _fake_export(bundle_path: str, output_path: str, compare_bundle_path: str | None = None):
        called["count"] += 1
        p = Path(output_path)
        p.write_text("<html></html>", encoding="utf-8")
        return p

    monkeypatch.setattr("qocc.trace.html_report.export_html_report", _fake_export)

    runner = CliRunner()
    result = runner.invoke(
        trace_run,
        [
            "--adapter",
            "qiskit",
            "--input",
            str(in_qasm),
            "--out",
            str(out_zip),
            "--html",
        ],
    )

    assert result.exit_code == 0
    assert called["count"] == 1
    assert "HTML report created" in result.output
