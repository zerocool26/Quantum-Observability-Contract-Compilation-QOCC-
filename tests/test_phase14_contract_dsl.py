"""Phase 14 tests: contract DSL parsing and wiring."""

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
        json.dumps({"input": {"depth": 12}, "compiled": {"depth": 8, "gates_2q": 4, "total_gates": 20}}),
        encoding="utf-8",
    )
    (root / "contracts.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "contract_results.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "trace.jsonl").write_text(
        json.dumps({"trace_id": "t", "span_id": "s", "name": "n", "start_time": "x", "status": "OK"}) + "\n",
        encoding="utf-8",
    )
    (root / "circuits" / "input.qasm").write_text("OPENQASM 3.0; qubit[1] q; h q[0];", encoding="utf-8")
    (root / "circuits" / "selected.qasm").write_text("OPENQASM 3.0; qubit[1] q; h q[0];", encoding="utf-8")
    return root


def test_parse_contract_dsl_basic() -> None:
    from qocc.contracts.dsl import parse_contract_dsl

    text = """
# QOCC Contract DSL v1
contract tvd_check:
  type: distribution
  tolerance: tvd <= 0.05
  confidence: 0.99
  shots: 4096 .. 65536

contract depth_budget:
  type: cost
  assert: depth <= 50
  assert: two_qubit_gates <= 100
"""

    specs = parse_contract_dsl(text)
    assert len(specs) == 2
    assert specs[0].name == "tvd_check"
    assert specs[0].type == "distribution"
    assert specs[0].tolerances["tvd"] == 0.05
    assert specs[0].confidence["level"] == 0.99
    assert specs[0].resource_budget["min_shots"] == 4096
    assert specs[0].resource_budget["max_shots"] == 65536

    assert specs[1].type == "cost"
    assert specs[1].resource_budget["max_depth"] == 50
    assert specs[1].resource_budget["max_gates_2q"] == 100


def test_parse_contract_dsl_qec_mapping() -> None:
    from qocc.contracts.dsl import parse_contract_dsl

    text = """
contract qec_threshold:
  type: qec
  assert: logical_error_rate <= 1e-6 @ physical_error_rate = 1e-3
  assert: code_distance >= 5
  assert: syndrome_weight_budget <= 2.0
"""
    specs = parse_contract_dsl(text)
    assert len(specs) == 1
    s = specs[0]
    assert s.tolerances["logical_error_rate_threshold"] == 1e-6
    assert s.tolerances["code_distance"] == 5
    assert s.tolerances["syndrome_weight_budget"] == 2.0
    assert s.spec.get("contexts")


def test_parse_contract_dsl_error_has_line_column() -> None:
    from qocc.contracts.dsl import ContractDSLParseError, parse_contract_dsl

    text = """
contract bad:
  type cost
"""
    try:
        parse_contract_dsl(text)
        assert False, "Expected ContractDSLParseError"
    except ContractDSLParseError as exc:
        msg = str(exc)
        assert "line" in msg
        assert "column" in msg


def test_check_contract_accepts_qocc_file(tmp_path: Path) -> None:
    from qocc.api import check_contract

    bundle = _write_minimal_bundle(tmp_path / "bundleA", run_id="rA")
    contracts = tmp_path / "contracts.qocc"
    contracts.write_text(
        """
contract depth_budget:
  type: cost
  assert: depth <= 10
""",
        encoding="utf-8",
    )

    results = check_contract(str(bundle), str(contracts))
    assert len(results) == 1
    assert results[0]["name"] == "depth_budget"
    assert results[0]["passed"] is True


def test_contract_cli_accepts_qocc_file(tmp_path: Path) -> None:
    from qocc.cli.main import cli

    bundle = _write_minimal_bundle(tmp_path / "bundleB", run_id="rB")
    contracts = tmp_path / "contracts.qocc"
    contracts.write_text(
        """
contract depth_budget:
  type: cost
  assert: depth <= 10
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["contract", "check", "--bundle", str(bundle), "--contracts", str(contracts)])
    assert result.exit_code == 0
    assert "All contracts passed" in result.output


def test_contract_cli_reports_dsl_syntax_error(tmp_path: Path) -> None:
    from qocc.cli.main import cli

    bundle = _write_minimal_bundle(tmp_path / "bundleC", run_id="rC")
    contracts = tmp_path / "bad.qocc"
    contracts.write_text(
        """
contract broken:
  type cost
""",
        encoding="utf-8",
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["contract", "check", "--bundle", str(bundle), "--contracts", str(contracts)])
    assert result.exit_code == 1
    assert "line" in result.output.lower()
    assert "column" in result.output.lower()
