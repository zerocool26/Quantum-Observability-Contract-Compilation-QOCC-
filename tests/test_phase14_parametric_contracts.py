"""Phase 14 tests: parametric contract resolution at evaluation time."""

from __future__ import annotations

import json
from pathlib import Path


def _write_bundle(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "circuits").mkdir(parents=True, exist_ok=True)

    (root / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "0.1.0",
                "created_at": "2026-02-26T00:00:00Z",
                "run_id": "parametric",
                "adapter": "qiskit",
            }
        ),
        encoding="utf-8",
    )
    (root / "env.json").write_text(json.dumps({"os": "x", "python": "3.11"}), encoding="utf-8")
    (root / "seeds.json").write_text(json.dumps({"global_seed": 1, "rng_algorithm": "PCG64"}), encoding="utf-8")
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "input": {"depth": 12, "proxy_error_score": 0.0},
                "compiled": {"depth": 8, "gates_2q": 3, "proxy_error_score": 0.75},
                "baseline": {"tvd": 0.5},
            }
        ),
        encoding="utf-8",
    )
    (root / "contracts.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "contract_results.json").write_text(json.dumps([]), encoding="utf-8")
    (root / "trace.jsonl").write_text(
        json.dumps({"trace_id": "t", "span_id": "s", "name": "n", "start_time": "x", "status": "OK"}) + "\n",
        encoding="utf-8",
    )
    return root


def test_check_contract_resolves_parametric_budget_expression(tmp_path: Path) -> None:
    from qocc.api import check_contract

    bundle = _write_bundle(tmp_path / "bundleA")
    contracts = [
        {
            "name": "depth_budget",
            "type": "cost",
            "resource_budget": {"max_depth": "input_depth - 2"},
        }
    ]

    results = check_contract(str(bundle), contracts)
    assert len(results) == 1
    assert results[0]["passed"] is True
    assert results[0]["details"]["checks"]["depth"]["limit"] == 10.0


def test_check_contract_resolves_symbolic_reference(tmp_path: Path) -> None:
    from qocc.api import check_contract

    bundle = _write_bundle(tmp_path / "bundleB")
    contracts = [
        {
            "name": "proxy_budget",
            "type": "cost",
            "spec": {"error_budget": 0.2},
            "tolerances": {"max_proxy_error": "1 - error_budget"},
        }
    ]

    results = check_contract(str(bundle), contracts)
    assert results[0]["passed"] is True
    assert results[0]["details"]["checks"]["proxy_error"]["limit"] == 0.8


def test_check_contract_unresolved_symbol_fails_contract(tmp_path: Path) -> None:
    from qocc.api import check_contract

    bundle = _write_bundle(tmp_path / "bundleC")
    contracts = [
        {
            "name": "bad_symbol",
            "type": "cost",
            "resource_budget": {"max_depth": "missing_symbol + 1"},
        }
    ]

    results = check_contract(str(bundle), contracts)
    assert results[0]["passed"] is False
    assert "Failed to resolve" in results[0]["details"]["error"]


def test_dsl_parser_accepts_expression_rhs() -> None:
    from qocc.contracts.dsl import parse_contract_dsl

    specs = parse_contract_dsl(
        """
contract symbolic:
  type: cost
  assert: depth <= compiled_depth + 10
  assert: proxy_error_score <= 1 - error_budget
  tolerance: tvd <= 0.1 * baseline_tvd
"""
    )

    assert len(specs) == 1
    s = specs[0]
    assert s.resource_budget["max_depth"] == "compiled_depth + 10"
    assert s.tolerances["max_proxy_error"] == "1 - error_budget"
    assert s.tolerances["tvd"] == "0.1 * baseline_tvd"
