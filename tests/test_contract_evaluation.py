"""Tests for contract evaluation pipeline (wired check_contract)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from qocc.api import check_contract, _evaluate_cost_contract
from qocc.contracts.eval_sampling import _counts_to_observable_values
from qocc.contracts.spec import ContractSpec


def _make_bundle(metrics: dict | None = None, root: str | None = None) -> dict:
    """Create a minimal fake bundle dict."""
    return {
        "manifest": {"run_id": "test123", "adapter": "qiskit"},
        "metrics": metrics or {"input": {}, "compiled": {}},
        "seeds": {"global_seed": 42},
        "_root": root,
    }


def test_cost_contract_passes():
    """Cost contract passes when metrics are within limits."""
    spec = ContractSpec(
        name="depth_limit",
        type="cost",
        tolerances={"max_depth": 50, "max_gates_2q": 100},
    )
    compiled_metrics = {"depth": 30, "gates_2q": 40}

    result = _evaluate_cost_contract(spec, compiled_metrics)
    assert result.passed is True
    assert len(result.details["violations"]) == 0


def test_cost_contract_fails():
    """Cost contract fails when metrics exceed limits."""
    spec = ContractSpec(
        name="depth_limit",
        type="cost",
        tolerances={"max_depth": 10, "max_gates_2q": 5},
    )
    compiled_metrics = {"depth": 30, "gates_2q": 40}

    result = _evaluate_cost_contract(spec, compiled_metrics)
    assert result.passed is False
    assert len(result.details["violations"]) == 2


def test_cost_contract_partial_checks():
    """Cost contract only checks limits that are specified."""
    spec = ContractSpec(
        name="partial",
        type="cost",
        tolerances={"max_depth": 100},
    )
    compiled_metrics = {"depth": 50, "gates_2q": 1000}  # gates_2q has no limit

    result = _evaluate_cost_contract(spec, compiled_metrics)
    assert result.passed is True


def test_counts_to_observable_values():
    """Bitstring counts convert to ±1 parity values."""
    counts = {"00": 3, "01": 2, "11": 1}
    values = _counts_to_observable_values(counts)

    # "00" → parity 0 → +1 (3 times)
    # "01" → parity 1 → -1 (2 times)
    # "11" → parity 0 → +1 (1 time)
    assert values.count(1.0) == 4
    assert values.count(-1.0) == 2
    assert len(values) == 6


def test_check_contract_cost_type():
    """check_contract dispatches cost contracts correctly."""
    bundle = _make_bundle(metrics={
        "input": {"depth": 5},
        "compiled": {"depth": 30, "gates_2q": 20},
    })
    contracts = [
        {
            "name": "depth_ok",
            "type": "cost",
            "tolerances": {"max_depth": 50},
        }
    ]
    results = check_contract(bundle, contracts)
    assert len(results) == 1
    assert results[0]["passed"] is True


def test_check_contract_cost_failure():
    """Cost contract fails when over budget."""
    bundle = _make_bundle(metrics={
        "input": {"depth": 5},
        "compiled": {"depth": 100, "gates_2q": 20},
    })
    contracts = [
        {
            "name": "depth_tight",
            "type": "cost",
            "tolerances": {"max_depth": 10},
        }
    ]
    results = check_contract(bundle, contracts)
    assert results[0]["passed"] is False


def test_check_contract_unknown_type():
    """Unknown contract type returns error."""
    bundle = _make_bundle()
    contracts = [{"name": "x", "type": "nonexistent"}]
    results = check_contract(bundle, contracts)
    assert results[0]["passed"] is False
    assert "Unknown contract type" in results[0]["details"]["error"]


def test_check_contract_distribution_no_sim():
    """Distribution contract without simulation data returns failure."""
    bundle = _make_bundle()
    contracts = [
        {"name": "dist_test", "type": "distribution", "tolerances": {"tvd": 0.1}}
    ]
    results = check_contract(bundle, contracts)
    assert results[0]["passed"] is False
    assert "simulation" in results[0]["details"]["error"].lower() or \
           "No simulation" in results[0]["details"]["error"]


def test_check_contract_multiple():
    """Multiple contracts are evaluated independently."""
    bundle = _make_bundle(metrics={
        "input": {},
        "compiled": {"depth": 50, "gates_2q": 10},
    })
    contracts = [
        {"name": "pass_test", "type": "cost", "tolerances": {"max_depth": 100}},
        {"name": "fail_test", "type": "cost", "tolerances": {"max_depth": 10}},
    ]
    results = check_contract(bundle, contracts)
    assert len(results) == 2
    assert results[0]["passed"] is True
    assert results[1]["passed"] is False
