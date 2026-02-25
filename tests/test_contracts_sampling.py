"""Tests for sampling-based contract evaluation and statistics."""

from __future__ import annotations

import pytest
import numpy as np

from qocc.contracts.spec import ContractSpec, ContractResult
from qocc.contracts.stats import (
    total_variation_distance,
    tvd_bootstrap_ci,
    expectation_ci_hoeffding,
    expectation_bootstrap_ci,
)
from qocc.contracts.eval_sampling import (
    evaluate_distribution_contract,
    evaluate_observable_contract,
)


class TestTVD:
    def test_identical_distributions(self):
        counts = {"00": 500, "11": 500}
        assert total_variation_distance(counts, counts) == pytest.approx(0.0)

    def test_completely_different(self):
        a = {"00": 1000}
        b = {"11": 1000}
        assert total_variation_distance(a, b) == pytest.approx(1.0)

    def test_partial_overlap(self):
        a = {"00": 700, "11": 300}
        b = {"00": 300, "11": 700}
        tvd = total_variation_distance(a, b)
        assert 0.0 < tvd < 1.0
        assert tvd == pytest.approx(0.4)

    def test_empty_counts(self):
        assert total_variation_distance({}, {"00": 100}) == 1.0
        assert total_variation_distance({"00": 100}, {}) == 1.0


class TestTVDBootstrap:
    def test_ci_contains_point(self):
        a = {"00": 500, "11": 500}
        b = {"00": 480, "11": 520}
        ci = tvd_bootstrap_ci(a, b, confidence=0.95, seed=42)
        assert ci["lower"] <= ci["point"] <= ci["upper"]
        assert ci["confidence"] == 0.95


class TestExpectationCI:
    def test_hoeffding_ci(self):
        values = [1.0] * 500 + [-1.0] * 500
        ci = expectation_ci_hoeffding(values, confidence=0.95)
        assert abs(ci["mean"]) < 0.01
        assert ci["lower"] < ci["mean"] < ci["upper"]
        assert ci["n"] == 1000

    def test_bootstrap_ci(self):
        rng = np.random.default_rng(42)
        values = rng.normal(0.5, 0.1, size=200).tolist()
        ci = expectation_bootstrap_ci(values, confidence=0.95, seed=42)
        assert abs(ci["mean"] - 0.5) < 0.05
        assert ci["lower"] < ci["upper"]

    def test_empty_values(self):
        ci = expectation_ci_hoeffding([], confidence=0.95)
        assert ci["n"] == 0


class TestDistributionContract:
    def test_identical_passes(self):
        spec = ContractSpec(
            name="dist_test",
            type="distribution",
            tolerances={"tvd": 0.1},
            confidence={"level": 0.95},
            resource_budget={"seed": 42, "n_bootstrap": 500},
        )
        counts = {"00": 500, "11": 500}
        result = evaluate_distribution_contract(spec, counts, counts)
        assert result.passed is True

    def test_very_different_fails(self):
        spec = ContractSpec(
            name="dist_test",
            type="distribution",
            tolerances={"tvd": 0.05},
            confidence={"level": 0.95},
            resource_budget={"seed": 42, "n_bootstrap": 500},
        )
        a = {"00": 900, "11": 100}
        b = {"00": 100, "11": 900}
        result = evaluate_distribution_contract(spec, a, b)
        assert result.passed is False


class TestObservableContract:
    def test_same_values_pass(self):
        spec = ContractSpec(
            name="obs_test",
            type="observable",
            tolerances={"epsilon": 1.0},
            confidence={"level": 0.95},
        )
        values = [0.5] * 100
        result = evaluate_observable_contract(spec, values, values)
        assert result.passed is True

    def test_very_different_fails(self):
        spec = ContractSpec(
            name="obs_test",
            type="observable",
            tolerances={"epsilon": 0.01},
            confidence={"level": 0.95},
        )
        a = [1.0] * 100
        b = [-1.0] * 100
        result = evaluate_observable_contract(spec, a, b)
        assert result.passed is False
