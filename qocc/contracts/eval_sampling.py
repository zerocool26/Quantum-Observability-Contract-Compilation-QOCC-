"""Sampling-based contract evaluation.

Evaluates observable and distribution contracts using shot-based simulation.
"""

from __future__ import annotations

from typing import Any

from qocc.contracts.spec import ContractResult, ContractSpec
from qocc.contracts.stats import (
    chi_square_test,
    expectation_ci_hoeffding,
    g_test,
    total_variation_distance,
    tvd_bootstrap_ci,
)


def evaluate_distribution_contract(
    spec: ContractSpec,
    counts_before: dict[str, int],
    counts_after: dict[str, int],
) -> ContractResult:
    """Evaluate a distribution-preservation contract via TVD.

    The contract passes if the upper CI of the TVD is below the tolerance.
    An optional ``spec.spec["test"]`` key can select ``"chi_square"`` or
    ``"g_test"`` instead of the default TVD bootstrap.
    """
    tolerance = spec.tolerances.get("tvd", 0.1)
    confidence_level = spec.confidence.get("level", 0.95)
    seed = int(spec.resource_budget.get("seed", 42))
    n_bootstrap = int(spec.resource_budget.get("n_bootstrap", 1000))

    # Choose test method
    test_method = spec.spec.get("test", "tvd")

    if test_method == "chi_square":
        alpha = 1.0 - confidence_level
        chi_result = chi_square_test(counts_before, counts_after, alpha=alpha)
        return ContractResult(
            name=spec.name,
            passed=chi_result["passed"],
            details={
                "method": "chi_square",
                "statistic": chi_result["statistic"],
                "p_value": chi_result["p_value"],
                "alpha": alpha,
            },
        )

    if test_method == "g_test":
        alpha = 1.0 - confidence_level
        g_result = g_test(counts_before, counts_after, alpha=alpha)
        return ContractResult(
            name=spec.name,
            passed=g_result["passed"],
            details={
                "method": "g_test",
                "statistic": g_result["statistic"],
                "p_value": g_result["p_value"],
                "alpha": alpha,
                "df": g_result.get("df"),
                "williams_correction": g_result.get("williams_correction"),
            },
        )

    # Default: TVD bootstrap
    tvd_point = total_variation_distance(counts_before, counts_after)
    ci = tvd_bootstrap_ci(
        counts_before,
        counts_after,
        confidence=confidence_level,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    passed = ci["upper"] <= tolerance

    return ContractResult(
        name=spec.name,
        passed=passed,
        details={
            "method": "tvd",
            "tvd_point": tvd_point,
            "tvd_ci": ci,
            "tolerance": tolerance,
            "shots_before": sum(counts_before.values()),
            "shots_after": sum(counts_after.values()),
        },
    )


def evaluate_observable_contract(
    spec: ContractSpec,
    values_before: list[float],
    values_after: list[float],
) -> ContractResult:
    """Evaluate an observable-preservation contract.

    For each: estimate expectation E, compute CI.
    Pass if |E1 - E2| <= eps considering CI bounds.
    """
    epsilon = spec.tolerances.get("epsilon", 0.05)
    confidence_level = spec.confidence.get("level", 0.95)

    ci_before = expectation_ci_hoeffding(values_before, confidence=confidence_level)
    ci_after = expectation_ci_hoeffding(values_after, confidence=confidence_level)

    diff = abs(ci_before["mean"] - ci_after["mean"])

    # Conservative bound: use CI widths
    half_width_before = (ci_before["upper"] - ci_before["lower"]) / 2
    half_width_after = (ci_after["upper"] - ci_after["lower"]) / 2
    conservative_diff = diff + half_width_before + half_width_after

    passed = conservative_diff <= epsilon

    return ContractResult(
        name=spec.name,
        passed=passed,
        details={
            "mean_before": ci_before["mean"],
            "mean_after": ci_after["mean"],
            "diff": diff,
            "conservative_diff": conservative_diff,
            "ci_before": ci_before,
            "ci_after": ci_after,
            "epsilon": epsilon,
        },
    )
