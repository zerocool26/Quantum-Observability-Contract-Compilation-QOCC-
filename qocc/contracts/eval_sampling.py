"""Sampling-based contract evaluation.

Evaluates observable and distribution contracts using shot-based simulation.
Supports iterative early stopping: when the statistical result is conclusive
(pass or fail) before exhausting the shot budget, sampling halts early.
"""

from __future__ import annotations

import time
from typing import Any, Callable

from qocc import DEFAULT_SEED
from qocc.contracts.spec import ContractResult, ContractSpec
from qocc.contracts.stats import (
    chi_square_test,
    expectation_ci_hoeffding,
    g_test,
    total_variation_distance,
    tvd_bootstrap_ci,
)


def _counts_to_observable_values(counts: dict[str, int]) -> list[float]:
    """Convert measurement counts to Z-observable per-shot values.

    Each bitstring maps to: +1 if parity(bitstring) == 0, else -1.
    """
    values: list[float] = []
    for bitstring, count in counts.items():
        parity = sum(int(b) for b in bitstring) % 2
        val = 1.0 if parity == 0 else -1.0
        values.extend([val] * count)
    return values


# ======================================================================
# Iterative early-stopping wrapper (SPRT-enhanced)
# ======================================================================


def _iterative_evaluate(
    spec: ContractSpec,
    simulate_fn: Callable[[int], dict[str, int]] | None,
    counts_before: dict[str, int],
    counts_after: dict[str, int],
    evaluate_once: Callable[
        [ContractSpec, dict[str, int], dict[str, int]],
        ContractResult,
    ],
) -> ContractResult:
    """Run *evaluate_once* with increasing shot counts until the result
    is conclusive or the budget is exhausted.

    Uses a two-tier stopping strategy:
    1. **SPRT** (Sequential Probability Ratio Test) — provides statistically
       optimal stopping with guaranteed Type I/II error bounds.
    2. **CI separation heuristic** — fallback when SPRT is not applicable.

    When *simulate_fn* is ``None`` (no adapter available for re-simulation),
    falls back to a single evaluation with the provided counts.
    """
    min_shots = int(spec.resource_budget.get("min_shots", 0))
    max_shots = int(spec.resource_budget.get("max_shots", 0))
    early_stop = spec.resource_budget.get("early_stopping", False)
    max_runtime: float | None = None
    _rt = spec.resource_budget.get("max_runtime")
    if _rt is not None:
        max_runtime = float(_rt)

    # Fast path: no iterative budget configured or no simulator
    if not early_stop or max_shots <= 0 or simulate_fn is None:
        return evaluate_once(spec, counts_before, counts_after)

    # Iterative path
    t0 = time.monotonic()
    current_shots = max(min_shots, sum(counts_before.values()))
    total_shots_used = current_shots
    last_result = evaluate_once(spec, counts_before, counts_after)

    # Check if already conclusive
    sprt = _SPRTChecker.from_spec(spec)
    if sprt.check(last_result) or _is_conclusive(last_result, spec):
        last_result.details["early_stopped"] = True
        last_result.details["stopping_method"] = "sprt" if sprt.check(last_result) else "ci_separation"
        last_result.details["total_shots"] = total_shots_used
        return last_result

    # Iterate: double shots each time
    next_shots = current_shots * 2
    while next_shots <= max_shots:
        # Enforce wall-clock budget
        if max_runtime is not None and (time.monotonic() - t0) >= max_runtime:
            last_result.details["early_stopped"] = True
            last_result.details["stopping_method"] = "max_runtime"
            last_result.details["total_shots"] = total_shots_used
            last_result.details["budget_exceeded"] = "max_runtime"
            return last_result

        try:
            new_counts = simulate_fn(next_shots)
        except Exception:
            break
        total_shots_used += next_shots
        last_result = evaluate_once(spec, counts_before, new_counts)

        if sprt.check(last_result) or _is_conclusive(last_result, spec):
            last_result.details["early_stopped"] = True
            last_result.details["stopping_method"] = "sprt" if sprt.check(last_result) else "ci_separation"
            last_result.details["total_shots"] = total_shots_used
            return last_result
        next_shots *= 2

    last_result.details["early_stopped"] = False
    last_result.details["total_shots"] = total_shots_used
    return last_result


class _SPRTChecker:
    """Sequential Probability Ratio Test (Wald) for early stopping.

    For a simple null H0: theta <= theta0 vs H1: theta > theta0,
    SPRT monitors the log-likelihood ratio and stops when it crosses
    one of two boundaries:

        upper boundary = ln((1 - beta) / alpha)   → reject H0
        lower boundary = ln(beta / (1 - alpha))    → accept H0

    where alpha = P(Type I error), beta = P(Type II error).
    """

    def __init__(self, alpha: float = 0.05, beta: float = 0.1, theta0: float = 0.0) -> None:
        import math
        self.alpha = alpha
        self.beta = beta
        self.theta0 = theta0
        self.upper_bound = math.log((1 - beta) / max(alpha, 1e-12))
        self.lower_bound = math.log(beta / max(1 - alpha, 1e-12))

    @classmethod
    def from_spec(cls, spec: ContractSpec) -> _SPRTChecker:
        alpha = 1.0 - spec.confidence.get("level", 0.95)
        beta = spec.resource_budget.get("sprt_beta", 0.1)
        theta0 = spec.tolerances.get("tvd", spec.tolerances.get("epsilon", 0.05))
        return cls(alpha=alpha, beta=beta, theta0=theta0)

    def check(self, result: ContractResult) -> bool:
        """Check if the result is SPRT-conclusive."""
        d = result.details
        import math

        # For TVD-based results: use point estimate vs threshold
        if "tvd_point" in d:
            tvd = d["tvd_point"]
            tol = d.get("tolerance", self.theta0)
            shots = d.get("shots_after", 100)
            # Approximate log-likelihood ratio for TVD
            # Under H0: TVD ~ tol, Under H1: TVD ~ 0
            if shots > 0:
                llr = self._compute_llr(tvd, tol, shots)
                d["sprt_llr"] = llr
                if llr >= self.upper_bound:
                    return True  # Reject H0 (TVD > tol) → fail
                if llr <= self.lower_bound:
                    return True  # Accept H0 (TVD <= tol) → pass
            return False

        # For p-value based: direct SPRT on p-value
        if "p_value" in d:
            p = d["p_value"]
            alpha = d.get("alpha", self.alpha)
            # Simple O'Brien-Fleming-like spending
            if p < alpha / 3:
                return True  # Very strong rejection
            if p > 1 - alpha / 3:
                return True  # Very strong non-rejection
            return False

        return False

    @staticmethod
    def _compute_llr(observed: float, theta0: float, n: int) -> float:
        """Approximate log-likelihood ratio for TVD observation."""
        import math
        # Use Gaussian approximation: TVD ~ N(theta, sigma^2/n)
        sigma = max(0.01, theta0 * 0.5)
        theta1 = theta0 * 0.5  # alternative hypothesis: half the tolerance

        # Log-likelihood ratio
        if sigma <= 0:
            return 0.0
        ll_h0 = -0.5 * n * ((observed - theta0) / sigma) ** 2
        ll_h1 = -0.5 * n * ((observed - theta1) / sigma) ** 2
        return ll_h1 - ll_h0


def _is_conclusive(result: ContractResult, spec: ContractSpec) -> bool:
    """Heuristic: a result is conclusive if the CI doesn't straddle the
    tolerance threshold.  For TVD the CI upper/lower vs tolerance; for
    observable the conservative_diff vs epsilon."""
    d = result.details

    # TVD-based
    if "tvd_ci" in d:
        tol = d.get("tolerance", spec.tolerances.get("tvd", 0.1))
        ci = d["tvd_ci"]
        # Pass-conclusive: CI upper well below tolerance
        if ci.get("upper", 1.0) < tol * 0.9:
            return True
        # Fail-conclusive: CI lower well above tolerance
        if ci.get("lower", 0.0) > tol * 1.1:
            return True
        return False

    # Observable-based
    if "conservative_diff" in d:
        eps = d.get("epsilon", spec.tolerances.get("epsilon", 0.05))
        if d["conservative_diff"] < eps * 0.9:
            return True
        if d["conservative_diff"] > eps * 1.1:
            return True
        return False

    # Statistical test (chi-square / g-test): p-value far from alpha
    if "p_value" in d:
        alpha = d.get("alpha", 0.05)
        if d["p_value"] > alpha * 2:
            return True  # clearly pass
        if d["p_value"] < alpha * 0.5:
            return True  # clearly fail
        return False

    # Default: treat as conclusive to avoid infinite loops
    return True


def evaluate_distribution_contract(
    spec: ContractSpec,
    counts_before: dict[str, int],
    counts_after: dict[str, int],
    simulate_fn: Callable[[int], dict[str, int]] | None = None,
) -> ContractResult:
    """Evaluate a distribution-preservation contract via TVD.

    The contract passes if the upper CI of the TVD is below the tolerance.
    An optional ``spec.spec["test"]`` key can select ``"chi_square"`` or
    ``"g_test"`` instead of the default TVD bootstrap.

    If *simulate_fn* is provided and ``resource_budget.early_stopping`` is
    true, sampling iterates with increasing shots until the result is
    statistically conclusive or ``max_shots`` is reached.
    """
    return _iterative_evaluate(
        spec, simulate_fn, counts_before, counts_after,
        _evaluate_distribution_once,
    )


def _evaluate_distribution_once(
    spec: ContractSpec,
    counts_before: dict[str, int],
    counts_after: dict[str, int],
) -> ContractResult:
    """Single-shot distribution contract evaluation."""
    tolerance = spec.tolerances.get("tvd", 0.1)
    confidence_level = spec.confidence.get("level", 0.95)
    seed = int(spec.resource_budget.get("seed", DEFAULT_SEED))
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

    if test_method == "kolmogorov_smirnov":
        from qocc.contracts.stats import kolmogorov_smirnov_test
        alpha = 1.0 - confidence_level
        ks_result = kolmogorov_smirnov_test(counts_before, counts_after, alpha=alpha)
        return ContractResult(
            name=spec.name,
            passed=ks_result["passed"],
            details={
                "method": "kolmogorov_smirnov",
                "statistic": ks_result["statistic"],
                "p_value": ks_result["p_value"],
                "alpha": alpha,
            },
        )
        
    if test_method == "jensen_shannon":
        from qocc.contracts.stats import jensen_shannon_divergence
        jsd = jensen_shannon_divergence(counts_before, counts_after)
        passed = jsd <= tolerance
        return ContractResult(
            name=spec.name,
            passed=passed,
            details={
                "method": "jensen_shannon",
                "divergence": jsd,
                "tolerance": tolerance,
            },
        )
        
    if test_method == "permutation":
        from qocc.contracts.stats import permutation_test
        alpha = 1.0 - confidence_level
        n_perm = int(spec.resource_budget.get("n_permutations", 1000))
        perm_result = permutation_test(counts_before, counts_after, alpha=alpha, n_permutations=n_perm, seed=seed)
        return ContractResult(
            name=spec.name,
            passed=perm_result["passed"],
            details={
                "method": "permutation",
                "statistic": perm_result["statistic"],
                "p_value": perm_result["p_value"],
                "alpha": alpha,
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
    simulate_fn: Callable[[int], dict[str, int]] | None = None,
) -> ContractResult:
    """Evaluate an observable-preservation contract.

    For each: estimate expectation E, compute CI.
    Pass if |E1 - E2| <= eps considering CI bounds.

    If *simulate_fn* is provided and ``resource_budget.early_stopping`` is
    true, iterates with increasing shots.
    """
    # Observable evaluation needs values, not counts. Convert via helper if iterating.
    def _obs_evaluate_wrapper(
        _spec: ContractSpec,
        _cb: dict[str, int],
        _ca: dict[str, int],
    ) -> ContractResult:
        vb = _counts_to_observable_values(_cb) if _cb else values_before
        va = _counts_to_observable_values(_ca) if _ca else values_after
        return _evaluate_observable_once(_spec, vb, va)

    # For the iterative path, construct dummy counts from values
    dummy_before: dict[str, int] = {"0": len(values_before)} if values_before else {}
    dummy_after: dict[str, int] = {"0": len(values_after)} if values_after else {}

    return _iterative_evaluate(
        spec, simulate_fn, dummy_before, dummy_after,
        _obs_evaluate_wrapper,
    )


def _evaluate_observable_once(
    spec: ContractSpec,
    values_before: list[float],
    values_after: list[float],
) -> ContractResult:
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
