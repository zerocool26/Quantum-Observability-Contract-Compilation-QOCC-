"""Statistical utilities for contract evaluation.

Provides:
- Total Variation Distance (TVD) estimation with CI
- Expectation value estimation with CI (Hoeffding / bootstrap)
- Chi-square / G-test wrappers
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from qocc import DEFAULT_SEED


# ======================================================================
# Total Variation Distance
# ======================================================================


def total_variation_distance(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
) -> float:
    """Compute TVD = 0.5 * Σ |p_i - q_i| between two count distributions."""
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())
    if total_a == 0 or total_b == 0:
        return 1.0

    all_keys = set(counts_a) | set(counts_b)
    tvd = 0.0
    for k in all_keys:
        p = counts_a.get(k, 0) / total_a
        q = counts_b.get(k, 0) / total_b
        tvd += abs(p - q)
    return 0.5 * tvd


def tvd_bootstrap_ci(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Bootstrap confidence interval for TVD.

    Returns:
        Dict with ``point``, ``lower``, ``upper``, ``confidence``.
    """
    rng = np.random.default_rng(seed)

    keys = sorted(set(counts_a) | set(counts_b))
    total_a = sum(counts_a.values())
    total_b = sum(counts_b.values())

    probs_a = np.array([counts_a.get(k, 0) / total_a for k in keys])
    probs_b = np.array([counts_b.get(k, 0) / total_b for k in keys])

    point = float(0.5 * np.sum(np.abs(probs_a - probs_b)))

    bootstrap_tvds = []
    for _ in range(n_bootstrap):
        samples_a = rng.multinomial(total_a, probs_a)
        samples_b = rng.multinomial(total_b, probs_b)
        p_a = samples_a / total_a
        p_b = samples_b / total_b
        bootstrap_tvds.append(float(0.5 * np.sum(np.abs(p_a - p_b))))

    alpha = 1.0 - confidence
    lower = float(np.percentile(bootstrap_tvds, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_tvds, 100 * (1 - alpha / 2)))

    return {
        "point": point,
        "lower": lower,
        "upper": upper,
        "confidence": confidence,
    }


# ======================================================================
# Expectation value CI
# ======================================================================


def expectation_ci_hoeffding(
    values: list[float] | np.ndarray,
    confidence: float = 0.95,
    value_range: float = 2.0,
) -> dict[str, Any]:
    """Hoeffding-bound CI for the mean of bounded values.

    Assumes values ∈ [-value_range/2, value_range/2] by default (e.g. eigenvalues ±1).

    Returns:
        Dict with ``mean``, ``lower``, ``upper``, ``confidence``, ``n``.
    """
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "confidence": confidence, "n": 0}

    mean = float(np.mean(arr))
    alpha = 1.0 - confidence
    # Hoeffding: P(|X̄ - μ| >= t) <= 2 exp(-2nt²/R²)
    # -> t = R * sqrt(ln(2/α) / (2n))
    t = value_range * math.sqrt(math.log(2.0 / alpha) / (2.0 * n))

    return {
        "mean": mean,
        "lower": mean - t,
        "upper": mean + t,
        "confidence": confidence,
        "n": n,
    }


def expectation_bootstrap_ci(
    values: list[float] | np.ndarray,
    confidence: float = 0.95,
    n_bootstrap: int = 1000,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Bootstrap CI for the mean."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return {"mean": 0.0, "lower": 0.0, "upper": 0.0, "confidence": confidence, "n": 0}

    mean = float(np.mean(arr))
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=n, replace=True)
        boot_means.append(float(np.mean(sample)))

    alpha = 1.0 - confidence
    lower = float(np.percentile(boot_means, 100 * alpha / 2))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))

    return {
        "mean": mean,
        "lower": lower,
        "upper": upper,
        "confidence": confidence,
        "n": n,
    }


# ======================================================================
# Chi-square test
# ======================================================================


def chi_square_test(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Chi-square goodness-of-fit test between two count distributions.

    Tests H0: the two distributions are the same.

    Returns:
        Dict with ``statistic``, ``p_value``, ``passed`` (True if NOT rejected).
    """
    from scipy import stats  # type: ignore[import-untyped]

    keys = sorted(set(counts_a) | set(counts_b))
    obs = np.array([counts_a.get(k, 0) for k in keys], dtype=float)
    exp = np.array([counts_b.get(k, 0) for k in keys], dtype=float)

    # Scale expected to match observed total
    total_obs = obs.sum()
    total_exp = exp.sum()
    if total_exp > 0:
        exp = exp * (total_obs / total_exp)

    # Remove bins where expected is 0
    mask = exp > 0
    obs = obs[mask]
    exp = exp[mask]

    if len(obs) <= 1:
        return {"statistic": 0.0, "p_value": 1.0, "passed": True}

    stat, p_value = stats.chisquare(obs, exp)
    return {
        "statistic": float(stat),
        "p_value": float(p_value),
        "passed": bool(p_value > alpha),
    }


def g_test(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
    alpha: float = 0.05,
) -> dict[str, Any]:
    """G-test (log-likelihood ratio) between two count distributions.

    Uses the Williams correction for small samples.

    Tests H0: the two distributions are the same.

    Returns:
        Dict with ``statistic``, ``p_value``, ``passed`` (True if NOT rejected).
    """
    keys = sorted(set(counts_a) | set(counts_b))
    obs = np.array([counts_a.get(k, 0) for k in keys], dtype=float)
    exp = np.array([counts_b.get(k, 0) for k in keys], dtype=float)

    total_obs = obs.sum()
    total_exp = exp.sum()
    if total_exp > 0:
        exp = exp * (total_obs / total_exp)

    mask = (exp > 0) & (obs > 0)
    obs_m = obs[mask]
    exp_m = exp[mask]

    if len(obs_m) <= 1:
        return {"statistic": 0.0, "p_value": 1.0, "passed": True}

    # G = 2 Σ O_i ln(O_i / E_i)
    g_stat = float(2.0 * np.sum(obs_m * np.log(obs_m / exp_m)))

    # Williams correction: q = 1 + (k+1)/(6*n) for k categories, n total
    k = len(obs_m)
    q = 1.0 + (k + 1) / (6.0 * total_obs) if total_obs > 0 else 1.0
    g_corrected = g_stat / q if q > 0 else g_stat

    # p-value from chi-square distribution with k-1 df
    try:
        from scipy import stats as sp_stats  # type: ignore[import-untyped]
        p_value = float(1.0 - sp_stats.chi2.cdf(g_corrected, df=k - 1))
    except ImportError:
        # Rough approximation without scipy
        p_value = 1.0 if g_corrected < k - 1 else 0.0

    return {
        "statistic": g_corrected,
        "statistic_raw": g_stat,
        "p_value": p_value,
        "passed": bool(p_value > alpha),
        "df": k - 1,
        "williams_correction": q,
    }
