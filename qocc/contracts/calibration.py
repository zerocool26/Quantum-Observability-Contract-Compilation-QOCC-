"""Calibration of confidence intervals for statistical contract evaluators.

This module provides tools to empirically validate that a given evaluation
method achieves its nominal confidence coverage, e.g. a 95% CI contains the
true parameter 95% of the time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np


def calibrate_ci_coverage(
    evaluator: Callable[[dict[str, int], dict[str, int]], dict[str, Any]],
    true_param: float,
    distribution_generator: Callable[[int], tuple[dict[str, int], dict[str, int]]],
    n_trials: int = 1000,
    confidence_level: float = 0.95,
    seed: int = 42,
) -> dict[str, Any]:
    """Run synthetic trials to measure actual CI coverage.

    Parameters:
        evaluator: A function taking `(counts_a, counts_b)` and returning a dict
                   with `lower` and `upper` keys representing the CI.
        true_param: The known true underlying parameter (e.g. true TVD).
        distribution_generator: A function taking a seed and generating
                                `(counts_a, counts_b)` sampled from the ground
                                truth distribution.
        n_trials: Number of synthetic trials to run.
        confidence_level: The nominal confidence level (e.g. 0.95).
        seed: Random seed.

    Returns:
        A dict with the calibration report (actual coverage, etc.).
    """
    rng = np.random.default_rng(seed)
    
    hits = 0
    miss_low = 0
    miss_high = 0
    
    trial_results = []
    
    for i in range(n_trials):
        seed_iter = int(rng.integers(0, 2**31))
        
        try:
            ca, cb = distribution_generator(seed_iter)
            res = evaluator(ca, cb)
            
            lower = res.get("lower", float("-inf"))
            upper = res.get("upper", float("inf"))
            
            hit = lower <= true_param <= upper
            if hit:
                hits += 1
            elif true_param < lower:
                miss_low += 1
            elif true_param > upper:
                miss_high += 1
                
            trial_results.append({
                "trial": i,
                "hit": hit,
                "lower": lower,
                "upper": upper
            })
        except Exception as e:
            # If evaluator fails, skip or count as miss?
            pass
            
    actual_coverage = hits / n_trials if n_trials > 0 else 0.0
    
    # Calculate binomial CI for the coverage proportion itself
    # Using normal approximation
    z = 1.96 # approx 95%
    if n_trials > 0:
        mc = actual_coverage * (1 - actual_coverage) / n_trials
        margin = z * np.sqrt(mc)
    else:
        margin = 0.0
        
    deviation = abs(actual_coverage - confidence_level)
    passed_calibration = deviation <= 0.02 + margin
    
    return {
        "nominal_confidence": confidence_level,
        "actual_coverage": actual_coverage,
        "coverage_margin_of_error": margin,
        "deviation": deviation,
        "hits": hits,
        "miss_low": miss_low,
        "miss_high": miss_high,
        "n_trials": n_trials,
        "passed_calibration": passed_calibration,
        "warning": "Coverage deviates by > 2%" if not passed_calibration else None
    }


def save_calibration_report(report: dict[str, Any], out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(report, indent=2), encoding="utf-8")
