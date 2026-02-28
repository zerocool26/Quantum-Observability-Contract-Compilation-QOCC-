"""Zero-noise extrapolation (ZNE) contract evaluator."""

from __future__ import annotations

from typing import Any

import numpy as np

from qocc.contracts.spec import ContractResult


def evaluate_zne_contract(
    spec: Any,
    ideal_value: float,
    scaled_expectations: list[dict[str, Any]],
) -> ContractResult:
    """Evaluate ZNE contract using Richardson extrapolation at noise scale 0.

    Parameters:
        spec: ContractSpec-like object.
        ideal_value: Ideal expectation (typically simulated baseline).
        scaled_expectations: Sequence of ``{"scale": float, "expectation": float}``.
    """
    if len(scaled_expectations) < 2:
        return ContractResult(
            name=spec.name,
            passed=False,
            details={
                "type": "zne",
                "error": "Need at least two noise-scale expectation points for extrapolation.",
                "ideal_value": ideal_value,
                "per_level": scaled_expectations,
            },
        )

    xs = np.array([float(x["scale"]) for x in scaled_expectations], dtype=float)
    ys = np.array([float(x["expectation"]) for x in scaled_expectations], dtype=float)

    degree = max(1, min(len(xs) - 1, int(spec.spec.get("fit_degree", len(xs) - 1))))
    try:
        coeffs = np.polyfit(xs, ys, deg=degree)
        extrapolated = float(np.polyval(coeffs, 0.0))
    except Exception as exc:
        return ContractResult(
            name=spec.name,
            passed=False,
            details={
                "type": "zne",
                "error": f"Richardson extrapolation failed: {exc}",
                "ideal_value": ideal_value,
                "per_level": scaled_expectations,
            },
        )

    tolerance = _resolve_tolerance(spec)
    abs_error = abs(extrapolated - float(ideal_value))
    passed = abs_error <= tolerance

    return ContractResult(
        name=spec.name,
        passed=passed,
        details={
            "type": "zne",
            "ideal_value": float(ideal_value),
            "extrapolated_value": extrapolated,
            "abs_error": abs_error,
            "tolerance": tolerance,
            "fit_degree": degree,
            "extrapolation_coefficients": [float(c) for c in coeffs],
            "per_level": scaled_expectations,
        },
    )


def _resolve_tolerance(spec: Any) -> float:
    for key in ("zne_abs_error", "abs_error", "epsilon"):
        v = spec.tolerances.get(key)
        if v is not None:
            return float(v)
    return float(spec.tolerances.get("tvd", 0.05))
