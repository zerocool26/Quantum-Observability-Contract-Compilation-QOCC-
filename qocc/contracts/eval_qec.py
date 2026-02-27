"""QEC contract evaluation.

Implements checks for logical error rate, code distance, and syndrome weight.
"""

from __future__ import annotations

from typing import Any

from qocc.contracts.spec import ContractResult, ContractSpec


def evaluate_qec_contract(
    spec: ContractSpec,
    bundle: dict[str, Any] | None = None,
    compiled_metrics: dict[str, Any] | None = None,
    simulation_metadata: dict[str, Any] | None = None,
) -> ContractResult:
    """Evaluate QEC contract checks.

    Supported checks:
    - logical_error_rate_threshold
    - code_distance
    - syndrome_weight_budget
    """
    bundle = bundle or {}
    compiled_metrics = compiled_metrics or {}
    simulation_metadata = simulation_metadata or {}

    tolerances = spec.tolerances or {}
    details: dict[str, Any] = {"type": "qec", "checks": {}}
    passed = True

    logical_rates = _first_dict(
        simulation_metadata.get("logical_error_rates"),
        bundle.get("logical_error_rates"),
        {},
    )
    decoder_stats = _first_dict(
        simulation_metadata.get("decoder_stats"),
        bundle.get("decoder_stats"),
        {},
    )

    ler_threshold = _first_number(
        tolerances.get("logical_error_rate_threshold"),
        spec.spec.get("logical_error_rate_threshold"),
    )
    if ler_threshold is not None:
        observed = _first_number(
            simulation_metadata.get("logical_error_rate"),
            logical_rates.get("logical_error_rate"),
            compiled_metrics.get("logical_error_rate"),
        )
        ok = observed is not None and observed <= ler_threshold
        details["checks"]["logical_error_rate_threshold"] = {
            "threshold": ler_threshold,
            "observed": observed,
            "passed": ok,
        }
        passed = passed and ok

    min_code_distance = _first_number(
        tolerances.get("code_distance"),
        spec.spec.get("code_distance"),
    )
    if min_code_distance is not None:
        observed_distance = _first_number(
            decoder_stats.get("code_distance"),
            simulation_metadata.get("code_distance"),
            compiled_metrics.get("code_distance"),
        )
        ok = observed_distance is not None and observed_distance >= min_code_distance
        details["checks"]["code_distance"] = {
            "min_required": min_code_distance,
            "observed": observed_distance,
            "passed": ok,
        }
        passed = passed and ok

    syndrome_budget = _first_number(
        tolerances.get("syndrome_weight_budget"),
        spec.spec.get("syndrome_weight_budget"),
    )
    if syndrome_budget is not None:
        observed_mean = _first_number(
            simulation_metadata.get("mean_syndrome_weight"),
            decoder_stats.get("mean_syndrome_weight"),
            _mean_from_distribution(simulation_metadata.get("syndrome_weight_distribution")),
            _mean_from_distribution(bundle.get("decoder_stats", {}).get("syndrome_weight_distribution") if isinstance(bundle.get("decoder_stats"), dict) else None),
        )
        ok = observed_mean is not None and observed_mean <= syndrome_budget
        details["checks"]["syndrome_weight_budget"] = {
            "max_allowed": syndrome_budget,
            "observed_mean": observed_mean,
            "passed": ok,
        }
        passed = passed and ok

    if not details["checks"]:
        passed = False
        details["error"] = "No QEC check thresholds configured in contract spec."

    return ContractResult(name=spec.name, passed=passed, details=details)


def _first_number(*values: Any) -> float | None:
    for v in values:
        if isinstance(v, (int, float)):
            return float(v)
    return None


def _first_dict(*values: Any) -> dict[str, Any]:
    for v in values:
        if isinstance(v, dict):
            return v
    return {}


def _mean_from_distribution(dist: Any) -> float | None:
    if not isinstance(dist, dict) or not dist:
        return None
    total = 0.0
    denom = 0.0
    for k, v in dist.items():
        try:
            weight = float(k)
            count = float(v)
        except Exception:
            continue
        total += weight * count
        denom += count
    if denom <= 0:
        return None
    return total / denom
