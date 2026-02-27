"""Surrogate scoring for candidate pipelines.

Computes a cheap proxy score from deterministic metrics:
    base_score = Σ metric_i * weight_i
    optional noise_score = gate_error + readout_error + decoherence_error
"""

from __future__ import annotations

from typing import Any

from qocc.adapters.base import MetricsSnapshot
from qocc.metrics.noise_model import NoiseModel


def _pick_noise_param(value: float | dict[str, float] | None, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, dict):
        vals = [float(v) for v in value.values()]
        return sum(vals) / len(vals) if vals else default
    return float(value)


def _compute_noise_score(metrics: dict[str, Any], noise_model: NoiseModel) -> float:
    gates_1q = float(metrics.get("gates_1q", 0.0) or 0.0)
    gates_2q = float(metrics.get("gates_2q", 0.0) or 0.0)
    depth = float(metrics.get("depth", 0.0) or 0.0)
    duration = float(metrics.get("duration", metrics.get("duration_estimate", 0.0)) or 0.0)

    p1 = _pick_noise_param(noise_model.single_qubit_error)
    p2 = _pick_noise_param(noise_model.two_qubit_error)
    pr = _pick_noise_param(noise_model.readout_error)
    t1 = _pick_noise_param(noise_model.t1, default=0.0)
    t2 = _pick_noise_param(noise_model.t2, default=0.0)

    gate_error = (gates_1q * p1) + (gates_2q * p2)
    readout_error = pr

    if t1 > 0.0 and t2 > 0.0:
        t_eff = max(min(t1, t2), 1e-9)
        decoherence_error = (duration / t_eff)
    elif t1 > 0.0:
        decoherence_error = (duration / max(t1, 1e-9))
    else:
        decoherence_error = depth * 1e-4

    return gate_error + readout_error + decoherence_error


def surrogate_score(
    metrics: dict[str, Any] | MetricsSnapshot,
    weights: dict[str, float] | None = None,
    noise_model: dict[str, Any] | NoiseModel | None = None,
) -> dict[str, Any]:
    """Compute a surrogate score from metrics.

    Parameters:
        metrics: Metrics dict or MetricsSnapshot.
        weights: ``{"depth": w1, "gates_2q": w2, "duration": w3, "proxy_error": w4, "noise_score": w5}``.
        noise_model: Optional provider-agnostic noise model payload.

    Returns:
        Dict with ``score`` (scalar) and ``breakdown`` (per-term).
    """
    if isinstance(metrics, MetricsSnapshot):
        m = metrics.to_dict()
    else:
        m = dict(metrics)

    if weights is None:
        weights = {
            "depth": 1.0,
            "gates_2q": 5.0,
            "duration": 0.001,
            "proxy_error": 100.0,
            "noise_score": 1.0,
        }

    model: NoiseModel | None
    if noise_model is None:
        model = None
    elif isinstance(noise_model, NoiseModel):
        model = noise_model
    else:
        model = NoiseModel.from_dict(noise_model)

    breakdown: dict[str, float] = {}
    total = 0.0

    for key, w in weights.items():
        if key == "noise_score":
            continue
        val = m.get(key)
        if val is not None:
            term = float(val) * w
        else:
            term = 0.0
        breakdown[key] = term
        total += term

    if model is not None:
        raw_noise = _compute_noise_score(m, model)
        weighted_noise = raw_noise * float(weights.get("noise_score", 1.0))
        breakdown["noise_score"] = weighted_noise
        total += weighted_noise

    result = {
        "score": total,
        "breakdown": breakdown,
        "weights": weights,
    }
    if model is not None:
        result["noise_model_hash"] = model.stable_hash()
    return result


def rank_candidates(
    candidates: list[Any],
    weights: dict[str, float] | None = None,
    noise_model: dict[str, Any] | NoiseModel | None = None,
) -> list[Any]:
    """Score and sort candidates by surrogate score (ascending = best).

    Mutates each candidate's ``surrogate_score`` field and returns
    the sorted list.
    """
    for c in candidates:
        result = surrogate_score(c.metrics, weights, noise_model=noise_model)
        c.surrogate_score = result["score"]

    return sorted(candidates, key=lambda c: c.surrogate_score)
