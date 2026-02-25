"""Surrogate scoring for candidate pipelines.

Computes a cheap proxy score from deterministic metrics:
  score = Σ count(gate_type) * p_error(gate_type) + depth * decoherence_weight + duration * duration_weight
"""

from __future__ import annotations

from typing import Any

from qocc.adapters.base import MetricsSnapshot


def surrogate_score(
    metrics: dict[str, Any] | MetricsSnapshot,
    weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Compute a surrogate score from metrics.

    Parameters:
        metrics: Metrics dict or MetricsSnapshot.
        weights: ``{"depth": w1, "gates_2q": w2, "duration": w3, "proxy_error": w4}``.

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
        }

    breakdown: dict[str, float] = {}
    total = 0.0

    for key, w in weights.items():
        val = m.get(key)
        if val is not None:
            term = float(val) * w
        else:
            term = 0.0
        breakdown[key] = term
        total += term

    return {
        "score": total,
        "breakdown": breakdown,
        "weights": weights,
    }


def rank_candidates(
    candidates: list[Any],
    weights: dict[str, float] | None = None,
) -> list[Any]:
    """Score and sort candidates by surrogate score (ascending = best).

    Mutates each candidate's ``surrogate_score`` field and returns
    the sorted list.
    """
    for c in candidates:
        result = surrogate_score(c.metrics, weights)
        c.surrogate_score = result["score"]

    return sorted(candidates, key=lambda c: c.surrogate_score)
