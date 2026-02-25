"""Pluggable gate-duration models for different hardware backends."""

from __future__ import annotations

from typing import Any


# Pre-defined duration models (gate_name -> duration in nanoseconds)

SUPERCONDUCTING_DEFAULTS: dict[str, float] = {
    "x": 25.0,
    "sx": 25.0,
    "rz": 0.0,  # virtual Z
    "cx": 300.0,
    "cz": 250.0,
    "ecr": 280.0,
    "id": 25.0,
    "default": 50.0,
}

TRAPPED_ION_DEFAULTS: dict[str, float] = {
    "rx": 10_000.0,
    "ry": 10_000.0,
    "rz": 0.1,  # nearly free
    "xx": 200_000.0,
    "zz": 200_000.0,
    "ms": 200_000.0,
    "default": 15_000.0,
}


def get_duration_model(name: str) -> dict[str, float]:
    """Return a named duration model."""
    models: dict[str, dict[str, float]] = {
        "superconducting": SUPERCONDUCTING_DEFAULTS,
        "trapped_ion": TRAPPED_ION_DEFAULTS,
    }
    if name not in models:
        raise KeyError(f"Unknown duration model: {name!r}. Available: {list(models)}")
    return models[name]


def custom_duration_model(overrides: dict[str, float]) -> dict[str, float]:
    """Create a custom duration model from user-supplied gate durations."""
    model = dict(SUPERCONDUCTING_DEFAULTS)
    model.update(overrides)
    return model
