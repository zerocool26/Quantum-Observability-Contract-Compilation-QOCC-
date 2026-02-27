"""Metrics subpackage — circuit analysis and cost estimation."""

from __future__ import annotations

__all__ = [
    "compute_metrics",
    "check_topology",
    "NoiseModel",
    "NoiseModelRegistry",
]

from qocc.metrics.compute import compute_metrics
from qocc.metrics.noise_model import NoiseModel, NoiseModelRegistry
from qocc.metrics.topology import check_topology
