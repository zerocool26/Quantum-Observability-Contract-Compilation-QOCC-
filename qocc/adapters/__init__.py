"""Adapter subpackage — vendor-specific quantum circuit adapters."""

from __future__ import annotations

__all__ = [
    "get_adapter",
    "AdapterBase",
    "SimulationSpec",
    "SimulationResult",
    "ExecutionResult",
]

from qocc.adapters.base import (
    BaseAdapter as AdapterBase,
    ExecutionResult,
    SimulationResult,
    SimulationSpec,
    get_adapter,
)
