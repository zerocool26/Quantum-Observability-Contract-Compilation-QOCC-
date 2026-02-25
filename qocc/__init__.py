"""QOCC — Quantum Observability + Contract-Based Compilation."""

from __future__ import annotations

__version__ = "0.1.0"

from qocc.api import check_contract, compare_bundles, run_trace, search_compile

__all__ = [
    "__version__",
    "run_trace",
    "compare_bundles",
    "check_contract",
    "search_compile",
]
