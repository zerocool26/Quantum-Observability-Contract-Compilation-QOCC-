"""QOCC — Quantum Observability + Contract-Based Compilation."""

from __future__ import annotations

__version__ = "0.1.0"

#: Default random seed used across the project.
DEFAULT_SEED: int = 42

#: Default RNG algorithm identifier recorded in provenance metadata.
DEFAULT_RNG_ALGORITHM: str = "PCG64"

from qocc.api import check_contract, compare_bundles, run_trace, search_compile
from qocc.trace.jupyter_widget import compare_interactive, search_dashboard, show_bundle

__all__ = [
    "__version__",
    "DEFAULT_SEED",
    "DEFAULT_RNG_ALGORITHM",
    "run_trace",
    "compare_bundles",
    "check_contract",
    "search_compile",
    "show_bundle",
    "compare_interactive",
    "search_dashboard",
]
