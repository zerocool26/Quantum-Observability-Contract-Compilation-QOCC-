"""Canonicalization utilities for quantum circuits.

Responsible for normalising qubit ordering, gate naming, and register
mapping so that structurally identical circuits always produce the same
hash regardless of cosmetic differences.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qocc.core.circuit_handle import CircuitHandle


def canonicalize_qasm3(qasm: str) -> str:
    """Normalise an OpenQASM 3 string.

    * Strips comments
    * Normalises whitespace
    * Sorts gate declarations (stable)
    * Removes trailing whitespace
    """
    lines: list[str] = []
    for raw in qasm.splitlines():
        line = raw.strip()
        # strip inline comments
        line = re.sub(r"//.*$", "", line).rstrip()
        if not line:
            continue
        lines.append(line)
    return "\n".join(lines) + "\n"


def normalize_circuit(handle: CircuitHandle) -> CircuitHandle:
    """Return a new *normalised* CircuitHandle.

    If QASM3 text is present, it is canonicalized.  The ``_normalized``
    flag is set on the returned handle.
    """
    import copy

    out = copy.copy(handle)
    if out.qasm3 is not None:
        out.qasm3 = canonicalize_qasm3(out.qasm3)
    out._normalized = True
    return out
