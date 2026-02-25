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

    * Strips comments (// and /* ... */)
    * Normalises whitespace (collapse runs, trim lines)
    * Sorts gate declarations (stable)
    * Canonicalises qubit register naming (q[n] form)
    * Removes trailing whitespace
    * Stabilises parameter formatting
    """
    # Strip block comments
    qasm = re.sub(r"/\*.*?\*/", "", qasm, flags=re.DOTALL)

    lines: list[str] = []
    for raw in qasm.splitlines():
        line = raw.strip()
        # strip inline comments
        line = re.sub(r"//.*$", "", line).rstrip()
        if not line:
            continue
        # Normalise whitespace runs to single space
        line = re.sub(r"\s+", " ", line)
        lines.append(line)

    # Separate into header block, gate declarations, and body
    header_lines: list[str] = []  # OPENQASM, include, qubit, bit, creg, qreg
    gate_decl_lines: list[str] = []  # gate ... { ... }
    body_lines: list[str] = []  # everything else

    header_keywords = {"OPENQASM", "include", "qubit", "bit", "creg", "qreg", "input", "output"}

    i = 0
    while i < len(lines):
        line = lines[i]
        first_word = line.split()[0].rstrip(";") if line.split() else ""

        # Gate declaration blocks: gate foo(params) q0, q1 { ... }
        if first_word == "gate":
            # Collect until closing brace
            block = [line]
            if "{" in line and "}" in line:
                gate_decl_lines.append(" ".join(block))
                i += 1
                continue
            elif "{" in line:
                i += 1
                while i < len(lines) and "}" not in lines[i]:
                    block.append(lines[i])
                    i += 1
                if i < len(lines):
                    block.append(lines[i])
                gate_decl_lines.append(" ".join(block))
                i += 1
                continue
            else:
                gate_decl_lines.append(line)
                i += 1
                continue
        elif first_word in header_keywords:
            header_lines.append(line)
        else:
            body_lines.append(line)
        i += 1

    # Sort gate declarations alphabetically for stability
    gate_decl_lines.sort()

    # Sort body lines that are purely gate applications within commuting groups
    # (lines that operate on disjoint qubits can be reordered)
    body_lines = _sort_commuting_gates(body_lines)

    # Normalise floating-point parameter representations
    all_lines = header_lines + gate_decl_lines + body_lines
    normalised: list[str] = []
    for line in all_lines:
        line = _normalise_float_params(line)
        normalised.append(line)

    return "\n".join(normalised) + "\n"


def _sort_commuting_gates(lines: list[str]) -> list[str]:
    """Sort gate application lines into canonical order where safe.

    Two gate lines commute if they operate on disjoint qubit sets.
    Within each commutation window we sort by (qubit indices, gate name).
    Non-gate lines (barriers, measures, conditions) act as barriers that
    flush the current window.
    """
    result: list[str] = []
    window: list[tuple[str, set[str], str]] = []  # (sort_key, qubits, original_line)

    def _flush() -> None:
        if not window:
            return
        # Topological sort respecting qubit dependencies within window
        # Simple approach: group by non-overlapping qubit sets
        sorted_batch = sorted(window, key=lambda x: x[0])
        for _, _, line in sorted_batch:
            result.append(line)
        window.clear()

    barrier_words = {"barrier", "measure", "reset", "if", "while", "for"}

    for line in lines:
        first_word = line.split("(")[0].split()[0].rstrip(";") if line.split() else ""

        if first_word in barrier_words or "=" in line.split("//")[0]:
            _flush()
            result.append(line)
            continue

        # Extract qubit references for sorting
        qubits = set(re.findall(r"[a-zA-Z_]\w*\[\d+\]", line))
        if not qubits:
            qubits = set(re.findall(r"\bq\d+\b", line))

        # Check for conflicts with current window
        conflicts = False
        for _, wq, _ in window:
            if qubits & wq:
                conflicts = True
                break

        if conflicts:
            _flush()

        # Sort key: smallest qubit index, then gate name
        qubit_indices = sorted(int(x) for x in re.findall(r"\[(\d+)\]", line))
        sort_key = (tuple(qubit_indices) if qubit_indices else (999,), first_word, line)
        window.append((sort_key, qubits, line))

    _flush()
    return result


def _normalise_float_params(line: str) -> str:
    """Normalise floating-point numbers to consistent precision."""
    def _fmt_float(m: re.Match) -> str:
        val = float(m.group(0))
        # Use 10 significant digits for stability
        formatted = f"{val:.10g}"
        return formatted

    # Match floating-point numbers (not integers)
    return re.sub(r"(?<!\w)-?\d+\.\d+(?:[eE][+-]?\d+)?", _fmt_float, line)


def normalize_circuit(handle: CircuitHandle) -> CircuitHandle:
    """Return a new *normalised* CircuitHandle.

    If QASM3 text is present, it is canonicalized.  The ``_normalized``
    flag is set on the returned handle.  Uses ``copy.deepcopy`` to avoid
    sharing mutable state with the original.
    """
    import copy

    out = copy.deepcopy(handle)
    if out.qasm3 is not None:
        out.qasm3 = canonicalize_qasm3(out.qasm3)
    out._normalized = True
    return out
