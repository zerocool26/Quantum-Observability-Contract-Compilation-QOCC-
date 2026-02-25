"""QOCC end-to-end example: Qiskit trace demo.

Creates a GHZ circuit, runs an instrumented compilation trace,
and produces a Trace Bundle.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    try:
        from qiskit import QuantumCircuit  # type: ignore[import-untyped]
    except ImportError:
        print("Qiskit is required. Install with: pip install 'qocc[qiskit]'")
        sys.exit(1)

    from qocc.api import run_trace
    from qocc.core.circuit_handle import PipelineSpec

    # 1. Create a 3-qubit GHZ circuit
    qc = QuantumCircuit(3, name="ghz_3")
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(1, 2)

    print("=== QOCC Qiskit Trace Demo ===")
    print(f"Circuit: {qc.name}, {qc.num_qubits} qubits")
    print(qc)
    print()

    # 2. Define pipeline
    pipeline = PipelineSpec(
        adapter="qiskit",
        optimization_level=2,
        parameters={"seed": 42},
    )

    # 3. Run trace
    output = str(Path(__file__).parent / "output" / "qiskit_ghz_bundle.zip")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    result = run_trace(
        adapter_name="qiskit",
        input_source=qc,
        pipeline=pipeline,
        output=output,
    )

    print(f"Bundle created: {result['bundle_zip']}")
    print(f"Run ID:         {result['run_id']}")
    print(f"Input hash:     {result['input_hash'][:16]}…")
    print(f"Compiled hash:  {result['compiled_hash'][:16]}…")
    print(f"Spans:          {result['num_spans']}")
    print()
    print("Metrics before compilation:")
    for k, v in result["metrics_before"].items():
        if k != "gate_histogram":
            print(f"  {k}: {v}")
    print()
    print("Metrics after compilation:")
    for k, v in result["metrics_after"].items():
        if k != "gate_histogram":
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
