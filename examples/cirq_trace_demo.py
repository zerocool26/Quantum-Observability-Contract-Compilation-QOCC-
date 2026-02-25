"""QOCC end-to-end example: Cirq trace demo.

Creates a Bell-state circuit in Cirq, runs an instrumented compilation
trace, and produces a Trace Bundle.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> None:
    try:
        import cirq  # type: ignore[import-untyped]
    except ImportError:
        print("Cirq is required. Install with: pip install 'qocc[cirq]'")
        sys.exit(1)

    from qocc.api import run_trace
    from qocc.core.circuit_handle import PipelineSpec

    # 1. Create a Bell-state circuit
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit([
        cirq.H(q0),
        cirq.CNOT(q0, q1),
    ])

    print("=== QOCC Cirq Trace Demo ===")
    print(f"Circuit: Bell state, {len(circuit.all_qubits())} qubits")
    print(circuit)
    print()

    # 2. Define pipeline
    pipeline = PipelineSpec(
        adapter="cirq",
        optimization_level=1,
        parameters={"seed": 42},
    )

    # 3. Run trace
    output = str(Path(__file__).parent / "output" / "cirq_bell_bundle.zip")
    Path(output).parent.mkdir(parents=True, exist_ok=True)

    result = run_trace(
        adapter_name="cirq",
        input_source=circuit,
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
