# QOCC — Quantum Observability + Contract-Based Compilation

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**QOCC** is a vendor-agnostic, reproducible, trace-first layer that instruments quantum program workflows end-to-end and supports contract-defined correctness + cost optimization via closed-loop compilation/search.

## Features

- **Observability** — OpenTelemetry-style traces for quantum compilation, simulation, mitigation, decoding, and execution
- **Contracts** — Machine-checkable semantic constraints + explicit cost objectives
- **Closed-loop optimization** — Generate candidate pipelines, score cheaply, validate with simulation, choose best under contracts
- **Trace Bundles** — Portable "repro packages" that can be rerun, compared, regression-tested, and shared

## Quick Start

```bash
pip install -e ".[all]"

# Run a trace
qocc trace run --adapter qiskit --input examples/ghz.qasm --pipeline examples/pipeline_examples/qiskit_default.json --out bundle.zip

# Compare bundles
qocc trace compare bundleA.zip bundleB.zip --report reports/

# Check contracts
qocc contract check --bundle bundle.zip --contracts examples/contracts_examples.json
```

## Architecture

```
Input Circuit → Adapter (ingest/normalize) → Compilation → Metrics → Contract Eval → Bundle Export
                                                ↓
                                        Trace Emitter (spans/events)
```

Every stage emits structured spans. The resulting Trace Bundle contains everything needed to reproduce, compare, and debug.

## Supported Backends

| Backend | Status |
|---------|--------|
| Qiskit  | ✅ MVP |
| Cirq    | ✅ MVP |
| pytket  | 🔜 Planned |
| CUDA-Q  | 🔜 Optional |
| Stim/PyMatching | 🔜 QEC mode |

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check .
mypy qocc/
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
