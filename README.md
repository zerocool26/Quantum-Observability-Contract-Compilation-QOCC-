# QOCC — Quantum Observability + Contract-Based Compilation

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

**QOCC** is a vendor-agnostic, reproducible, trace-first layer that instruments quantum program workflows end-to-end and supports contract-defined correctness + cost optimization via closed-loop compilation/search.

## Features

- **Observability** — OpenTelemetry-style traces for quantum compilation, simulation, mitigation, decoding, and execution
- **Contracts** — Machine-checkable semantic constraints + explicit cost objectives
- **Closed-loop optimization** — Generate candidate pipelines, score cheaply, validate with simulation, choose best under contracts
- **Trace Bundles** — Portable "repro packages" that can be rerun, compared, regression-tested, and shared
- **Nondeterminism detection** — Compile multiple times and verify output stability
- **Content-addressed caching** — Cache compilation results keyed by circuit hash + pipeline spec + backend version
- **Plugin system** — Register custom adapters and evaluators via Python entry points

## Quick Start

```bash
pip install -e ".[all]"

# Run a trace
qocc trace run --adapter qiskit --input examples/ghz.qasm \
    --pipeline examples/pipeline_examples/qiskit_default.json --out bundle.zip

# Compare bundles
qocc trace compare bundleA.zip bundleB.zip --report reports/

# Check contracts
qocc contract check --bundle bundle.zip --contracts examples/contracts_examples.json

# Compilation search
qocc compile search --adapter qiskit --input examples/ghz.qasm --topk 5 --out search.zip

# Compilation search with Pareto multi-objective selection
qocc compile search --adapter qiskit --input examples/ghz.qasm --mode pareto --out search.zip

# Detect nondeterminism (compile 5 times)
qocc trace run --adapter qiskit --input examples/ghz.qasm --repeat 5 --out nd.zip

# Replay a bundle
qocc trace replay bundle.zip --out replayed.zip
```

## Architecture

```
Input Circuit → Adapter (ingest/normalize) → Compilation → Metrics → Contract Eval → Bundle Export
                                                ↓
                                        Trace Emitter (spans/events)
                                                ↓
                     Content-Addressed Cache ←→ Nondeterminism Detection
```

Every stage emits structured spans with per-pass granularity. The resulting Trace Bundle contains everything needed to reproduce, compare, and debug.

## Contract Types

| Type | Evaluator | Description |
|------|-----------|-------------|
| `distribution` | TVD / chi-square / G-test | Output distribution preserved within tolerance |
| `observable` | Hoeffding CI | Z-observable expectation preserved within ε |
| `clifford` | Stabilizer tableau | Exact Clifford equivalence (falls back to distribution for non-Clifford) |
| `exact` | Statevector fidelity | Exact statevector equivalence |
| `cost` | Resource budget | Depth, 2Q gates, total gates, duration, proxy error within limits |

### Contract Example

```json
[
  {"name": "tvd-check", "type": "distribution", "tolerances": {"tvd": 0.1},
   "confidence": {"level": 0.95}, "resource_budget": {"n_bootstrap": 1000}},
  {"name": "depth-budget", "type": "cost", "tolerances": {"max_depth": 50}},
  {"name": "g-test", "type": "distribution", "spec": {"test": "g_test"},
   "confidence": {"level": 0.99}}
]
```

### Early Stopping

Set `resource_budget.early_stopping: true` with `min_shots` and `max_shots` to enable iterative sampling that halts when pass/fail is statistically certain:

```json
{"name": "adaptive", "type": "distribution", "tolerances": {"tvd": 0.1},
 "resource_budget": {"early_stopping": true, "min_shots": 256, "max_shots": 8192}}
```

## Compilation Search

The `search_compile()` API and `qocc compile search` CLI implement the full closed-loop pipeline from §3 of the spec:

1. **Generate** candidates by varying optimization level, seeds, and parameters
2. **Compile** each candidate (with per-candidate caching)
3. **Score** cheaply with a surrogate cost model
4. **Validate** top-k candidates via simulation
5. **Evaluate** contracts on validated candidates
6. **Select** the best candidate (single-best or Pareto frontier)

### Pareto Multi-Objective Selection

```python
from qocc.api import search_compile

result = search_compile(
    adapter_name="qiskit",
    input_source="circuit.qasm",
    mode="pareto",  # multi-objective Pareto frontier
    contracts=[{"name": "tvd", "type": "distribution", "tolerances": {"tvd": 0.1}}],
)
```

## Caching

QOCC uses a content-addressed compilation cache keyed by `SHA-256(circuit_hash || pipeline_dict || backend_version)`. Cache hits are recorded in `cache_index.json` inside the bundle for reproducibility auditing.

```python
from qocc.core.cache import CompilationCache
cache = CompilationCache()
print(cache.stats())  # {"size": 42, "hits": 10, "misses": 5}
```

## Plugin System

Register custom adapters and contract evaluators via Python entry points:

```toml
# pyproject.toml
[project.entry-points."qocc.adapters"]
my_backend = "my_package:MyAdapter"

[project.entry-points."qocc.evaluators"]
my_evaluator = "my_package:my_eval_function"
```

Or register programmatically:

```python
from qocc.adapters.base import register_adapter
from qocc.contracts.registry import register_evaluator

register_adapter("my_backend", MyAdapter)
register_evaluator("my_eval", my_eval_function)
```

## Nondeterminism Detection

Run a circuit through the compilation pipeline multiple times and detect stochastic variation:

```python
from qocc.api import run_trace

result = run_trace("qiskit", "circuit.qasm", repeat=5)
if result.get("nondeterminism", {}).get("reproducible") is False:
    print("WARNING: compilation is nondeterministic!")
```

## Bundle Replay

Replay a previously recorded bundle to verify reproducibility:

```bash
qocc trace replay bundle.zip --out replayed.zip
qocc trace compare bundle.zip replayed.zip --report diff/
```

## Supported Backends

| Backend | Status |
|---------|--------|
| Qiskit  | ✅ Full (per-stage spans, statevector sim) |
| Cirq    | ✅ Full (per-pass spans, statevector sim) |
| pytket  | 🔜 Planned |
| CUDA-Q  | 🔜 Optional |
| Stim/PyMatching | 🔜 QEC mode |

## Development

```bash
pip install -e ".[dev]"
pytest                 # ~130 tests
ruff check .
mypy qocc/
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
