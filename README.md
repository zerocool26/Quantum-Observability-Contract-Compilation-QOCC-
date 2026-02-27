# QOCC — Quantum Observability + Contract-Based Compilation

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

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

# Check contracts from DSL file
qocc contract check --bundle bundle.zip --contracts examples/contracts.qocc

# Compilation search
qocc compile search --adapter qiskit --input examples/ghz.qasm --topk 5 --out search.zip

# Compilation search with Pareto multi-objective selection
qocc compile search --adapter qiskit --input examples/ghz.qasm --mode pareto --out search.zip

# Random search strategy
qocc compile search --adapter qiskit --input examples/ghz.qasm --strategy random --out search.zip

# Bayesian adaptive search (UCB acquisition)
qocc compile search --adapter qiskit --input examples/ghz.qasm --strategy bayesian --out search.zip

# Noise-aware surrogate scoring (provider-agnostic noise model JSON)
qocc compile search --adapter qiskit --input examples/ghz.qasm --noise-model examples/noise_model.json --out search.zip

# Detect nondeterminism (compile 5 times)
qocc trace run --adapter qiskit --input examples/ghz.qasm --repeat 5 --out nd.zip

# Replay a bundle
qocc trace replay bundle.zip --out replayed.zip

# Generate interactive HTML report from an existing bundle
qocc trace html --bundle bundle.zip --out report.html

# Generate HTML report directly during trace run
qocc trace run --adapter qiskit --input examples/ghz.qasm --html

# Optional notebook visualization dependencies
pip install -e ".[jupyter]"

# Run trace and auto-ingest into regression DB
qocc trace run --adapter qiskit --input examples/ghz.qasm --db

# Regression DB workflows
qocc db ingest bundle.zip
qocc db query --adapter qiskit --since 2026-01-01
qocc db tag bundle.zip --tag baseline

# Notebook helpers (inside Python/Jupyter)
python -c "import qocc; print(qocc.show_bundle('bundle.zip'))"
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

### Contract DSL

`qocc contract check --contracts` supports both JSON and `.qocc` DSL files.

```text
contract depth_budget:
    type: cost
    assert: depth <= 50
    assert: two_qubit_gates <= 100

contract tvd_check:
    type: distribution
    tolerance: tvd <= 0.05
    confidence: 0.99
    shots: 4096 .. 65536

contract parametric_budget:
    type: cost
    assert: depth <= input_depth - 2
    assert: proxy_error_score <= 1 - error_budget
```

Parametric values are resolved at evaluation time from bundle metrics and
contract fields (for example `input_depth`, `compiled_depth`, `baseline_tvd`,
and symbolic references like `error_budget`).

### Contract Composition

Composition is supported via JSON envelopes:

```json
{
    "name": "combined",
    "op": "all_of",
    "contracts": [
        {"name": "depth", "type": "cost", "resource_budget": {"max_depth": 50}},
        {"name": "dist", "type": "distribution", "tolerances": {"tvd": 0.05}}
    ]
}
```

Supported ops: `all_of`, `any_of`, `best_effort`, `with_fallback`.

- `best_effort` records inner contract results but does not fail overall.
- `with_fallback` switches to fallback when primary returns a
    `NotImplementedError`-class failure.

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

### Early Stopping (SPRT)

Set `resource_budget.early_stopping: true` with `min_shots` and `max_shots` to enable iterative sampling that halts when pass/fail is statistically certain. Uses a two-tier strategy: **SPRT** (Sequential Probability Ratio Test) for guaranteed Type I/II error bounds, with a CI-separation heuristic as fallback.

```json
{"name": "adaptive", "type": "distribution", "tolerances": {"tvd": 0.1},
 "resource_budget": {"early_stopping": true, "min_shots": 256, "max_shots": 8192,
                      "sprt_beta": 0.1}}
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

## OpenTelemetry Export

Export traces as OTLP-compatible JSON for ingestion by Jaeger, Grafana Tempo, Datadog, or any OpenTelemetry collector:

```python
from qocc.trace.exporters import export_otlp_json, export_to_otel_sdk

# OTLP JSON file (works standalone)
export_otlp_json(spans, "traces.otlp.json", service_name="qocc")

# Bridge to OpenTelemetry Python SDK (when opentelemetry-sdk installed)
export_to_otel_sdk(spans)  # Spans appear in any configured OTel exporter
```

## Caching

QOCC uses a content-addressed compilation cache keyed by `SHA-256(circuit_hash || pipeline_dict || backend_version || extra)` where `extra` can include search seed and noise model provenance hash. Cache hits are recorded in `cache_index.json` inside the bundle for reproducibility auditing. Cache hits now **skip recompilation entirely** by deserialising cached results.

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
| pytket  | ✅ Full (pass-sequence compile spans, deterministic JSON hash) |
| CUDA-Q  | 🔜 Optional |
| Stim/PyMatching | ✅ QEC mode (DEM + decoder stats metadata) |

## Development

```bash
pip install -e ".[dev]"
pytest                 # ~303 tests
ruff check .
mypy qocc/
```

## License

Apache 2.0 — see [LICENSE](LICENSE).
