Advanced Development Prompt: QOCC — Phase 10+ Roadmap & Next-Generation Expansion

You are continuing development of QOCC (Quantum Observability + Contract-Based Compilation), an open-source, vendor-agnostic, trace-first observability and contract compilation system for quantum workflows. You have full context on Phases 1–9. Everything in `prompt1.md` remains in force as the foundational specification.

---

## Current State Summary (Phases 1–9 Complete)

All core systems are operational and hardened:
- **Adapters**: Qiskit (full), Cirq (full), pytket/Stim/CUDA-Q (stubs)
- **Contracts**: distribution (TVD/chi-square/G-test), observable (Hoeffding CI), clifford (stabilizer), exact (statevector fidelity), cost (resource budget)
- **Search**: Bayesian UCB, random, Pareto multi-objective, SPRT early stopping
- **Tracing**: OpenTelemetry-compatible OTLP JSON export, thread-safe TraceEmitter, per-stage spans
- **CLI**: `trace run`, `trace compare`, `trace replay`, `contract check`, `compile search`, `validate`
- **Infrastructure**: Content-addressed cache, plugin system (entry-point based), ZipSlip-hardened bundle I/O, path-injection guards, PEP 561 `py.typed` marker
- **Tests**: 333 passing, 5 skipped

---

## Phase 10 — Full Backend Expansion

### 10.1 pytket Adapter (Production-Grade)

Implement a complete pytket adapter (`qocc/adapters/tket_adapter.py`) that goes beyond the current stub:

- `ingest(input)`: Accept `.qasm`, `pytket.Circuit`, or JSON-serialised tket circuits
- `normalize(circuit)`: Apply `pytket.passes.SequencePass([RemoveRedundancies(), CommuteThroughMultis()])` as the canonical normalization; preserve register names
- `compile(circuit, pipeline_spec)`: Accept tket `PassManager`-style config; emit one span per `BasePass` in the sequence with `pass_name`, `time_ms`, and `metrics_delta`
- `simulate(circuit, sim_spec)`: Use `pytket.extensions.qulacs` or `pytket.extensions.projectq` when installed (mark as optional dep `tket` extra)
- `get_metrics(circuit)`: Extract `n_qubits`, `n_gates`, `n_2q_gates`, `depth`, `gate_histogram`, `two_qubit_depth`
- `hash(circuit)`: Canonicalize via tket's built-in JSON serialization + SHA-256; must be deterministic across process restarts
- `describe_backend()`: Return compiler version, active backend extension, and pass set hash

Register under `qocc.adapters` entry-point as `tket`.

### 10.2 Stim / PyMatching / Sinter Adapter (QEC Mode)

Implement `qocc/adapters/stim_adapter.py` for quantum error correction workflows:

- **Circuit model**: Stim circuits (`.stim` files or `stim.Circuit` objects); treat each Stim instruction as a "gate" for metric purposes
- `ingest(input)`: Accept `.stim` files or `stim.Circuit`; populate `CircuitHandle` with `framework = "stim"`, `qasm3 = None` (Stim has no QASM3 equivalent — store Stim text in `metadata["stim_text"]`)
- `compile(circuit, pipeline_spec)`: Apply Stim detector error model (DEM) generation as the "compilation" step; record each logical/physical qubit mapping decision as a span
- `simulate(circuit, sim_spec)`: Use `stim.TableauSimulator` for exact simulation; use `stim.CompiledSampler` for shot sampling; record logical error rate, syndrome weight distribution, and decoder round counts
- **PyMatching integration**: When `pymatching` is installed, run MWPM decoding; emit spans for `build_matching_graph`, `decode_batch`, `logical_error_rate`; record matching graph edge weights and node counts in trace attributes
- **Sinter integration**: When `sinter` is installed, wrap `sinter.collect()` as a `simulate()` backend; record `shots`, `errors`, `discards`, `seconds`, `custom_counts` in metrics
- **QEC contract type**: Add `ContractType.QEC` with evaluator `eval_qec.py`:
  - `logical_error_rate_threshold`: circuit must achieve logical error rate below threshold at given physical error rate
  - `code_distance`: Stim circuit must encode a distance-d code (verify via minimum-weight detector path)
  - `syndrome_weight_budget`: mean syndrome weight per round must be within tolerance
- **Bundle additions for QEC**: Add `dem.json` (detector error model), `logical_error_rates.json`, `decoder_stats.json` to the bundle schema

### 10.3 CUDA-Q Adapter (Optional GPU Backend)

Implement `qocc/adapters/cudaqadapter.py` for NVIDIA CUDA-Q:

- `ingest(input)`: Accept CUDA-Q kernel functions or `.qasm` files; wrap in `CircuitHandle` with `framework = "cudaq"`
- `simulate(circuit, sim_spec)`: Use `cudaq.sample()` / `cudaq.observe()` with configurable `shots_count`; record GPU device ID, VRAM usage, and kernel compilation time as trace attributes
- `compile(circuit, pipeline_spec)`: Map to CUDA-Q target selection (`nvidia`, `tensornet`, `density-matrix-cpu`, etc.); emit one span per target with compile latency
- **GPU memory contract**: Add `resource_budget.max_vram_mb` to `CostSpec`; CUDA-Q adapter must enforce it via `cudaq.get_state()` memory introspection
- Mark entire adapter as optional extra `[cudaq]`; gracefully import-guard with `logger.warning` if not installed

---

## Phase 11 — Hardware Execution Layer

### 11.1 Hardware Execution Adapter Interface

Extend `AdapterBase` with an optional `execute()` method for real quantum hardware:

```python
def execute(
    self,
    circuit: CircuitHandle,
    backend_spec: dict[str, Any],
    shots: int = 1024,
    emitter: TraceEmitter | None = None,
) -> ExecutionResult:
    """Submit to real hardware and return results with job metadata."""
```

- `ExecutionResult` dataclass: `job_id`, `backend_name`, `shots`, `counts`, `metadata`, `queue_time_s`, `run_time_s`, `error_mitigation_applied`
- Every hardware submission must emit spans: `job_submit`, `queue_wait`, `job_complete`, `result_fetch`
- Record `job_id`, `provider`, `backend_version`, `basis_gates`, `coupling_map_hash` in trace attributes
- Job polling must be non-blocking: emit `job_polling` events at configurable intervals; record total wall time
- Hardware results feed directly into contract evaluation (distribution contract on real counts)

### 11.2 IBM Quantum / Qiskit Runtime Adapter

Implement `qocc/adapters/ibm_adapter.py`:

- Uses `qiskit_ibm_runtime` (optional extra `[ibm]`)
- Supports both `SamplerV2` and `EstimatorV2` primitives
- Records `session_id`, `backend_name`, `backend_version`, `job_id`, `error_budget` in trace
- Applies `qiskit.transpile()` for hardware-native gate decomposition before submission; records transpile metrics as a compilation span
- Stores raw job result JSON in bundle under `hardware/ibm_result.json`
- Implements `execute()` with automatic job polling and timeout handling

### 11.3 Asynchronous Job Tracking

Add `qocc trace watch --bundle bundle.zip --poll-interval 5` CLI command:

- Monitors pending hardware jobs recorded in a bundle
- Polls provider APIs and updates `hardware/` results in-place
- Emits completion events to the existing trace when jobs finish
- Supports `--timeout` and `--on-complete "qocc contract check ..."` for chained automation

---

## Phase 12 — Advanced Search & Optimization

### 12.1 Evolutionary / Genetic Search Strategy

Add `--strategy evolutionary` to `qocc compile search`:

- Implement `qocc/search/evolutionary.py` with tournament selection, single-point crossover on pipeline param vectors, and Gaussian mutation
- Population size, mutation rate, and crossover rate are configurable in `SearchSpaceConfig`
- Fitness function = surrogate score (same as Bayesian mode, pluggable)
- Elitism: carry forward top-k individuals unchanged each generation
- Emit one span per generation with `generation`, `best_score`, `population_diversity` attributes
- Terminate on: max generations, convergence (std-dev of fitness < threshold), or wall-clock budget

### 12.2 Noise-Aware Surrogate Scoring

Extend `qocc/search/scorer.py` with noise-aware scoring:

- Accept a `noise_model` parameter (provider-agnostic dict with `single_qubit_error`, `two_qubit_error`, `readout_error`, `t1`, `t2` per qubit)
- Compute: `noise_score = Σ_gates p_error(gate, qubit) + Σ_idle_periods (1 - exp(-t_idle/T2))`
- Integrate T1/T2 decoherence into duration-weighted error estimate
- Store noise model hash in surrogate score provenance; different noise models are different cache keys
- Add `NoiseModel` dataclass to `qocc/metrics/noise_model.py` with JSON schema `noise_model.schema.json`

### 12.3 Transfer Learning / Historical Prior for Bayesian Search

Extend `BayesianSearchOptimizer` with cross-circuit transfer:

- Persist UCB model state (observed X/Y arrays) as `search_history.json` in the user cache dir
- When starting a new search, load historical observations for the same adapter + backend version
- Treat previous results as warm-start prior observations (downweighted by age: `weight = exp(-days_old / half_life)`)
- Emit `prior_loaded` and `prior_size` as trace attributes on the Bayesian optimizer span
- Configurable via `--prior-half-life` CLI flag (default: 30 days)

### 12.4 Multi-Circuit Batch Search Mode

Add `qocc compile batch` CLI command and `batch_search_compile()` API:

- Accept a manifest JSON listing N circuits with per-circuit contract specs and pipeline hints
- Run search in parallel across all circuits (configurable worker count)
- Shared cache across the batch — if circuit A and circuit B share a common subcircuit, the compiled subcircuit is reused
- Output a batch bundle with per-circuit result summaries and a cross-circuit metric table
- Emit a top-level batch span with `n_circuits`, `n_cache_hits`, `total_candidates_evaluated`

---

## Phase 13 — Observability & Visualization Upgrades

### 13.1 Interactive HTML Trace Viewer

Implement `qocc/trace/html_report.py` with `export_html_report(bundle_path, output_path)`:

- Self-contained single-file HTML (no external CDN dependencies for air-gapped environments)
- **Flame chart**: Gantt-style timeline of all spans, color-coded by module (adapters=blue, contracts=green, search=orange, cache=gray); hover shows full attribute dict
- **Metric dashboard**: Per-candidate bar charts for depth, 2Q gates, proxy error score, duration
- **Contract results panel**: Pass/fail badges with CI visualization (confidence interval bars)
- **Diff view**: Side-by-side metric table when comparing two bundles (green/red for improvements/regressions)
- **Circuit diff**: Show gate histogram overlaid for input vs. selected candidate
- Activate via `qocc trace html --bundle bundle.zip --out report.html` or `--html` flag on `trace run`

### 13.2 Jupyter Widget Integration

Implement `qocc/trace/jupyter_widget.py` as an optional extra `[jupyter]`:

- Uses `ipywidgets` + `plotly` for interactive in-notebook visualization
- `qocc.show_bundle(bundle_path)` renders the flame chart inline
- `qocc.compare_interactive(bundle_a, bundle_b)` renders side-by-side metric sliders
- `qocc.search_dashboard(search_result)` renders candidate Pareto scatter plot with hover tooltips
- All widgets are read-only views — no state mutation from the widget

### 13.3 Regression Tracking Database

Implement `qocc/core/regression_db.py` with `RegressionDatabase`:

- SQLite-backed (zero server dependency): stores bundle summary rows (run_id, adapter, circuit_hash, metrics snapshot, contract results, timestamp)
- `db.ingest(bundle_path)` — parse bundle, extract key metrics, write one row per candidate
- `db.query(circuit_hash=..., adapter=..., since=...)` — return DataFrame-compatible dict list
- `db.detect_regressions(new_bundle, baseline_tag=...)` — compare new bundle against tagged historical baseline; flag metrics that regressed beyond a configurable delta threshold
- CLI: `qocc db ingest bundle.zip`, `qocc db query --circuit-hash abc123 --since 2025-01-01`, `qocc db tag bundle.zip --tag baseline`
- Integration: `qocc trace run --db` auto-ingests result into default DB at `~/.qocc/regression.db`

---

## Phase 14 — Contract System Expansion

### 14.1 Contract DSL (Domain-Specific Language)

Add a human-readable contract definition language parsed into `ContractSpec` objects:

```
# QOCC Contract DSL v1
contract tvd_check:
  type: distribution
  tolerance: tvd <= 0.05
  confidence: 0.99
  shots: 4096 .. 65536  # min..max for SPRT

contract depth_budget:
  type: cost
  assert: depth <= 50
  assert: two_qubit_gates <= 100

contract qec_threshold:
  type: qec
  assert: logical_error_rate <= 1e-6 @ physical_error_rate = 1e-3
  assert: code_distance >= 5
```

- Parser: `qocc/contracts/dsl.py` with `parse_contract_dsl(text: str) -> list[ContractSpec]`
- CLI: `--contracts` flag on `contract check` accepts both JSON and `.qocc` DSL files
- Error messages must pinpoint line/column of invalid syntax

### 14.2 Parametric Contracts

Extend `ContractSpec` to support parametric tolerances:

- `tolerance: tvd <= 0.1 * baseline_tvd` — tolerance relative to a stored baseline metric
- `tolerance: depth <= compiled_depth + 10` — tolerance relative to compiled output
- `assert: fidelity >= 1 - error_budget` — symbolic references to other contract fields
- Parametric values resolved at evaluation time against the current bundle metrics

### 14.3 Contract Composition and Inheritance

Add contract composition operators:

- `all_of([c1, c2, c3])` — passes only when all sub-contracts pass (AND)
- `any_of([c1, c2, c3])` — passes when at least one passes (OR, with logging)
- `best_effort(contract)` — evaluates contract but never causes overall failure; records result only
- `with_fallback(primary, fallback)` — use `fallback` evaluation strategy if `primary` raises `NotImplementedError`
- Serializable to/from JSON using `{"op": "all_of", "contracts": [...]}` envelope

### 14.4 Contract Result Caching

Cache contract evaluation results (not just compilation) in `CompilationCache`:

- Cache key = `SHA-256(circuit_hash || contract_spec_hash || shots || seed)`
- Avoids re-running expensive sampling when the same circuit + contract was already evaluated
- Cached results include: pass/fail, point estimate, CI, test statistic, shot count used
- Respect `max_cache_age` parameter — do not use cached results older than N days (stale noise model assumption)

---

## Phase 15 — Noise Modeling & Error Mitigation Integration

### 15.1 Noise Model Registry

Implement `qocc/metrics/noise_model.py` with `NoiseModelRegistry`:

- Load noise models from JSON files (schema: `noise_model.schema.json`)
- Built-in: uniform depolarizing model, thermal relaxation model, readout error model
- User-provided: accept `noise_model` path on CLI (`--noise-model hardware_calibration.json`)
- Validate all models against the schema before use
- Record noise model hash in trace attributes for reproducibility

### 15.2 Zero-Noise Extrapolation (ZNE) Contract

Add `ContractType.ZNE`:

- Runs circuit at noise scale factors `[1.0, 1.5, 2.0, 2.5]` (configurable)
- Fits Richardson extrapolation to estimate zero-noise expectation value
- Contract passes if extrapolated value is within tolerance of ideal (simulated) value
- Emit one span per noise level; store extrapolation coefficients in contract result details

### 15.3 Error Mitigation Pipeline Stage

Add optional `mitigation` stage to the compilation pipeline:

- Supported methods (adapter-configurable): `twirling`, `pec` (probabilistic error cancellation), `zne`, `m3_readout`
- Emit mitigation as a first-class span with method, parameters, and overhead factor
- Record mitigation overhead (shot multiplier, runtime multiplier) in metrics
- `MitigationSpec` dataclass: `method`, `params`, `overhead_budget`; serializable to/from JSON

---

## Phase 16 — CI/CD & Developer Experience

### 16.1 GitHub Actions Templates

Create `examples/ci/`:

- `qocc_baseline.yml` — on push: run trace, check contracts, auto-ingest to regression DB; fail PR if regressions detected
- `qocc_benchmark.yml` — nightly: run batch search across benchmark suite, post metric table as GitHub Actions summary
- `qocc_pr_check.yml` — on PR: compare compile output against `main` baseline bundle; post diff table as PR comment using `gh` CLI
- Each template is parameterized with `workflow_dispatch` inputs for adapter, circuit path, contract file

### 16.2 Pre-Commit Hook Integration

Add `examples/ci/pre_commit_config.yaml` with a QOCC hook:

```yaml
- repo: local
  hooks:
    - id: qocc-contract-check
      name: QOCC Contract Check
      entry: qocc contract check --bundle .qocc_baseline.zip --contracts contracts/ci_contracts.qocc
      language: system
      pass_filenames: false
```

Document hook usage in `CONTRIBUTING.md`.

### 16.3 `qocc init` CLI Command

Add `qocc init` interactive setup wizard:

- Detects installed backends (Qiskit, Cirq, pytket, Stim) and pre-fills adapter config
- Generates `contracts/default_contracts.qocc` with sensible defaults based on detected backend
- Creates `pipeline_examples/` with backend-appropriate default pipeline JSON
- Generates `.github/workflows/qocc_ci.yml` template
- Offers to run `qocc trace run` immediately on a demo circuit to validate setup
- Stores project config in `pyproject.toml` under `[tool.qocc]`

### 16.4 Developer Documentation Site

Generate a documentation scaffold in `docs/` (Sphinx or MkDocs-compatible):

- API reference auto-generated from docstrings
- Tutorials: "Your first trace bundle", "Writing contracts", "Debugging regressions", "Adding a custom adapter"
- Architecture deep-dive: trace model, bundle format, search pipeline
- Contract reference: all contract types with statistical methodology explained
- CLI reference: all commands, flags, exit codes

---

## Phase 17 — Bundle Format Evolution & Interoperability

### 17.1 Bundle Signing & Provenance

Implement `qocc/core/signing.py`:

- Sign bundles with Ed25519 (using `cryptography` package as optional extra `[signing]`)
- `sign_bundle(bundle_path, private_key_path)` — adds `signature.json` to bundle containing: signer identity, timestamp, manifest hash, and Ed25519 signature over `SHA-256(manifest.json)`
- `verify_bundle(bundle_path, public_key_path)` — returns `VerificationResult` with valid/invalid status and signer info
- CLI: `qocc bundle sign --key mykey.pem bundle.zip`, `qocc bundle verify bundle.zip`
- Signed bundles include signer in `manifest.json` `provenance` field
- Do NOT store private keys in bundles; only the public key fingerprint

### 17.2 OpenTelemetry Native Export

Extend `qocc/trace/exporters.py` with a gRPC OTLP exporter:

- `export_otlp_grpc(spans, endpoint, headers)` — streams spans to any OTLP-compatible collector (Jaeger, Grafana Tempo, Datadog Agent, OpenTelemetry Collector) via gRPC
- Uses `opentelemetry-exporter-otlp-proto-grpc` as optional dep `[otel]`
- Add `--otel-endpoint` flag to `trace run` for real-time span streaming during compilation
- Spans must carry QOCC-specific semantic conventions as resource attributes: `quantum.adapter`, `quantum.circuit_hash`, `quantum.n_qubits`

### 17.3 Bundle Diff Format (Machine-Readable)

Extend `compare_bundles()` to output a structured `BundleDiff` format:

- `BundleDiff` dataclass: `metric_deltas` (dict), `contract_regressions` (list), `pass_log_diffs` (list), `env_diffs` (dict), `circuit_hash_change` (bool), `regression_cause` (enum: TOOL_VERSION, PASS_PARAM, SEED, ROUTING, UNKNOWN)
- Schema: `bundle_diff.schema.json`
- CLI: `qocc trace compare --format diff bundle_a.zip bundle_b.zip` outputs `diff.json` with the `BundleDiff` structure
- Machine-readable diff feeds directly into regression DB and CI checks

### 17.4 Bundle Compression & Streaming

- Support streaming bundle creation (write zip incrementally, not all-at-once) for very large trace files
- Add `--compress {none,zstd,lz4}` flag; default `zstd` when `zstandard` installed, else `deflate`
- Add `--max-bundle-size-mb` guard to abort (with clean error) if bundle would exceed size limit
- Implement `ArtifactStore.stream_bundle(callback)` for real-time bundle streaming to object storage

---

## Phase 18 — Advanced Statistical Features

### 18.1 Quantum Process Tomography Contract

Add `ContractType.QPT` for process tomography verification:

- Uses randomized benchmarking (RB) or direct state tomography inputs
- Estimate process fidelity between compiled and ideal unitary
- `qpt_fidelity >= threshold` as the pass condition
- Statistical rigor: report confidence interval on process fidelity estimate
- Efficient implementation for small circuits (≤ 5 qubits) using state tomography; RB for larger

### 18.2 Robust Statistical Tests

Extend `qocc/contracts/stats.py`:

- **Kolmogorov-Smirnov test** as an alternative to chi-square for distribution comparison
- **Jensen-Shannon divergence** as an additional distance metric (smooth, symmetric, bounded)
- **Permutation test** for distribution equivalence (non-parametric, exact for small shot counts)
- **FDR correction** (Benjamini-Hochberg) for multi-contract evaluation — controls false discovery rate when checking many contracts simultaneously
- All new tests available as `spec.test` values in contract JSON; document each with: null hypothesis, assumptions, recommended shot count, known limitations

### 18.3 Confidence Interval Calibration

Add `qocc/contracts/calibration.py`:

- `calibrate_ci_coverage(evaluator, n_trials, true_param)` — empirically validate that the CI method achieves stated coverage probability
- Runs `n_trials` synthetic experiments and measures actual CI coverage
- Stores calibration results in `calibration_report.json`; warns if coverage deviates from nominal by >2%
- Supports Hoeffding, bootstrap, and normal approximation methods

---

## Phase 19 — Multi-Backend Circuit Portability

### 19.1 Cross-Adapter Equivalence Testing

Add `qocc.cross_check(circuit, adapters, contract)` API:

- Compiles and simulates the same circuit through multiple adapters simultaneously
- Checks distribution contract between every pair of adapter outputs
- Outputs a cross-adapter compatibility matrix (N×N pass/fail table)
- CLI: `qocc cross-check --adapters qiskit,cirq,tket --input circuit.qasm --contract dist_contract.qocc`
- Useful for verifying that a circuit is portable across quantum SDKs

### 19.2 Circuit Format Bridge

Implement `qocc/core/format_bridge.py`:

- `convert(circuit_handle, target_format)` — convert between QASM2, QASM3, Qiskit IR, Cirq JSON, tket JSON, Stim
- Uses each adapter's `ingest()`/`export()` as conversion primitives
- Records conversion provenance as a span (source format → target format, lossless/lossy flag)
- Lossy conversions (e.g., non-Clifford → Stim) emit a `WARNING` event in trace with `lost_operations` list

### 19.3 Hardware Topology Mapping

Extend `qocc/metrics/topology.py`:

- `TopologyGraph` dataclass: nodes (qubits), edges (couplings), `t1`, `t2`, `readout_fidelity` per node
- `TopologyGraph.from_ibm_backend(backend)` — load live topology from IBM Quantum
- `TopologyGraph.from_json(path)` — load from JSON file (schema: `topology.schema.json`)
- Violations contract: `ContractType.TOPOLOGY_VIOLATIONS` — passes only if 0 SWAP insertions required beyond budget
- Routing analysis: report which gates required SWAP insertion and estimated routing overhead

---

## Cross-Cutting Requirements (All Phases)

### Engineering Standards (Strictly Enforced)

- **Type hints**: All new code must have complete type annotations; `mypy --strict` on new modules
- **Docstrings**: All public functions, classes, and modules require Google-style docstrings
- **Test coverage**: Every new feature must have tests before merging; aim for >90% line coverage on new modules
- **Schema-first**: Every new bundle file must have a JSON schema in `qocc/schemas/` registered in `schemas.py`
- **Backward compatibility**: Bundle format changes must increment schema version; old bundles must remain loadable
- **Trace-first**: Every new stage must emit spans with `name`, `start_time`, `end_time`, `attributes`; no silent operations
- **No hidden state**: All configuration must be serializable to bundle files; bundles must be self-contained
- **Security**: No shell injection (use `subprocess` with list args, not strings); validate all user inputs at API boundaries

### Performance Targets

- `trace run` on a 20-qubit Qiskit circuit: < 2 seconds wall time (compilation + bundle write)
- `compile search` with 50 candidates, Bayesian strategy: < 30 seconds (simulation excluded)
- `trace compare` on two 10 MB bundles: < 500 ms
- Cache lookup (hit path): < 10 ms
- HTML report generation: < 5 seconds for 1000-span traces

### Dependency Policy

- Core QOCC (`pip install qocc`) must have ≤5 hard dependencies: `click`, `jsonschema`, `numpy`, `rich`, `platformdirs`
- All backend integrations are optional extras: `[qiskit]`, `[cirq]`, `[tket]`, `[stim]`, `[cudaq]`, `[ibm]`, `[otel]`, `[jupyter]`, `[signing]`
- No new mandatory dependencies without explicit justification in PR description
- All optional deps must be gracefully import-guarded with informative `ImportError` messages

### Test Strategy

- **Unit tests**: Pure function tests with no I/O; mock all filesystem and adapter calls
- **Integration tests**: Use the mock adapter (already present) for full pipeline tests without real backends
- **Property-based tests**: Use `hypothesis` for hash stability, contract monotonicity, and metric bound invariants
- **Regression tests**: Golden-file tests that compare output bundles against checked-in expected values
- **Performance benchmarks**: `pytest-benchmark` for critical hot paths (hashing, cache lookup, span serialization)

---

## Immediate Next Actions (Phase 10 Sprint)

Execute in this order — each must pass all existing tests before starting the next:

1. **Implement full pytket adapter** with per-pass span emission and unit tests
2. **Implement Stim/PyMatching adapter** with QEC contract type and sinter integration
3. **Add noise model registry** and wire into surrogate scorer
4. **Implement regression tracking database** (SQLite) with `qocc db` CLI
5. **Build HTML trace viewer** as a self-contained single-file report
6. **Add GitHub Actions CI templates** to `examples/ci/`
7. **Implement `qocc init` wizard**
8. **Update ROADMAP.md, CHANGELOG.md, README.md** to reflect new capabilities

After each phase, run the full test suite (`pytest -v`) and confirm all 333+ tests still pass before proceeding.

---

## Output Required from You (the Implementer)

For each phase implemented:

1. All new source files with complete, production-ready implementations
2. Updated JSON schemas for any new bundle files
3. New test files (`tests/test_phase10_*.py`, etc.) with ≥25 tests per phase
4. Updated `CHANGELOG.md` entries (Unreleased section per phase)
5. Updated `ROADMAP.md` checkboxes
6. Updated `README.md` feature table and quick-start examples
7. Updated `pyproject.toml` for any new optional extras or dependencies

The project must remain fully installable and all tests must pass after every phase.
