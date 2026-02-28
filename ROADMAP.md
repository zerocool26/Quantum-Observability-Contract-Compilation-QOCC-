# QOCC Roadmap

## MVP (v0.1) — Quantum Trace Pack
**Target: 4–8 weeks**

### Done when:
- [x] Qiskit + Cirq ingestion, normalization, and compilation
- [x] Valid trace bundle emission with stable hashes
- [x] Metrics computed and stored per stage
- [x] CLI `trace run` works on example circuits
- [x] `trace compare` identifies deltas between bundles
- [x] Unit tests for hash stability + bundle integrity pass
- [x] JSON schemas for all bundle files
- [x] Two end-to-end examples (Qiskit + Cirq)

### Key deliverables:
- `qocc trace run` CLI command
- `qocc trace compare` CLI command
- `qocc.run_trace()` Python API
- `qocc.compare_bundles()` Python API
- Trace Bundle format (zip with manifest, env, seeds, circuits, metrics, trace)
- Deterministic hashing via QASM canonicalization

---

## v0.2 — Contracts
**Target: 8–16 weeks**

### Done when:
- [x] Observable contract (expectation preservation with CI)
- [x] Distribution contract (TVD with bootstrap CI)
- [x] Clifford/stabilizer contract (exact equivalence)
- [x] Contract evaluation with proper statistical rigor
- [x] Results stored and reported in bundle
- [x] `contract check` CLI returns nonzero on failure
- [x] Resource budgets (max shots, time, memory) enforced
- [x] Early stopping when pass/fail is statistically certain
- [x] Tests validate contract pass/fail behavior

### Key deliverables:
- `qocc contract check` CLI command
- `qocc.check_contract()` Python API
- `ContractSpec` / `ContractResult` data structures
- Sampling-based evaluation with Hoeffding and bootstrap CIs
- Chi-square and G-test (with Williams correction) alternatives
- Integration with adapter simulation backends

---

## v0.3 — Closed-Loop Compilation Search
**Target: 16–32 weeks**

### Done when:
- [x] Search generates ≥ N candidates by varying compilation params
- [x] Surrogate scoring ranks candidates cheaply
- [x] Expensive validation (simulation) runs on top-k
- [x] Selection returns best candidate satisfying all contracts
- [x] Bundle contains full candidate table + selection reasoning
- [x] Content-addressed caching reduces repeated runs measurably
- [x] Compare highlights regressions in chosen candidate over time

### Key deliverables:
- `qocc compile search` CLI command
- `qocc.search_compile()` Python API
- `Candidate` / `SearchSpaceConfig` data structures
- Surrogate scorer with pluggable cost models
- Cache index with hit/miss tracing
- Multi-objective Pareto selection mode (`--mode pareto`)

---

## v0.4 — Implemented Extensions

### Done:
- [x] Multi-objective Pareto selection
- [x] Plugin system for custom adapters and evaluators (entry-point based)
- [x] ASCII timeline visualization
- [x] Nondeterminism detection (`--repeat N`)
- [x] Bundle replay (`qocc trace replay`)
- [x] Regression-cause analysis in `compare_bundles`
- [x] Per-stage trace spans in adapter `compile()`
- [x] Statevector metadata from `simulate()`
- [x] 11 JSON schemas with full bundle validation
- [x] Exact statevector equivalence contract
- [x] Cost/resource budget contract

---

## Future (v0.5+)

### Additional adapters:
- [x] pytket adapter
- [ ] CUDA-Q adapter (optional)
- [x] Stim/PyMatching/sinter (QEC mode)
- [x] IBM Quantum Runtime adapter (`execute()` + job polling spans)
- [x] Asynchronous hardware job tracking (`qocc trace watch` with timeout and on-complete hook)
- [x] Evolutionary search strategy (`--strategy evolutionary` with per-generation spans)
- [x] Bayesian transfer-learning prior (`search_history.json`, half-life weighting)
- [x] Multi-circuit batch search (`qocc compile batch`, `batch_search_compile()`)
- [x] Zero-noise extrapolation contract (`ContractType.ZNE`)
- [x] Error mitigation pipeline stage (`MitigationSpec`, mitigation span, overhead telemetry)
- [x] Project bootstrap wizard (`qocc init` with contracts/pipeline/CI scaffolding)

### Advanced features:
- [ ] GPU simulation backend integration
- [ ] QEC sampling mode
- [x] CI/CD integration guides and GitHub Actions templates
- [x] OpenTelemetry OTLP JSON export (bridge to existing observability stacks)
- [x] Bayesian optimization for compilation search (`--strategy bayesian`)
- [x] Noise-model-aware surrogate scoring + provenance hashing
- [x] SPRT early stopping (statistically optimal termination)
- [x] Parallel candidate compilation (ThreadPoolExecutor)
- [x] Random search strategy (`--strategy random`)
- [x] Regression tracking database (`qocc db` + `trace run --db`)
- [x] Interactive HTML trace viewer (`qocc trace html` + `trace run --html`)
- [x] Jupyter widget integration (`qocc.show_bundle`, `qocc.compare_interactive`, `qocc.search_dashboard`)
- [x] Contract DSL support (`.qocc` files in `contract check` and API)
- [x] Parametric contracts (evaluation-time symbolic tolerance resolution)
- [x] Contract composition (`all_of`, `any_of`, `best_effort`, `with_fallback`)
- [x] Contract result caching (keyed by circuit/spec/shots/seed with max-age policy)
- [x] Hardware execution adapter interface (`ExecutionResult`, optional `execute()`, hardware count ingestion in contract checks)
- [x] Developer documentation site scaffold (`docs/`, API/tutorials/architecture/contracts/CLI references)
- [x] Bundle signing & provenance (`qocc bundle sign/verify`, Ed25519 signatures)

## v0.6 — Phase 17-19 Advanced Features

### Done:
- [x] Bundle Diff Format (Machine Readable structured diff format)
- [x] Native gRPC OTLP Tracing span streams
- [x] ZSTD/LZ4 Streaming Bundle compression and export mapping
- [x] Advanced Statistics: Permutation tests, FDR correction (Benjamini-Hochberg), KS-Test, JS Divergence inside contract evaluation
- [x] CI Coverage Calibration module (empirically validates CI rules and math coverage)
- [x] QPT (Quantum Process Tomography) proxy contracts and RB bounds
- [x] Cross-adapter compatibility CLI (`qocc cross-check` mapping N×N matrix)
- [x] `TopologyGraph` metrics mapping loading from IBMQ / JSON configs
- [x] ContractType.TOPOLOGY_VIOLATIONS for routing integrity
- [x] OpenQASM Bridge / Format converter API (`convert()`)
---

## Versioning

QOCC follows [Semantic Versioning](https://semver.org/):
- **0.x.y** — Pre-1.0, API may change between minor versions
- **1.0.0** — Stable API for Python + CLI + bundle format
