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
- [ ] pytket adapter
- [ ] CUDA-Q adapter (optional)
- [ ] Stim/PyMatching/sinter (QEC mode)

### Advanced features:
- [ ] GPU simulation backend integration
- [ ] QEC sampling mode
- [ ] CI/CD integration guides and GitHub Actions templates
- [x] OpenTelemetry OTLP JSON export (bridge to existing observability stacks)
- [x] Bayesian optimization for compilation search (`--strategy bayesian`)
- [x] SPRT early stopping (statistically optimal termination)
- [x] Parallel candidate compilation (ThreadPoolExecutor)
- [x] Random search strategy (`--strategy random`)

---

## v0.5 — Phase 5 Improvements

### Done:
- [x] Cache actually skips recompilation (CompileResult.from_dict)
- [x] Seeds threaded into compilation pipeline
- [x] QASM canonicalization rewritten (commuting gate sort, float normalisation)
- [x] Deep copy in normalize_circuit prevents shared mutable state
- [x] Parallel compilation via ThreadPoolExecutor
- [x] OpenTelemetry OTLP JSON export for Jaeger/Grafana/Datadog
- [x] OpenTelemetry SDK bridge (when opentelemetry-sdk installed)
- [x] Bayesian UCB-based search optimizer (numpy-only, no sklearn)
- [x] Random search strategy with deduplication
- [x] SPRT (Sequential Probability Ratio Test) early stopping
- [x] ContractType enum with validation
- [x] Metric key alignment (visualization matches compute output)
- [x] Circular dependency fix (eval_sampling no longer imports api)
- [x] Memory tracking via tracemalloc in Qiskit adapter
- [x] --strategy CLI flag for compile search
- [x] 75 new Phase 5 tests (233 total, 1 skipped)

---

## Versioning

QOCC follows [Semantic Versioning](https://semver.org/):
- **0.x.y** — Pre-1.0, API may change between minor versions
- **1.0.0** — Stable API for Python + CLI + bundle format
