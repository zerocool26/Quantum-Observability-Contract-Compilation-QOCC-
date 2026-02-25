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
- [ ] Observable contract (expectation preservation with CI)
- [ ] Distribution contract (TVD with bootstrap CI)
- [ ] Clifford/stabilizer contract (exact equivalence)
- [ ] Contract evaluation with proper statistical rigor
- [ ] Results stored and reported in bundle
- [ ] `contract check` CLI returns nonzero on failure
- [ ] Resource budgets (max shots, time, memory) enforced
- [ ] Early stopping when pass/fail is statistically certain
- [ ] Tests validate contract pass/fail behavior

### Key deliverables:
- `qocc contract check` CLI command
- `qocc.check_contract()` Python API
- `ContractSpec` / `ContractResult` data structures
- Sampling-based evaluation with Hoeffding and bootstrap CIs
- Integration with adapter simulation backends

---

## v0.3 — Closed-Loop Compilation Search
**Target: 16–32 weeks**

### Done when:
- [ ] Search generates ≥ N candidates by varying compilation params
- [ ] Surrogate scoring ranks candidates cheaply
- [ ] Expensive validation (simulation) runs on top-k
- [ ] Selection returns best candidate satisfying all contracts
- [ ] Bundle contains full candidate table + selection reasoning
- [ ] Content-addressed caching reduces repeated runs measurably
- [ ] Compare highlights regressions in chosen candidate over time

### Key deliverables:
- `qocc compile search` CLI command
- `qocc.search_compile()` Python API
- `Candidate` / `SearchSpaceConfig` data structures
- Surrogate scorer with pluggable cost models
- Cache index with hit/miss tracing

---

## Future (v0.4+)

### Additional adapters:
- [ ] pytket adapter
- [ ] CUDA-Q adapter (optional)
- [ ] Stim/PyMatching/sinter (QEC mode)

### Advanced features:
- [ ] GPU simulation backend integration
- [ ] QEC sampling mode
- [ ] Multi-objective Pareto selection
- [ ] Plugin system for custom adapters and evaluators
- [ ] Visualization / plotting in bundles
- [ ] CI/CD integration guides and GitHub Actions templates
- [ ] OpenTelemetry export (bridge to existing observability stacks)

---

## Versioning

QOCC follows [Semantic Versioning](https://semver.org/):
- **0.x.y** — Pre-1.0, API may change between minor versions
- **1.0.0** — Stable API for Python + CLI + bundle format
