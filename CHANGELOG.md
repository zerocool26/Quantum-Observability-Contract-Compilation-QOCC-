# Changelog

All notable changes to the QOCC project are documented here.

## [Unreleased] — Phase 15 (Error Mitigation Pipeline Stage)

### Added
- **`MitigationSpec` dataclass** (`qocc.core.circuit_handle`) with
  serializable `method`, `params`, and `overhead_budget` fields.
- **Mitigation pipeline stage integration** in `run_trace()` and
  `search_compile()` candidate compilation flow.
- **First-class `mitigation` span** with method, parameters, overhead budget,
  `shot_multiplier`, `runtime_multiplier`, and combined `overhead_factor`.
- **Mitigation telemetry fields in metrics**:
  `mitigation`, `mitigation_shot_multiplier`,
  `mitigation_runtime_multiplier`, `mitigation_overhead_factor`.
- **Phase 15.3 tests** in `tests/test_phase15_mitigation.py`.

### Changed
- `PipelineSpec` now supports optional top-level `mitigation` config and
  preserves it through `to_dict()`/`from_dict()` round-trips.
- Metrics schema definitions now include mitigation overhead keys.

### Validation
- Full test suite: **454 passed, 5 skipped, 3 warnings**.

## [Unreleased] — Phase 10 (Backend Expansion, Part 1)

### Added
- **Production `tket` adapter** (`qocc.adapters.tket_adapter`) implementing:
  ingest (`.qasm`, JSON, native circuit), canonical normalization
  (`RemoveRedundancies` + `CommuteThroughMultis`), per-pass compile spans,
  optional-extension simulation hooks (`qulacs`/`projectq`), deterministic
  JSON-based hashing, backend metadata (`active_extension`, `pass_set_hash`).
- **Adapter fallback wiring** — `get_adapter("tket")` now lazy-imports
  the built-in `TketAdapter`.
- **Packaging support for pytket** — new optional dependency extra
  `qocc[tket]` and adapter entry-point registration under
  `[project.entry-points."qocc.adapters"]`.
- **CLI support** — `qocc trace run --adapter tket` is now accepted.
- **Phase 10 tket test suite** (`tests/test_phase10_tket_adapter.py`) with
  mocked pytket objects covering ingest/normalize/compile/simulate/export,
  helper utilities, adapter fallback, and backend metadata.

### Validation
- Full test suite passes after changes.

## [Unreleased] — Phase 10 (Backend Expansion, Part 2)

### Added
- **Production `stim` adapter** (`qocc.adapters.stim_adapter`) implementing
  ingest (`.stim` path, raw text, native circuit), DEM generation during
  compile, shot sampling, syndrome-weight distribution, logical error-rate
  metadata, and optional `pymatching`/`sinter` integration hooks.
- **QEC contract type** (`ContractType.QEC`) and evaluator
  (`qocc.contracts.eval_qec:evaluate_qec_contract`) with checks for:
  logical error-rate threshold, code distance floor, and syndrome weight budget.
- **QEC bundle schema artifacts**: `dem.json`, `logical_error_rates.json`,
  `decoder_stats.json` with schema files in `schemas/` and validation wiring.

### Changed
- `check_contract()` and search-time candidate contract evaluation now support
  `qec` contracts.
- Search validator now preserves simulation metadata in
  `candidate.validation_result` for downstream evaluators.
- `qocc trace run --adapter` now accepts `stim`; adapter registry fallback
  supports `get_adapter("stim")`.

### Packaging
- Added optional extra: `qocc[stim]`.
- Added adapter entry-point: `stim = qocc.adapters.stim_adapter:StimAdapter`.
- Added evaluator entry-point: `qec = qocc.contracts.eval_qec:evaluate_qec_contract`.

## [Unreleased] — Phase 10 (Search Intelligence, Part 3)

### Added
- **Noise model registry** (`qocc.metrics.noise_model`) with provider-agnostic
  `NoiseModel` dataclass, built-in presets, schema-validated file loading,
  and deterministic `stable_hash()` for provenance.
- **Noise model schema**: `schemas/noise_model.schema.json` plus in-memory
  registry export support via `qocc.core.schemas.NOISE_MODEL_SCHEMA`.
- **CLI support** for noise-aware search scoring:
  `qocc compile search --noise-model noise.json`.

### Changed
- `SearchSpaceConfig` now supports optional `noise_model` payload.
- Surrogate scoring now accepts optional noise model context and computes a
  `noise_score` term (gate error + readout + decoherence) with model hash
  attached to scoring metadata.
- Search cache provenance now factors in `noise_model_hash`, so different
  noise assumptions produce distinct cache keys/hits.

### Validation
- Added `tests/test_phase10_noise_model.py`.
- Full test suite: **398 passed, 5 skipped**.

## [Unreleased] — Phase 13 (Regression Tracking Database)

### Added
- **SQLite regression database**: `qocc.core.regression_db.RegressionDatabase`
  with `ingest()`, `query()`, `tag()`, and `detect_regressions()` APIs.
- **New CLI command group**: `qocc db` with:
  - `qocc db ingest <bundle>`
  - `qocc db query [--circuit-hash ...] [--adapter ...] [--since ...]`
  - `qocc db tag <bundle> --tag baseline`
- **Trace integration**: `qocc trace run --db [--db-path ...]` now auto-ingests
  newly generated bundles into the regression database.
- **Phase 10 regression DB tests**: `tests/test_phase10_regression_db.py`.

### Validation
- Full test suite: **403 passed, 5 skipped**.

## [Unreleased] — Phase 13 (Interactive HTML Trace Viewer)

### Added
- **Interactive HTML report exporter**: `qocc.trace.html_report.export_html_report()`
  generates a self-contained report (no CDN dependencies) with:
  - flame-chart timeline with per-span attribute hover
  - metric dashboard for candidate/compiled metrics
  - contract pass/fail panel with CI bar rendering when available
  - optional compare-bundle diff table
  - circuit gate-histogram input vs compiled overlay table
- **New CLI command**: `qocc trace html --bundle ... --out ... [--compare ...]`.
- **Trace-run integration**: `qocc trace run --html [--html-out ...]`.
- **Phase 13 HTML report tests**: `tests/test_phase13_html_report.py`.

### Validation
- Full test suite: **406 passed, 5 skipped**.

## [Unreleased] — Phase 13 (Jupyter Widget Integration)

### Added
- **Jupyter widget module**: `qocc.trace.jupyter_widget` using optional
  `ipywidgets` + `plotly` dependencies (`qocc[jupyter]`).
- **Notebook helper APIs** exposed at top-level package:
  - `qocc.show_bundle(bundle_path)`
  - `qocc.compare_interactive(bundle_a, bundle_b)`
  - `qocc.search_dashboard(search_result)`
- **Phase 13.2 tests**: `tests/test_phase13_jupyter_widget.py`.

### Validation
- Full test suite: **411 passed, 5 skipped**.

## [Unreleased] — Phase 14 (Contract DSL)

### Added
- **Contract DSL parser**: `qocc.contracts.dsl.parse_contract_dsl(text)` with
  location-aware syntax errors (`line`, `column`) via `ContractDSLParseError`.
- **CLI support**: `qocc contract check --contracts` now accepts both `.json`
  and `.qocc` files.
- **API support**:
  - `check_contract(..., contract_spec="*.qocc")`
  - `search_compile(..., contracts="*.qocc")`
- **Contracts package export**: `parse_contract_dsl` re-exported from
  `qocc.contracts`.
- **Phase 14 tests**: `tests/test_phase14_contract_dsl.py`.

### Validation
- Full test suite: **417 passed, 5 skipped**.

## [Unreleased] — Phase 14 (Parametric Contracts)

### Added
- **Parametric expression resolver**: `qocc.contracts.parametric.resolve_contract_spec`
  with safe arithmetic expression evaluation (`+`, `-`, `*`, `/`, `**`, `%`) and
  symbol binding from runtime context.
- **Runtime context support** for symbolic tolerances/budgets including:
  - compiled metrics (`depth`, `gates_2q`, `compiled_depth`, ...)
  - input metrics (`input_depth`, ...)
  - baseline metrics (`baseline_*` when provided in bundle metrics)
  - contract-local numeric fields (e.g., `error_budget` from `spec`).
- **Evaluation-time integration**:
  - `check_contract()` resolves parametric fields before evaluator dispatch.
  - `search_compile()` resolves parametric fields per candidate before contract checks.
- **DSL extension**: constraint RHS now accepts expressions (not only numeric literals).
- **Phase 14 parametric tests**: `tests/test_phase14_parametric_contracts.py`.

### Validation
- Full test suite: **421 passed, 5 skipped**.

## [Unreleased] — Phase 14 (Contract Composition)

### Added
- **Composition evaluator**: `qocc.contracts.composition.evaluate_contract_entry`
  with JSON-envelope support for:
  - `all_of([c1, c2, ...])`
  - `any_of([c1, c2, ...])`
  - `best_effort(contract)`
  - `with_fallback(primary, fallback)`
- **Leaf-flatten helper**: `iter_leaf_contract_dicts()` for simulation prep and
  contract metadata handling.
- **`check_contract()` integration** for top-level composed contract trees.
- **`search_compile()` integration** for composed contracts during per-candidate
  validation/evaluation.
- **NotImplemented fallback semantics**: custom evaluator failures tagged as
  `NotImplementedError: ...` trigger `with_fallback` secondary evaluation.
- **Phase 14 composition tests**: `tests/test_phase14_contract_composition.py`.

### Validation
- Full test suite: **424 passed, 5 skipped**.

## [Unreleased] — Phase 14 (Contract Result Caching)

### Added
- **Contract result cache integration** in `check_contract()` and
  `search_compile()` leaf evaluation paths.
- **Cache key formula** per Phase 14.4:
  `SHA-256(circuit_hash || contract_spec_hash || shots || seed)`.
- **Staleness control**: `check_contract(..., max_cache_age_days=...)` and CLI
  flag `qocc contract check --max-cache-age-days ...` to ignore old entries.
- Cached payload stores pass/fail/details including `shot_count_used` metadata.
- **Phase 14 cache tests**: `tests/test_phase14_contract_cache.py`.

### Validation
- Full test suite: **427 passed, 5 skipped**.

## [Unreleased] — Phase 11 (Hardware Execution Adapter Interface)

### Added
- **`ExecutionResult` dataclass** in `qocc.adapters.base` with required fields:
  `job_id`, `backend_name`, `shots`, `counts`, `metadata`, `queue_time_s`,
  `run_time_s`, and `error_mitigation_applied`.
- **Optional adapter `execute()` interface** on `BaseAdapter` for real hardware
  submission with trace guidance for required spans:
  `job_submit`, `queue_wait`, `job_complete`, `result_fetch`, plus
  `job_polling` events during asynchronous polling.
- **Adapters package export**: `ExecutionResult` is re-exported from
  `qocc.adapters`.
- **Phase 11 tests**: `tests/test_phase11_hardware_execution.py`.

### Changed
- `check_contract()` now ingests optional hardware count payloads from bundle
  metadata (`hardware.input_counts`/`baseline_counts`,
  `hardware.counts`/`hardware.result.counts`) so distribution-style contract
  checks can run directly on real-device counts.

## [Unreleased] — Phase 11 (IBM Quantum Runtime Adapter)

### Added
- **IBM adapter**: `qocc.adapters.ibm_adapter.IBMAdapter` with runtime hardware
  `execute()` implementation using `qiskit_ibm_runtime` primitives.
- **Primitive support**: `SamplerV2` and `EstimatorV2` submission paths.
- **Hardware execution tracing** with required spans:
  `job_submit`, `queue_wait`, `job_complete`, `result_fetch`, plus
  `job_polling` events during queue wait.
- **Hardware transpilation span**: `compile/transpile_hardware` records
  pre/post transpile depth and size metrics.
- **Execution metadata** records runtime provenance fields including
  `job_id`, `provider`, `backend_version`, `basis_gates`, and
  `coupling_map_hash`, with raw runtime result payload embedded in metadata.
- **Phase 11.2 tests**: `tests/test_phase11_ibm_adapter.py`.

### Changed
- Adapter resolver fallback now supports `get_adapter("ibm")`.
- Added optional dependency extra: `qocc[ibm]` (`qiskit-ibm-runtime`).
- Added adapter entry-point registration: `ibm`.
- CLI `qocc trace run --adapter` now accepts `ibm`.

## [Unreleased] — Phase 11 (Asynchronous Job Tracking)

### Added
- **Watch engine**: `qocc.trace.watch.watch_bundle_jobs()` for polling pending
  hardware jobs recorded in `hardware/pending_jobs.json`.
- **New CLI command**: `qocc trace watch --bundle ... --poll-interval ...`
  with support for:
  - `--timeout` to bound watch duration
  - `--on-complete` command hook for chained automation
- **In-place bundle updates** during watch:
  - per-job result files: `hardware/<job_id>_result.json`
  - aggregate payload: `hardware/hardware.json`
  - appended completion spans in `trace.jsonl`
- **IBM polling integration**: `poll_ibm_job()` helper for retrieving status and
  results of submitted IBM runtime jobs by `job_id`.
- **Phase 11.3 tests**: `tests/test_phase11_watch.py`.

### Changed
- `ArtifactStore.load_bundle()` now loads optional hardware payloads from
  `hardware.json` or `hardware/hardware.json` into `bundle["hardware"]`.

## [Unreleased] — Phase 12 (Evolutionary Search Strategy)

### Added
- **Evolutionary optimizer module**: `qocc.search.evolutionary` with
  tournament selection, single-point crossover, Gaussian mutation, and elitism.
- **Generation diversity metric** via `population_diversity()`.
- **Search config support** for evolutionary parameters:
  `evolutionary_population_size`, `evolutionary_max_generations`,
  `evolutionary_mutation_rate`, `evolutionary_crossover_rate`,
  `evolutionary_tournament_size`, `evolutionary_elitism`,
  `evolutionary_convergence_std`, `evolutionary_wall_clock_s`,
  `evolutionary_mutation_sigma`.
- **API integration** in `search_compile()` with generation-loop execution,
  per-generation span emission (`evolutionary_generation`), and termination
  by max generations, convergence, wall-clock budget, or population exhaustion.
- **CLI support**: `qocc compile search --strategy evolutionary`.
- **Phase 12 tests**: `tests/test_phase12_evolutionary.py`.

### Changed
- `generate_candidates()` now recognizes strategy `evolutionary` and returns
  an initial population sized by `evolutionary_population_size`.

## [Unreleased] — Phase 12 (Bayesian Transfer-Learning Prior)

### Added
- **Persistent Bayesian history** in `~/.qocc/search_history.json` (or custom
  `bayesian_history_path`) storing observed parameter vectors and scores.
- **Historical prior loading** for matching adapter + backend version with
  exponential age decay weighting:
  `weight = exp(-days_old / half_life_days)`.
- **Config/CLI support** for half-life tuning:
  `bayesian_prior_half_life_days` and `qocc compile search --prior-half-life`.
- **Trace attributes** on `bayesian_optimizer` span:
  `prior_loaded`, `prior_size`, and `history_appended`.
- **Phase 12.3 tests**: `tests/test_phase12_bayesian_prior.py`.

### Changed
- `search_compile()` now runs an adaptive Bayesian loop with observation
  persistence across rounds and runs.
- `BayesianSearchOptimizer` now uses weighted observations in UCB estimation.

## [Unreleased] — Phase 12 (Multi-Circuit Batch Search)

### Added
- **Batch API**: `qocc.api.batch_search_compile(manifest, output, workers)` for
  manifest-driven multi-circuit search runs.
- **CLI command**: `qocc compile batch --manifest ... [--workers N] [--out ...]`.
- **Batch bundle outputs**:
  - `batch_results.json` (per-circuit result objects)
  - `cross_circuit_metrics.json` (cross-circuit rows + aggregate summary)
- **Top-level batch trace span**: `batch_search` with required attributes:
  `n_circuits`, `n_cache_hits`, `total_candidates_evaluated`.
- **Phase 12.4 tests**: `tests/test_phase12_batch.py`.

### Changed
- `search_compile()` return payload now includes `cache_hits` and `cache_misses`
  for upstream aggregation in batch mode.

## [Unreleased] — Phase 15 (Zero-Noise Extrapolation Contract)

### Added
- **New contract type**: `ContractType.ZNE` (`"zne"`).
- **ZNE evaluator module**: `qocc.contracts.eval_zne.evaluate_zne_contract`
  implementing Richardson extrapolation to noise scale 0.
- ZNE result details include:
  - `per_level` (noise scale and expectation values)
  - `extrapolation_coefficients`
  - `extrapolated_value`, `ideal_value`, `abs_error`, `tolerance`.
- **Search-time ZNE tracing**: one span per noise level (`zne/noise_level`).
- **Phase 15.2 tests**: `tests/test_phase15_zne_contract.py`.

### Changed
- `check_contract()` and `search_compile()` now dispatch `type="zne"` contracts.
- `schemas/contracts.schema.json` now accepts `"zne"` in contract type enum.

## [Unreleased] — Phase 9 (Correctness, Performance & Hygiene)

### Fixed (High)
- **CircuitHandle hash instability** — `stable_hash()` now caches its result
  on first call via `_stable_hash_cache`, preventing hash drift when `qasm3`
  is set after the object is used as a dict key.  `deepcopy()` correctly
  resets the cache so normalised copies get their own fresh hash.
- **Thread-unsafe `TraceEmitter` reads** — `finished_spans()` and `to_dicts()`
  now acquire `self._lock` before reading the span list, preventing
  concurrent-modification issues.
- **Deprecated `np.random.RandomState`** — all 6 source-file usages replaced
  with `np.random.default_rng()` (Generator API); `rng.randint()` calls
  updated to `rng.integers()`.
- **Redundant exception tuples** — two `except (NotImplementedError, Exception)`
  clauses in `api.py` collapsed to `except Exception`.

### Fixed (Medium)
- **Tooling config mismatch** — Ruff `target-version` ("py310" → "py311") and
  mypy `python_version` ("3.10" → "3.11") now match `requires-python ≥ 3.11`.
- **CI Python 3.14 prerelease risk** — added `allow-prereleases: true` to
  `setup-python@v5` step so nightly / pre-release 3.14 installs succeed.
- **Silent error swallowing** — 6 bare `except: pass` blocks (in
  `adapters/base.py`, `contracts/registry.py`, `core/cache.py`) replaced
  with `logger.debug(…, exc_info=True)` for diagnosability.
- **`render_timeline` crash on tiny width** — widths below 20 are now clamped
  to 20, preventing division-by-zero in the bar renderer.

### Added
- **`CompilationCache` context manager** — `__enter__`/`__exit__`/`close()`
  allow deterministic cleanup; `close()` clears per-key locks and
  `evict_lru()` prunes lock entries for evicted keys.
- **run_id path-injection guard** — `ArtifactStore.write_manifest()` now
  validates `run_id` against `^[a-zA-Z0-9_-]+$`; rejects `../evil` etc.
- **Entry-point idempotency** — adapter and evaluator discovery functions
  (`_discover_entry_point_adapters`, `_discover_entry_point_evaluators`) now
  use module-level `_ep_*_discovered` flags so entry-point scanning runs
  at most once.
- **PEP 561 `py.typed` marker** — downstream consumers can now run mypy
  against QOCC without `--ignore-missing-imports`.
- **`DEFAULT_SEED` consistency** — replaced all remaining hardcoded `seed=42`
  values across `api.py`, `stats.py`, `eval_sampling.py`, `validator.py`,
  `commands_trace.py`, `qiskit_adapter.py`, and `search/space.py` with the
  canonical `qocc.DEFAULT_SEED` constant.
- **Phase 9 test suite** (`tests/test_phase9_features.py`) — 30 new tests
  covering: hash caching, DEFAULT_SEED, default_rng migration, thread-safe
  emitter, cache context manager, width clamping, run_id sanitization,
  entry-point idempotency, py.typed marker, tooling config consistency,
  redundant exception removal, search-space RNG.
- 333 tests passing, 5 skipped.

---

## [Unreleased] — Phase 8 (Deep Audit & Hardening)

### Fixed (Critical / High)
- **`qocc validate` crash on zip bundles** — fixed `ImportError` (`load_bundle`
  → `ArtifactStore.load_bundle()`) and corrected return-value handling (dict →
  Path extraction).
- **Visualization `parent_span_id` mismatch** — `render_timeline()` now reads
  both `parent_span_id` (canonical) and `parent_id` (legacy) so child spans
  render correctly as nested in the flame chart.
- **`tracemalloc` scope hole** — moved `tracemalloc` init before the loop so
  `_tm_was_tracing` is always defined even with an empty `contract_spec` list.
- **Duplicate `_counts_to_observable_values`** — removed the copy in `api.py`;
  now imported from `qocc.contracts.eval_sampling`.
- **Unused `pydantic` dependency** — removed from `pyproject.toml`; pydantic
  was never imported anywhere.

### Fixed (Medium)
- **Dead `_compare_params` code** removed from `commands_compare.py`.
- **Duplicated `_load_schema` / `_SCHEMA_DIR`** — `commands_validate.py` now
  imports them from `qocc.cli.validation`.
- **`ContractSpec` unknown type warning** — `__post_init__` now emits
  `warnings.warn()` when an unrecognised contract type is used without a
  custom evaluator.

### Added
- **`__all__` exports** on all 7 subpackage `__init__.py` files with proper
  imports (`core`, `adapters`, `contracts`, `trace`, `search`, `metrics`, `cli`).
- **`DEFAULT_SEED` constant** (`qocc.DEFAULT_SEED = 42`) for a single source of
  truth instead of hardcoded seed values.
- **Input validation** — `run_trace()` rejects `repeat < 1`;
  `search_compile()` rejects `top_k < 1` and clamps `top_k` to the number
  of ranked candidates with a log message.
- **Phase 8 test suite** (`tests/test_phase8_features.py`) — 29 new tests
  covering: zip validate, parent_span_id, tracemalloc scope, dedup, contract
  warnings, `__all__` exports, DEFAULT_SEED, input validation, replay module,
  topology module, pydantic removal, dead-code removal.
- **Python 3.14 in CI matrix** — added to GitHub Actions test matrix.

### Changed
- **Minimum Python bumped to 3.11** — `pyproject.toml` and README badge updated
  to match what CI actually tests.
- **CI `schema-validate` job** now installs the package (`pip install -e .`)
  instead of only `jsonschema`.
- **README** test count updated to ~303.
- 303 tests passing, 5 skipped.

---

## [Unreleased] — Phase 7 (Hardening & DX)

### Added
- **`qocc validate` CLI** — validates every JSON file in a bundle against its
  QOCC schema; supports `--format json` / `--format table` and `--strict`.
- **`qocc trace compare`** — canonical path for bundle comparison; the old
  `qocc compare` still works but shows a deprecation warning.
- **CI `typecheck` job** — runs `mypy qocc/ --ignore-missing-imports` as part
  of the GitHub Actions workflow.
- **Resource budget `max_runtime`** — the SPRT iterative-evaluation loop
  now respects `resource_budget.max_runtime` (seconds); iteration stops
  early and `budget_exceeded: "max_runtime"` is set in the result details.
- **`max_memory_mb` parameter** on `check_contract()` — tracks peak memory
  via `tracemalloc` and annotates results with `_budget.peak_memory_mb`.
- **Cross-thread span parentage** — `_compile_one_candidate()` now passes
  `parent=compile_parent` explicitly so candidate spans in worker threads
  have a correct `parent_span_id` (in addition to the existing span link).
- **Phase 7 test suite** (`tests/test_phase7_features.py`) covering all
  new features: validate CLI, compare deprecation, JSON-pure stdout,
  concurrent cache writes, ZipSlip rejection, max_runtime budget,
  cross-thread span parentage, CI configuration, CLI registration.

### Changed
- **Cache atomic writes** — `CompilationCache.put()` and `get()` now use
  `_atomic_write_text()` (temp-file → `os.replace()`) so concurrent or
  interrupted writes never leave partial JSON on disk.
- **Per-key locking** — `CompilationCache` lazily creates a `threading.Lock`
  per cache key; concurrent `put()` calls on the same key are serialised.
- **Zip extraction hardening** — `ArtifactStore.load_bundle()` rejects
  zip members whose resolved path escapes the extraction root (ZipSlip
  protection) and extracts into a unique per-process directory.
- **JSON-pure compare output** — `compare --format json` now routes all
  Rich-formatted human-readable output to stderr; stdout contains only
  the `json.dumps()` payload.

---

## Phase 6 (Deep Audit)

### Added
- Thread-safe `TraceEmitter` with per-thread active-span stacks.
- Structured logging throughout trace, cache, and contract modules.
- `--format json` on `qocc compare` (Rich output + JSON payload).
- Idle-penalty duration model (`IdlePenaltyDuration`).
- CLI JSON schema validation (`qocc.cli.validation`).
- Span links across compilation candidates.
- Cache key `extra` parameter (captures seeds/config).
- GitHub Actions CI (`lint` + `test` + `schema-validate` jobs).
- RNG algorithm recorded in `seeds.json`.
- 256 tests passing, 5 skipped.

---

## Phases 1–5 (Initial Build)

### Added
- Full QOCC implementation from spec (`prompt1.md`).
- Adapters: Qiskit, Cirq, Tket (stub), Stim (stub), CUDA-Q (stub).
- Contract system: distribution, observable, clifford, exact, cost types.
- Iterative SPRT-enhanced evaluation with early stopping.
- Bayesian surrogate-assisted compilation search.
- Trace Bundle format with 13 JSON schemas.
- CLI: `trace run`, `trace timeline`, `trace replay`, `compare`, `contract check`, `compile search`.
- Compilation cache with LRU eviction.
- 233 tests passing, 1 skipped at end of Phase 5.
