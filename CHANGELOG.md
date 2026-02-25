# Changelog

All notable changes to the QOCC project are documented here.

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
