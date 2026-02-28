# API Reference

This file is generated from docstrings and signatures.

## `qocc.api`

QOCC public Python API.

### `batch_search_compile(manifest: 'str | dict[str, Any]', output: 'str | None' = None, workers: 'int | None' = None) -> 'dict[str, Any]'`

Run search compilation across multiple circuits from a manifest.

### `check_contract(bundle_or_circuits: 'str | dict[str, Any]', contract_spec: 'str | list[dict[str, Any]]', adapter_name: 'str | None' = None, simulation_shots: 'int' = 1024, simulation_seed: 'int' = 42, max_memory_mb: 'float | None' = None, max_cache_age_days: 'float | None' = None) -> 'list[dict[str, Any]]'`

Evaluate contracts against a bundle or circuits.

### `compare_bundles(bundle_a: 'str | dict[str, Any]', bundle_b: 'str | dict[str, Any]', report_dir: 'str | None' = None) -> 'dict[str, Any]'`

Compare two Trace Bundles and produce a diff report.

### `run_trace(adapter_name: 'str', input_source: 'str | Any', pipeline: 'PipelineSpec | dict[str, Any] | str | None' = None, output: 'str | None' = None, seeds: 'dict[str, Any] | None' = None, repeat: 'int' = 1) -> 'dict[str, Any]'`

Run an instrumented compilation trace and produce a Trace Bundle.

### `search_compile(adapter_name: 'str', input_source: 'str | Any', search_config: 'dict[str, Any] | str | None' = None, contracts: 'list[dict[str, Any]] | str | None' = None, output: 'str | None' = None, top_k: 'int' = 5, simulation_shots: 'int' = 1024, simulation_seed: 'int' = 42, mode: 'str' = 'single') -> 'dict[str, Any]'`

Closed-loop compilation search (v3 entrypoint).

## `qocc.core.circuit_handle`

IR-neutral circuit wrapper for QOCC.

### `class BackendInfo`

Describes an adapter backend.

Methods:

- `to_dict(self) -> 'dict[str, Any]'` — 

### `class CircuitHandle`

Vendor-agnostic wrapper around a quantum circuit.

Methods:

- `stable_hash(self) -> 'str'` — Produce a deterministic SHA-256 hash of the canonical representation.
- `to_dict(self) -> 'dict[str, Any]'` — Return a JSON-safe dict (no native circuit).

### `class MitigationSpec`

Error mitigation configuration for an optional pipeline stage.

Methods:

- `to_dict(self) -> 'dict[str, Any]'` — 

### `class PassLogEntry`

A single compilation-pass log entry.

Methods:

- `to_dict(self) -> 'dict[str, Any]'` — 

### `class PipelineSpec`

Describes a compilation pipeline configuration.

Methods:

- `stable_hash(self) -> 'str'` — 
- `to_dict(self) -> 'dict[str, Any]'` — 

## `qocc.adapters.base`

Base adapter interface.

### `class BaseAdapter`

Abstract base adapter all backend adapters must implement.

Methods:

- `compile(self, circuit: 'CircuitHandle', pipeline: 'PipelineSpec', emitter: 'Any | None' = None) -> 'CompileResult'` — Compile/transpile the circuit with the given pipeline spec.
- `describe_backend(self) -> 'BackendInfo'` — Return backend/version information.
- `execute(self, circuit: 'CircuitHandle', backend_spec: 'dict[str, Any]', shots: 'int' = 1024, emitter: "'TraceEmitter' | None" = None) -> 'ExecutionResult'` — Submit to real hardware and return counts + job metadata.
- `export(self, circuit: 'CircuitHandle', fmt: 'str' = 'qasm3') -> 'str'` — Export the circuit to the given format string.
- `get_metrics(self, circuit: 'CircuitHandle') -> 'MetricsSnapshot'` — Compute and return a metrics snapshot for the given circuit.
- `hash(self, circuit: 'CircuitHandle') -> 'str'` — Return a stable hash of the circuit (via normalization + serialization).
- `ingest(self, source: 'str | Any') -> 'CircuitHandle'` — Load a circuit from a file path, QASM string, or native object.
- `name(self) -> 'str'` — Return adapter name (e.g. ``'qiskit'``).
- `normalize(self, circuit: 'CircuitHandle') -> 'CircuitHandle'` — Canonicalize the circuit (ordering, naming, register mapping).
- `simulate(self, circuit: 'CircuitHandle', spec: 'SimulationSpec') -> 'SimulationResult'` — Run simulation. Optional in MVP — adapters may raise NotImplementedError.

### `class CompileResult`

Result of compilation — a circuit + pass log.

Methods:

- `to_dict(self) -> 'dict[str, Any]'` — 

### `class ExecutionResult`

Result of a real hardware execution run.

Methods:

- `to_dict(self) -> 'dict[str, Any]'` — 

### `class MetricsSnapshot`

Immutable metrics for a circuit at a point in time.

Methods:

- `get(self, key: 'str', default: 'Any' = None) -> 'Any'` — 
- `to_dict(self) -> 'dict[str, Any]'` — 

### `class SimulationResult`

Result of a simulation run.

Methods:

- `to_dict(self) -> 'dict[str, Any]'` — 

### `class SimulationSpec`

Configuration for running a simulation.

Methods:

- `to_dict(self) -> 'dict[str, Any]'` — 

### `get_adapter(name: 'str') -> 'BaseAdapter'`

Instantiate and return the adapter registered under *name*.

### `register_adapter(name: 'str', cls: 'type[BaseAdapter]') -> 'None'`

Register an adapter class under *name*.

## `qocc.contracts.spec`

Contract specification and result data structures.

### `class ContractResult`

Result of evaluating a contract.

Methods:

- `to_dict(self) -> 'dict[str, Any]'` — 

### `class ContractSpec`

Specification for a semantic or cost contract.

Methods:

- `to_dict(self) -> 'dict[str, Any]'` — 

### `class ContractType`

Valid contract types.

### `class CostResult`

Result of cost evaluation.

Methods:

- `to_dict(self) -> 'dict[str, Any]'` — 

### `class CostSpec`

Cost/optimization objective specification.

Methods:

- `to_dict(self) -> 'dict[str, Any]'` — 

## `qocc.search.space`

Search space definition for candidate pipeline generation.

### `class BayesianSearchOptimizer`

Surrogate-model-guided adaptive search.

Methods:

- `initial_candidates(self) -> 'list[Candidate]'` — Generate the initial random exploratory batch.
- `load_prior(self, backend_version: 'str', half_life_days: 'float | None' = None) -> 'int'` — Load weighted historical observations for same adapter/backend.
- `observe(self, candidates: 'list[Candidate]') -> 'None'` — Record observed (compiled + scored) candidates.
- `persist_history(self, backend_version: 'str') -> 'int'` — Append current-run observations to persisted search history.
- `suggest(self, batch_size: 'int' = 4) -> 'list[Candidate]'` — Suggest the next batch of candidates using UCB acquisition.

### `class Candidate`

A single candidate pipeline with its outputs and scores.

Methods:

- `to_dict(self) -> 'dict[str, Any]'` — 

### `class SearchSpaceConfig`

Defines the search space for closed-loop compilation.

Methods:

- `to_dict(self) -> 'dict[str, Any]'` — 

### `generate_candidates(config: 'SearchSpaceConfig') -> 'list[Candidate]'`

Enumerate candidate pipelines from the search space.
