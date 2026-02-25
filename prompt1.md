Advanced Build Prompt: QOCC — Quantum Observability + Contract-Based Compilation

You are building an open-source system named QOCC (Quantum Observability + Contract Compilation). QOCC is a vendor-agnostic, reproducible, trace-first layer that instruments quantum program workflows end-to-end and supports contract-defined correctness + cost optimization via closed-loop compilation/search.

Mission

Make quantum development debuggable, reproducible, and optimizable across toolchains by introducing:

Observability: OpenTelemetry-style traces for quantum compilation, simulation, mitigation, decoding, and execution.

Contracts: Machine-checkable semantic constraints + explicit cost objectives.

Closed-loop optimization: Generate many candidate pipelines, score cheaply with surrogate models, validate with simulation/QEC sampling, choose best under contracts.

Trace Bundles: Portable “repro packages” that can be rerun, compared, regression-tested, and shared.

Primary Users

Researchers benchmarking compilation strategies

Engineers debugging regressions across versions/backends

Teams needing reproducible artifacts for papers or CI

People building toolchains who need standardized telemetry + contracts

Non-Goals (explicit)

Not a new quantum SDK.

Not a universal large-scale simulator.

Not proprietary calibration ingestion; QOCC accepts user-provided noise models + metadata.

0) Hard Requirements (Non-Negotiable)
Determinism + Reproducibility

Every run must produce a Trace Bundle with:

complete environment snapshot (OS, Python, package versions, git SHAs)

all seeds used by compilation/simulation/sampling and any RNG

canonical circuit hashes for input and outputs

stable ordering of spans/events and metrics

If a backend/tool is nondeterministic, QOCC must:

detect nondeterminism via repeated runs (configurable)

record nondeterminism evidence and a “repro confidence” score

store multiple samples (hashes) if needed

Vendor-Agnostic Adapters

Must support at least:

Qiskit

Cirq
Later:

pytket

CUDA-Q (optional)

Stim/PyMatching/sinter (QEC mode)

Adapters must implement a shared interface (below).

Trace-First Architecture

Everything is instrumented. Every stage must emit spans:

parse/ingest

normalization/canonicalization

compilation/transpilation

candidate generation

scoring

simulation

contract evaluation

selection

reporting/bundle export

1) Core Concepts
1.1 Semantic Contract (Correctness)

A Semantic Contract is a machine-checkable requirement. It has:

spec: what is to be preserved and how it’s tested

evaluator: which method is used (exact/sampling/stabilizer)

tolerances: numeric thresholds

confidence: confidence intervals / significance levels

resource_budget: max shots / time / memory

Minimum contract types to implement (v2)

Observable contract

preserve expectation values of specified observables within tolerance

e.g., |E1 - E2| <= eps with CI or exact check

Distribution contract

preserve output measurement distribution within tolerance

test via:

total variation distance estimate + CI, or

chi-square / G-test with correction

must state: fixed shot count, significance level alpha

Clifford/Stabilizer contract

for Clifford circuits, verify exact equivalence efficiently (where possible)

or compare stabilizer tableau invariants

fall back to sampling if non-Clifford

1.2 Cost Contract (Optimization Objective)

A Cost Contract declares objective(s):

single objective (minimize gate count)

multi-objective (Pareto frontier)

constraints (depth <= X)

weights (w1depth + w22Q + w3duration + w4proxy_error)

Must support:

deterministic metrics (depth, counts)

pluggable duration models

pluggable noise proxy models

1.3 Provenance Trace (Observability)

Trace model inspired by OpenTelemetry:

trace_id, span_id, parent_span_id

name, start_time, end_time

attributes: toolchain, versions, seeds, inputs, outputs, metrics

events: warnings, exceptions, nondeterminism flags

links: connect spans across candidate pipelines

2) Trace Bundle Format (Exportable Artifact)

A Trace Bundle is a directory or zip containing:

Required files

manifest.json — bundle metadata (schema version, created_at, tool versions, run id)

trace.jsonl — span/event log (JSON Lines)

env.json — environment + dependency snapshot

seeds.json — all seeds, RNG algorithm identifiers

circuits/

input.<format> (qasm3 + framework native dump)

normalized.<format>

candidates/ (each candidate’s circuits + metadata)

selected.<format>

metrics.json — metrics snapshots per stage/candidate

contracts.json — contract specs

contract_results.json — results with stats

reports/summary.md — human-readable report auto-generated

Optional files

cache_index.json — cache keys used and hits/misses

plots/ — metric charts, diff visualizations (optional in MVP)

Strong constraint

Everything in the bundle must be sufficient to:

reconstruct the full pipeline config

rerun evaluation (if tools installed)

compare against another bundle deterministically

3) Public API + CLI Requirements
3.1 Python API (stable)

Provide:

Primary entrypoints

qocc.run_trace(...) -> TraceBundle

qocc.compare_bundles(bundle_a, bundle_b) -> CompareReport

qocc.check_contract(bundle_or_circuits, contract_spec) -> ContractResult

qocc.search_compile(...) -> SearchResult (v3)

Key data structures

CircuitHandle (IR-neutral wrapper)

PipelineSpec (toolchain config, passes, parameters)

ContractSpec / ContractResult

CostSpec / CostResult

Candidate (pipeline + output + metrics + score)

TraceEmitter (span + event API)

ArtifactStore (writes bundle files consistently)

3.2 CLI (required)

qocc trace run --adapter qiskit --input foo.qasm --pipeline pipelines/qiskit_default.json --out bundle.zip

qocc trace compare bundleA.zip bundleB.zip --report reports/

qocc contract check --bundle bundle.zip --contracts contracts.json

qocc compile search --bundle bundle.zip --search search_spec.json --topk 10 --out search_bundle.zip (v3)

CLI must be scriptable and CI-friendly (exit codes reflect pass/fail).

4) Adapter Interface (Critical)

Each adapter must implement:

ingest(input) -> CircuitHandle

normalize(circuit) -> CircuitHandle
(canonicalizes ordering, naming, register mapping)

export(circuit, format) -> bytes/str

compile(circuit, pipeline_spec) -> CircuitHandle + PassLog

simulate(circuit, sim_spec) -> Result (optional in MVP; needed in v2/v3)

get_metrics(circuit) -> MetricsSnapshot

hash(circuit) -> stable_hash (via normalization + serialization)

describe_backend() -> BackendInfo (versioned)

Pass logging must preserve:

pass name

parameters

order

time/memory (best effort)

warnings/errors

5) Metrics (Minimum Set)

Compute these per circuit, per stage, per candidate:

width (#qubits)

#gates total

#1Q gates, #2Q gates

depth (overall and 2Q depth if possible)

gate histogram by type

topology violations count (given coupling map)

duration estimate:

duration = Σ count(gate_type)*duration(gate_type) + Σ idle_penalties

proxy error score:

weighted sum (user-configurable)

can incorporate simple per-gate error rates if provided

Store metrics as immutable snapshots.

6) Contract Evaluation (v2 detail)
6.1 Statistical rigor

When sampling:

record: shots, seed, sampler, backend, confidence_level

output:

point estimate

confidence interval

test statistic if hypothesis test used

pass boolean under specified tolerances

Distribution contract default

Use Total Variation Distance estimate between histograms:

TVD = 0.5 * Σ |p_i - q_i|

Provide a conservative CI via bootstrap or normal approx

Pass if upper_CI <= tvd_tolerance

Observable contract default

For each observable:

estimate expectation E

CI via Hoeffding or bootstrap

Pass if |E1 - E2| <= eps considering CI overlap or conservative bound

Clifford/stabilizer contract

If circuit is Clifford (detected), use exact equivalence checks where feasible.

Else fallback to distribution/observable sampling.

6.2 Resource budgets

Each contract must accept:

max shots

max runtime

max memory

early stopping if pass/fail is statistically certain

All truncations must be recorded in trace.

7) Closed-Loop Compilation Search (v3 detail)
7.1 Candidate generation

Implement a search space that can vary:

transpiler optimization level

routing method / placement strategy

gate decomposition choices

scheduling/timing assumptions

mitigation transforms (optional, if integrated)

pass ordering (limited permutations)

Candidate spec must be serializable + hashable.

7.2 Surrogate scoring (cheap)

For each candidate:

compute deterministic metrics

compute proxy error score:

proxy_error = Σ count(gate_type)*p_error(gate_type) + depth * decoherence_weight + duration * duration_weight

output a scalar score and per-term breakdown

7.3 Validation (expensive, top-k)

Select top-k candidates by surrogate score, then validate:

simulation (exact for small circuits)

sampling for medium circuits

GPU sim backend if available (optional)

QEC sampling mode (Stim + decoder) when circuit is QEC tagged

7.4 Selection

Choose best candidate that:

satisfies all semantic contracts

minimizes cost contract (single or Pareto)
If none satisfy:

output “infeasible” with best-effort nearest candidates + failure reasons

7.5 Caching

Implement caching keyed by:

normalized circuit hash

pipeline spec hash

backend version hash

contract spec hash

seeds

Cache must be content-addressed and stored in:

bundle local cache folder OR

user cache dir

Cache hits/misses must be traced.

8) Reporting + Comparison (Critical UX)
8.1 Bundle summary report

Auto-generate reports/summary.md with:

input circuit identifiers + hashes

pipeline configuration

candidate table with metrics + scores

contract results table

selected candidate explanation

reproducibility notes (nondeterminism warnings)

environment snapshot summary

8.2 Compare bundles

Implement qocc trace compare to produce:

circuit hash diffs

metric diffs (absolute + percent)

pass log diffs (added/removed/reordered)

contract diffs

environment/version diffs

highlight likely cause of regression:

changed tool version

changed pass param

changed routing choice

changed seed/nondeterminism

Comparison output must be both JSON and Markdown.

9) Repo Layout (Required)

Provide a repo scaffold:

qocc/
  pyproject.toml
  qocc/
    __init__.py
    core/
      circuit_handle.py
      canonicalize.py
      hashing.py
      artifacts.py
      schemas.py
    trace/
      emitter.py
      span.py
      exporters.py
    adapters/
      base.py
      qiskit_adapter.py
      cirq_adapter.py
    metrics/
      compute.py
      duration_models.py
      topology.py
    contracts/
      spec.py
      eval_sampling.py
      eval_exact.py
      eval_clifford.py
      stats.py
    search/
      space.py
      scorer.py
      validator.py
      selector.py
    cli/
      main.py
      commands_trace.py
      commands_compare.py
      commands_contract.py
      commands_search.py
  tests/
    test_trace_bundle_roundtrip.py
    test_hash_stability.py
    test_metrics.py
    test_contracts_sampling.py
    test_compare.py
  examples/
    qiskit_trace_demo.py
    cirq_trace_demo.py
    contracts_examples.json
    pipeline_examples/
      qiskit_default.json
      cirq_default.json
    search_examples.json
10) Milestones + “Definition of Done”
MVP (4–8 weeks): Quantum Trace Pack

Done when:

supports Qiskit + Cirq ingestion + normalization + compilation

emits valid trace bundle with stable hashes

metrics computed and stored

CLI trace run works on example circuits

trace compare identifies deltas

unit tests for hash stability + bundle integrity pass

v2 (8–16 weeks): Contracts

Done when:

implements 3 contract types

contract evaluation works with sampling and CI

results stored + reported in bundle

contract check CLI returns nonzero exit on failure

tests validate contract pass/fail behavior

v3 (16–32 weeks): Closed-loop optimization

Done when:

search generates ≥ N candidates

surrogate scoring ranks candidates

expensive validation runs on top-k

selection returns best satisfying contracts

bundle contains full candidate table + selected reasoning

caching reduces repeated runs measurably

compare highlights regressions in chosen candidate over time

11) Output Required from the Implementer (You)

You must produce:

A complete repo scaffold (files created with skeleton code)

JSON schema definitions for all bundle files

CLI implementation for MVP

Qiskit + Cirq adapters minimal viable

Metrics implementation

Bundle writer + deterministic hashing

Two end-to-end examples that create bundles

Tests that enforce determinism and bundle completeness

A ROADMAP.md and CONTRIBUTING.md

12) Engineering Standards

Type hints everywhere

Strict linting/formatting

Errors must be traceable: every exception becomes a trace event with stack + context

Backends must be version-pinned in examples

Avoid heavy deps unless optional extras

Provide plugin hooks for adding new adapters and new contract evaluators