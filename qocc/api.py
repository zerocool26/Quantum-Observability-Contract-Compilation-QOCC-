"""QOCC public Python API.

Provides the primary entrypoints:
  - ``run_trace(...)`` → TraceBundle
  - ``compare_bundles(...)`` → CompareReport
  - ``check_contract(...)`` → ContractResult
  - ``search_compile(...)`` → SearchResult (v3)
"""

from __future__ import annotations

import difflib
import json
import logging
import os
import platform
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from qocc.adapters.base import get_adapter
from qocc.core.artifacts import ArtifactStore
from qocc.core.cache import CompilationCache
from qocc.core.circuit_handle import PipelineSpec
from qocc.trace.emitter import TraceEmitter

logger = logging.getLogger("qocc")


def run_trace(
    adapter_name: str,
    input_source: str | Any,
    pipeline: PipelineSpec | dict[str, Any] | str | None = None,
    output: str | None = None,
    seeds: dict[str, Any] | None = None,
    repeat: int = 1,
) -> dict[str, Any]:
    """Run an instrumented compilation trace and produce a Trace Bundle.

    Parameters:
        adapter_name: Adapter to use (``"qiskit"`` or ``"cirq"``).
        input_source: File path, QASM string, or native circuit object.
        pipeline: Pipeline spec (dict, PipelineSpec, or path to JSON).
        output: Output path for the bundle zip (optional; uses temp dir if None).
        seeds: Seed configuration.
        repeat: Number of times to compile for nondeterminism detection (>=2 enables detection).

    Returns:
        Dictionary with bundle metadata and file path.
    """
    # Setup
    run_id = uuid.uuid4().hex[:12]
    emitter = TraceEmitter()

    if seeds is None:
        seeds = {"global_seed": 42, "rng_algorithm": "MT19937", "stage_seeds": {}}

    # Inject global_seed into pipeline parameters so adapters use it
    _seed_value = seeds.get("global_seed", 42)

    # Build output dir
    if output:
        out_path = Path(output)
        if out_path.suffix == ".zip":
            bundle_dir = out_path.with_suffix("")
            zip_path = out_path
        else:
            bundle_dir = out_path
            zip_path = out_path.with_suffix(".zip")
    else:
        bundle_dir = Path(tempfile.mkdtemp(prefix="qocc_bundle_"))
        zip_path = bundle_dir.with_suffix(".zip")

    store = ArtifactStore(bundle_dir)

    # Pipeline spec
    if isinstance(pipeline, str):
        p = Path(pipeline)
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            pipeline_spec = PipelineSpec.from_dict(data)
        else:
            pipeline_spec = PipelineSpec(adapter=adapter_name)
    elif isinstance(pipeline, dict):
        pipeline_spec = PipelineSpec.from_dict(pipeline)
    elif isinstance(pipeline, PipelineSpec):
        pipeline_spec = pipeline
    else:
        pipeline_spec = PipelineSpec(adapter=adapter_name)

    # Inject seed from seeds dict into pipeline parameters
    if "seed" not in pipeline_spec.parameters:
        pipeline_spec.parameters["seed"] = _seed_value

    adapter = get_adapter(adapter_name)

    # 1. Ingest
    with emitter.span("ingest", attributes={"adapter": adapter_name}) as span:
        handle = adapter.ingest(input_source)
        span.set_attribute("circuit_name", handle.name)
        span.set_attribute("num_qubits", handle.num_qubits)

    # 2. Normalize
    with emitter.span("normalize") as span:
        normalized = adapter.normalize(handle)
        span.set_attribute("hash_before", handle.stable_hash()[:16])
        span.set_attribute("hash_after", normalized.stable_hash()[:16])

    # 3. Store input circuit (original) and normalized circuit
    with emitter.span("store_input") as span:
        if handle.qasm3:
            store.write_circuit("input.qasm", handle.qasm3)
        if normalized.qasm3:
            store.write_circuit("normalized.qasm", normalized.qasm3)

    # 4. Compile (with cache lookup)
    cache = CompilationCache()
    cache_key = CompilationCache.cache_key(
        circuit_hash=normalized.stable_hash(),
        pipeline_dict=pipeline_spec.to_dict(),
        backend_version=adapter.describe_backend().version,
        extra={"seeds": seeds},
    )
    cache_index: list[dict[str, Any]] = []

    with emitter.span("compile", attributes={"pipeline": pipeline_spec.to_dict()}) as span:
        cached = cache.get(cache_key)
        if cached is not None:
            span.set_attribute("cache_hit", True)
            span.add_event("cache_hit", key=cache_key[:16])
            cache_index.append({
                "key": cache_key[:16],
                "hit": True,
                "circuit_hash": normalized.stable_hash()[:16],
                "pipeline_hash": pipeline_spec.stable_hash()[:16],
                "timestamp": time.time(),
            })
            # Restore from cache — skip actual compilation
            from qocc.adapters.base import CompileResult as _CR
            compile_result = _CR.from_dict(cached)
            # If the cached handle has no native circuit, re-ingest from QASM
            if compile_result.circuit.native_circuit is None and compile_result.circuit.qasm3:
                try:
                    compile_result.circuit = adapter.ingest(compile_result.circuit.qasm3)
                except Exception as exc:
                    logger.warning("Cache re-ingest failed, recompiling: %s", exc)
                    span.add_event("cache_reingest_failed", error=str(exc))
                    compile_result = adapter.compile(normalized, pipeline_spec, emitter=emitter)
        else:
            span.set_attribute("cache_hit", False)
            span.add_event("cache_miss", key=cache_key[:16])
            compile_result = adapter.compile(normalized, pipeline_spec, emitter=emitter)
            # Store in cache
            cache.put(
                cache_key,
                compile_result.to_dict(),
                circuit_qasm=compile_result.circuit.qasm3,
                metadata={
                    "circuit_hash": normalized.stable_hash(),
                    "pipeline_hash": pipeline_spec.stable_hash(),
                },
            )
            cache_index.append({
                "key": cache_key[:16],
                "hit": False,
                "circuit_hash": normalized.stable_hash()[:16],
                "pipeline_hash": pipeline_spec.stable_hash()[:16],
                "timestamp": time.time(),
            })

        compiled = compile_result.circuit
        span.set_attribute("compiled_hash", compiled.stable_hash()[:16])
        span.set_attribute("pass_count", len(compile_result.pass_log))

    # 5. Store compiled circuit
    with emitter.span("store_compiled") as span:
        if compiled.qasm3:
            store.write_circuit("selected.qasm", compiled.qasm3)

    # 6. Compute metrics
    with emitter.span("compute_metrics") as span:
        metrics_before = adapter.get_metrics(normalized)
        metrics_after = adapter.get_metrics(compiled)
        span.set_attribute("metrics_before", metrics_before.to_dict())
        span.set_attribute("metrics_after", metrics_after.to_dict())

    # 7. Nondeterminism detection (if repeat >= 2)
    nondet_report: dict[str, Any] | None = None
    if repeat >= 2:
        from qocc.core.nondeterminism import detect_nondeterminism

        with emitter.span("nondeterminism_detection", attributes={"repeat": repeat}) as span:
            nd = detect_nondeterminism(adapter, normalized, pipeline_spec, num_runs=repeat)
            nondet_report = nd.to_dict()
            span.set_attribute("reproducible", nd.reproducible)
            span.set_attribute("unique_hashes", nd.unique_hashes)
            span.set_attribute("confidence", nd.confidence)
            # Record as span event per spec §1.3
            if not nd.reproducible:
                span.add_event(
                    "nondeterminism_detected",
                    unique_hashes=nd.unique_hashes,
                    confidence=nd.confidence,
                    hash_counts=nd.hash_counts,
                )

    # 8. Write bundle files
    with emitter.span("write_bundle") as span:
        store.write_manifest(run_id, extra={
            "adapter": adapter_name,
            "pipeline": pipeline_spec.to_dict(),
        })
        store.write_env()
        store.write_seeds(seeds)
        store.write_metrics({
            "input": metrics_before.to_dict(),
            "compiled": metrics_after.to_dict(),
            "pass_log": [p.to_dict() for p in compile_result.pass_log],
        })
        store.write_trace(emitter.to_dicts())

        # Cache index for reproducibility auditing
        store.write_cache_index(cache_index)

        # Write empty contracts / contract_results so bundles validate
        store.write_contracts([])
        store.write_contract_results([])

        # Write nondeterminism report if available
        if nondet_report:
            store.write_json("nondeterminism.json", nondet_report)

        # Generate summary report
        summary = _generate_summary(
            run_id=run_id,
            adapter_name=adapter_name,
            input_handle=normalized,
            compiled_handle=compiled,
            metrics_before=metrics_before.to_dict(),
            metrics_after=metrics_after.to_dict(),
            pipeline_spec=pipeline_spec,
            pass_log=compile_result.pass_log,
            nondet_report=nondet_report,
        )
        store.write_summary_report(summary)

    # 9. Zip
    with emitter.span("export_zip") as span:
        final_zip = store.export_zip(zip_path)
        span.set_attribute("zip_path", str(final_zip))

    # Re-write trace with all spans (including the zip span)
    store.write_trace(emitter.to_dicts())

    result_dict: dict[str, Any] = {
        "run_id": run_id,
        "bundle_dir": str(bundle_dir),
        "bundle_zip": str(final_zip),
        "input_hash": normalized.stable_hash(),
        "compiled_hash": compiled.stable_hash(),
        "metrics_before": metrics_before.to_dict(),
        "metrics_after": metrics_after.to_dict(),
        "num_spans": len(emitter.finished_spans()),
    }
    if nondet_report:
        result_dict["nondeterminism"] = nondet_report

    return result_dict


def compare_bundles(
    bundle_a: str | dict[str, Any],
    bundle_b: str | dict[str, Any],
    report_dir: str | None = None,
) -> dict[str, Any]:
    """Compare two Trace Bundles and produce a diff report.

    Parameters:
        bundle_a: Path to bundle zip/dir, or pre-loaded bundle dict.
        bundle_b: Path to bundle zip/dir, or pre-loaded bundle dict.
        report_dir: Optional output directory for reports.

    Returns:
        Comparison report dict.
    """
    if isinstance(bundle_a, str):
        bundle_a = ArtifactStore.load_bundle(bundle_a)
    if isinstance(bundle_b, str):
        bundle_b = ArtifactStore.load_bundle(bundle_b)

    report: dict[str, Any] = {
        "bundle_a": bundle_a.get("manifest", {}),
        "bundle_b": bundle_b.get("manifest", {}),
        "diffs": {},
    }

    # Metrics diff
    metrics_a = bundle_a.get("metrics", {})
    metrics_b = bundle_b.get("metrics", {})
    metrics_diff: dict[str, Any] = {}

    for stage in ["input", "compiled"]:
        ma = metrics_a.get(stage, {})
        mb = metrics_b.get(stage, {})
        stage_diff: dict[str, Any] = {}
        all_keys = set(list(ma.keys()) + list(mb.keys()))
        for k in sorted(all_keys):
            va = ma.get(k)
            vb = mb.get(k)
            if va != vb:
                diff_entry: dict[str, Any] = {"a": va, "b": vb}
                if isinstance(va, (int, float)) and isinstance(vb, (int, float)) and va != 0:
                    diff_entry["pct_change"] = ((vb - va) / abs(va)) * 100
                stage_diff[k] = diff_entry
        if stage_diff:
            metrics_diff[stage] = stage_diff

    report["diffs"]["metrics"] = metrics_diff

    # Environment diff
    env_a = bundle_a.get("env", {})
    env_b = bundle_b.get("env", {})
    env_diff: dict[str, Any] = {}
    for k in ["os", "python", "git_sha"]:
        if env_a.get(k) != env_b.get(k):
            env_diff[k] = {"a": env_a.get(k), "b": env_b.get(k)}
    # Package diffs
    pkgs_a = env_a.get("packages", {})
    pkgs_b = env_b.get("packages", {})
    pkg_diffs: dict[str, Any] = {}
    for pkg in sorted(set(list(pkgs_a.keys()) + list(pkgs_b.keys()))):
        va = pkgs_a.get(pkg)
        vb = pkgs_b.get(pkg)
        if va != vb:
            pkg_diffs[pkg] = {"a": va, "b": vb}
    if pkg_diffs:
        env_diff["packages"] = pkg_diffs

    report["diffs"]["environment"] = env_diff

    # Contract diffs
    cr_a = bundle_a.get("contract_results", [])
    cr_b = bundle_b.get("contract_results", [])
    report["diffs"]["contracts"] = {"a": cr_a, "b": cr_b}

    # Pass-log structural diff
    pass_log_a = metrics_a.get("pass_log", [])
    pass_log_b = metrics_b.get("pass_log", [])
    passes_a = [p.get("pass_name", "") for p in pass_log_a]
    passes_b = [p.get("pass_name", "") for p in pass_log_b]
    pass_diff: dict[str, Any] = {}
    if passes_a != passes_b:
        sm = difflib.SequenceMatcher(None, passes_a, passes_b)
        opcodes = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag != "equal":
                opcodes.append({
                    "action": tag,
                    "a": passes_a[i1:i2],
                    "b": passes_b[j1:j2],
                    "a_range": [i1, i2],
                    "b_range": [j1, j2],
                })
        added = [p for p in passes_b if p not in passes_a]
        removed = [p for p in passes_a if p not in passes_b]
        pass_diff = {
            "passes_a": passes_a,
            "passes_b": passes_b,
            "added": added,
            "removed": removed,
            "opcodes": opcodes,
        }
    report["diffs"]["pass_log"] = pass_diff

    # Circuit hash diffs
    circuit_hashes: dict[str, Any] = {}
    for label, bndl in [("a", bundle_a), ("b", bundle_b)]:
        root = bndl.get("_root")
        hashes: dict[str, str | None] = {"input": None, "compiled": None}
        if root:
            root_p = Path(root)
            for name, sub in [("input", "circuits/input.qasm"), ("compiled", "circuits/selected.qasm")]:
                fpath = root_p / sub
                if fpath.exists():
                    from qocc.core.hashing import hash_string
                    hashes[name] = hash_string(fpath.read_text(encoding="utf-8"))[:16]
        circuit_hashes[label] = hashes
    hash_changed = {
        k: {"a": circuit_hashes["a"].get(k), "b": circuit_hashes["b"].get(k)}
        for k in ["input", "compiled"]
        if circuit_hashes["a"].get(k) != circuit_hashes["b"].get(k)
    }
    report["diffs"]["circuit_hashes"] = hash_changed

    # Seeds diff
    seeds_a = bundle_a.get("seeds", {})
    seeds_b = bundle_b.get("seeds", {})
    seeds_diff: dict[str, Any] = {}
    if seeds_a != seeds_b:
        for k in sorted(set(list(seeds_a.keys()) + list(seeds_b.keys()))):
            va = seeds_a.get(k)
            vb = seeds_b.get(k)
            if va != vb:
                seeds_diff[k] = {"a": va, "b": vb}
    report["diffs"]["seeds"] = seeds_diff

    # ── Regression-cause analysis ────────────────────────────
    regression = _analyze_regression_causes(bundle_a, bundle_b, metrics_diff, env_diff)
    report["regression_analysis"] = regression

    # Generate markdown report
    md = _generate_comparison_md(report)
    report["markdown"] = md

    if report_dir:
        out = Path(report_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "comparison.json").write_text(
            json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
        )
        (out / "comparison.md").write_text(md, encoding="utf-8")

    return report


def check_contract(
    bundle_or_circuits: str | dict[str, Any],
    contract_spec: str | list[dict[str, Any]],
    adapter_name: str | None = None,
    simulation_shots: int = 1024,
    simulation_seed: int = 42,
) -> list[dict[str, Any]]:
    """Evaluate contracts against a bundle or circuits.

    Dispatches to the appropriate evaluator based on contract type:
    - ``"distribution"`` → TVD sampling test
    - ``"observable"``   → expectation CI test
    - ``"clifford"``     → stabilizer tableau comparison
    - ``"exact"``        → statevector fidelity check
    - ``"cost"``         → resource budget checks

    Parameters:
        bundle_or_circuits: Bundle path or loaded bundle dict.
        contract_spec: Path to contracts JSON or list of contract spec dicts.
        adapter_name: Adapter to use for simulation (inferred from bundle if None).
        simulation_shots: Default shots for distribution contracts.
        simulation_seed: Default seed for simulation.

    Returns:
        List of contract result dicts.
    """
    from qocc.contracts.spec import ContractSpec, ContractResult, CostSpec, CostResult
    from qocc.contracts.eval_sampling import (
        evaluate_distribution_contract,
        evaluate_observable_contract,
    )
    from qocc.contracts.eval_exact import (
        evaluate_exact_equivalence,
        evaluate_unitary_equivalence,
    )
    from qocc.contracts.eval_clifford import (
        evaluate_clifford_contract,
        is_clifford_circuit,
    )

    # Load bundle
    if isinstance(bundle_or_circuits, str):
        bundle = ArtifactStore.load_bundle(bundle_or_circuits)
    else:
        bundle = bundle_or_circuits

    # Load contract specs
    if isinstance(contract_spec, str):
        p = Path(contract_spec)
        specs_data = json.loads(p.read_text(encoding="utf-8"))
    else:
        specs_data = contract_spec

    specs = [ContractSpec.from_dict(s) for s in specs_data]
    results: list[dict[str, Any]] = []

    # Try to extract simulation data from bundle
    bundle_metrics = bundle.get("metrics", {})
    input_metrics = bundle_metrics.get("input", {})
    compiled_metrics = bundle_metrics.get("compiled", {})

    # Attempt to get adapter for simulation
    adapter = None
    if adapter_name is None:
        manifest = bundle.get("manifest", {})
        adapter_name = manifest.get("adapter")
    if adapter_name:
        try:
            adapter = get_adapter(adapter_name)
        except (KeyError, ImportError):
            adapter = None

    # Attempt to reconstruct circuit handles from bundle
    bundle_root = bundle.get("_root")
    input_handle = None
    compiled_handle = None
    if bundle_root:
        root = Path(bundle_root)
        input_qasm = root / "circuits" / "input.qasm"
        compiled_qasm = root / "circuits" / "selected.qasm"
        if adapter and input_qasm.exists():
            try:
                input_handle = adapter.ingest(str(input_qasm))
            except Exception as exc:
                logger.warning("Failed to ingest input circuit from bundle: %s", exc)
        if adapter and compiled_qasm.exists():
            try:
                compiled_handle = adapter.ingest(str(compiled_qasm))
            except Exception as exc:
                logger.warning("Failed to ingest compiled circuit from bundle: %s", exc)

    # Pre-run simulation if adapter is available and we need distribution/observable data
    # Clamp shots to minimum of all per-contract max_shots budgets
    sim_counts_input: dict[str, int] | None = None
    sim_counts_compiled: dict[str, int] | None = None
    need_simulation = any(
        s.type in ("distribution", "observable") for s in specs
    )

    if need_simulation and adapter and input_handle and compiled_handle:
        from qocc.adapters.base import SimulationSpec

        # Determine effective shots: min(simulation_shots, per-contract max_shots)
        effective_shots = simulation_shots
        for s in specs:
            if s.type in ("distribution", "observable"):
                budget_max = s.resource_budget.get("max_shots")
                if budget_max is not None:
                    effective_shots = min(effective_shots, int(budget_max))

        # Determine max_runtime budget (seconds) — use the minimum across contracts
        max_runtime: float | None = None
        for s in specs:
            if s.type in ("distribution", "observable"):
                rt = s.resource_budget.get("max_runtime")
                if rt is not None:
                    if max_runtime is None:
                        max_runtime = float(rt)
                    else:
                        max_runtime = min(max_runtime, float(rt))

        sim_spec = SimulationSpec(
            shots=effective_shots,
            seed=simulation_seed,
        )

        def _run_sim(handle: Any) -> dict[str, int] | None:
            try:
                r = adapter.simulate(handle, sim_spec)
                return r.counts
            except (NotImplementedError, Exception):
                return None

        if max_runtime is not None:
            # Run simulation in a thread with timeout
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                f_in = pool.submit(_run_sim, input_handle)
                f_out = pool.submit(_run_sim, compiled_handle)
                try:
                    sim_counts_input = f_in.result(timeout=max_runtime)
                except (concurrent.futures.TimeoutError, Exception):
                    sim_counts_input = None
                try:
                    sim_counts_compiled = f_out.result(timeout=max_runtime)
                except (concurrent.futures.TimeoutError, Exception):
                    sim_counts_compiled = None
        else:
            sim_counts_input = _run_sim(input_handle)
            sim_counts_compiled = _run_sim(compiled_handle)

    for spec in specs:
        result: ContractResult

        # ── Check evaluator registry first (plugin dispatch) ─
        if spec.evaluator not in ("auto", ""):
            from qocc.contracts.registry import get_evaluator

            custom_fn = get_evaluator(spec.evaluator)
            if custom_fn is not None:
                try:
                    result = custom_fn(
                        spec,
                        counts_before=sim_counts_input,
                        counts_after=sim_counts_compiled,
                        input_handle=input_handle,
                        compiled_handle=compiled_handle,
                        adapter=adapter,
                        bundle=bundle,
                    )
                    results.append(result.to_dict())
                    continue
                except Exception as exc:
                    result = ContractResult(
                        name=spec.name,
                        passed=False,
                        details={"error": f"Custom evaluator {spec.evaluator!r} failed: {exc}"},
                    )
                    results.append(result.to_dict())
                    continue

        # ── Distribution contract ────────────────────────────
        if spec.type == "distribution":
            if sim_counts_input and sim_counts_compiled:
                result = evaluate_distribution_contract(
                    spec, sim_counts_input, sim_counts_compiled,
                )
            else:
                result = ContractResult(
                    name=spec.name,
                    passed=False,
                    details={
                        "type": "distribution",
                        "error": "No simulation data available. "
                                 "Install adapter extras and provide circuit files.",
                    },
                )

        # ── Observable contract ──────────────────────────────
        elif spec.type == "observable":
            if sim_counts_input and sim_counts_compiled:
                # Convert counts to expectation values
                # (assume computational-basis Z observable: +1 for |0⟩, -1 for |1⟩)
                values_before = _counts_to_observable_values(sim_counts_input)
                values_after = _counts_to_observable_values(sim_counts_compiled)
                result = evaluate_observable_contract(
                    spec, values_before, values_after,
                )
            else:
                result = ContractResult(
                    name=spec.name,
                    passed=False,
                    details={
                        "type": "observable",
                        "error": "No simulation data. Install adapter extras.",
                    },
                )

        # ── Clifford contract ────────────────────────────────
        elif spec.type == "clifford":
            if input_handle and compiled_handle:
                result = evaluate_clifford_contract(
                    spec, input_handle, compiled_handle,
                    counts_before=sim_counts_input,
                    counts_after=sim_counts_compiled,
                )
            else:
                result = ContractResult(
                    name=spec.name,
                    passed=False,
                    details={
                        "type": "clifford",
                        "error": "Circuit handles not available for Clifford check.",
                    },
                )

        # ── Exact statevector contract ───────────────────────
        elif spec.type == "exact":
            sv_before = None
            sv_after = None
            if adapter and input_handle and compiled_handle:
                try:
                    from qocc.adapters.base import SimulationSpec as SS

                    sv_spec = SS(shots=0, method="statevector")
                    sv_res_in = adapter.simulate(input_handle, sv_spec)
                    sv_res_out = adapter.simulate(compiled_handle, sv_spec)
                    sv_before = sv_res_in.metadata.get("statevector")
                    sv_after = sv_res_out.metadata.get("statevector")
                except (NotImplementedError, Exception):
                    pass

            if sv_before is not None and sv_after is not None:
                result = evaluate_exact_equivalence(spec, sv_before, sv_after)
            else:
                result = ContractResult(
                    name=spec.name,
                    passed=False,
                    details={
                        "type": "exact",
                        "error": "Statevector simulation not available.",
                    },
                )

        # ── Cost/resource budget contract ────────────────────
        elif spec.type == "cost":
            result = _evaluate_cost_contract(spec, compiled_metrics)

        # ── Unknown type ─────────────────────────────────────
        else:
            result = ContractResult(
                name=spec.name,
                passed=False,
                details={
                    "type": spec.type,
                    "error": f"Unknown contract type: {spec.type!r}",
                },
            )

        results.append(result.to_dict())

    return results


def _counts_to_observable_values(counts: dict[str, int]) -> list[float]:
    """Convert measurement counts to Z-observable per-shot values.

    Each bitstring maps to: +1 if parity(bitstring) == 0, else -1.
    """
    values: list[float] = []
    for bitstring, count in counts.items():
        parity = sum(int(b) for b in bitstring) % 2
        val = 1.0 if parity == 0 else -1.0
        values.extend([val] * count)
    return values


def _evaluate_cost_contract(
    spec: Any,
    compiled_metrics: dict[str, Any],
) -> Any:
    """Evaluate cost/resource budget constraints.

    Checks compiled metrics against budget limits in the contract spec.
    """
    from qocc.contracts.spec import ContractResult

    violations: list[str] = []
    checks: dict[str, Any] = {}

    budget = spec.resource_budget
    tolerances = spec.tolerances

    # Check depth limit
    max_depth = budget.get("max_depth") or tolerances.get("max_depth")
    if max_depth is not None:
        actual = compiled_metrics.get("depth", 0)
        ok = actual <= max_depth
        checks["depth"] = {"limit": max_depth, "actual": actual, "passed": ok}
        if not ok:
            violations.append(f"depth {actual} > {max_depth}")

    # Check 2Q gate limit
    max_2q = budget.get("max_gates_2q") or tolerances.get("max_gates_2q")
    if max_2q is not None:
        actual = compiled_metrics.get("gates_2q", 0)
        ok = actual <= max_2q
        checks["gates_2q"] = {"limit": max_2q, "actual": actual, "passed": ok}
        if not ok:
            violations.append(f"gates_2q {actual} > {max_2q}")

    # Check total gate limit
    max_gates = budget.get("max_total_gates") or tolerances.get("max_total_gates")
    if max_gates is not None:
        actual = compiled_metrics.get("total_gates", 0)
        ok = actual <= max_gates
        checks["total_gates"] = {"limit": max_gates, "actual": actual, "passed": ok}
        if not ok:
            violations.append(f"total_gates {actual} > {max_gates}")

    # Check duration estimate
    max_duration = budget.get("max_duration_ns") or tolerances.get("max_duration_ns")
    if max_duration is not None:
        actual = compiled_metrics.get("duration_estimate_ns", 0)
        ok = actual <= max_duration
        checks["duration_ns"] = {"limit": max_duration, "actual": actual, "passed": ok}
        if not ok:
            violations.append(f"duration {actual} ns > {max_duration} ns")

    # Check proxy error score
    max_error = tolerances.get("max_proxy_error")
    if max_error is not None:
        actual = compiled_metrics.get("proxy_error_score", 0)
        ok = actual <= max_error
        checks["proxy_error"] = {"limit": max_error, "actual": actual, "passed": ok}
        if not ok:
            violations.append(f"proxy_error {actual} > {max_error}")

    passed = len(violations) == 0

    return ContractResult(
        name=spec.name,
        passed=passed,
        details={
            "type": "cost",
            "checks": checks,
            "violations": violations,
        },
    )


def search_compile(
    adapter_name: str,
    input_source: str | Any,
    search_config: dict[str, Any] | str | None = None,
    contracts: list[dict[str, Any]] | str | None = None,
    output: str | None = None,
    top_k: int = 5,
    simulation_shots: int = 1024,
    simulation_seed: int = 42,
    mode: str = "single",
) -> dict[str, Any]:
    """Closed-loop compilation search (v3 entrypoint).

    Generates candidate pipelines, compiles each, scores with a cheap
    surrogate, validates the top-k with simulation, evaluates contracts,
    and selects the best.

    Parameters:
        adapter_name: Adapter to use.
        input_source: Circuit file, QASM string, or native object.
        search_config: Search space config dict or path to JSON.
        contracts: Contract specs (list of dicts or path to JSON).
        output: Output bundle path.
        top_k: Number of top candidates to validate expensively.
        simulation_shots: Default simulation shots.
        simulation_seed: Default simulation seed.
        mode: Selection mode — ``"single"`` (best surrogate) or
              ``"pareto"`` (multi-objective Pareto frontier).

    Returns:
        SearchResult dict with selected candidate and full rankings.
    """
    from qocc.search.space import SearchSpaceConfig, Candidate, generate_candidates
    from qocc.search.scorer import surrogate_score, rank_candidates
    from qocc.search.validator import validate_candidates
    from qocc.search.selector import select_best
    from qocc.adapters.base import SimulationSpec
    from qocc.trace.emitter import TraceEmitter
    from qocc.contracts.spec import ContractSpec

    emitter = TraceEmitter()
    adapter = get_adapter(adapter_name)

    # ── Load search config ───────────────────────────────────
    if isinstance(search_config, str):
        p = Path(search_config)
        if p.exists():
            cfg_data = json.loads(p.read_text(encoding="utf-8"))
        else:
            cfg_data = {"adapter": adapter_name}
    elif isinstance(search_config, dict):
        cfg_data = search_config
    else:
        cfg_data = {"adapter": adapter_name}

    cfg_data.setdefault("adapter", adapter_name)
    config = SearchSpaceConfig.from_dict(cfg_data)

    # ── Load contracts ───────────────────────────────────────
    contract_specs: list[ContractSpec] = []
    if isinstance(contracts, str):
        cp = Path(contracts)
        contract_specs = [ContractSpec.from_dict(s) for s in json.loads(cp.read_text(encoding="utf-8"))]
    elif isinstance(contracts, list):
        contract_specs = [ContractSpec.from_dict(s) for s in contracts]

    # ── 1. Ingest & normalize ────────────────────────────────
    with emitter.span("ingest") as span:
        handle = adapter.ingest(input_source)
        span.set_attribute("circuit_name", handle.name)

    with emitter.span("normalize") as span:
        normalized = adapter.normalize(handle)
        span.set_attribute("hash", normalized.stable_hash()[:16])
        normalize_span_id = span.span_id

    # ── 2. Generate candidates ───────────────────────────────
    with emitter.span("generate_candidates") as span:
        candidates = generate_candidates(config)
        span.set_attribute("num_candidates", len(candidates))

    # ── 3. Compile each candidate (with cache) ──────────────
    circuit_handles: dict[str, Any] = {}
    cache = CompilationCache()
    cache_index: list[dict[str, Any]] = []

    def _compile_one_candidate(candidate: Any, parent_span_id: str | None = None) -> tuple[Any, Any, list[dict[str, Any]]]:
        """Compile a single candidate and return (candidate, handle, cache_entries).

        Creates a per-candidate span linked to *parent_span_id*.
        """
        local_cache_entries: list[dict[str, Any]] = []
        cand_span = emitter.start_span(
            f"compile/candidate/{candidate.candidate_id[:8]}",
            attributes={
                "candidate_id": candidate.candidate_id,
                "pipeline": str(candidate.pipeline.to_dict()),
            },
        )
        # Link back to the parent compile span
        if parent_span_id:
            cand_span.add_link(emitter.trace_id, parent_span_id, relationship="child_of_batch")
        try:
            c_key = CompilationCache.cache_key(
                circuit_hash=normalized.stable_hash(),
                pipeline_dict=candidate.pipeline.to_dict(),
                backend_version=adapter.describe_backend().version,
                extra={"seed": candidate.pipeline.parameters.get("seed")},
            )
            cached = cache.get(c_key)
            if cached is not None:
                local_cache_entries.append({
                    "key": c_key[:16], "hit": True,
                    "candidate_id": candidate.candidate_id,
                    "timestamp": time.time(),
                })
                from qocc.adapters.base import CompileResult as _CR
                result = _CR.from_dict(cached)
                if result.circuit.native_circuit is None and result.circuit.qasm3:
                    try:
                        result.circuit = adapter.ingest(result.circuit.qasm3)
                    except Exception as exc:
                        logger.warning("Cache re-ingest failed for candidate %s: %s", candidate.candidate_id[:8], exc)
                        result = adapter.compile(normalized, candidate.pipeline)
            else:
                local_cache_entries.append({
                    "key": c_key[:16], "hit": False,
                    "candidate_id": candidate.candidate_id,
                    "timestamp": time.time(),
                })
                result = adapter.compile(normalized, candidate.pipeline)
                cache.put(
                    c_key, result.to_dict(),
                    circuit_qasm=result.circuit.qasm3,
                    metadata={"candidate_id": candidate.candidate_id},
                )

            compiled = result.circuit
            m = adapter.get_metrics(compiled)
            candidate.metrics = m.to_dict()
            cand_span.set_attribute("cache_hit", local_cache_entries[-1]["hit"] if local_cache_entries else False)
            emitter.finish_span(cand_span, status="OK")
            return candidate, compiled, local_cache_entries
        except Exception as exc:
            logger.error("Candidate %s compilation failed: %s", candidate.candidate_id[:8], exc, exc_info=True)
            candidate.metrics = {"error": str(exc)}
            cand_span.record_exception(exc)
            emitter.finish_span(cand_span, status="ERROR")
            return candidate, None, local_cache_entries

    with emitter.span("compile_candidates") as compile_parent:
        # Use parallel compilation for speed when many candidates exist
        import concurrent.futures

        max_workers = min(len(candidates), os.cpu_count() or 4, 8)
        parent_sid = compile_parent.span_id

        if len(candidates) > 1 and max_workers > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(_compile_one_candidate, c, parent_sid): c
                    for c in candidates
                }
                for future in concurrent.futures.as_completed(futures):
                    cand, compiled_handle, c_entries = future.result()
                    cache_index.extend(c_entries)
                    if compiled_handle is not None:
                        circuit_handles[cand.candidate_id] = compiled_handle
        else:
            # Serial fallback for single candidate
            for candidate in candidates:
                cand, compiled_handle, c_entries = _compile_one_candidate(candidate, parent_sid)
                cache_index.extend(c_entries)
                if compiled_handle is not None:
                    circuit_handles[cand.candidate_id] = compiled_handle

        compile_parent.set_attribute("compiled_count", len(circuit_handles))
        compile_parent.set_attribute("parallel_workers", max_workers)

    # ── 4. Surrogate score & rank ────────────────────────────
    with emitter.span("score_and_rank") as span:
        ranked = rank_candidates(candidates)
        span.set_attribute("ranking_complete", True)

    # ── 5. Validate top-k ────────────────────────────────────
    with emitter.span("validate_top_k", attributes={"top_k": top_k}) as span:
        sim_spec = SimulationSpec(shots=simulation_shots, seed=simulation_seed)
        validated = validate_candidates(ranked, adapter, circuit_handles, top_k=top_k, sim_spec=sim_spec)
        span.set_attribute("validated_count", len([v for v in validated if v.validated]))

    # ── 6. Evaluate contracts on validated candidates ────────
    if contract_specs:
        with emitter.span("evaluate_contracts") as span:
            for candidate in validated:
                if not candidate.validated:
                    continue
                compiled_handle = circuit_handles.get(candidate.candidate_id)
                if not compiled_handle:
                    continue

                # Run contract check with the compiled metrics
                candidate_contracts = check_contract(
                    {"metrics": {"compiled": candidate.metrics}, "manifest": {}, "_root": None},
                    [s.to_dict() for s in contract_specs],
                    adapter_name=adapter_name,
                )
                candidate.contract_results = candidate_contracts

            span.set_attribute("contracts_evaluated", len(contract_specs))

    # ── 7. Select best candidate ─────────────────────────────
    with emitter.span("select_best", attributes={"mode": mode}) as span:
        selection = select_best(validated, require_validated=True, mode=mode)
        span.set_attribute("feasible", selection.feasible)
        if selection.selected:
            span.set_attribute("selected_id", selection.selected.candidate_id)

    # ── 8. Write output bundle ───────────────────────────────
    run_id = uuid.uuid4().hex[:12]
    if output:
        out_path = Path(output)
        if out_path.suffix == ".zip":
            bundle_dir = out_path.with_suffix("")
            zip_path = out_path
        else:
            bundle_dir = out_path
            zip_path = out_path.with_suffix(".zip")
    else:
        bundle_dir = Path(tempfile.mkdtemp(prefix="qocc_search_"))
        zip_path = bundle_dir.with_suffix(".zip")

    store = ArtifactStore(bundle_dir)

    store.write_manifest(run_id, extra={
        "type": "search",
        "adapter": adapter_name,
        "num_candidates": len(candidates),
        "top_k": top_k,
    })
    store.write_env()
    store.write_trace(emitter.to_dicts())
    store.write_cache_index(cache_index)

    # Write input & normalized circuits
    if handle.qasm3:
        store.write_circuit("input.qasm", handle.qasm3)
    if normalized.qasm3:
        store.write_circuit("normalized.qasm", normalized.qasm3)

    # Write candidate circuits
    for cid, ch in circuit_handles.items():
        if ch.qasm3:
            store.write_circuit(f"candidates/{cid}.qasm", ch.qasm3)

    # Write rankings table
    rankings = [c.to_dict() for c in ranked]
    store.write_json("search_rankings.json", rankings)

    # Write selected candidate details
    store.write_json("search_result.json", selection.to_dict())

    # Write contracts and contract results
    contract_spec_dicts = [s.to_dict() for s in contract_specs]
    all_contract_results: list[dict[str, Any]] = []
    for c in validated:
        for cr in c.contract_results:
            all_contract_results.append(cr)
    store.write_contracts(contract_spec_dicts)
    store.write_contract_results(all_contract_results)

    # Store selected circuit if available
    if selection.selected:
        sel_handle = circuit_handles.get(selection.selected.candidate_id)
        if sel_handle and sel_handle.qasm3:
            store.write_circuit("selected.qasm", sel_handle.qasm3)

    # Generate search summary report
    search_summary = _generate_search_summary(
        run_id=run_id,
        adapter_name=adapter_name,
        num_candidates=len(candidates),
        top_k=top_k,
        selection=selection,
        ranked=ranked,
        contract_specs=contract_specs,
        all_contract_results=all_contract_results,
    )
    store.write_summary_report(search_summary)

    store.export_zip(zip_path)

    return {
        "run_id": run_id,
        "bundle_dir": str(bundle_dir),
        "bundle_zip": str(zip_path),
        "num_candidates": len(candidates),
        "num_validated": len([c for c in validated if c.validated]),
        "feasible": selection.feasible,
        "selected": selection.selected.to_dict() if selection.selected else None,
        "selection_reason": selection.reason,
        "top_rankings": [c.to_dict() for c in ranked[:top_k]],
    }


def _analyze_regression_causes(
    bundle_a: dict[str, Any],
    bundle_b: dict[str, Any],
    metrics_diff: dict[str, Any],
    env_diff: dict[str, Any],
) -> dict[str, Any]:
    """Identify likely causes of compilation regressions.

    Heuristics:
    1. Tool version change → likely regression source.
    2. Pass-log differences → different pass pipeline.
    3. Seed differences → stochastic pass nondeterminism.
    4. Circuit hash change in input → different input (not a regression).
    5. Metric regressions → quantified for reporting.
    """
    causes: list[dict[str, Any]] = []
    severity = "none"  # none | low | medium | high | critical

    # 1. Tool version changes
    pkgs_a = bundle_a.get("env", {}).get("packages", {})
    pkgs_b = bundle_b.get("env", {}).get("packages", {})
    tool_changes = []
    for pkg_name in ["qiskit", "qiskit-terra", "cirq-core", "qocc"]:
        va = pkgs_a.get(pkg_name)
        vb = pkgs_b.get(pkg_name)
        if va and vb and va != vb:
            tool_changes.append({"package": pkg_name, "a": va, "b": vb})
    if tool_changes:
        causes.append({
            "type": "tool_version_change",
            "description": "Framework/tool version changed between runs",
            "packages": tool_changes,
            "likelihood": "high",
        })
        severity = max(severity, "high", key=_severity_ranking)

    # 2. Pass-log differences
    metrics_a = bundle_a.get("metrics", {})
    metrics_b = bundle_b.get("metrics", {})
    pass_log_a = metrics_a.get("pass_log", [])
    pass_log_b = metrics_b.get("pass_log", [])
    passes_a = [p.get("pass_name", "") for p in pass_log_a]
    passes_b = [p.get("pass_name", "") for p in pass_log_b]
    if passes_a != passes_b:
        causes.append({
            "type": "pass_pipeline_change",
            "description": "Compilation pass pipeline differs",
            "passes_a": passes_a,
            "passes_b": passes_b,
            "likelihood": "high",
        })
        severity = max(severity, "high", key=_severity_ranking)

    # 3. Seed differences
    seeds_a = bundle_a.get("seeds", {})
    seeds_b = bundle_b.get("seeds", {})
    if seeds_a != seeds_b:
        causes.append({
            "type": "seed_change",
            "description": "Seeds differ — stochastic passes may produce different output",
            "seeds_a": seeds_a,
            "seeds_b": seeds_b,
            "likelihood": "medium",
        })
        severity = max(severity, "medium", key=_severity_ranking)

    # 4. Input circuit hash change
    manifest_a = bundle_a.get("manifest", {})
    manifest_b = bundle_b.get("manifest", {})
    # Check if using different pipelines (adapter / pipeline spec)
    pipe_a = manifest_a.get("pipeline", {})
    pipe_b = manifest_b.get("pipeline", {})
    if pipe_a != pipe_b:
        causes.append({
            "type": "pipeline_spec_change",
            "description": "Pipeline specification differs",
            "pipeline_a": pipe_a,
            "pipeline_b": pipe_b,
            "likelihood": "high",
        })

    # 5. Quantify metric regressions
    regressions: list[dict[str, Any]] = []
    for stage, stage_diff in metrics_diff.items():
        for metric_name, diff_info in stage_diff.items():
            pct = diff_info.get("pct_change", 0)
            if pct > 10:  # 10% regression threshold
                regressions.append({
                    "metric": f"{stage}.{metric_name}",
                    "pct_change": pct,
                    "a": diff_info["a"],
                    "b": diff_info["b"],
                })
    if regressions:
        worst_pct = max(r["pct_change"] for r in regressions)
        if worst_pct > 50:
            severity = max(severity, "critical", key=_severity_ranking)
        elif worst_pct > 25:
            severity = max(severity, "high", key=_severity_ranking)
        else:
            severity = max(severity, "medium", key=_severity_ranking)

    # 6. OS / Python change
    os_changed = env_diff.get("os")
    python_changed = env_diff.get("python")
    if os_changed or python_changed:
        causes.append({
            "type": "environment_change",
            "description": "OS or Python version changed",
            "os": os_changed,
            "python": python_changed,
            "likelihood": "low",
        })

    if not causes and not regressions:
        summary = "No regressions detected."
    elif not causes:
        summary = (
            f"Detected {len(regressions)} metric regressions but no obvious "
            f"root cause. Consider seed nondeterminism or environmental factors."
        )
    else:
        root = causes[0]["type"].replace("_", " ")
        summary = f"Most likely cause: {root}. {len(regressions)} metrics regressed."

    return {
        "severity": severity,
        "causes": causes,
        "regressions": regressions,
        "summary": summary,
    }


def _severity_ranking(s: str) -> int:
    return {"none": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}.get(s, 0)


# ======================================================================
# Report generation helpers
# ======================================================================


def _generate_summary(
    run_id: str,
    adapter_name: str,
    input_handle: Any,
    compiled_handle: Any,
    metrics_before: dict[str, Any],
    metrics_after: dict[str, Any],
    pipeline_spec: Any,
    pass_log: list[Any],
    nondet_report: dict[str, Any] | None = None,
    contract_results: list[dict[str, Any]] | None = None,
) -> str:
    """Generate a Markdown summary report for a trace bundle."""
    lines = [
        f"# QOCC Trace Bundle Summary",
        f"",
        f"**Run ID:** `{run_id}`",
        f"**Adapter:** {adapter_name}",
        f"",
        f"## Environment",
        f"",
        f"- **OS:** {platform.platform()}",
        f"- **Python:** {sys.version.split()[0]}",
        f"- **QOCC:** 0.1.0",
        f"",
        f"## Input Circuit",
        f"",
        f"- **Name:** {input_handle.name}",
        f"- **Qubits:** {input_handle.num_qubits}",
        f"- **Hash:** `{input_handle.stable_hash()[:16]}`",
        f"",
        f"## Compiled Circuit",
        f"",
        f"- **Name:** {compiled_handle.name}",
        f"- **Qubits:** {compiled_handle.num_qubits}",
        f"- **Hash:** `{compiled_handle.stable_hash()[:16]}`",
        f"",
        f"## Pipeline Configuration",
        f"",
        f"```json",
        json.dumps(pipeline_spec.to_dict(), indent=2),
        f"```",
        f"",
        f"## Metrics Comparison",
        f"",
        f"| Metric | Before | After |",
        f"|--------|--------|-------|",
    ]

    for key in sorted(set(list(metrics_before.keys()) + list(metrics_after.keys()))):
        if key == "gate_histogram":
            continue
        vb = metrics_before.get(key, "—")
        va = metrics_after.get(key, "—")
        lines.append(f"| {key} | {vb} | {va} |")

    lines.extend([
        f"",
        f"## Pass Log",
        f"",
    ])

    for entry in pass_log:
        d = entry.to_dict() if hasattr(entry, "to_dict") else entry
        lines.append(f"- **{d['pass_name']}** (order {d['order']}, {d.get('duration_ms', '?')} ms)")

    # Contract results table
    if contract_results:
        lines.extend([
            f"",
            f"## Contract Results",
            f"",
            f"| Contract | Type | Passed |",
            f"|----------|------|--------|",
        ])
        for cr in contract_results:
            icon = "✅" if cr.get("passed") else "❌"
            lines.append(f"| {cr.get('name', '?')} | {cr.get('details', {}).get('type', '?')} | {icon} |")

    # Reproducibility / nondeterminism
    lines.extend([
        f"",
        f"## Reproducibility",
        f"",
    ])

    if nondet_report:
        reproducible = nondet_report.get("reproducible", True)
        if not reproducible:
            lines.append(
                f"> ⚠ **Nondeterminism detected:** {nondet_report.get('unique_hashes', '?')} "
                f"unique hashes in {nondet_report.get('num_runs', '?')} runs "
                f"(confidence {nondet_report.get('confidence', 0) * 100:.0f}%)"
            )
        else:
            lines.append(
                f"Compilation is reproducible ({nondet_report.get('num_runs', '?')} runs, "
                f"confidence {nondet_report.get('confidence', 0) * 100:.0f}%)."
            )
    else:
        lines.append("Re-run with `qocc trace run --repeat N` to detect nondeterminism.")

    lines.append("")
    return "\n".join(lines) + "\n"


def _generate_search_summary(
    run_id: str,
    adapter_name: str,
    num_candidates: int,
    top_k: int,
    selection: Any,
    ranked: list[Any],
    contract_specs: list[Any],
    all_contract_results: list[dict[str, Any]],
) -> str:
    """Generate a Markdown summary report for a search bundle."""
    lines = [
        "# QOCC Search Compilation Summary",
        "",
        f"**Run ID:** `{run_id}`",
        f"**Adapter:** {adapter_name}",
        f"**Candidates:** {num_candidates} generated, top {top_k} validated",
        f"**Feasible:** {'Yes' if selection.feasible else 'No'}",
        "",
        "## Environment",
        "",
        f"- **OS:** {platform.platform()}",
        f"- **Python:** {sys.version.split()[0]}",
        "",
        "## Selection",
        "",
        f"**Reason:** {selection.reason}",
        "",
    ]

    if selection.selected:
        s = selection.selected
        lines.extend([
            f"### Selected Candidate: `{s.candidate_id}`",
            "",
            f"- **Surrogate score:** {s.surrogate_score:.4f}",
            f"- **Validated:** {s.validated}",
            "",
        ])

    # Pareto frontier
    if selection.pareto_frontier:
        lines.extend([
            "## Pareto Frontier",
            "",
            f"**{len(selection.pareto_frontier)}** non-dominated candidates.",
            "",
        ])

    # Candidate rankings table
    lines.extend([
        "## Candidate Rankings",
        "",
        "| # | ID | Opt Level | Score | Depth | 2Q Gates | Validated | Contracts |",
        "|---|-----|-----------|-------|-------|----------|-----------|-----------|",
    ])

    for i, c in enumerate(ranked[:20], 1):
        cid = c.candidate_id[:12]
        opt = c.pipeline.to_dict().get("optimization_level", "?")
        score = f"{c.surrogate_score:.4f}" if c.surrogate_score < float("inf") else "—"
        depth = c.metrics.get("depth", "—")
        g2q = c.metrics.get("gates_2q", "—")
        val = "✅" if c.validated else "—"
        n_pass = sum(1 for r in c.contract_results if r.get("passed")) if c.contract_results else "—"
        n_total = len(c.contract_results) if c.contract_results else 0
        contracts_str = f"{n_pass}/{n_total}" if n_total else "—"
        lines.append(f"| {i} | `{cid}` | {opt} | {score} | {depth} | {g2q} | {val} | {contracts_str} |")

    # Contract results
    if all_contract_results:
        lines.extend([
            "",
            "## Contract Results",
            "",
            "| Contract | Type | Passed |",
            "|----------|------|--------|",
        ])
        for cr in all_contract_results:
            icon = "✅" if cr.get("passed") else "❌"
            lines.append(f"| {cr.get('name', '?')} | {cr.get('details', {}).get('type', '?')} | {icon} |")

    lines.append("")
    return "\n".join(lines) + "\n"


def _generate_comparison_md(report: dict[str, Any]) -> str:
    """Generate a Markdown comparison report."""
    lines = [
        "# QOCC Bundle Comparison",
        "",
        "## Bundles",
        "",
        f"- **A:** {report['bundle_a'].get('run_id', 'unknown')}",
        f"- **B:** {report['bundle_b'].get('run_id', 'unknown')}",
        "",
    ]

    diffs = report.get("diffs", {})

    # Metrics
    metrics_diff = diffs.get("metrics", {})
    if metrics_diff:
        lines.extend(["## Metric Differences", ""])
        for stage, stage_diff in metrics_diff.items():
            lines.append(f"### {stage.title()}")
            lines.extend(["", "| Metric | Bundle A | Bundle B | Change |", "|--------|----------|----------|--------|"])
            for k, v in stage_diff.items():
                pct = f"{v.get('pct_change', 0):.1f}%" if "pct_change" in v else "—"
                lines.append(f"| {k} | {v['a']} | {v['b']} | {pct} |")
            lines.append("")
    else:
        lines.extend(["## Metrics", "", "No metric differences detected.", ""])

    # Circuit hash diffs
    hash_diffs = diffs.get("circuit_hashes", {})
    if hash_diffs:
        lines.extend(["## Circuit Hash Differences", ""])
        for name, hd in hash_diffs.items():
            lines.append(f"- **{name}:** `{hd.get('a', '—')}` → `{hd.get('b', '—')}`")
        lines.append("")

    # Seeds diff
    seeds_diff = diffs.get("seeds", {})
    if seeds_diff:
        lines.extend(["## Seed Differences", ""])
        for k, v in seeds_diff.items():
            lines.append(f"- **{k}:** `{v.get('a', '—')}` → `{v.get('b', '—')}`")
        lines.append("")

    # Pass-log diff
    pass_diff = diffs.get("pass_log", {})
    if pass_diff:
        lines.extend(["## Pass-Log Differences", ""])
        added = pass_diff.get("added", [])
        removed = pass_diff.get("removed", [])
        if added:
            lines.append(f"- **Added:** {', '.join(added)}")
        if removed:
            lines.append(f"- **Removed:** {', '.join(removed)}")
        opcodes = pass_diff.get("opcodes", [])
        if opcodes:
            lines.extend(["", "| Action | Bundle A | Bundle B |", "|--------|----------|----------|"])
            for op in opcodes:
                a_str = ", ".join(op.get("a", []))
                b_str = ", ".join(op.get("b", []))
                lines.append(f"| {op['action']} | {a_str} | {b_str} |")
        lines.append("")

    # Environment
    env_diff = diffs.get("environment", {})
    if env_diff:
        lines.extend(["## Environment Differences", ""])
        for k, v in env_diff.items():
            if k != "packages":
                lines.append(f"- **{k}:** `{v['a']}` → `{v['b']}`")
        pkg_diff = env_diff.get("packages", {})
        if pkg_diff:
            lines.extend(["", "### Package Differences", "", "| Package | A | B |", "|---------|---|---|"])
            for pkg, v in sorted(pkg_diff.items()):
                lines.append(f"| {pkg} | {v.get('a', '—')} | {v.get('b', '—')} |")
        lines.append("")

    # Regression Analysis
    regression = report.get("regression_analysis", {})
    if regression:
        sev = regression.get("severity", "none")
        lines.extend([
            "## Regression Analysis",
            "",
            f"**Severity:** {sev.upper()}",
            f"**Summary:** {regression.get('summary', 'N/A')}",
            "",
        ])
        causes = regression.get("causes", [])
        if causes:
            lines.extend(["### Likely Causes", ""])
            for c in causes:
                lines.append(f"- **{c['type']}** ({c.get('likelihood', '?')}): {c['description']}")
            lines.append("")

        regressions = regression.get("regressions", [])
        if regressions:
            lines.extend([
                "### Metric Regressions", "",
                "| Metric | A | B | Change |",
                "|--------|---|---|--------|",
            ])
            for r in regressions:
                lines.append(
                    f"| {r['metric']} | {r['a']} | {r['b']} | +{r['pct_change']:.1f}% |"
                )
            lines.append("")

    return "\n".join(lines) + "\n"
