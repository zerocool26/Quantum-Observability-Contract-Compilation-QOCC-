"""CLI commands for closed-loop compilation search (v3)."""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group("compile")
def compile_group() -> None:
    """Compilation search commands (v3)."""


@compile_group.command("search")
@click.option("--adapter", "-a", type=str, default="qiskit", help="Adapter name.")
@click.option("--input", "-i", "input_source", type=click.Path(exists=True), required=True,
              help="Input circuit file.")
@click.option("--bundle", "-b", type=click.Path(exists=True), default=None,
              help="Input Trace Bundle to optimise (alternative to --input).")
@click.option("--search", "-s", "search_spec", type=click.Path(exists=True), default=None,
              help="Search space specification JSON.")
@click.option("--contracts", "-c", type=click.Path(exists=True), default=None,
              help="Contracts specification JSON.")
@click.option("--topk", "-k", type=int, default=5, help="Number of candidates to validate.")
@click.option("--shots", type=int, default=1024, help="Simulation shots for validation.")
@click.option("--mode", "-m", type=click.Choice(["single", "pareto"]), default="single",
              help="Selection mode: single (best surrogate) or pareto (multi-objective).")
@click.option("--strategy", type=click.Choice(["grid", "random", "bayesian", "evolutionary"]), default="grid",
              help="Search strategy: grid (exhaustive), random (sampling), bayesian (adaptive), evolutionary (genetic).")
@click.option("--noise-model", type=click.Path(exists=True), default=None,
              help="Optional noise model JSON file for noise-aware surrogate scoring.")
@click.option("--prior-half-life", type=float, default=30.0,
              help="Half-life in days for Bayesian historical prior weighting.")
@click.option("--out", "-o", "output", type=click.Path(), default=None,
              help="Output bundle path.")
def compile_search(
    adapter: str,
    input_source: str | None,
    bundle: str | None,
    search_spec: str | None,
    contracts: str | None,
    topk: int,
    shots: int,
    mode: str,
    strategy: str,
    noise_model: str | None,
    prior_half_life: float,
    output: str | None,
) -> None:
    """Generate, compile, score, validate, and select best pipeline.

    Runs the full closed-loop compilation search from §3 of the QOCC spec.
    """
    from qocc.api import search_compile

    console.print("[bold blue]QOCC Compilation Search[/bold blue]")

    # Determine input source
    circuit_source = input_source
    if not circuit_source and bundle:
        from qocc.core.artifacts import ArtifactStore
        from pathlib import Path

        b = ArtifactStore.load_bundle(bundle)
        root = b.get("_root")
        if root:
            input_qasm = Path(root) / "circuits" / "input.qasm"
            if input_qasm.exists():
                circuit_source = str(input_qasm)
        if not circuit_source:
            console.print("[red]Error: Could not extract circuit from bundle.[/red]")
            sys.exit(1)

    if not circuit_source:
        console.print("[red]Error: Provide --input or --bundle.[/red]")
        sys.exit(1)

    # Load and validate search config
    from qocc.cli.validation import validate_json_file

    search_config = None
    if search_spec:
        search_config = validate_json_file(search_spec, "search_space", strict=False)
    if search_config is None:
        search_config = {}
    # Inject strategy from CLI flag
    search_config.setdefault("strategy", strategy)
    search_config.setdefault("bayesian_prior_half_life_days", prior_half_life)
    if noise_model:
        search_config["noise_model"] = validate_json_file(noise_model, "noise_model", strict=False)

    # Load and validate contracts
    contract_data = None
    if contracts:
        contract_data = validate_json_file(contracts, "contracts", strict=False)

    console.print(f"  Adapter: {adapter}")
    console.print(f"  Input: {circuit_source}")
    console.print(f"  Top-K: {topk}")

    try:
        result = search_compile(
            adapter_name=adapter,
            input_source=circuit_source,
            search_config=search_config,
            contracts=contract_data,
            output=output,
            top_k=topk,
            simulation_shots=shots,
            mode=mode,
        )
    except Exception as exc:
        console.print(f"[red]Search failed: {exc}[/red]")
        sys.exit(1)

    # Display results
    console.print(f"\n[green]Search complete![/green]")
    console.print(f"  Candidates: {result['num_candidates']}")
    console.print(f"  Validated:  {result['num_validated']}")
    console.print(f"  Feasible:   {result['feasible']}")

    selected = result.get("selected")
    if selected:
        console.print(f"\n[bold green]Selected: {selected['candidate_id'][:12]}[/bold green]")
        console.print(f"  Score: {selected['surrogate_score']:.4f}")
        console.print(f"  Opt Level: {selected['pipeline'].get('optimization_level', '?')}")

    # Rankings table
    rankings = result.get("top_rankings", [])
    if rankings:
        table = Table(title=f"Top-{topk} Candidates")
        table.add_column("#", justify="right")
        table.add_column("ID")
        table.add_column("Opt Level", justify="right")
        table.add_column("Score", justify="right")
        table.add_column("Validated")
        table.add_column("Contracts")

        for i, c in enumerate(rankings, 1):
            contracts_status = "—"
            if c.get("contract_results"):
                all_pass = all(r.get("passed", False) for r in c["contract_results"])
                contracts_status = "[green]PASS[/green]" if all_pass else "[red]FAIL[/red]"

            table.add_row(
                str(i),
                c["candidate_id"][:12],
                str(c["pipeline"].get("optimization_level", "?")),
                f"{c['surrogate_score']:.2f}",
                "[green]YES[/green]" if c.get("validated") else "[dim]no[/dim]",
                contracts_status,
            )

        console.print()
        console.print(table)

    console.print(f"\n  Bundle: {result.get('bundle_zip', result.get('bundle_dir'))}")
    console.print(f"  Reason: {result['selection_reason']}")


@compile_group.command("batch")
@click.option("--manifest", "manifest_path", type=click.Path(exists=True), required=True,
              help="Batch manifest JSON listing circuits and per-circuit options.")
@click.option("--workers", type=int, default=None,
              help="Optional parallel worker count across circuits.")
@click.option("--out", "-o", "output", type=click.Path(), default=None,
              help="Output batch bundle path.")
def compile_batch(
    manifest_path: str,
    workers: int | None,
    output: str | None,
) -> None:
    """Run batch compilation search across multiple circuits from a manifest."""
    from qocc.api import batch_search_compile

    console.print("[bold blue]QOCC Batch Compilation Search[/bold blue]")
    console.print(f"  Manifest: {manifest_path}")
    if workers is not None:
        console.print(f"  Workers: {workers}")

    try:
        result = batch_search_compile(
            manifest=manifest_path,
            output=output,
            workers=workers,
        )
    except Exception as exc:
        console.print(f"[red]Batch search failed: {exc}[/red]")
        sys.exit(1)

    rows = result.get("cross_circuit_metrics", {}).get("rows", [])
    table = Table(title="Batch Results")
    table.add_column("Circuit")
    table.add_column("Status")
    table.add_column("Candidates", justify="right")
    table.add_column("Validated", justify="right")
    table.add_column("Feasible")

    for row in rows:
        table.add_row(
            str(row.get("id", "?")),
            str(row.get("status", "?")),
            str(row.get("num_candidates", "—")),
            str(row.get("num_validated", "—")),
            str(row.get("feasible", "—")),
        )

    console.print(table)
    console.print(f"\n  Batch bundle: {result.get('bundle_zip', result.get('bundle_dir'))}")
