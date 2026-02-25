"""CLI commands for trace operations."""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group("trace")
def trace() -> None:
    """Trace-related commands."""


@trace.command("run")
@click.option("--adapter", "-a", required=True, type=click.Choice(["qiskit", "cirq"]),
              help="Backend adapter to use.")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Input circuit file (QASM).")
@click.option("--pipeline", "-p", type=click.Path(exists=True), default=None,
              help="Pipeline specification JSON file.")
@click.option("--out", "-o", "output", type=click.Path(), default=None,
              help="Output bundle path (zip or directory).")
@click.option("--seed", type=int, default=42, help="Global seed.")
@click.option("--repeat", "-n", type=int, default=1,
              help="Repeat compilation N times for nondeterminism detection (>=2).")
def trace_run(
    adapter: str,
    input_path: str,
    pipeline: str | None,
    output: str | None,
    seed: int,
    repeat: int,
) -> None:
    """Run an instrumented compilation trace and produce a Trace Bundle."""
    from qocc.api import run_trace

    seeds = {"global_seed": seed, "rng_algorithm": "MT19937", "stage_seeds": {}}

    console.print(f"[bold blue]QOCC Trace Run[/bold blue]")
    console.print(f"  Adapter: {adapter}")
    console.print(f"  Input:   {input_path}")
    console.print(f"  Seed:    {seed}")
    if repeat >= 2:
        console.print(f"  Repeat:  {repeat} (nondeterminism detection)")

    try:
        result = run_trace(
            adapter_name=adapter,
            input_source=input_path,
            pipeline=pipeline,
            output=output,
            seeds=seeds,
            repeat=repeat,
        )
    except ImportError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        console.print(f"Install the adapter with: pip install 'qocc[{adapter}]'")
        sys.exit(1)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    console.print()
    console.print(f"[green]✓[/green] Bundle created: {result['bundle_zip']}")
    console.print(f"  Run ID:         {result['run_id']}")
    console.print(f"  Input hash:     {result['input_hash'][:16]}…")
    console.print(f"  Compiled hash:  {result['compiled_hash'][:16]}…")
    console.print(f"  Spans recorded: {result['num_spans']}")

    # Show nondeterminism result if present
    nondet = result.get("nondeterminism")
    if nondet:
        if nondet["reproducible"]:
            console.print(
                f"  [green]✓ Reproducible[/green] "
                f"({nondet['num_runs']} runs, confidence {nondet['confidence']:.2%})"
            )
        else:
            console.print(
                f"  [red]✗ NONDETERMINISTIC[/red] "
                f"({nondet['unique_hashes']} unique hashes in {nondet['num_runs']} runs)"
            )

    # Show metrics table
    table = Table(title="Metrics Comparison")
    table.add_column("Metric")
    table.add_column("Before", justify="right")
    table.add_column("After", justify="right")

    mb = result["metrics_before"]
    ma = result["metrics_after"]
    for key in sorted(set(list(mb.keys()) + list(ma.keys()))):
        if key == "gate_histogram":
            continue
        table.add_row(key, str(mb.get(key, "—")), str(ma.get(key, "—")))

    console.print(table)


@trace.command("timeline")
@click.argument("bundle", type=click.Path(exists=True))
@click.option("--width", "-w", type=int, default=100, help="Terminal width.")
@click.option("--attrs/--no-attrs", default=False, help="Show span attributes.")
def trace_timeline(bundle: str, width: int, attrs: bool) -> None:
    """Display an ASCII timeline visualization of a trace bundle."""
    from qocc.trace.visualization import render_timeline_from_bundle

    timeline = render_timeline_from_bundle(bundle, width=width, show_attributes=attrs)
    console.print(timeline)


# ── compare subcommand (canonical location) ──────────────────
from qocc.cli.commands_compare import compare as _trace_compare_cmd
trace.add_command(_trace_compare_cmd, "compare")


@trace.command("replay")
@click.argument("bundle", type=click.Path(exists=True))
@click.option("--out", "-o", "output", type=click.Path(), default=None,
              help="Output path for replay bundle.")
def trace_replay(bundle: str, output: str | None) -> None:
    """Replay a trace bundle to verify reproducibility."""
    from qocc.core.replay import replay_bundle

    console.print(f"[bold blue]QOCC Bundle Replay[/bold blue]")
    console.print(f"  Original: {bundle}")

    try:
        result = replay_bundle(bundle, output=output)
    except Exception as exc:
        console.print(f"[red]Replay failed: {exc}[/red]")
        sys.exit(1)

    console.print(f"  Run ID:   {result.original_run_id}")

    if result.bit_exact:
        console.print(f"  [green]✓ BIT-EXACT match[/green]")
    else:
        console.print(f"  [yellow]⚠ Differences detected:[/yellow]")
        console.print(f"    Input hash match:    {'✓' if result.input_hash_match else '✗'}")
        console.print(f"    Compiled hash match: {'✓' if result.compiled_hash_match else '✗'}")
        console.print(f"    Metrics match:       {'✓' if result.metrics_match else '✗'}")
        if result.diff:
            for k, v in result.diff.items():
                console.print(f"    {k}: {v}")

    if result.replay_bundle:
        console.print(f"  Replay bundle: {result.replay_bundle}")
