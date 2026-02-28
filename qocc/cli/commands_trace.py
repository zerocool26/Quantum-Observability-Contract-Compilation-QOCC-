"""CLI commands for trace operations."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from qocc import DEFAULT_RNG_ALGORITHM, DEFAULT_SEED

console = Console()


@click.group("trace")
def trace() -> None:
    """Trace-related commands."""


@trace.command("run")
@click.option("--adapter", "-a", required=True, type=click.Choice(["qiskit", "cirq", "tket", "stim", "ibm"]),
              help="Backend adapter to use.")
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True),
              help="Input circuit file (QASM).")
@click.option("--pipeline", "-p", type=click.Path(exists=True), default=None,
              help="Pipeline specification JSON file.")
@click.option("--out", "-o", "output", type=click.Path(), default=None,
              help="Output bundle path (zip or directory).")
@click.option("--seed", type=int, default=DEFAULT_SEED, help="Global seed.")
@click.option("--repeat", "-n", type=int, default=1,
              help="Repeat compilation N times for nondeterminism detection (>=2).")
@click.option("--db/--no-db", "ingest_db", default=False,
              help="Auto-ingest resulting bundle into regression DB.")
@click.option("--db-path", type=click.Path(), default=None,
              help="Path to regression sqlite database (used with --db).")
@click.option("--html/--no-html", "emit_html", default=False,
              help="Generate interactive HTML report for the output bundle.")
@click.option("--html-out", type=click.Path(), default=None,
              help="Output path for HTML report (used with --html).")
def trace_run(
    adapter: str,
    input_path: str,
    pipeline: str | None,
    output: str | None,
    seed: int,
    repeat: int,
    ingest_db: bool,
    db_path: str | None,
    emit_html: bool,
    html_out: str | None,
) -> None:
    """Run an instrumented compilation trace and produce a Trace Bundle."""
    from qocc.api import run_trace

    seeds = {
        "global_seed": seed,
        "rng_algorithm": DEFAULT_RNG_ALGORITHM,
        "stage_seeds": {},
    }

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

    if ingest_db:
        from qocc.core.regression_db import RegressionDatabase

        db = RegressionDatabase(db_path)
        ingest_result = db.ingest(result["bundle_zip"])
        console.print("[green]✓ Ingested into regression DB[/green]")
        console.print(f"  Rows: {ingest_result['rows_ingested']}")
        console.print(f"  DB:   {ingest_result['db_path']}")

    if emit_html:
        from qocc.trace.html_report import export_html_report

        default_html = str(Path(result["bundle_zip"]).with_suffix(".html"))
        html_path = export_html_report(result["bundle_zip"], html_out or default_html)
        console.print("[green]✓ HTML report created[/green]")
        console.print(f"  Report: {html_path}")


@trace.command("timeline")
@click.argument("bundle", type=click.Path(exists=True))
@click.option("--width", "-w", type=int, default=100, help="Terminal width.")
@click.option("--attrs/--no-attrs", default=False, help="Show span attributes.")
def trace_timeline(bundle: str, width: int, attrs: bool) -> None:
    """Display an ASCII timeline visualization of a trace bundle."""
    from qocc.trace.visualization import render_timeline_from_bundle

    timeline = render_timeline_from_bundle(bundle, width=width, show_attributes=attrs)
    console.print(timeline)


@trace.command("html")
@click.option("--bundle", "bundle_path", required=True, type=click.Path(exists=True),
              help="Bundle zip or directory to render.")
@click.option("--out", "output_path", required=True, type=click.Path(),
              help="Output HTML report path.")
@click.option("--compare", "compare_bundle", type=click.Path(exists=True), default=None,
              help="Optional second bundle for side-by-side diff view.")
def trace_html(bundle_path: str, output_path: str, compare_bundle: str | None) -> None:
    """Generate an interactive self-contained HTML report for a trace bundle."""
    from qocc.trace.html_report import export_html_report

    report_path = export_html_report(bundle_path, output_path, compare_bundle_path=compare_bundle)
    console.print("[green]✓ HTML report created[/green]")
    console.print(f"  Report: {report_path}")


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

    input_status = getattr(
        result,
        "input_hash_status",
        "matched" if result.input_hash_match else "mismatched",
    )
    compiled_status = getattr(
        result,
        "compiled_hash_status",
        "matched" if result.compiled_hash_match else "mismatched",
    )

    if result.bit_exact:
        console.print(f"  [green]✓ BIT-EXACT match[/green]")
        console.print(f"    Input hash status:    {input_status}")
        console.print(f"    Compiled hash status: {compiled_status}")
    else:
        console.print(f"  [yellow]⚠ Differences detected:[/yellow]")
        console.print(f"    Input hash status:    {input_status}")
        console.print(f"    Compiled hash status: {compiled_status}")
        console.print(f"    Metrics match:       {'✓' if result.metrics_match else '✗'}")
        if result.diff:
            for k, v in result.diff.items():
                console.print(f"    {k}: {v}")

    if result.replay_bundle:
        console.print(f"  Replay bundle: {result.replay_bundle}")


@trace.command("watch")
@click.option("--bundle", "bundle_path", required=True, type=click.Path(exists=True),
              help="Bundle zip or directory containing hardware/pending_jobs.json.")
@click.option("--poll-interval", type=float, default=5.0,
              help="Polling interval in seconds for provider job status checks.")
@click.option("--timeout", type=float, default=None,
              help="Optional timeout in seconds for watch operation.")
@click.option("--on-complete", "on_complete", type=str, default=None,
              help="Optional command to run after one or more jobs complete. Use {bundle} placeholder.")
def trace_watch(
    bundle_path: str,
    poll_interval: float,
    timeout: float | None,
    on_complete: str | None,
) -> None:
    """Watch pending hardware jobs in a bundle and update results in-place."""
    from qocc.trace.watch import watch_bundle_jobs

    console.print("[bold blue]QOCC Hardware Watch[/bold blue]")
    console.print(f"  Bundle: {bundle_path}")
    console.print(f"  Poll interval: {poll_interval}s")
    if timeout is not None:
        console.print(f"  Timeout: {timeout}s")

    try:
        summary = watch_bundle_jobs(
            bundle_path=bundle_path,
            poll_interval_s=poll_interval,
            timeout_s=timeout,
            on_complete=on_complete,
        )
    except Exception as exc:
        console.print(f"[red]Watch failed:[/red] {exc}")
        sys.exit(1)

    console.print("[green]✓ Watch complete[/green]")
    console.print(f"  Completed jobs: {summary.get('completed', 0)}")
    console.print(f"  Failed jobs:    {summary.get('failed', 0)}")
    console.print(f"  Pending jobs:   {summary.get('pending', 0)}")

    hook = summary.get("on_complete")
    if isinstance(hook, dict):
        rc = hook.get("returncode")
        if rc == 0:
            console.print("[green]✓ on-complete command succeeded[/green]")
        else:
            console.print(f"[yellow]⚠ on-complete command returned {rc}[/yellow]")
