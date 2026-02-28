"""CLI commands for bundle comparison."""

from __future__ import annotations

import json
import sys
import warnings

import click
from rich.console import Console

console = Console()
_err_console = Console(stderr=True)


def _run_compare(
    bundle_a: str,
    bundle_b: str,
    report_dir: str | None,
    output_format: str,
) -> None:
    """Core comparison logic shared by ``trace compare`` and the legacy alias."""
    from qocc.api import compare_bundles

    # In JSON/diff mode every human-readable message goes to stderr so stdout is
    # pure, machine-parseable JSON.
    out = _err_console if output_format in ("json", "diff") else console

    out.print("[bold blue]QOCC Bundle Comparison[/bold blue]")
    out.print(f"  Bundle A: {bundle_a}")
    out.print(f"  Bundle B: {bundle_b}")

    try:
        result = compare_bundles(bundle_a, bundle_b, report_dir=report_dir)
    except Exception as exc:
        out.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    # ── JSON output ────────────────────────────────────────────
    if output_format == "json":
        json_str = json.dumps(result, indent=2, default=str)
        click.echo(json_str)  # stdout only
        return

    # ── DIFF output ────────────────────────────────────────────
    if output_format == "diff":
        b_diff = result.get("bundle_diff", {})
        json_str = json.dumps(b_diff, indent=2, default=str)
        click.echo(json_str)  # stdout only
        return

    # ── Text output ────────────────────────────────────────────
    diffs = result.get("diffs", {})
    metrics_diff = diffs.get("metrics", {})
    env_diff = diffs.get("environment", {})

    if not metrics_diff and not env_diff:
        console.print("[green]✓ Bundles are identical[/green]")
        return

    if metrics_diff:
        console.print("\n[yellow]Metric differences:[/yellow]")
        for stage, sdiff in metrics_diff.items():
            console.print(f"  [{stage}]")
            for k, v in sdiff.items():
                pct = f" ({v.get('pct_change', 0):.1f}%)" if "pct_change" in v else ""
                console.print(f"    {k}: {v['a']} → {v['b']}{pct}")

    if env_diff:
        console.print("\n[yellow]Environment differences:[/yellow]")
        for k, v in env_diff.items():
            if k != "packages":
                console.print(f"  {k}: {v['a']} → {v['b']}")

    if report_dir:
        console.print(f"\n[green]Reports written to {report_dir}/[/green]")


@click.command("compare")
@click.argument("bundle_a", type=click.Path(exists=True))
@click.argument("bundle_b", type=click.Path(exists=True))
@click.option("--report", "-r", "report_dir", type=click.Path(), default=None,
              help="Output directory for reports.")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json", "diff"]),
              default="text", help="Output format: text (default), json, or diff.")
def compare(bundle_a: str, bundle_b: str, report_dir: str | None,
            output_format: str) -> None:
    """Compare two Trace Bundles and highlight differences."""
    _run_compare(bundle_a, bundle_b, report_dir, output_format)


# ── Backward-compatible top-level alias (deprecated) ──────────
@click.command("compare", hidden=False,
               deprecated=True)
@click.argument("bundle_a", type=click.Path(exists=True))
@click.argument("bundle_b", type=click.Path(exists=True))
@click.option("--report", "-r", "report_dir", type=click.Path(), default=None,
              help="Output directory for reports.")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json", "diff"]),
              default="text", help="Output format: text (default), json, or diff.")
def compare_legacy(bundle_a: str, bundle_b: str, report_dir: str | None,
                   output_format: str) -> None:
    """Compare two Trace Bundles (DEPRECATED — use ``qocc trace compare``)."""
    _err_console.print(
        "[yellow]Warning:[/yellow] 'qocc compare' is deprecated. "
        "Use 'qocc trace compare' instead.",
    )
    _run_compare(bundle_a, bundle_b, report_dir, output_format)
