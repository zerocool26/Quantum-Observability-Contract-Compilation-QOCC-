"""CLI commands for bundle comparison."""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console

console = Console()


@click.command("compare")
@click.argument("bundle_a", type=click.Path(exists=True))
@click.argument("bundle_b", type=click.Path(exists=True))
@click.option("--report", "-r", "report_dir", type=click.Path(), default=None,
              help="Output directory for reports.")
@click.option("--format", "-f", "output_format", type=click.Choice(["text", "json"]),
              default="text", help="Output format: text (default) or json.")
def compare(bundle_a: str, bundle_b: str, report_dir: str | None, output_format: str) -> None:
    """Compare two Trace Bundles and highlight differences."""
    from qocc.api import compare_bundles

    console.print("[bold blue]QOCC Bundle Comparison[/bold blue]")
    console.print(f"  Bundle A: {bundle_a}")
    console.print(f"  Bundle B: {bundle_b}")

    try:
        result = compare_bundles(bundle_a, bundle_b, report_dir=report_dir)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    # JSON output mode — emit *only* JSON, no Rich decoration
    if output_format == "json":
        json_str = json.dumps(result, indent=2, default=str)
        click.echo(json_str)
        if report_dir:
            from pathlib import Path
            Path(report_dir).mkdir(parents=True, exist_ok=True)
            (Path(report_dir) / "comparison.json").write_text(
                json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
            )
        return

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
