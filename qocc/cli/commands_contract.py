"""CLI commands for contract checking."""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group("contract")
def contract() -> None:
    """Contract-related commands."""


@contract.command("check")
@click.option("--bundle", "-b", required=True, type=click.Path(exists=True),
              help="Trace Bundle to check.")
@click.option("--contracts", "-c", required=True, type=click.Path(exists=True),
              help="Contract specifications JSON file.")
def contract_check(bundle: str, contracts: str) -> None:
    """Evaluate contracts against a Trace Bundle.

    Exit code 0 = all pass, 1 = at least one failure.
    """
    from qocc.api import check_contract

    console.print("[bold blue]QOCC Contract Check[/bold blue]")
    console.print(f"  Bundle:    {bundle}")
    console.print(f"  Contracts: {contracts}")

    # Validate contracts JSON against schema
    from qocc.cli.validation import validate_json_file
    validate_json_file(contracts, "contracts")

    try:
        results = check_contract(bundle, contracts)
    except Exception as exc:
        console.print(f"[red]Error:[/red] {exc}")
        sys.exit(1)

    table = Table(title="Contract Results")
    table.add_column("Contract")
    table.add_column("Type")
    table.add_column("Passed", justify="center")

    any_failed = False
    for r in results:
        passed = r.get("passed", False)
        if not passed:
            any_failed = True
        status = "[green]✓[/green]" if passed else "[red]✗[/red]"
        table.add_row(
            r.get("name", "?"),
            r.get("details", {}).get("type", "?"),
            status,
        )

    console.print(table)

    if any_failed:
        console.print("[red]Some contracts failed.[/red]")
        sys.exit(1)
    else:
        console.print("[green]All contracts passed.[/green]")
