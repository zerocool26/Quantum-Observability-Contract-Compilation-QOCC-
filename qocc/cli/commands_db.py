"""CLI commands for regression database operations."""

from __future__ import annotations

import json

import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group("db")
def db_group() -> None:
    """Regression database commands."""


@db_group.command("ingest")
@click.argument("bundle", type=click.Path(exists=True))
@click.option("--db-path", type=click.Path(), default=None, help="Path to regression sqlite database.")
def db_ingest(bundle: str, db_path: str | None) -> None:
    """Ingest a bundle into the regression database."""
    from qocc.core.regression_db import RegressionDatabase

    db = RegressionDatabase(db_path)
    result = db.ingest(bundle)
    console.print("[green]✓ Ingested bundle[/green]")
    console.print(f"  Run ID: {result['run_id']}")
    console.print(f"  Adapter: {result.get('adapter')}")
    console.print(f"  Rows: {result['rows_ingested']}")
    console.print(f"  DB: {result['db_path']}")


@db_group.command("query")
@click.option("--circuit-hash", type=str, default=None, help="Filter by circuit hash.")
@click.option("--adapter", type=str, default=None, help="Filter by adapter name.")
@click.option("--since", type=str, default=None, help="Filter rows with timestamp >= ISO date/time.")
@click.option("--db-path", type=click.Path(), default=None, help="Path to regression sqlite database.")
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table", help="Output format.")
def db_query(circuit_hash: str | None, adapter: str | None, since: str | None, db_path: str | None, fmt: str) -> None:
    """Query historical rows from the regression database."""
    from qocc.core.regression_db import RegressionDatabase

    db = RegressionDatabase(db_path)
    rows = db.query(circuit_hash=circuit_hash, adapter=adapter, since=since)

    if fmt == "json":
        click.echo(json.dumps(rows, indent=2, default=str))
        return

    table = Table(title=f"Regression Query ({len(rows)} rows)")
    table.add_column("Run ID")
    table.add_column("Adapter")
    table.add_column("Circuit Hash")
    table.add_column("Candidate")
    table.add_column("Score", justify="right")
    table.add_column("Timestamp")

    for row in rows[:200]:
        score = row.get("surrogate_score")
        table.add_row(
            str(row.get("run_id", ""))[:12],
            str(row.get("adapter", "")),
            str((row.get("circuit_hash") or ""))[:16],
            str((row.get("candidate_id") or "compiled"))[:16],
            "" if score is None else f"{float(score):.4f}",
            str(row.get("timestamp", "")),
        )

    console.print(table)


@db_group.command("tag")
@click.argument("bundle", type=click.Path(exists=True))
@click.option("--tag", "tag_name", required=True, type=str, help="Tag name to assign (e.g., baseline).")
@click.option("--db-path", type=click.Path(), default=None, help="Path to regression sqlite database.")
def db_tag(bundle: str, tag_name: str, db_path: str | None) -> None:
    """Assign a tag to a bundle run in the regression database."""
    from qocc.core.regression_db import RegressionDatabase

    db = RegressionDatabase(db_path)
    result = db.tag(bundle, tag_name)
    console.print("[green]✓ Tagged run[/green]")
    console.print(f"  Run ID: {result['run_id']}")
    console.print(f"  Tag: {result['tag']}")
    console.print(f"  DB: {result['db_path']}")
