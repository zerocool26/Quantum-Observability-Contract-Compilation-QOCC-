"""``qocc validate`` — validate bundle JSON files against QOCC schemas."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.table import Table

from qocc.cli.validation import _SCHEMA_DIR, _load_schema

logger = logging.getLogger("qocc.cli")

# Map bundle filenames → schema names (without .schema.json suffix)
_FILE_SCHEMA_MAP: dict[str, str] = {
    "manifest.json": "manifest",
    "env.json": "env",
    "seeds.json": "seeds",
    "metrics.json": "metrics",
    "contracts.json": "contracts",
    "contract_results.json": "contract_results",
    "nondeterminism.json": "nondeterminism",
    "cache_index.json": "cache_index",
    "search_space.json": "search_space",
    "search_result.json": "search_result",
    "search_rankings.json": "search_rankings",
    "dem.json": "dem",
    "logical_error_rates.json": "logical_error_rates",
    "decoder_stats.json": "decoder_stats",
    "signature.json": "signature",
}


def _validate_one(filepath: Path, schema_name: str) -> list[str]:
    """Return a list of error messages (empty = valid)."""
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        return [f"JSON parse error: {exc}"]

    schema = _load_schema(schema_name)
    if schema is None:
        return [f"Schema '{schema_name}' not found"]

    try:
        import jsonschema
    except ImportError:
        return ["jsonschema package not installed — cannot validate"]

    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))

    msgs: list[str] = []
    for err in errors:
        path = ".".join(str(p) for p in err.absolute_path) or "(root)"
        msgs.append(f"[{path}] {err.message}")
    return msgs


@click.command("validate")
@click.argument("bundle", type=click.Path(exists=True))
@click.option("--format", "fmt", type=click.Choice(["table", "json"]), default="table",
              help="Output format.")
@click.option("--strict", is_flag=True, default=False,
              help="Exit with non-zero status if any file is invalid.")
def validate(bundle: str, fmt: str, strict: bool) -> None:
    """Validate bundle JSON files against QOCC schemas.

    BUNDLE may be a directory or a .zip file.
    """
    bundle_path = Path(bundle)
    err_console = Console(stderr=True)

    # Resolve bundle root
    if bundle_path.is_file() and bundle_path.suffix == ".zip":
        from qocc.core.artifacts import ArtifactStore

        loaded = ArtifactStore.load_bundle(str(bundle_path))
        _root = loaded.get("_root")
        if not _root:
            raise click.ClickException("Failed to extract bundle zip")
        bundle_root = Path(_root)
    elif bundle_path.is_dir():
        bundle_root = bundle_path
    else:
        raise click.ClickException(f"Not a bundle directory or zip: {bundle}")

    results: list[dict[str, Any]] = []
    total_errors = 0

    for filename, schema_name in _FILE_SCHEMA_MAP.items():
        fpath = bundle_root / filename
        if not fpath.exists():
            results.append({
                "file": filename,
                "schema": schema_name,
                "status": "skipped",
                "errors": [],
            })
            continue

        errs = _validate_one(fpath, schema_name)
        total_errors += len(errs)
        results.append({
            "file": filename,
            "schema": schema_name,
            "status": "invalid" if errs else "valid",
            "errors": errs,
        })

    # Output
    if fmt == "json":
        click.echo(json.dumps({"results": results, "total_errors": total_errors}, indent=2))
    else:
        table = Table(title=f"Bundle Validation: {bundle_path.name}")
        table.add_column("File", style="cyan")
        table.add_column("Schema", style="magenta")
        table.add_column("Status")
        table.add_column("Errors", style="red")

        for r in results:
            status_style = {
                "valid": "[green]valid[/green]",
                "invalid": "[red]INVALID[/red]",
                "skipped": "[dim]skipped[/dim]",
            }.get(r["status"], r["status"])

            err_text = "\n".join(r["errors"][:3]) if r["errors"] else ""
            if len(r["errors"]) > 3:
                err_text += f"\n... +{len(r['errors']) - 3} more"
            table.add_row(r["file"], r["schema"], status_style, err_text)

        err_console.print(table)
        if total_errors:
            err_console.print(f"\n[red]{total_errors} validation error(s) found.[/red]")
        else:
            err_console.print("\n[green]All present files are valid.[/green]")

    if strict and total_errors:
        raise SystemExit(1)
