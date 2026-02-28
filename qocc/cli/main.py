"""QOCC CLI — main entry point.

Usage::

    qocc trace run --adapter qiskit --input foo.qasm --out bundle.zip
    qocc trace compare bundleA.zip bundleB.zip --report reports/
    qocc contract check --bundle bundle.zip --contracts contracts.json
    qocc compile search --bundle bundle.zip --search search_spec.json (v3)
"""

from __future__ import annotations

import click

from qocc.cli.commands_trace import trace
from qocc.cli.commands_compare import compare_legacy
from qocc.cli.commands_contract import contract
from qocc.cli.commands_db import db_group
from qocc.cli.commands_search import compile_group
from qocc.cli.commands_init import init_project
from qocc.cli.commands_bundle import bundle_group
from qocc.cli.commands_validate import validate
from qocc.cli.commands_cross_check import cli_cross_check

@click.group()
@click.version_option(package_name="qocc")
def cli() -> None:
    """QOCC — Quantum Observability + Contract-Based Compilation."""


cli.add_command(trace)
cli.add_command(compare_legacy, "compare")  # deprecated alias
cli.add_command(contract)
cli.add_command(compile_group)
cli.add_command(validate)
cli.add_command(db_group)
cli.add_command(init_project)
cli.add_command(bundle_group)
cli.add_command(cli_cross_check, "cross-check")

if __name__ == "__main__":
    cli()
