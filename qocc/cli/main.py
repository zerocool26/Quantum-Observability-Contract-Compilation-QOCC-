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
from qocc.cli.commands_compare import compare
from qocc.cli.commands_contract import contract
from qocc.cli.commands_search import compile_group


@click.group()
@click.version_option(package_name="qocc")
def cli() -> None:
    """QOCC — Quantum Observability + Contract-Based Compilation."""


cli.add_command(trace)
cli.add_command(compare)
cli.add_command(contract)
cli.add_command(compile_group)


if __name__ == "__main__":
    cli()
