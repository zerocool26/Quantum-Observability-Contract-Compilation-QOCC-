"""CLI command for project bootstrap and setup wizard."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console

from qocc import DEFAULT_RNG_ALGORITHM, DEFAULT_SEED

console = Console()

_BACKEND_IMPORTS: dict[str, str] = {
    "qiskit": "qiskit",
    "cirq": "cirq",
    "tket": "pytket",
    "stim": "stim",
}


def _detect_backends() -> list[str]:
    found: list[str] = []
    for adapter, module_name in _BACKEND_IMPORTS.items():
        if importlib.util.find_spec(module_name) is not None:
            found.append(adapter)
    return found


def _default_pipeline(adapter: str) -> dict[str, Any]:
    routing = "sabre" if adapter in {"qiskit", "tket"} else "stochastic"
    return {
        "adapter": adapter,
        "optimization_level": 2,
        "passes": [],
        "parameters": {
            "seed": DEFAULT_SEED,
            "routing_method": routing,
        },
    }


def _default_contracts(adapter: str) -> str:
    lines = [
        "contract dist_stability:",
        "    type: distribution",
        "    tolerance: tvd <= 0.08",
        "    confidence: 0.95",
        "    shots: 2048 .. 16384",
        "",
        "contract depth_budget:",
        "    type: cost",
        "    assert: depth <= input_depth + 5",
        "    assert: two_qubit_gates <= input_gates_2q + 12",
    ]
    if adapter in {"qiskit", "cirq", "tket", "stim"}:
        lines.extend(
            [
                "",
                "contract zne_guard:",
                "    type: zne",
                "    tolerance: abs_error <= 0.10",
            ]
        )
    return "\n".join(lines) + "\n"


def _render_ci_workflow(default_adapter: str, default_circuit: str, default_contracts: str) -> str:
    return f"""name: QOCC CI\n\non:\n  push:\n    branches: [\"main\"]\n  workflow_dispatch:\n    inputs:\n      adapter:\n        description: \"QOCC adapter\"\n        required: false\n        default: \"{default_adapter}\"\n      circuit_path:\n        description: \"Input circuit path\"\n        required: false\n        default: \"{default_circuit}\"\n      contract_file:\n        description: \"Contracts file path\"\n        required: false\n        default: \"{default_contracts}\"\n\npermissions:\n  contents: read\n\njobs:\n  qocc-ci:\n    runs-on: ubuntu-latest\n    steps:\n      - uses: actions/checkout@v4\n      - uses: actions/setup-python@v5\n        with:\n          python-version: \"3.12\"\n      - name: Install\n        run: |\n          python -m pip install --upgrade pip\n          pip install -e \".[all]\"\n      - name: Baseline trace + DB ingest\n        run: |\n          ADAPTER=\"${{{{ github.event.inputs.adapter || '{default_adapter}' }}}}\"\n          CIRCUIT=\"${{{{ github.event.inputs.circuit_path || '{default_circuit}' }}}}\"\n          qocc trace run --adapter \"$ADAPTER\" --input \"$CIRCUIT\" --out .qocc_baseline.zip --db --db-path .qocc_regression.db\n      - name: Contract check\n        run: |\n          CONTRACTS=\"${{{{ github.event.inputs.contract_file || '{default_contracts}' }}}}\"\n          qocc contract check --bundle .qocc_baseline.zip --contracts \"$CONTRACTS\"\n"""


def _upsert_tool_qocc(pyproject_path: Path, config: dict[str, Any]) -> None:
    section_lines = ["[tool.qocc]"]
    for key, value in config.items():
        if isinstance(value, bool):
            encoded = "true" if value else "false"
        elif isinstance(value, (int, float)):
            encoded = str(value)
        else:
            encoded = json.dumps(str(value))
        section_lines.append(f"{key} = {encoded}")
    section = "\n".join(section_lines) + "\n"

    if pyproject_path.exists():
        text = pyproject_path.read_text(encoding="utf-8")
        lines = text.splitlines()

        start = None
        end = None
        for i, line in enumerate(lines):
            if line.strip() == "[tool.qocc]":
                start = i
                break

        if start is not None:
            end = len(lines)
            for j in range(start + 1, len(lines)):
                if lines[j].startswith("["):
                    end = j
                    break
            new_lines = lines[:start] + section.splitlines() + lines[end:]
            pyproject_path.write_text("\n".join(new_lines).rstrip() + "\n", encoding="utf-8")
            return

        content = text.rstrip() + "\n\n" + section
        pyproject_path.write_text(content, encoding="utf-8")
        return

    pyproject_path.write_text(
        "[build-system]\nrequires = [\"hatchling\"]\nbuild-backend = \"hatchling.build\"\n\n"
        "[project]\nname = \"qocc-project\"\nversion = \"0.0.1\"\nrequires-python = \">=3.11\"\n\n"
        + section,
        encoding="utf-8",
    )


def _write_text(path: Path, content: str, force: bool) -> bool:
    if path.exists() and not force:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return True


@click.command("init")
@click.option("--project-root", type=click.Path(file_okay=False, dir_okay=True), default=".", show_default=True,
              help="Target project root.")
@click.option("--adapter", type=click.Choice(["qiskit", "cirq", "tket", "stim", "ibm"]), default=None,
              help="Preferred default adapter.")
@click.option("--yes", "assume_yes", is_flag=True, default=False,
              help="Run non-interactively with sensible defaults.")
@click.option("--run-demo/--no-run-demo", default=None,
              help="Run a demo trace at the end of setup.")
@click.option("--force", is_flag=True, default=False,
              help="Overwrite generated files if they already exist.")
def init_project(
    project_root: str,
    adapter: str | None,
    assume_yes: bool,
    run_demo: bool | None,
    force: bool,
) -> None:
    """Interactive setup wizard for QOCC projects."""
    root = Path(project_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    detected = _detect_backends()
    if detected:
        console.print(f"[green]Detected backends:[/green] {', '.join(detected)}")
    else:
        console.print("[yellow]No optional backends detected; defaulting to qiskit config.[/yellow]")

    default_adapter = adapter or (detected[0] if detected else "qiskit")
    if not assume_yes and adapter is None:
        default_adapter = click.prompt(
            "Select default adapter",
            type=click.Choice(["qiskit", "cirq", "tket", "stim", "ibm"]),
            default=default_adapter,
            show_choices=True,
        )

    default_circuit = "examples/ghz.qasm"
    default_contracts_rel = "contracts/default_contracts.qocc"
    default_pipeline_rel = f"pipeline_examples/{default_adapter}_default.json"
    default_workflow_rel = ".github/workflows/qocc_ci.yml"

    contracts_text = _default_contracts(default_adapter)
    pipeline_payload = _default_pipeline(default_adapter)
    workflow_text = _render_ci_workflow(default_adapter, default_circuit, default_contracts_rel)

    created: list[str] = []
    skipped: list[str] = []

    contracts_path = root / default_contracts_rel
    if _write_text(contracts_path, contracts_text, force=force):
        created.append(default_contracts_rel)
    else:
        skipped.append(default_contracts_rel)

    pipeline_path = root / default_pipeline_rel
    if _write_text(pipeline_path, json.dumps(pipeline_payload, indent=2) + "\n", force=force):
        created.append(default_pipeline_rel)
    else:
        skipped.append(default_pipeline_rel)

    workflow_path = root / default_workflow_rel
    if _write_text(workflow_path, workflow_text, force=force):
        created.append(default_workflow_rel)
    else:
        skipped.append(default_workflow_rel)

    pyproject_path = root / "pyproject.toml"
    _upsert_tool_qocc(
        pyproject_path,
        {
            "adapter": default_adapter,
            "default_circuit": default_circuit,
            "default_contracts": default_contracts_rel,
            "default_pipeline": default_pipeline_rel,
            "ci_workflow": default_workflow_rel,
            "seed": DEFAULT_SEED,
            "rng_algorithm": DEFAULT_RNG_ALGORITHM,
        },
    )
    created.append("pyproject.toml [tool.qocc]")

    if run_demo is None:
        run_demo = False if assume_yes else click.confirm("Run demo trace now?", default=False)

    if run_demo:
        try:
            from qocc.api import run_trace

            demo_input = root / default_circuit
            if not demo_input.exists():
                demo_input = Path(__file__).resolve().parents[2] / default_circuit

            run_trace(
                adapter_name=default_adapter,
                input_source=str(demo_input),
                pipeline=str(pipeline_path),
                output=str(root / ".qocc_baseline.zip"),
                seeds={
                    "global_seed": DEFAULT_SEED,
                    "rng_algorithm": DEFAULT_RNG_ALGORITHM,
                    "stage_seeds": {},
                },
                repeat=1,
            )
            created.append(".qocc_baseline.zip")
        except Exception as exc:
            console.print(f"[yellow]Demo trace skipped due to error:[/yellow] {exc}")

    console.print("\n[bold blue]QOCC init complete[/bold blue]")
    if created:
        console.print("[green]Created/updated:[/green]")
        for item in created:
            console.print(f"  - {item}")
    if skipped:
        console.print("[yellow]Skipped (already exists):[/yellow]")
        for item in skipped:
            console.print(f"  - {item}")
