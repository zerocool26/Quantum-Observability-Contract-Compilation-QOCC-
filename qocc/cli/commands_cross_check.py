import json
import logging
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table

from qocc.core.cross_check import cross_check

logger = logging.getLogger(__name__)

@click.command(name="cross-check")
@click.option("--adapters", help="Comma-separated list of adapters to compare (e.g. qiskit,cirq,tket)", required=True)
@click.option("--input", "input_source", help="Input circuit file", required=True)
@click.option("--contract", "contract_file", help="Contract JSON file to evaluate across adapters", required=True)
@click.option("--shots", default=1000, help="Simulation shots", type=int)
@click.option("--seed", default=42, help="Simulation seed", type=int)
def cli_cross_check(adapters: str, input_source: str, contract_file: str, shots: int, seed: int):
    """Evaluate portability of a circuit across multiple quantum SDKs."""
    console = Console()
    
    adapter_list = [a.strip() for a in adapters.split(",") if a.strip()]
    if len(adapter_list) < 2:
        console.print("[red]Error: Must specify at least two adapters for cross-check[/red]")
        raise click.Abort()
        
    contract_p = Path(contract_file)
    if not contract_p.exists():
        console.print(f"[red]Error: Contract file {contract_file} not found[/red]")
        raise click.Abort()
        
    try:
        specs = json.loads(contract_p.read_text())
        if isinstance(specs, dict):
            specs = [specs]
    except Exception as e:
        console.print(f"[red]Error parsing contract file: {e}[/red]")
        raise click.Abort()
        
    src_p = Path(input_source)
    if not src_p.exists():
        console.print(f"[red]Error: Input file {input_source} not found[/red]")
        raise click.Abort()
        
    try:
        source_content = src_p.read_text()
    except Exception:
        # Some loaders like qiskit.qasm can take paths directly, but our wrappers usually like strings
        source_content = str(src_p)

    console.print(f"Running cross-check across: {', '.join(adapter_list)}")
    result = cross_check(source_content, adapter_list, specs, shots, seed)
    
    table = Table(title="Cross-Adapter Portability Matrix")
    table.add_column("Baseline \\ Candidate", style="cyan")
    
    for a in adapter_list:
        table.add_column(a, justify="center")
        
    matrix = result["matrix"]
    for adapter_a in adapter_list:
        row = [adapter_a]
        for adapter_b in adapter_list:
            if adapter_a == adapter_b:
                row.append("[dim]-[/dim]")
                continue
            
            cell_results = matrix.get(adapter_a, {}).get(adapter_b, [])
            if not cell_results:
                row.append("[yellow]N/A[/yellow]")
                continue
                
            all_passed = True
            for r in cell_results:
                if isinstance(r, dict) and not r.get("passed", False):
                    all_passed = False
                    
            if all_passed:
                row.append("[green]PASS[/green]")
            else:
                row.append("[red]FAIL[/red]")
                
        table.add_row(*row)
        
    console.print(table)
    
    # Save raw json locally
    output_file = Path("cross_check_result.json")
    output_file.write_text(json.dumps(result, indent=2))
    console.print(f"Raw cross-check results written to [bold]{output_file}[/bold]")
