"""CLI commands for bundle operations (sign/verify)."""

from __future__ import annotations

import json
import sys

import click
from rich.console import Console

console = Console()


@click.group("bundle")
def bundle_group() -> None:
    """Bundle-level operations."""


@bundle_group.command("sign")
@click.option("--key", "private_key_path", required=True, type=click.Path(exists=True),
              help="Path to Ed25519 private key PEM file.")
@click.option("--signer", "signer_identity", type=str, default=None,
              help="Optional signer identity to store in signature metadata.")
@click.argument("bundle", type=click.Path(exists=True))
def bundle_sign(private_key_path: str, signer_identity: str | None, bundle: str) -> None:
    """Sign a bundle and embed signature metadata."""
    from qocc.core.signing import sign_bundle

    try:
        payload = sign_bundle(bundle, private_key_path, signer_identity=signer_identity)
    except Exception as exc:
        console.print(f"[red]Bundle sign failed:[/red] {exc}")
        sys.exit(1)

    console.print("[green]✓ Bundle signed[/green]")
    console.print(f"  Signer: {payload.get('signer')}")
    console.print(f"  Manifest hash: {str(payload.get('manifest_hash', ''))[:16]}…")
    console.print(f"  Key fingerprint: {str(payload.get('public_key_fingerprint', ''))[:16]}…")


@bundle_group.command("verify")
@click.option("--key", "public_key_path", required=True, type=click.Path(exists=True),
              help="Path to Ed25519 public key PEM file.")
@click.option("--format", "fmt", type=click.Choice(["text", "json"]), default="text",
              help="Output format.")
@click.argument("bundle", type=click.Path(exists=True))
def bundle_verify(public_key_path: str, fmt: str, bundle: str) -> None:
    """Verify a bundle signature using a public key."""
    from qocc.core.signing import verify_bundle

    try:
        result = verify_bundle(bundle, public_key_path)
    except Exception as exc:
        console.print(f"[red]Bundle verify failed:[/red] {exc}")
        sys.exit(1)

    if fmt == "json":
        click.echo(json.dumps(result.to_dict(), indent=2))
    else:
        if result.valid:
            console.print("[green]✓ Signature valid[/green]")
        else:
            console.print("[red]✗ Signature invalid[/red]")
        console.print(f"  Signer: {result.signer}")
        console.print(f"  Reason: {result.reason}")
        if result.public_key_fingerprint:
            console.print(f"  Key fingerprint: {result.public_key_fingerprint[:16]}…")

    if not result.valid:
        sys.exit(1)
