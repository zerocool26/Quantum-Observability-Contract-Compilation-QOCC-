"""CLI input JSON validation against QOCC schemas."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import click

logger = logging.getLogger("qocc.cli")

_SCHEMA_DIR = Path(__file__).resolve().parent.parent.parent / "schemas"


def _load_schema(name: str) -> dict[str, Any] | None:
    """Load a JSON schema by name (without extension)."""
    schema_path = _SCHEMA_DIR / f"{name}.schema.json"
    if not schema_path.exists():
        logger.debug("Schema file not found: %s", schema_path)
        return None
    with open(schema_path, encoding="utf-8") as f:
        return json.load(f)


def validate_json_file(
    filepath: str,
    schema_name: str,
    *,
    strict: bool = True,
) -> dict[str, Any] | list[Any]:
    """Load a JSON file and validate it against a QOCC schema.

    Parameters:
        filepath: Path to the JSON file.
        schema_name: Schema name (e.g. ``"contracts"``, ``"search_space"``).
        strict: If ``True`` (default), abort on validation errors.
                If ``False``, warn but return data anyway.

    Returns:
        The parsed JSON data.

    Raises:
        click.ClickException: On parse or validation errors (when *strict*).
    """
    # Parse JSON
    try:
        with open(filepath, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise click.ClickException(f"Invalid JSON in {filepath}: {exc}") from exc

    # Validate against schema
    schema = _load_schema(schema_name)
    if schema is None:
        logger.debug("No schema '%s' found — skipping validation", schema_name)
        return data

    try:
        import jsonschema
    except ImportError:
        logger.debug("jsonschema not installed — skipping validation")
        return data

    validator = jsonschema.Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(data), key=lambda e: list(e.absolute_path))

    if errors:
        msgs = []
        for err in errors[:5]:
            path = ".".join(str(p) for p in err.absolute_path) or "(root)"
            msgs.append(f"  [{path}] {err.message}")
        summary = "\n".join(msgs)
        full = f"Validation errors for {filepath} (schema: {schema_name}):\n{summary}"
        if strict:
            raise click.ClickException(full)
        else:
            logger.warning(full)

    return data
