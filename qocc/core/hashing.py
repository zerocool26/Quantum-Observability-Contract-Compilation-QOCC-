"""Deterministic hashing utilities.

All hashes are SHA-256 over a canonical byte representation.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def hash_bytes(data: bytes) -> str:
    """SHA-256 hex digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def hash_string(data: str) -> str:
    """SHA-256 hex digest of a UTF-8 string."""
    return hash_bytes(data.encode("utf-8"))


def hash_dict(d: dict[str, Any]) -> str:
    """SHA-256 hex digest of a dict serialized as sorted JSON."""
    payload = json.dumps(d, sort_keys=True, default=str).encode("utf-8")
    return hash_bytes(payload)


def hash_circuit_handle(handle: Any) -> str:
    """Delegate to ``CircuitHandle.stable_hash()``."""
    return handle.stable_hash()
