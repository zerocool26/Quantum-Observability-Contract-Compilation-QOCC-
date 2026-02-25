"""Content-addressed compilation cache.

Caches compilation results keyed by:
  hash(normalized_circuit) + hash(pipeline_spec) + hash(backend_version)

Avoids redundant compilation when the same circuit + pipeline + toolchain
has already been processed.
"""

from __future__ import annotations

import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from qocc.core.hashing import hash_dict, hash_string


class CompilationCache:
    """Content-addressed cache for compilation artifacts.

    By default, the cache lives in ``~/.qocc/cache/``. Each entry is a
    directory named by the cache key (a composite hash).

    Cache layout::

        ~/.qocc/cache/
            <key>/
                meta.json    # metadata (input hash, pipeline hash, timestamp)
                result.json  # serialized CompileResult
                circuit.qasm # compiled QASM (if available)
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = Path.home() / ".qocc" / "cache"
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Key computation
    # ------------------------------------------------------------------

    @staticmethod
    def cache_key(
        circuit_hash: str,
        pipeline_dict: dict[str, Any],
        backend_version: str = "",
        extra: dict[str, Any] | None = None,
    ) -> str:
        """Compute a deterministic cache key.

        Parameters:
            circuit_hash: Stable hash of the normalized input circuit.
            pipeline_dict: Pipeline spec as a dict.
            backend_version: Adapter/backend version string.
            extra: Any additional parameters to factor into the key.

        Returns:
            A SHA-256 hex string.
        """
        key_data: dict[str, Any] = {
            "circuit_hash": circuit_hash,
            "pipeline": pipeline_dict,
            "backend_version": backend_version,
        }
        if extra:
            key_data["extra"] = extra
        return hash_dict(key_data)

    # ------------------------------------------------------------------
    # Lookup / store
    # ------------------------------------------------------------------

    def get(self, key: str) -> dict[str, Any] | None:
        """Look up a cached compilation result.

        Returns:
            The cached result dict, or ``None`` on miss.
        """
        entry_dir = self.root / key
        result_path = entry_dir / "result.json"
        if not result_path.exists():
            return None
        try:
            data = json.loads(result_path.read_text(encoding="utf-8"))
            # Update last-access timestamp
            meta_path = entry_dir / "meta.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                meta["last_accessed"] = time.time()
                meta["access_count"] = meta.get("access_count", 0) + 1
                meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
            return data
        except (json.JSONDecodeError, OSError):
            return None

    def put(
        self,
        key: str,
        result: dict[str, Any],
        circuit_qasm: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Store a compilation result in the cache.

        Parameters:
            key: Cache key (from ``cache_key()``).
            result: Serialized compilation result dict.
            circuit_qasm: Optional QASM output to cache.
            metadata: Additional metadata to store.

        Returns:
            Path to the cache entry directory.
        """
        entry_dir = self.root / key
        entry_dir.mkdir(parents=True, exist_ok=True)

        # Write result
        (entry_dir / "result.json").write_text(
            json.dumps(result, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

        # Write QASM if available
        if circuit_qasm:
            (entry_dir / "circuit.qasm").write_text(circuit_qasm, encoding="utf-8")

        # Write metadata
        meta: dict[str, Any] = {
            "created": time.time(),
            "last_accessed": time.time(),
            "access_count": 0,
        }
        if metadata:
            meta.update(metadata)
        (entry_dir / "meta.json").write_text(
            json.dumps(meta, indent=2, default=str) + "\n",
            encoding="utf-8",
        )

        return entry_dir

    def has(self, key: str) -> bool:
        """Check if a key is in the cache."""
        return (self.root / key / "result.json").exists()

    def evict(self, key: str) -> bool:
        """Remove a cache entry.

        Returns:
            True if something was removed.
        """
        entry_dir = self.root / key
        if entry_dir.exists():
            shutil.rmtree(entry_dir)
            return True
        return False

    def clear(self) -> int:
        """Remove all cache entries. Returns the number removed."""
        count = 0
        for child in self.root.iterdir():
            if child.is_dir() and (child / "result.json").exists():
                shutil.rmtree(child)
                count += 1
        return count

    def stats(self) -> dict[str, Any]:
        """Return cache statistics."""
        entries = 0
        total_size = 0
        oldest = float("inf")
        newest = 0.0

        for child in self.root.iterdir():
            if child.is_dir() and (child / "result.json").exists():
                entries += 1
                for f in child.rglob("*"):
                    if f.is_file():
                        total_size += f.stat().st_size
                meta_path = child / "meta.json"
                if meta_path.exists():
                    try:
                        m = json.loads(meta_path.read_text(encoding="utf-8"))
                        created = m.get("created", 0)
                        oldest = min(oldest, created)
                        newest = max(newest, created)
                    except Exception:
                        pass

        return {
            "entries": entries,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "oldest_timestamp": oldest if entries > 0 else None,
            "newest_timestamp": newest if entries > 0 else None,
        }

    def evict_lru(self, max_entries: int = 1000) -> int:
        """Evict least-recently-used entries to keep at most *max_entries*.

        Returns:
            Number of entries evicted.
        """
        entries: list[tuple[float, Path]] = []
        for child in self.root.iterdir():
            if child.is_dir() and (child / "result.json").exists():
                last_access = 0.0
                meta_path = child / "meta.json"
                if meta_path.exists():
                    try:
                        m = json.loads(meta_path.read_text(encoding="utf-8"))
                        last_access = m.get("last_accessed", 0)
                    except Exception:
                        pass
                entries.append((last_access, child))

        if len(entries) <= max_entries:
            return 0

        # Sort by last access (oldest first)
        entries.sort(key=lambda x: x[0])
        to_evict = entries[: len(entries) - max_entries]
        for _, path in to_evict:
            shutil.rmtree(path)
        return len(to_evict)
