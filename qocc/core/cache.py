"""Content-addressed compilation cache.

Caches compilation results keyed by:
  hash(normalized_circuit) + hash(pipeline_spec) + hash(backend_version)

Avoids redundant compilation when the same circuit + pipeline + toolchain
has already been processed.

Thread- and process-safe: uses per-key file locking and atomic writes
(write to temp → rename) so concurrent ``put()`` calls never produce
partial JSON on disk.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import threading
import time
from pathlib import Path
from typing import Any

from qocc.core.hashing import hash_dict, hash_string

logger = logging.getLogger("qocc.cache")


def _atomic_write_text(path: Path, content: str) -> None:
    """Write *content* to *path* atomically via temp-file + rename.

    On POSIX ``os.replace`` is atomic.  On Windows it is nearly so and
    avoids ever leaving a half-written file on disk.
    """
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.replace(tmp, path)
    except BaseException:
        # Clean up tmp on failure
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


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

    All writes are **atomic** (write to tempfile → ``os.replace``).  A
    per-key ``threading.Lock`` prevents concurrent in-process writes from
    interleaving.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        if cache_dir is None:
            cache_dir = Path.home() / ".qocc" / "cache"
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        # Per-key threading locks (process-level concurrency)
        self._key_locks: dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()  # protects _key_locks dict

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "CompilationCache":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def close(self) -> None:
        """Release per-key locks and reset internal state."""
        with self._meta_lock:
            self._key_locks.clear()

    # ------------------------------------------------------------------
    # Key computation
    # ------------------------------------------------------------------

    def _lock_for_key(self, key: str) -> threading.Lock:
        """Return (and lazily create) the per-key threading lock."""
        with self._meta_lock:
            if key not in self._key_locks:
                self._key_locks[key] = threading.Lock()
            return self._key_locks[key]

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
        lock = self._lock_for_key(key)
        with lock:
            try:
                data = json.loads(result_path.read_text(encoding="utf-8"))
                # Update last-access timestamp (atomic)
                meta_path = entry_dir / "meta.json"
                if meta_path.exists():
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                    meta["last_accessed"] = time.time()
                    meta["access_count"] = meta.get("access_count", 0) + 1
                    _atomic_write_text(meta_path, json.dumps(meta, indent=2) + "\n")
                return data
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("Cache read failed for key %s: %s", key[:16], exc)
                return None

    def put(
        self,
        key: str,
        result: dict[str, Any],
        circuit_qasm: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Store a compilation result in the cache (atomic + locked).

        Parameters:
            key: Cache key (from ``cache_key()``).
            result: Serialized compilation result dict.
            circuit_qasm: Optional QASM output to cache.
            metadata: Additional metadata to store.

        Returns:
            Path to the cache entry directory.
        """
        lock = self._lock_for_key(key)
        with lock:
            entry_dir = self.root / key
            entry_dir.mkdir(parents=True, exist_ok=True)

            # Write result (atomic)
            _atomic_write_text(
                entry_dir / "result.json",
                json.dumps(result, indent=2, default=str) + "\n",
            )

            # Write QASM if available (atomic)
            if circuit_qasm:
                _atomic_write_text(entry_dir / "circuit.qasm", circuit_qasm)

            # Write metadata (atomic)
            meta: dict[str, Any] = {
                "created": time.time(),
                "last_accessed": time.time(),
                "access_count": 0,
            }
            if metadata:
                meta.update(metadata)
            _atomic_write_text(
                entry_dir / "meta.json",
                json.dumps(meta, indent=2, default=str) + "\n",
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
                        logger.debug("Corrupt meta.json in %s", child, exc_info=True)

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
                        logger.debug("Corrupt meta.json in %s", child, exc_info=True)
                entries.append((last_access, child))

        if len(entries) <= max_entries:
            return 0

        # Sort by last access (oldest first)
        entries.sort(key=lambda x: x[0])
        to_evict = entries[: len(entries) - max_entries]
        evicted_keys: list[str] = []
        for _, path in to_evict:
            evicted_keys.append(path.name)
            shutil.rmtree(path)

        # Prune per-key locks for evicted entries
        if evicted_keys:
            with self._meta_lock:
                for k in evicted_keys:
                    self._key_locks.pop(k, None)

        return len(to_evict)
