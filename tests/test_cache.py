"""Tests for the content-addressed compilation cache."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from qocc.core.cache import CompilationCache


@pytest.fixture
def cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "test_cache"


@pytest.fixture
def cache(cache_dir: Path) -> CompilationCache:
    return CompilationCache(cache_dir)


def test_cache_key_deterministic():
    """Same inputs produce the same cache key."""
    k1 = CompilationCache.cache_key("abc123", {"opt": 2}, "qiskit-1.0")
    k2 = CompilationCache.cache_key("abc123", {"opt": 2}, "qiskit-1.0")
    assert k1 == k2


def test_cache_key_differs_on_circuit():
    """Different circuit hash → different key."""
    k1 = CompilationCache.cache_key("abc", {"opt": 2}, "v1")
    k2 = CompilationCache.cache_key("def", {"opt": 2}, "v1")
    assert k1 != k2


def test_cache_key_differs_on_pipeline():
    """Different pipeline → different key."""
    k1 = CompilationCache.cache_key("abc", {"opt": 1}, "v1")
    k2 = CompilationCache.cache_key("abc", {"opt": 3}, "v1")
    assert k1 != k2


def test_put_and_get(cache: CompilationCache):
    """Store and retrieve a result."""
    key = CompilationCache.cache_key("abc", {"level": 2}, "v1")
    data = {"compiled_hash": "xyz", "depth": 10}

    cache.put(key, data, circuit_qasm="OPENQASM 3;")
    result = cache.get(key)

    assert result is not None
    assert result["compiled_hash"] == "xyz"


def test_get_miss(cache: CompilationCache):
    """Cache miss returns None."""
    assert cache.get("nonexistent_key") is None


def test_has(cache: CompilationCache):
    """has() returns True for existing entries, False otherwise."""
    key = CompilationCache.cache_key("abc", {}, "v1")
    assert cache.has(key) is False

    cache.put(key, {"data": True})
    assert cache.has(key) is True


def test_evict(cache: CompilationCache):
    """Evict removes a cached entry."""
    key = CompilationCache.cache_key("abc", {}, "v1")
    cache.put(key, {"data": True})
    assert cache.has(key) is True

    removed = cache.evict(key)
    assert removed is True
    assert cache.has(key) is False


def test_evict_nonexistent(cache: CompilationCache):
    """Evicting a missing entry returns False."""
    assert cache.evict("missing") is False


def test_clear(cache: CompilationCache):
    """Clear removes all entries."""
    for i in range(5):
        key = CompilationCache.cache_key(f"circuit_{i}", {}, "v1")
        cache.put(key, {"i": i})

    count = cache.clear()
    assert count == 5


def test_stats(cache: CompilationCache):
    """Stats returns entry count and size."""
    for i in range(3):
        key = CompilationCache.cache_key(f"c{i}", {}, "v1")
        cache.put(key, {"i": i})

    s = cache.stats()
    assert s["entries"] == 3
    assert s["total_size_bytes"] > 0


def test_access_count_increments(cache: CompilationCache):
    """Reading a cached entry increments the access count."""
    key = CompilationCache.cache_key("abc", {}, "v1")
    cache.put(key, {"data": True})

    cache.get(key)
    cache.get(key)

    meta_path = cache.root / key / "meta.json"
    meta = json.loads(meta_path.read_text())
    assert meta["access_count"] == 2


def test_evict_lru(cache: CompilationCache):
    """LRU eviction keeps at most max_entries."""
    import time

    for i in range(10):
        key = CompilationCache.cache_key(f"c{i}", {}, "v1")
        cache.put(key, {"i": i})
        # Slight delay to ensure different timestamps
        time.sleep(0.01)

    evicted = cache.evict_lru(max_entries=5)
    assert evicted == 5
    assert cache.stats()["entries"] == 5
