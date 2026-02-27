"""SQLite-backed regression tracking for QOCC bundles."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qocc.core.artifacts import ArtifactStore
from qocc.core.hashing import hash_string


class RegressionDatabase:
    """Stores bundle summaries for historical regression tracking."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        if db_path is None:
            db_path = Path.home() / ".qocc" / "regression.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS bundle_rows (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    adapter TEXT,
                    circuit_hash TEXT,
                    candidate_id TEXT,
                    surrogate_score REAL,
                    bundle_path TEXT,
                    timestamp TEXT,
                    metrics_json TEXT NOT NULL,
                    contract_results_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_bundle_rows_run_id ON bundle_rows(run_id);
                CREATE INDEX IF NOT EXISTS idx_bundle_rows_hash ON bundle_rows(circuit_hash);
                CREATE INDEX IF NOT EXISTS idx_bundle_rows_adapter ON bundle_rows(adapter);
                CREATE TABLE IF NOT EXISTS run_tags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    tag TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    UNIQUE(run_id, tag)
                );
                """
            )

    def ingest(self, bundle_path: str | Path) -> dict[str, Any]:
        bundle = ArtifactStore.load_bundle(bundle_path)
        root = Path(bundle.get("_root", ""))
        manifest = bundle.get("manifest", {})
        run_id = str(manifest.get("run_id", "unknown"))
        adapter = manifest.get("adapter")
        timestamp = manifest.get("created_at")
        if not timestamp:
            timestamp = datetime.now(timezone.utc).isoformat()

        circuit_hash = self._resolve_circuit_hash(root)
        rows = self._extract_rows(bundle, root)

        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            for row in rows:
                conn.execute(
                    """
                    INSERT INTO bundle_rows (
                        run_id, adapter, circuit_hash, candidate_id, surrogate_score,
                        bundle_path, timestamp, metrics_json, contract_results_json, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        run_id,
                        adapter,
                        circuit_hash,
                        row.get("candidate_id"),
                        row.get("surrogate_score"),
                        str(bundle_path),
                        timestamp,
                        json.dumps(row.get("metrics", {}), sort_keys=True),
                        json.dumps(row.get("contract_results", []), sort_keys=True),
                        created_at,
                    ),
                )

        return {
            "run_id": run_id,
            "adapter": adapter,
            "circuit_hash": circuit_hash,
            "rows_ingested": len(rows),
            "db_path": str(self.db_path),
        }

    def query(
        self,
        circuit_hash: str | None = None,
        adapter: str | None = None,
        since: str | None = None,
    ) -> list[dict[str, Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        if circuit_hash:
            clauses.append("circuit_hash = ?")
            params.append(circuit_hash)
        if adapter:
            clauses.append("adapter = ?")
            params.append(adapter)
        if since:
            clauses.append("timestamp >= ?")
            params.append(since)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT run_id, adapter, circuit_hash, candidate_id, surrogate_score,
                   bundle_path, timestamp, metrics_json, contract_results_json, created_at
            FROM bundle_rows
            {where}
            ORDER BY timestamp DESC, created_at DESC
        """
        out: list[dict[str, Any]] = []
        with self._connect() as conn:
            for row in conn.execute(sql, params):
                out.append(
                    {
                        "run_id": row["run_id"],
                        "adapter": row["adapter"],
                        "circuit_hash": row["circuit_hash"],
                        "candidate_id": row["candidate_id"],
                        "surrogate_score": row["surrogate_score"],
                        "bundle_path": row["bundle_path"],
                        "timestamp": row["timestamp"],
                        "metrics": json.loads(row["metrics_json"]),
                        "contract_results": json.loads(row["contract_results_json"]),
                        "created_at": row["created_at"],
                    }
                )
        return out

    def tag(self, bundle_path: str | Path, tag: str) -> dict[str, Any]:
        bundle = ArtifactStore.load_bundle(bundle_path)
        run_id = str(bundle.get("manifest", {}).get("run_id", "unknown"))
        created_at = datetime.now(timezone.utc).isoformat()

        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO run_tags(run_id, tag, created_at) VALUES (?, ?, ?)",
                (run_id, tag, created_at),
            )

        return {"run_id": run_id, "tag": tag, "db_path": str(self.db_path)}

    def detect_regressions(
        self,
        new_bundle: str | Path,
        baseline_tag: str,
        delta_threshold: float = 0.05,
    ) -> dict[str, Any]:
        bundle = ArtifactStore.load_bundle(new_bundle)
        root = Path(bundle.get("_root", ""))
        manifest = bundle.get("manifest", {})
        run_id = str(manifest.get("run_id", "unknown"))
        adapter = manifest.get("adapter")
        circuit_hash = self._resolve_circuit_hash(root)

        new_rows = self._extract_rows(bundle, root)
        if not new_rows:
            return {
                "run_id": run_id,
                "baseline_tag": baseline_tag,
                "regressions": [],
                "reason": "no_rows",
            }

        baseline_rows = self._baseline_rows(baseline_tag, adapter, circuit_hash)
        if not baseline_rows:
            return {
                "run_id": run_id,
                "baseline_tag": baseline_tag,
                "regressions": [],
                "reason": "no_baseline",
            }

        baseline_by_candidate = {
            r["candidate_id"]: r for r in baseline_rows
        }

        regressions: list[dict[str, Any]] = []
        for row in new_rows:
            candidate_id = row.get("candidate_id")
            baseline = baseline_by_candidate.get(candidate_id)
            if baseline is None and candidate_id is not None:
                baseline = baseline_by_candidate.get(None)
            if baseline is None:
                continue

            current_metrics = row.get("metrics", {})
            baseline_metrics = baseline.get("metrics", {})
            metric_regressions: dict[str, Any] = {}
            for key, cur in current_metrics.items():
                base = baseline_metrics.get(key)
                if not isinstance(cur, (int, float)) or not isinstance(base, (int, float)):
                    continue
                if base == 0:
                    worsened = cur > 0 and abs(cur - base) >= delta_threshold
                    rel = None
                else:
                    rel = (cur - base) / abs(base)
                    worsened = rel > delta_threshold
                if worsened:
                    metric_regressions[key] = {
                        "baseline": base,
                        "current": cur,
                        "relative_delta": rel,
                    }

            if metric_regressions:
                regressions.append(
                    {
                        "candidate_id": candidate_id,
                        "metrics": metric_regressions,
                    }
                )

        return {
            "run_id": run_id,
            "adapter": adapter,
            "circuit_hash": circuit_hash,
            "baseline_tag": baseline_tag,
            "delta_threshold": delta_threshold,
            "regressions": regressions,
            "has_regressions": bool(regressions),
        }

    def _baseline_rows(self, tag: str, adapter: str | None, circuit_hash: str | None) -> list[dict[str, Any]]:
        clauses = ["t.tag = ?"]
        params: list[Any] = [tag]
        if adapter:
            clauses.append("b.adapter = ?")
            params.append(adapter)
        if circuit_hash:
            clauses.append("b.circuit_hash = ?")
            params.append(circuit_hash)

        where = " AND ".join(clauses)
        sql = f"""
            SELECT b.*
            FROM bundle_rows b
            JOIN run_tags t ON t.run_id = b.run_id
            WHERE {where}
            ORDER BY b.timestamp DESC, b.created_at DESC
        """

        out: list[dict[str, Any]] = []
        seen_candidate: set[str | None] = set()
        with self._connect() as conn:
            for row in conn.execute(sql, params):
                cid = row["candidate_id"]
                if cid in seen_candidate:
                    continue
                seen_candidate.add(cid)
                out.append(
                    {
                        "candidate_id": cid,
                        "metrics": json.loads(row["metrics_json"]),
                        "contract_results": json.loads(row["contract_results_json"]),
                    }
                )
        return out

    def _extract_rows(self, bundle: dict[str, Any], root: Path) -> list[dict[str, Any]]:
        rankings = self._read_json_if_exists(root / "search_rankings.json")
        if isinstance(rankings, list) and rankings:
            rows: list[dict[str, Any]] = []
            for entry in rankings:
                if not isinstance(entry, dict):
                    continue
                rows.append(
                    {
                        "candidate_id": entry.get("candidate_id"),
                        "surrogate_score": entry.get("surrogate_score"),
                        "metrics": entry.get("metrics", {}),
                        "contract_results": entry.get("contract_results", []),
                    }
                )
            if rows:
                return rows

        metrics = bundle.get("metrics", {}).get("compiled", {})
        contract_results = bundle.get("contract_results", [])
        return [
            {
                "candidate_id": None,
                "surrogate_score": None,
                "metrics": metrics,
                "contract_results": contract_results,
            }
        ]

    def _resolve_circuit_hash(self, root: Path) -> str | None:
        for rel in ("circuits/input.qasm", "circuits/selected.qasm", "circuits/normalized.qasm"):
            p = root / rel
            if p.exists():
                return hash_string(p.read_text(encoding="utf-8"))

        stim_path = root / "circuits" / "selected.stim"
        if stim_path.exists():
            return hash_string(stim_path.read_text(encoding="utf-8"))
        return None

    def _read_json_if_exists(self, path: Path) -> Any | None:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
