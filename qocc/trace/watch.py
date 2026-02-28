"""Hardware job watch utilities for trace bundles."""

from __future__ import annotations

import json
import shlex
import subprocess
import tempfile
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from qocc.trace.emitter import TraceEmitter


def watch_bundle_jobs(
    bundle_path: str,
    poll_interval_s: float = 5.0,
    timeout_s: float | None = None,
    on_complete: str | None = None,
) -> dict[str, Any]:
    """Poll pending hardware jobs and update bundle artifacts in-place.

    Supports bundle directories and zip bundles. For zip bundles, updates are
    applied to an extracted temp directory and then written back to the zip path.
    """
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be > 0")

    bundle_input = Path(bundle_path)
    root, rewrite_zip = _prepare_bundle_root(bundle_input)
    hardware_dir = root / "hardware"
    hardware_dir.mkdir(parents=True, exist_ok=True)

    pending_fp = hardware_dir / "pending_jobs.json"
    pending = _load_pending_jobs(pending_fp)

    if not pending:
        summary = {
            "bundle": str(bundle_input),
            "pending": 0,
            "completed": 0,
            "failed": 0,
            "results": [],
        }
        if rewrite_zip:
            _rewrite_zip(bundle_input, root)
        return summary

    started = time.perf_counter()
    completed_results: list[dict[str, Any]] = []
    completed_count = 0
    failed_count = 0

    while True:
        active = [j for j in pending if str(j.get("status", "pending")).lower() not in {"done", "completed", "failed", "cancelled", "error"}]
        if not active:
            break

        if timeout_s is not None and (time.perf_counter() - started) >= timeout_s:
            raise TimeoutError(f"Timed out while watching hardware jobs after {timeout_s:.1f}s")

        for entry in active:
            update = _poll_job_entry(entry)
            entry["last_polled_at"] = datetime.now(timezone.utc).isoformat()
            entry["status"] = update["status"]
            if update.get("message"):
                entry["message"] = update["message"]

            if update.get("done"):
                finished_at = datetime.now(timezone.utc).isoformat()
                entry["finished_at"] = finished_at
                span_payload = {
                    "job_id": str(entry.get("job_id", "")),
                    "provider": str(entry.get("provider", entry.get("adapter", "unknown"))),
                    "status": str(update.get("status", "unknown")),
                }
                if update.get("error"):
                    span_payload["error"] = str(update["error"])

                _append_completion_span(root / "trace.jsonl", "job_complete", span_payload)

                if update.get("error"):
                    failed_count += 1
                    continue

                if update.get("result"):
                    result_payload = dict(update["result"])
                    result_payload.setdefault("job_id", entry.get("job_id"))
                    result_payload.setdefault("backend_name", entry.get("backend_name"))
                    result_payload.setdefault("provider", entry.get("provider", "unknown"))
                    result_payload["finished_at"] = finished_at
                    completed_results.append(result_payload)
                    completed_count += 1
                    _write_result_file(hardware_dir, result_payload)

        pending_fp.write_text(json.dumps(pending, indent=2, default=str) + "\n", encoding="utf-8")
        if any(str(j.get("status", "pending")).lower() not in {"done", "completed", "failed", "cancelled", "error"} for j in pending):
            time.sleep(poll_interval_s)

    _update_hardware_aggregate(root, completed_results)

    if rewrite_zip:
        _rewrite_zip(bundle_input, root)

    hook_result = None
    if on_complete and completed_count > 0:
        hook_result = _run_on_complete(on_complete, str(bundle_input))

    return {
        "bundle": str(bundle_input),
        "pending": len([j for j in pending if str(j.get("status", "")).lower() not in {"done", "completed", "failed", "cancelled", "error"}]),
        "completed": completed_count,
        "failed": failed_count,
        "results": completed_results,
        "on_complete": hook_result,
    }


def _prepare_bundle_root(bundle_path: Path) -> tuple[Path, bool]:
    if bundle_path.is_dir():
        return bundle_path, False

    if bundle_path.suffix.lower() != ".zip":
        raise ValueError(f"Bundle path must be directory or .zip file: {bundle_path}")

    if not bundle_path.exists():
        raise FileNotFoundError(str(bundle_path))

    temp_root = Path(tempfile.mkdtemp(prefix="qocc_watch_"))
    with zipfile.ZipFile(bundle_path, "r") as zf:
        root = temp_root.resolve()
        for member in zf.namelist():
            target = (temp_root / member).resolve()
            if not str(target).startswith(str(root)):
                raise ValueError(f"Unsafe zip member rejected (ZipSlip): {member!r}")
        zf.extractall(temp_root)
    return temp_root, True


def _rewrite_zip(zip_path: Path, root: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(root.rglob("*")):
            if file.is_file():
                zf.write(file, file.relative_to(root))


def _load_pending_jobs(pending_fp: Path) -> list[dict[str, Any]]:
    if not pending_fp.exists():
        return []
    data = json.loads(pending_fp.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("hardware/pending_jobs.json must be a list of job entries")
    return [d for d in data if isinstance(d, dict)]


def _poll_job_entry(job: dict[str, Any]) -> dict[str, Any]:
    provider = str(job.get("provider", job.get("adapter", ""))).lower()
    if provider == "ibm":
        from qocc.adapters.ibm_adapter import poll_ibm_job

        backend_spec = dict(job.get("backend_spec", {}))
        if "backend_name" not in backend_spec and job.get("backend_name"):
            backend_spec["backend_name"] = job.get("backend_name")
        shots = int(job.get("shots", backend_spec.get("shots", 1024)))
        return poll_ibm_job(backend_spec=backend_spec, job_id=str(job.get("job_id", "")), shots=shots)

    return {
        "status": "failed",
        "done": True,
        "error": f"Unsupported hardware provider for watch: {provider or 'unknown'}",
    }


def _append_completion_span(trace_file: Path, name: str, attributes: dict[str, Any]) -> None:
    trace_file.parent.mkdir(parents=True, exist_ok=True)
    emitter = TraceEmitter()
    with emitter.span(name, attributes=attributes):
        pass
    with trace_file.open("a", encoding="utf-8") as f:
        for record in emitter.to_dicts():
            f.write(json.dumps(record, default=str) + "\n")


def _write_result_file(hardware_dir: Path, result_payload: dict[str, Any]) -> None:
    job_id = str(result_payload.get("job_id", "unknown"))
    out = hardware_dir / f"{job_id}_result.json"
    out.write_text(json.dumps(result_payload, indent=2, default=str) + "\n", encoding="utf-8")


def _update_hardware_aggregate(root: Path, new_results: list[dict[str, Any]]) -> None:
    if not new_results:
        return

    hardware_dir = root / "hardware"
    hardware_dir.mkdir(parents=True, exist_ok=True)

    aggregate_fp = hardware_dir / "hardware.json"
    aggregate: dict[str, Any] = {}
    if aggregate_fp.exists():
        try:
            loaded = json.loads(aggregate_fp.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                aggregate = loaded
        except Exception:
            aggregate = {}

    results = aggregate.get("results")
    if not isinstance(results, list):
        results = []
    results.extend(new_results)
    aggregate["results"] = results

    latest = new_results[-1]
    if isinstance(latest.get("counts"), dict):
        aggregate["counts"] = latest["counts"]
    if isinstance(latest.get("metadata"), dict):
        aggregate["metadata"] = latest["metadata"]
    if latest.get("backend_name"):
        aggregate["backend_name"] = latest["backend_name"]
    aggregate["job_id"] = latest.get("job_id")

    aggregate_fp.write_text(json.dumps(aggregate, indent=2, default=str) + "\n", encoding="utf-8")


def _run_on_complete(command: str, bundle_path: str) -> dict[str, Any]:
    cmd = command.replace("{bundle}", bundle_path)
    argv = shlex.split(cmd)
    if not argv:
        raise ValueError("--on-complete command cannot be empty")
    proc = subprocess.run(argv, capture_output=True, text=True)
    return {
        "command": argv,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }
