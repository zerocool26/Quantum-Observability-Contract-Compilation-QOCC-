"""Artifact store — writes Trace Bundle directories consistently."""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import importlib.metadata
import re

_SAFE_RUN_ID = re.compile(r"^[a-zA-Z0-9_\-]+$")


class ArtifactStore:
    """Manages writing files into a Trace Bundle directory.

    The bundle layout follows the spec in §2 of the QOCC design doc.
    """

    def __init__(self, bundle_dir: str | Path) -> None:
        self.root = Path(bundle_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "circuits").mkdir(exist_ok=True)
        (self.root / "circuits" / "candidates").mkdir(exist_ok=True)
        (self.root / "reports").mkdir(exist_ok=True)

    # ------------------------------------------------------------------
    # Writers
    # ------------------------------------------------------------------

    def write_json(self, name: str, data: Any) -> Path:
        """Write a JSON file into the bundle root."""
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
        return p

    def write_text(self, name: str, text: str) -> Path:
        """Write a plain-text file into the bundle root."""
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")
        return p

    def write_jsonl(self, name: str, records: list[dict[str, Any]]) -> Path:
        """Write JSON Lines file."""
        p = self.root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, default=str) + "\n")
        return p

    # ------------------------------------------------------------------
    # Bundle standard files
    # ------------------------------------------------------------------

    def write_manifest(self, run_id: str, extra: dict[str, Any] | None = None) -> Path:
        if not _SAFE_RUN_ID.match(run_id):
            raise ValueError(
                f"Invalid run_id {run_id!r}. "
                "Must match [a-zA-Z0-9_-]+."
            )
        manifest: dict[str, Any] = {
            "schema_version": "0.1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "run_id": run_id,
            "qocc_version": _qocc_version(),
        }
        if extra:
            manifest.update(extra)
        return self.write_json("manifest.json", manifest)

    def write_env(self) -> Path:
        """Capture environment snapshot."""
        env: dict[str, Any] = {
            "os": platform.platform(),
            "python": sys.version,
            "python_executable": sys.executable,
            "packages": _installed_packages(),
            "git_sha": _git_sha(),
        }
        return self.write_json("env.json", env)

    def write_seeds(self, seeds: dict[str, Any]) -> Path:
        return self.write_json("seeds.json", seeds)

    def write_metrics(self, metrics: Any) -> Path:
        if hasattr(metrics, "to_dict"):
            data = metrics.to_dict()
        else:
            data = metrics
        return self.write_json("metrics.json", data)

    def write_contracts(self, specs: list[dict[str, Any]]) -> Path:
        return self.write_json("contracts.json", specs)

    def write_contract_results(self, results: list[dict[str, Any]]) -> Path:
        return self.write_json("contract_results.json", results)

    def write_trace(self, spans: list[dict[str, Any]]) -> Path:
        return self.write_jsonl("trace.jsonl", spans)

    def write_circuit(self, subpath: str, content: str) -> Path:
        return self.write_text(f"circuits/{subpath}", content)

    def write_summary_report(self, md: str) -> Path:
        return self.write_text("reports/summary.md", md)

    def write_cache_index(self, entries: list[dict[str, Any]]) -> Path:
        """Write cache hit/miss index for reproducibility auditing."""
        return self.write_json("cache_index.json", entries)

    # ------------------------------------------------------------------
    # Zip export
    # ------------------------------------------------------------------

    def export_zip(self, zip_path: str | Path) -> Path:
        """Pack the bundle directory into a zip file."""
        zp = Path(zip_path)
        with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as zf:
            for file in sorted(self.root.rglob("*")):
                if file.is_file():
                    zf.write(file, file.relative_to(self.root))
        return zp

    @staticmethod
    def load_bundle(path: str | Path) -> dict[str, Any]:
        """Load a bundle from a zip or directory and return parsed standard files.

        When loading a zip, contents are extracted to a unique temporary
        directory under the zip's parent.  **ZipSlip protection** rejects
        any archive member whose resolved path escapes the extraction root.
        """
        path = Path(path)
        if path.suffix == ".zip":
            # Use a unique extraction directory to avoid clobbering siblings
            extract_dir = path.with_suffix("") / f"_qocc_{os.getpid()}"
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)

            with zipfile.ZipFile(path, "r") as zf:
                # ZipSlip protection: reject paths that escape extract_dir
                resolved_root = extract_dir.resolve()
                for member in zf.namelist():
                    target = (extract_dir / member).resolve()
                    if not str(target).startswith(str(resolved_root)):
                        raise ValueError(
                            f"Unsafe zip member rejected (ZipSlip): {member!r}"
                        )
                zf.extractall(extract_dir)
            path = extract_dir

        bundle: dict[str, Any] = {}
        for name in ["manifest.json", "env.json", "seeds.json", "metrics.json",
                      "contracts.json", "contract_results.json"]:
            fp = path / name
            if fp.exists():
                bundle[name.replace(".json", "")] = json.loads(fp.read_text(encoding="utf-8"))

        trace_path = path / "trace.jsonl"
        if trace_path.exists():
            spans = []
            for line in trace_path.read_text(encoding="utf-8").strip().splitlines():
                spans.append(json.loads(line))
            bundle["trace"] = spans

        bundle["_root"] = str(path)
        return bundle


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _qocc_version() -> str:
    try:
        return importlib.metadata.version("qocc")
    except importlib.metadata.PackageNotFoundError:
        return "0.1.0-dev"


def _installed_packages() -> dict[str, str]:
    """Return {name: version} for all installed packages."""
    pkgs: dict[str, str] = {}
    for dist in importlib.metadata.distributions():
        name = dist.metadata["Name"]
        version = dist.metadata["Version"]
        if name and version:
            pkgs[name] = version
    return dict(sorted(pkgs.items()))


def _git_sha() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None
