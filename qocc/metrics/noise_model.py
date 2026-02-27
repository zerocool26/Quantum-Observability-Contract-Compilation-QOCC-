"""Noise model datatypes and registry.

Provider-agnostic noise model representation for surrogate scoring.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class NoiseModel:
    """Provider-agnostic noise model for search scoring."""

    single_qubit_error: float | dict[str, float] = 0.0
    two_qubit_error: float | dict[str, float] = 0.0
    readout_error: float | dict[str, float] = 0.0
    t1: float | dict[str, float] | None = None
    t2: float | dict[str, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "single_qubit_error": self.single_qubit_error,
            "two_qubit_error": self.two_qubit_error,
            "readout_error": self.readout_error,
            "t1": self.t1,
            "t2": self.t2,
            **self.extra,
        }

    def stable_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NoiseModel":
        known = {"single_qubit_error", "two_qubit_error", "readout_error", "t1", "t2"}
        extra = {k: v for k, v in data.items() if k not in known}
        return cls(
            single_qubit_error=data.get("single_qubit_error", 0.0),
            two_qubit_error=data.get("two_qubit_error", 0.0),
            readout_error=data.get("readout_error", 0.0),
            t1=data.get("t1"),
            t2=data.get("t2"),
            extra=extra,
        )


class NoiseModelRegistry:
    """Registry/loader for built-in and external noise models."""

    def __init__(self) -> None:
        self._builtins: dict[str, NoiseModel] = {
            "uniform_depolarizing": NoiseModel(
                single_qubit_error=0.001,
                two_qubit_error=0.01,
                readout_error=0.02,
            ),
            "thermal_relaxation": NoiseModel(
                single_qubit_error=0.001,
                two_qubit_error=0.012,
                readout_error=0.015,
                t1=100_000.0,
                t2=80_000.0,
            ),
            "readout_error": NoiseModel(
                single_qubit_error=0.0,
                two_qubit_error=0.0,
                readout_error=0.03,
            ),
        }

    def get_builtin(self, name: str) -> NoiseModel:
        if name not in self._builtins:
            raise KeyError(f"Unknown built-in noise model: {name}")
        return self._builtins[name]

    def list_builtins(self) -> list[str]:
        return sorted(self._builtins)

    def load_from_file(self, path: str | Path) -> NoiseModel:
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        self.validate(data)
        return NoiseModel.from_dict(data)

    def validate(self, data: dict[str, Any]) -> None:
        schema_path = Path(__file__).resolve().parent.parent.parent / "schemas" / "noise_model.schema.json"
        schema = json.loads(schema_path.read_text(encoding="utf-8"))

        try:
            import jsonschema
        except ImportError as exc:
            raise ImportError("jsonschema is required to validate noise models.") from exc

        jsonschema.Draft202012Validator(schema).validate(data)
