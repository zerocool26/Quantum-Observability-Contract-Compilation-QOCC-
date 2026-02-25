"""Contract evaluator plugin registry.

Custom evaluators can be registered programmatically via
``register_evaluator()`` or automatically via the ``qocc.evaluators``
entry-point group in ``pyproject.toml``::

    [project.entry-points."qocc.evaluators"]
    my_eval = "my_package:evaluate_my_contract"

An evaluator callable has the signature::

    def evaluate(spec: ContractSpec, **kwargs) -> ContractResult:
        ...
"""

from __future__ import annotations

from typing import Any, Callable

from qocc.contracts.spec import ContractResult, ContractSpec

EvaluatorFn = Callable[..., ContractResult]

_EVALUATOR_REGISTRY: dict[str, EvaluatorFn] = {}


def register_evaluator(name: str, fn: EvaluatorFn) -> None:
    """Register a custom contract evaluator under *name*."""
    _EVALUATOR_REGISTRY[name] = fn


def get_evaluator(name: str) -> EvaluatorFn | None:
    """Look up a registered evaluator by name.

    Returns ``None`` if not found (caller should fall back to built-in dispatch).
    """
    if name in _EVALUATOR_REGISTRY:
        return _EVALUATOR_REGISTRY[name]

    # Auto-discover from entry points
    _discover_entry_point_evaluators()
    return _EVALUATOR_REGISTRY.get(name)


def list_evaluators() -> list[str]:
    """Return names of all registered evaluators."""
    _discover_entry_point_evaluators()
    return sorted(_EVALUATOR_REGISTRY)


def _discover_entry_point_evaluators() -> None:
    """Auto-discover evaluators from the ``qocc.evaluators`` entry-point group."""
    import importlib.metadata

    try:
        eps = importlib.metadata.entry_points()
        group = eps.select(group="qocc.evaluators") if hasattr(eps, "select") else eps.get("qocc.evaluators", [])
        for ep in group:
            try:
                fn = ep.load()
                if ep.name not in _EVALUATOR_REGISTRY:
                    _EVALUATOR_REGISTRY[ep.name] = fn
            except Exception:
                pass
    except Exception:
        pass
