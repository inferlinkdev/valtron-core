"""ExperimentRegistry: which recipe class a task_type name, or a raw data
sample, refers to.

``config_wizard.py``'s ``api_analyze_data`` hardcodes an if/elif over three
task types today, deciding which one a raw uploaded dataset "looks like"
from its first record's ``label`` shape. This module makes that dispatch
mechanical: ``classification.py``/``extraction.py``/``summarization.py``
each register themselves once, at import time, via ``@register_experiment``,
and ``ExperimentRegistry.sniff_best_match(data)`` replaces the hardcoded
if/elif. A new experiment type registers itself the same way; zero edits
here or to any existing recipe.

Not yet consumed anywhere: this only adds the registry and the three
registrations. Pointing ``config_wizard.py`` at it is a separate, later
commit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, TypeVar

from valtron_core.evaluation.model_eval import ModelEval

SniffFn = Callable[[list[dict[str, Any]]], bool]
ExperimentT = TypeVar("ExperimentT", bound=type[ModelEval])


def label_looks_like_json(label: Any) -> bool:
    """True if a raw label value is, or would parse as, a JSON object or array.

    A plain string label (``"positive"``) is ``False``; a dict/list label, or
    a string that already contains serialized JSON (``'{"a": 1}'``), is
    ``True``. Shared by every sniff predicate that cares about this exact
    distinction (``ClassificationExperiment``: ``False``,
    ``ExtractionExperiment``: ``True``), the same rule
    ``utilities/config_wizard.py``'s ``api_analyze_data`` applies inline today.
    """
    if isinstance(label, (dict, list)):
        return True
    try:
        parsed = json.loads(label)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(parsed, (dict, list))


@dataclass(frozen=True)
class _Registration:
    task_type: str
    experiment_class: "type[ModelEval]"
    sniff: "SniffFn | None"


class ExperimentRegistry:
    """Maps a task_type name to the ``ModelEval`` subclass that implements it.

    Populated by ``@register_experiment``, once per module import; nothing
    else mutates it. A class-level dict rather than instances, since there is
    exactly one registry for the whole process, the same way there is exactly
    one set of experiment types this codebase knows about.
    """

    _by_task_type: "dict[str, _Registration]" = {}

    @classmethod
    def register(
        cls,
        task_type: str,
        experiment_class: "type[ModelEval]",
        *,
        sniff: "SniffFn | None" = None,
    ) -> None:
        """Register ``experiment_class`` under ``task_type``.

        Re-registering the same class under a task_type it already owns is a
        true no-op: the existing registration (sniff predicate included) is
        left exactly as it was, rather than replaced by this call's own
        ``sniff`` (harmless if a module is imported more than once, e.g.
        under certain test setups, as long as that second call doesn't
        silently downgrade an already-registered sniff predicate to the
        default ``None`` by omitting it). Registering a *different* class
        under a task_type already claimed is an error, since that would
        silently make dispatch depend on import order.
        """
        existing = cls._by_task_type.get(task_type)
        if existing is not None:
            if existing.experiment_class is not experiment_class:
                raise ValueError(
                    f"task_type {task_type!r} is already registered to "
                    f"{existing.experiment_class.__name__}, cannot also register "
                    f"{experiment_class.__name__}."
                )
            return
        cls._by_task_type[task_type] = _Registration(task_type, experiment_class, sniff)

    @classmethod
    def get(cls, task_type: str) -> "type[ModelEval] | None":
        """The registered class for ``task_type``, or ``None`` if nothing registered it."""
        registration = cls._by_task_type.get(task_type)
        return registration.experiment_class if registration else None

    @classmethod
    def task_types(cls) -> "list[str]":
        """Every registered task_type name, in registration order."""
        return list(cls._by_task_type)

    @classmethod
    def sniff_best_match(cls, data: "list[dict[str, Any]]") -> "str | None":
        """The task_type of the first registered experiment whose sniff predicate
        matches ``data``, in registration order; ``None`` if none do (including
        when nothing registered a sniff predicate at all).
        """
        for task_type, registration in cls._by_task_type.items():
            if registration.sniff is not None and registration.sniff(data):
                return task_type
        return None


def register_experiment(
    task_type: str, *, sniff: "SniffFn | None" = None
) -> "Callable[[ExperimentT], ExperimentT]":
    """Class decorator: register this ``ModelEval`` subclass under ``task_type``.

    ``sniff(data) -> bool`` decides whether a raw uploaded dataset "looks
    like" this task type, for ``ExperimentRegistry.sniff_best_match()``; omit
    it for a type that should only ever be selected explicitly, never
    auto-detected.
    """

    def decorator(experiment_class: "ExperimentT") -> "ExperimentT":
        ExperimentRegistry.register(task_type, experiment_class, sniff=sniff)
        return experiment_class

    return decorator
