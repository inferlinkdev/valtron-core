from __future__ import annotations
from typing import Any

from valtron_core.evaluation.comparison_functions import MetricCategory, element_compare_category
from valtron_core.evaluation.json_eval.schema import (
    FieldConfig,
    LeafMetricConfig,
    _list_mc,
)


_BUILTIN_METRIC_NAMES: frozenset[str] = frozenset({
    # legacy (deprecated)
    "exact", "threshold", "comparator",
    # standalone metrics
    "exact_compare", "text_similarity", "llm", "embedding",
})


def _check_builtin_metric_category(
    metric_name: str, params: dict[str, Any]
) -> tuple[MetricCategory, str]:
    """Return ``(category, description)`` for a built-in metric.

    ``category`` is one of ``"local"``, ``"llm"``, or ``"embedding"`` and is used by both
    the pre-flight safety check and the auto-LLM-alignment routing for unordered lists.

    DEVELOPER NOTE — when adding a new metric to ``JsonEvaluator.metric_registry``:
      1. Add a branch here returning the correct category.
    Omitting this raises ``NotImplementedError`` at pre-flight time.

    :param metric_name: The built-in metric name.
    :param params: The metric's params dict.
    :return: A ``(category, description)`` tuple.
    """
    if metric_name in ("exact", "threshold", "exact_compare"):
        return "local", ""

    if metric_name == "text_similarity":
        if params.get("metric") == "cosine":
            model = params.get("embedding_model", "text-embedding-3-small")
            return "embedding", (
                f"text_similarity with cosine metric calls the embedding API (model='{model}')"
            )
        return "local", ""

    if metric_name == "comparator":
        element_compare = params.get("element_compare", "exact")
        return element_compare_category(element_compare, params)

    if metric_name == "llm":
        model = params.get("model", "gpt-4o-mini")
        return "llm", f"LLM metric (model='{model}')"

    if metric_name == "embedding":
        model = params.get("model", "text-embedding-3-small")
        return "embedding", f"embedding metric (model='{model}')"
    raise NotImplementedError(
        f"Built-in metric '{metric_name}' has no category declaration.\n"
        "When adding a new metric to JsonEvaluator.metric_registry you MUST:\n"
        "  1. Add a branch in _check_builtin_metric_category() in validation.py\n"
        "     returning the correct category ('local' | 'llm' | 'embedding').\n"
        "This check exists to prevent accidental n²-cost list evaluations."
    )


def _scan_item_logic_for_expensive_metrics(
    config: FieldConfig,
    path: str,
    custom_metric_names: frozenset[str],
    relative_path: str = "",
) -> list[dict[str, str]]:
    """Recursively scan a FieldConfig subtree for expensive metrics.

    ``relative_path`` tracks the position within the current list's item_logic,
    used to populate ``allow_expensive_comparisons_for`` suggestions.  For a list
    of primitives (item_logic is itself a leaf) the relative_path is ``"$item"``.

    Returns a list of issue dicts with keys:
      metric_path, relative_path, type ("builtin" | "custom"), metric, description
    """
    issues: list[dict[str, str]] = []

    if config.type == "leaf":
        mc = config.metric_config
        if mc is None:
            return issues
        assert isinstance(mc, LeafMetricConfig)
        # For a list of primitives the item_logic root is the leaf itself; use "$item"
        display_rel = relative_path if relative_path else "$item"
        if mc.metric in custom_metric_names:
            issues.append({
                "metric_path": path,
                "relative_path": display_rel,
                "type": "custom",
                "metric": mc.metric,
                "category": "custom",
                "description": (
                    f"custom metric '{mc.metric}' — its implementation is user-defined "
                    "and may call a 3rd-party service"
                ),
            })
        elif mc.metric in _BUILTIN_METRIC_NAMES:
            category, desc = _check_builtin_metric_category(mc.metric, mc.params)
            if category != "local":
                issues.append({
                    "metric_path": path,
                    "relative_path": display_rel,
                    "type": "builtin",
                    "metric": mc.metric,
                    "category": category,
                    "description": desc,
                })
        else:
            raise ValueError(
                f"Unknown metric '{mc.metric}' at '{path}'.\n"
                f"It is neither a built-in metric {sorted(_BUILTIN_METRIC_NAMES)} "
                f"nor registered in FieldMetricsConfig.custom_metrics.\n"
            )

    elif config.type == "object" and config.fields:
        for key, fc in config.fields.items():
            sub_rel = f"{relative_path}.{key}" if relative_path else key
            issues.extend(
                _scan_item_logic_for_expensive_metrics(
                    fc, f"{path}.{key}", custom_metric_names, sub_rel
                )
            )

    elif config.type == "list":
        mc = _list_mc(config)
        if mc.item_logic:
            issues.extend(
                _scan_item_logic_for_expensive_metrics(
                    mc.item_logic, f"{path}[]", custom_metric_names, relative_path
                )
            )

    return issues


def _find_expensive_lists_recursive(
    config: FieldConfig,
    path: str,
    custom_metric_names: frozenset[str],
    issues: list[dict[str, str]],
) -> None:
    """Walk the config tree; append issue dicts for each offending list field."""
    if config.type == "list":
        mc = _list_mc(config)
        if not mc.ordered:
            allowed = set(mc.allow_expensive_comparisons_for or [])
            if mc.item_logic:
                for issue in _scan_item_logic_for_expensive_metrics(
                    mc.item_logic, f"{path}[]", custom_metric_names
                ):
                    if issue["relative_path"] not in allowed:
                        issue["list_path"] = path
                        issues.append(issue)
        # Always recurse into item_logic in case there are nested lists
        if mc.item_logic:
            _find_expensive_lists_recursive(
                mc.item_logic, f"{path}[]", custom_metric_names, issues
            )

    elif config.type == "object" and config.fields:
        for key, fc in config.fields.items():
            _find_expensive_lists_recursive(fc, f"{path}.{key}", custom_metric_names, issues)


def _collect_llm_models_recursive(config: FieldConfig) -> set[str]:
    models: set[str] = set()
    if config.type == "leaf" and config.metric_config is not None:
        mc = config.metric_config
        params: dict[str, Any] = getattr(mc, "params", {})
        # Legacy: comparator metric with element_compare="llm"
        if params.get("element_compare") == "llm":
            models.add(params.get("llm_model", "gpt-4o-mini"))
        # New: llm metric registered directly
        if getattr(mc, "metric", None) == "llm":
            models.add(params.get("model", "gpt-4o-mini"))
    elif config.type == "object" and config.fields:
        for fc in config.fields.values():
            models.update(_collect_llm_models_recursive(fc))
    elif config.type == "list" and config.metric_config is not None:
        item_logic = getattr(config.metric_config, "item_logic", None)
        if item_logic is not None:
            models.update(_collect_llm_models_recursive(item_logic))
    return models


def collect_field_metric_llm_models(config_dict: dict[str, Any]) -> set[str]:
    """Return the set of LLM model names used as judges in field metrics.

    Walks the FieldConfig tree and collects every model name where
    element_compare='llm' is set in a leaf metric's params.

    :param config_dict: A serialized FieldConfig dict.
    :return: Set of LLM model names referenced by LLM-judge leaves.
    """
    config = FieldConfig.model_validate(config_dict)
    return _collect_llm_models_recursive(config)


def _item_logic_has_llm_judge_leaf(item_logic: FieldConfig | None) -> bool:
    """Return True if any leaf below ``item_logic`` uses an LLM-judge metric.

    Walks the nested FieldConfig subtree of a list's item_logic and reports whether
    at least one reachable leaf has ``metric == "comparator"`` with
    ``params.element_compare == "llm"``.  Nested lists are descended into so a deeply
    buried LLM-judge leaf still triggers the auto-alignment path on its enclosing list.

    :param item_logic: The list's ``item_logic`` FieldConfig, or None.
    :return: True if any leaf below uses an LLM judge.
    """
    if item_logic is None:
        return False

    return bool(_collect_llm_models_recursive(item_logic))


def find_expensive_unordered_list_fields(
    config_dict: dict[str, Any],
    custom_metric_names: set[str] | None = None,
) -> list[dict[str, str]]:
    """Scan a FieldConfig dict for unordered list fields that use expensive
    (3rd-party API) metrics and have not explicitly opted in via
    ``allow_expensive_list_comparison: true``.

    Returns a list of issue dicts, each containing:
      - ``list_path``   - path to the offending list field
      - ``metric_path`` - path within item_logic to the expensive leaf metric
      - ``type``        - ``"builtin"`` or ``"custom"``
      - ``metric``      - metric name
      - ``description`` - human-readable description of the issue
    """
    issues: list[dict[str, str]] = []
    config = FieldConfig.model_validate(config_dict)
    _find_expensive_lists_recursive(
        config, "root", frozenset(custom_metric_names or set()), issues
    )
    return issues
