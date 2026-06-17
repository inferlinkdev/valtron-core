from __future__ import annotations
from typing import Literal, Any, Callable, cast
from pydantic import BaseModel, Field, ConfigDict, model_validator
from concurrent.futures import ThreadPoolExecutor
import json
import copy
import logging
import math
import threading
import warnings
from litellm import completion, completion_cost, embedding, ModelResponse
import numpy as np
from scipy.optimize import linear_sum_assignment

from valtron_core.evaluation.comparison_functions import (
    Comparator,
    MetricCategory,
    element_compare_category,
)
from valtron_core.evaluation.comparisons import (
    _embedding_compare,
    _exact_compare,
    _llm_compare,
    _text_similarity_compare,
)

logger = logging.getLogger(__name__)


MAX_LIST_LENGTH_FOR_EXPENSIVE_COMPARE = 4

# --- Embedding alignment defaults (unordered lists with LLM-judge leaves) ---
# Unordered-list items are aligned by optimal one-to-one assignment over embedding cosine
# similarity (Hungarian). These knobs are overridable per list via ListMetricConfig.
DEFAULT_ALIGN_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MATCH_KEY_MODEL = "gpt-5.4-mini"
DEFAULT_ALIGN_LO = 0.35      # cosine floor: pairs below this are left unmatched
# Safety cap on the text embedded per item. Top-level-only rendering keeps items small; this
# bounds the request size even when a top-level string is unusually long, so a batched
# embedding over a long list can't grow into an oversized request.
MATCH_KEY_MAX_CHARS = 512


def _score_to_result(result: bool | float, params: dict[str, Any]) -> tuple[float, bool]:
    """Convert a comparison function's return value to a (score, is_correct) tuple."""
    if isinstance(result, bool):
        return (1.0 if result else 0.0), result
    threshold = params.get("comparison_threshold") or params.get("threshold")
    if threshold is not None:
        return result, result >= float(threshold)
    return result, True


def _run_comparator(
    expected: Any, actual: Any, params: dict[str, Any]
) -> tuple[float, bool, float, int]:
    """Run the legacy Comparator and return (score, is_correct, cost_usd, call_count)."""
    comp = Comparator(
        element_compare=params.get("element_compare", "exact"),
        text_similarity_threshold=params.get("text_similarity_threshold", None),
        text_similarity_metric=params.get("text_similarity_metric", "fuzz_ratio"),
        llm_model=params.get("llm_model", "gpt-4o-mini"),
        llm_prompt_template=params.get("llm_prompt_template", None),
        llm_prompt_extra_vars=params.get("_template_vars", None),
        embedding_model=params.get("embedding_model", "text-embedding-3-small"),
        embedding_threshold=params.get("embedding_threshold", None),
        case_sensitive=params.get("case_sensitive", False),
        ignore_spaces=params.get("ignore_spaces", False),
    )
    compare_result = comp.compare(expected, actual)

    category, _ = element_compare_category(params.get("element_compare", "exact"), params)
    uses_api = category != "local"
    cost_usd = comp.total_comparison_cost if uses_api else 0.0
    call_count = comp.comparison_count if uses_api else 0

    if isinstance(compare_result, bool):
        return (1.0 if compare_result else 0.0), compare_result, cost_usd, call_count

    threshold = params.get("comparison_threshold", None) or params.get("text_similarity_threshold", None)
    if threshold is not None:
        return compare_result, compare_result >= threshold, cost_usd, call_count

    return compare_result, True, cost_usd, call_count


def comparator_metric(expected: Any, actual: Any, params: dict[str, Any]) -> tuple[float, bool]:
    """Public comparator metric callable. Cost is not tracked; use JsonEvaluator for that."""
    warnings.warn(
        "The 'comparator' metric is deprecated; use 'exact_compare', 'text_similarity', "
        "'llm', or 'embedding' metrics directly instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    score, is_correct, _, _ = _run_comparator(expected, actual, params)
    return score, is_correct


class LeafMetricConfig(BaseModel):
    """Metric config for leaf (scalar) fields."""
    model_config = ConfigDict(extra='forbid')
    metric: str = "exact"
    params: dict[str, Any] = Field(default_factory=dict)


class ObjectMetricConfig(BaseModel):
    """Metric config for object fields."""
    model_config = ConfigDict(extra='forbid')
    propagation: str = "weighted_avg"


class ListMetricConfig(BaseModel):
    """Metric config for list fields."""
    model_config = ConfigDict(extra='forbid')
    ordered: bool = False
    match_threshold: float = 0.5
    item_logic: FieldConfig | None = None
    required_fields_to_match: list[str] | None = None
    allow_expensive_comparisons_for: list[str] | None = None

    # Embedding alignment knobs (only used for unordered lists whose item_logic contains an
    # LLM-judge leaf). match_key_fields names the identity-bearing fields to embed; when None
    # they are selected once per list by a small LLM call (cached), falling back to the item's
    # top-level scalar fields. align_lo is the cosine floor below which pairs stay unmatched.
    match_key_fields: list[str] | None = None
    match_key_model: str = DEFAULT_MATCH_KEY_MODEL
    align_embedding_model: str = DEFAULT_ALIGN_EMBEDDING_MODEL
    align_lo: float = DEFAULT_ALIGN_LO


class FieldConfig(BaseModel):
    type: Literal["object", "list", "leaf"] = "leaf"
    weight: float = 1.0
    optional: bool = False
    metric_config: LeafMetricConfig | ObjectMetricConfig | ListMetricConfig | None = None
    fields: dict[str, FieldConfig] | None = None

    @model_validator(mode='before')
    @classmethod
    def _route_metric_config(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        field_type = data.get('type', 'leaf')
        mc = data.get('metric_config') or {}
        if not isinstance(mc, dict):
            return data  # already a model instance, leave as-is

        if field_type == 'object':
            data['metric_config'] = {
                'propagation': mc.get('propagation', 'weighted_avg'),
            }
        elif field_type == 'list':
            list_keys = (
                'match_threshold', 'item_logic', 'required_fields_to_match',
                'allow_expensive_comparisons_for', 'match_key_fields', 'match_key_model',
                'align_embedding_model', 'align_lo',
            )
            filtered = {k: mc[k] for k in list_keys if k in mc}
            filtered['ordered'] = mc.get('ordered', False)  # always present for union discrimination
            data['metric_config'] = filtered
        else:  # leaf
            data['metric_config'] = {
                'metric': mc.get('metric', 'exact'),
                'params': mc.get('params', {}),
            }
        return data


# Resolve circular dependencies
ListMetricConfig.model_rebuild()
FieldConfig.model_rebuild()


def _leaf_mc(config: "FieldConfig") -> LeafMetricConfig:
    """Narrow a leaf field's metric config to :class:`LeafMetricConfig`.

    The ``_route_metric_config`` validator guarantees a leaf field always carries a
    ``LeafMetricConfig``; this helper encodes that invariant for the type checker.

    :param config: A leaf :class:`FieldConfig`.
    :return: The field's leaf metric config.
    """
    mc = config.metric_config
    assert isinstance(mc, LeafMetricConfig)
    return mc


def _object_mc(config: "FieldConfig") -> ObjectMetricConfig:
    """Narrow an object field's metric config to :class:`ObjectMetricConfig`.

    :param config: An object :class:`FieldConfig`.
    :return: The field's object metric config.
    """
    mc = config.metric_config
    assert isinstance(mc, ObjectMetricConfig)
    return mc


def _list_mc(config: "FieldConfig") -> ListMetricConfig:
    """Narrow a list field's metric config to :class:`ListMetricConfig`.

    :param config: A list :class:`FieldConfig`.
    :return: The field's list metric config.
    """
    mc = config.metric_config
    assert isinstance(mc, ListMetricConfig)
    return mc


class ExpensiveListComparisonError(Exception):
    """Raised when an unordered list field uses a 3rd-party metric without
    explicit opt-in via ``allow_expensive_list_comparison: true``."""


# ---------------------------------------------------------------------------
# Built-in metric category declarations
# ---------------------------------------------------------------------------
# Every metric in JsonEvaluator.metric_registry MUST be handled in
# _check_builtin_metric_category() below.  When you add a new built-in
# metric, add a branch there declaring its category.
# Omitting it raises NotImplementedError at pre-flight time.

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
        "  1. Add a branch in _check_builtin_metric_category() in json_eval.py\n"
        "     returning the correct category ('local' | 'llm' | 'embedding').\n"
        "This check exists to prevent accidental n²-cost list evaluations."
    )


def _scan_item_logic_for_expensive_metrics(
    config: "FieldConfig",
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
    config: "FieldConfig",
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


def _collect_llm_models_recursive(config: "FieldConfig") -> set[str]:
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


def _item_logic_has_llm_judge_leaf(item_logic: "FieldConfig | None") -> bool:
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


class _MatchKeyFields(BaseModel):
    """LLM response selecting the identity-bearing fields of a list item.

    Used once per list (cached) to decide which fields to embed when aligning items for an
    unordered list.  Embedding only the identity fields keeps the cosine signal sharp;
    boilerplate/enum fields would otherwise dilute it.

    :param fields: Field names (top level of the item) that together identify an item.
    """
    fields: list[str]


def _cosine(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two equal-length vectors.

    :param v1: First vector.
    :param v2: Second vector.
    :return: Cosine similarity in ``[-1, 1]``; ``0.0`` if either vector has zero magnitude.
    """
    dot = sum(a * b for a, b in zip(v1, v2))
    mag1 = math.sqrt(sum(a * a for a in v1))
    mag2 = math.sqrt(sum(b * b for b in v2))

    if mag1 == 0 or mag2 == 0:
        return 0.0

    return dot / (mag1 * mag2)


def _match_key_text(item: Any, fields: list[str] | None) -> str:
    """Build the text embedded to represent ``item`` when aligning candidates.

    An item is characterized by its **top-level elements only** — nested lists/dicts are not
    recursed into, since identity almost always lives at the top level and the nested content
    is mostly noise (and bulk) for matching. Resolution:

    * explicit ``fields`` (dict item): serialize just those fields (a nested field named
      explicitly is still honored);
    * no ``fields`` (dict item): serialize only the top-level *scalar* fields;
    * neither yields anything (e.g. an item whose identity is entirely nested), or a
      non-dict item: fall back to a whole-item rendering.

    The result is truncated to :data:`MATCH_KEY_MAX_CHARS` as a safety bound so a batched
    embedding over a long list cannot grow into an oversized request.

    :param item: The list item (dict or primitive).
    :param fields: Explicit identity field names to embed, or None to use top-level scalars.
    :return: A compact, length-bounded text representation suitable for embedding.
    """
    if not isinstance(item, dict):
        return json.dumps(item, default=str, ensure_ascii=False, sort_keys=True)[:MATCH_KEY_MAX_CHARS]

    if fields:
        keys = [f for f in fields if f in item and item[f] is not None]
    else:
        # Top-level (non-recursive) scalar fields only.
        keys = [k for k, v in item.items() if v is not None and isinstance(v, (str, int, float, bool))]

    parts: list[str] = []
    for k in keys:
        val = item[k]
        val_str = val if isinstance(val, str) else json.dumps(val, default=str, ensure_ascii=False)
        parts.append(f"{k}: {val_str}")

    if not parts:
        return json.dumps(item, default=str, ensure_ascii=False, sort_keys=True)[:MATCH_KEY_MAX_CHARS]

    return "\n".join(parts)[:MATCH_KEY_MAX_CHARS]


def _truncate_for_prompt(value: Any, limit: int = 200) -> str:
    """Serialize a field value and truncate it for inclusion in a sample prompt.

    :param value: The field value to render.
    :param limit: Maximum number of characters to keep.
    :return: A truncated string representation.
    """
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    if len(text) > limit:
        return text[:limit] + "…"

    return text


class AlignmentItem(BaseModel):
    e_idx: int
    a_idx: int
    score: float
    result: EvalResult


class EvalResult(BaseModel):
    path: str
    score: float
    weight: float

    is_correct: bool = False # For threshold-based metrics

    tp: float = 0.0  # True Positives (Sum of match scores)
    tn: float = 0.0
    fp: float = 0.0  # False Positives (Extra items / hallucinations)
    fn: float = 0.0  # False Negatives (Missing items)

    precision: float = 0.0
    recall: float = 0.0

    metric: str
    params: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict) 
    children: dict[str, EvalResult] = {}
    alignment: list[AlignmentItem] | None = None


class JsonEvaluator:
    def __init__(
        self,
        custom_metrics: dict[str, Callable[..., Any]] | None = None,
        custom_aggs: dict[str, Callable[..., Any]] | None = None,
    ):
        self._template_vars: dict[str, Any] = {}
        self._evaluation_cost_usd: float = 0.0
        self._evaluation_cost_lock = threading.Lock()
        # Per-path cache of match-key field selections (one LLM call per list, reused per run).
        self._match_key_cache: dict[str, list[str] | None] = {}

        # Metric Registry: (expected, actual, params) -> tuple[float, bool]:
        self.metric_registry = {
            # legacy (deprecated) metrics
            "exact": lambda e, a, p: 1.0 if e == a else 0.0,
            "threshold": lambda e, a, p: 1.0 if (a or 0) >= p.get("min", 0) else 0.0,
            "comparator": comparator_metric,
            # standalone metrics -- register by strategy name directly
            "exact_compare": lambda e, a, p: _score_to_result(
                _exact_compare(
                    str(e), str(a),
                    case_sensitive=p.get("case_sensitive", False),
                    ignore_spaces=p.get("ignore_spaces", False),
                ),
                p,
            ),
            "text_similarity": lambda e, a, p: _score_to_result(
                _text_similarity_compare(
                    str(e), str(a),
                    metric=p.get("metric", "fuzz_ratio"),
                    threshold=p.get("threshold"),
                    case_sensitive=p.get("case_sensitive", False),
                    ignore_spaces=p.get("ignore_spaces", False),
                    embedding_model=p.get("embedding_model", "text-embedding-3-small"),
                ),
                p,
            ),
            "llm": lambda e, a, p: _score_to_result(
                _llm_compare(
                    str(e), str(a),
                    model=p.get("model", "gpt-4o-mini"),
                    prompt_template=p.get("prompt_template"),
                    prompt_extra_vars=p.get("_template_vars"),
                ),
                p,
            ),
            "embedding": lambda e, a, p: _score_to_result(
                _embedding_compare(
                    str(e), str(a),
                    model=p.get("model", "text-embedding-3-small"),
                    threshold=p.get("threshold"),
                ),
                p,
            ),
        }
        if custom_metrics:
            self.metric_registry.update(custom_metrics)

        # Aggregator Registry: (List[EvalResult]) -> float
        self.agg_registry = {
            "weighted_avg": self._weighted_avg,
            "min": lambda items: min((res.score for res in items), default=0.0),
            "max": lambda items: max((res.score for res in items), default=0.0),
        }
        if custom_aggs:
            self.agg_registry.update(custom_aggs)

    def _weighted_avg(self, items: list[EvalResult]) -> float:
        total_weight = sum(res.weight for res in items)
        if total_weight == 0:
            return 0.0
        return sum(res.score * res.weight for res in items) / total_weight

    def _weighted_avg_field(self, items: list[EvalResult], field: str) -> float:
        total_weight = sum(res.weight for res in items)
        if total_weight == 0:
            return 0.0
        return sum(getattr(res, field) * res.weight for res in items) / total_weight

    def _record_evaluation_cost(self, cost_usd: float) -> None:
        with self._evaluation_cost_lock:
            self._evaluation_cost_usd += float(cost_usd or 0.0)

    @property
    def evaluation_cost(self) -> float:
        with self._evaluation_cost_lock:
            return self._evaluation_cost_usd

    def _comparator_metric(self, expected: Any, actual: Any, params: dict[str, Any]) -> tuple[float, bool]:
        score, is_correct, cost_usd, call_count = _run_comparator(expected, actual, params)
        if cost_usd:
            self._record_evaluation_cost(cost_usd)
        return score, is_correct

    def evaluate(
        self,
        config_dict: dict[str, Any] | str,
        expected: dict[str, Any] | str,
        actual: dict[str, Any] | str,
        extra_template_vars: dict[str, Any] | None = None,
    ) -> EvalResult:
        if isinstance(config_dict, str):
            config_dict = json.loads(config_dict)
        if isinstance(expected, str):
            expected = json.loads(expected)
        if isinstance(actual, str):
            actual = json.loads(actual)

        self._template_vars: dict[str, Any] = extra_template_vars or {}
        config = FieldConfig.model_validate(config_dict)
        return self._recurse(config, expected, actual, "root")

    def _recurse(self, config: FieldConfig, exp: Any, act: Any, path: str) -> EvalResult:
        if config.type == "object":
            return self._eval_object(config, exp, act, path)
        elif config.type == "list":
            return self._eval_list(config, exp, act, path)
        return self._eval_leaf(config, exp, act, path)

    def _eval_leaf(self, config: FieldConfig, exp: Any, act: Any, path: str) -> EvalResult:
        missing_exp = exp is None
        missing_act = act is None

        # Both missing → neutral
        if missing_exp and missing_act:
            return EvalResult(
                path=path,
                score=1.0,
                weight=0.0,
                metric="optional_missing_both",
                tp=0.0 if config.optional else 1.0,
                is_correct=True,
            )

        # Missing expected
        if missing_exp:
            return EvalResult(
                path=path,
                score=1.0 if config.optional else 0.0,
                weight=0.0 if config.optional else config.weight,
                metric="unexpected_field",
                fp=0.0 if config.optional else 1.0,
                is_correct=config.optional,
            )

        # Missing actual
        if missing_act:
            return EvalResult(
                path=path,
                score=1.0 if config.optional else 0.0,
                weight=0.0 if config.optional else config.weight,
                metric="missing_field",
                fn=0.0 if config.optional else 1.0,
                is_correct=False,
            )

        # Normal metric evaluation
        m_cfg = _leaf_mc(config)
        metric_fn = self.metric_registry.get(m_cfg.metric, self.metric_registry["exact"])
        effective_params = {**m_cfg.params, "_template_vars": self._template_vars}
        result = metric_fn(exp, act, effective_params)
        if isinstance(result, tuple):
            score, is_correct = result
        else:
            score = result
            is_correct = score == 1.0

        return EvalResult(
            path=path,
            score=score,
            weight=config.weight,
            metric=m_cfg.metric,
            tp=1.0 if is_correct else 0.0,
            fp=0.0 if is_correct else 1.0,
            fn=0.0 if is_correct else 1.0,
            precision=1.0 if is_correct else 0.0,
            recall=1.0 if is_correct else 0.0,
            is_correct=is_correct,
            params=m_cfg.params,          
        )


    def _eval_object(self, config: FieldConfig, exp: dict[str, Any], act: dict[str, Any], path: str) -> EvalResult:
        exp, act = (exp or {}), (act or {})
        o_cfg = _object_mc(config)
        fields = config.fields or {}
        child_results = {}
        eval_results = []
        for key in exp.keys():
            field_cfg = fields.get(key, FieldConfig())
            res = self._recurse(field_cfg, exp.get(key), act.get(key), f"{path}.{key}")
            child_results[key] = res
            eval_results.append(copy.deepcopy(res))

        agg_fn = self.agg_registry.get(o_cfg.propagation, self._weighted_avg)
        return EvalResult(
            path=path,
            score=agg_fn(eval_results),
            weight=config.weight,
            metric=f"agg:{o_cfg.propagation}",
            children=child_results,
            is_correct=all(res.is_correct for res in eval_results) and (bool(exp) or not bool(act)),
            precision=self._weighted_avg_field(eval_results, "precision"),
            recall=self._weighted_avg_field(eval_results, "recall"),
        )

    def _eval_list(self, config: FieldConfig, exp: list[Any], act: list[Any], path: str) -> EvalResult:
        m_cfg = _list_mc(config)
        exp, act = (exp or []), (act or [])

        if not m_cfg.ordered and m_cfg.item_logic:
            custom_names = frozenset(set(self.metric_registry.keys()) - _BUILTIN_METRIC_NAMES)
            allowed = set(m_cfg.allow_expensive_comparisons_for or [])
            unallowed_expensive = [
                issue for issue in _scan_item_logic_for_expensive_metrics(
                    m_cfg.item_logic, path, custom_names
                )
                if issue["relative_path"] not in allowed
            ]
            if unallowed_expensive:
                fields = ", ".join(f'"{i["relative_path"]}"' for i in unallowed_expensive)
                raise ValueError(
                    f"Unordered list at '{path}' uses 3rd-party metric(s) on [{fields}] without "
                    f"explicit opt-in. These comparisons call an external API and add cost per document. "
                    f"Add these paths to allow_expensive_comparisons_for on the list's metric_config if "
                    f"you accept the extra cost, or replace the metric(s) with ones that don't call "
                    f"3rd-party APIs.\n\n"
                )

        if m_cfg.ordered:
            return self._eval_list_ordered(config, exp, act, path, m_cfg)

        if _item_logic_has_llm_judge_leaf(m_cfg.item_logic):
            logger.info(
                "Unordered list at '%s': aligning items by embedding + Hungarian assignment "
                "(no LLM aligner calls) because at least one leaf below uses an LLM-judge metric.",
                path,
            )
            return self._eval_list_unordered_with_alignment(config, exp, act, path, m_cfg)

        return self._eval_list_unordered(config, exp, act, path, m_cfg)

    def _eval_list_ordered(self, config: FieldConfig, exp: list[Any], act: list[Any], path: str, m_cfg: "ListMetricConfig") -> EvalResult:
        item_logic = m_cfg.item_logic
        assert item_logic is not None
        alignments: list[AlignmentItem] = []
        min_len = min(len(exp), len(act))
        for i in range(min_len):
            res = self._recurse(item_logic, exp[i], act[i], f"{path}[{i}]")
            alignments.append(AlignmentItem(e_idx=i, a_idx=i, score=res.score, result=res))

        matched = sum(1 for a in alignments if a.result.is_correct)

        # Add unmatched expected items so leaf-level fn is counted
        for i in range(min_len, len(exp)):
            res = self._recurse(item_logic, exp[i], None, f"{path}[{i}]")
            alignments.append(AlignmentItem(e_idx=i, a_idx=-1, score=0.0, result=res))

        precision = matched / len(act) if act else 0
        recall = matched / len(exp) if exp else 0

        soft_tp = sum(a.result.score for a in alignments if a.a_idx >= 0)
        soft_precision = soft_tp / len(act) if act else 0.0
        soft_recall = soft_tp / len(exp) if exp else 0.0
        soft_f1 = (2 * soft_precision * soft_recall) / (soft_precision + soft_recall) if (soft_precision + soft_recall) > 0 else 0.0

        return EvalResult(
            path=path,
            score=soft_f1,
            weight=config.weight,
            metric="list_ordered_f1",
            alignment=alignments,
            precision=precision,
            recall=recall,
            fp=len(act) - matched,
            fn=len(exp) - matched,
            tp=matched,
            details={"matched_items": min_len},
            is_correct=(len(exp) == len(act) and all(a.result.is_correct for a in alignments)),
        )

    def _eval_list_unordered(self, config: FieldConfig, exp: list[Any], act: list[Any], path: str, m_cfg: "ListMetricConfig") -> EvalResult:
        item_logic = m_cfg.item_logic
        assert item_logic is not None
        potential_matches: list[AlignmentItem] = []
        for i, e_item in enumerate(exp):
            for j, a_item in enumerate(act):
                # Pre-filter: if required_fields_to_match is set, check those fields
                # first. If any required field doesn't match, skip this pair — the
                # actual item can't correspond to this expected item (counts as FP).
                if (
                    m_cfg.required_fields_to_match
                    and item_logic.fields
                    and isinstance(e_item, dict)
                    and isinstance(a_item, dict)
                ):
                    required_ok = True
                    for rf in m_cfg.required_fields_to_match:
                        rf_config = item_logic.fields.get(rf, FieldConfig())
                        rf_res = self._recurse(
                            rf_config,
                            e_item.get(rf),
                            a_item.get(rf),
                            f"{path}[{i}].{rf}",
                        )
                        if not rf_res.is_correct:
                            required_ok = False
                            break
                    if not required_ok:
                        continue

                res = self._recurse(item_logic, e_item, a_item, f"{path}[{i}]")
                potential_matches.append(AlignmentItem(e_idx=i, a_idx=j, score=res.score, result=res))

        potential_matches.sort(key=lambda x: x.score, reverse=True)
        matched_e: set[int] = set()
        matched_a: set[int] = set()
        alignments: list[AlignmentItem] = []

        for m in potential_matches:
            if m.e_idx not in matched_e and m.a_idx not in matched_a:
                if m.score >= m_cfg.match_threshold:
                    matched_e.add(m.e_idx)
                    matched_a.add(m.a_idx)
                    alignments.append(m)

        # Add unmatched expected items so leaf-level fn is counted
        for i in range(len(exp)):
            if i not in matched_e:
                res = self._recurse(item_logic, exp[i], None, f"{path}[{i}]")
                alignments.append(AlignmentItem(e_idx=i, a_idx=-1, score=0.0, result=res))

        precision = len(matched_a) / len(act) if act else 0
        recall = len(matched_e) / len(exp) if exp else 0

        soft_tp = sum(a.result.score for a in alignments if a.a_idx >= 0)
        soft_precision = soft_tp / len(act) if act else 0.0
        soft_recall = soft_tp / len(exp) if exp else 0.0
        soft_f1 = (2 * soft_precision * soft_recall) / (soft_precision + soft_recall) if (soft_precision + soft_recall) > 0 else 0.0

        all_correct = len(matched_e) == len(exp) and len(matched_a) == len(act) and all(a.result.is_correct for a in alignments)
        return EvalResult(
            path=path,
            score=1.0 if all_correct else soft_f1,
            weight=config.weight,
            metric="list_greedy_f1",
            alignment=alignments,
            precision=precision,
            recall=recall,
            fp=len(act) - len(matched_a),
            fn=len(exp) - len(matched_e),
            tp=len(matched_e),
            details={"matched_items": len(matched_e)},
            is_correct=all_correct,
        )

    def _select_match_key_fields(
        self,
        m_cfg: "ListMetricConfig",
        exp: list[Any],
        act: list[Any],
        path: str,
    ) -> list[str] | None:
        """Decide which item fields to embed when aligning candidates for this list.

        Resolution order: explicit ``match_key_fields`` on the config wins; otherwise a single
        small-LLM call picks the identity-bearing fields from a few sample items. The result is
        cached per evaluation path -- the item schema is constant across a run, so this is
        roughly one LLM call per list for the whole run. Any failure (non-dict items, API
        error, empty/invalid selection) caches None, which tells :func:`_match_key_text` to
        fall back to the item's top-level scalar fields.

        :param m_cfg: The list's metric config.
        :param exp: Expected list of items.
        :param act: Actual list of items.
        :param path: Evaluation path, used as the cache key and in log messages.
        :return: Field names to embed, or None to use the item's top-level scalar fields.
        """
        if m_cfg.match_key_fields:
            return m_cfg.match_key_fields

        if path in self._match_key_cache:
            return self._match_key_cache[path]

        samples = [it for it in (exp + act) if isinstance(it, dict)][:5]
        if not samples:
            self._match_key_cache[path] = None
            return None

        # Nested items: skip the LLM field selection and return None, which tells
        # _match_key_text to characterize the item by its top-level scalar fields only (no
        # recursion into nested lists/dicts). The LLM selection only earns its keep on flat
        # items, where it can drop low-information scalar fields (enums, flags) that would
        # otherwise dilute the cosine signal.
        if any(isinstance(v, (dict, list)) for it in samples for v in it.values()):
            self._match_key_cache[path] = None
            return None

        field_names = sorted({k for it in samples for k in it.keys()})
        if len(field_names) <= 1:
            selected = field_names or None
            self._match_key_cache[path] = selected
            return selected

        samples_repr = "\n\n".join(
            "ITEM:\n"
            + "\n".join(f"  {k}: {_truncate_for_prompt(it[k])}" for k in field_names if k in it)
            for it in samples[:3]
        )
        prompt = (
            "You are configuring an automated evaluation pipeline. Below are sample items from "
            "a list. Pick the subset of fields that together IDENTIFY an item -- the fields a "
            "human would use to tell one item apart from another (e.g. a title, name, or "
            "description). Exclude low-information fields whose values repeat across items "
            "(enums, booleans, status flags, priorities) and bookkeeping fields (ids, indices) "
            "unless they are the only identifier.\n\n"
            f"Available fields: {field_names}\n\n"
            f"Sample items:\n{samples_repr}\n\n"
            "Return JSON only, matching the schema (a list of field names drawn from the "
            "available fields)."
        )

        selected: list[str] | None = None
        try:
            response = cast(ModelResponse, completion(
                model=m_cfg.match_key_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format=_MatchKeyFields,
            ))
            try:
                self._record_evaluation_cost(completion_cost(completion_response=response))
            except Exception:  # noqa: BLE001
                logger.warning("match_key_cost_tracking_failed at '%s'", path)

            content = response.choices[0].message.content
            if not content:
                raise ValueError("empty match-key selection response")
            parsed = _MatchKeyFields.model_validate_json(content)
            valid = [f for f in parsed.fields if f in field_names]
            selected = valid or None
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Match-key field selection failed at '%s' (model=%s); embedding whole item: %s",
                path, m_cfg.match_key_model, e,
            )
            selected = None

        logger.info("Match-key fields for '%s': %s", path, selected if selected else "<whole item>")
        self._match_key_cache[path] = selected
        return selected

    def _embed_texts(self, texts: list[str], model: str, path: str) -> list[list[float]]:
        """Embed a batch of texts in a single API call and record the spend.

        :param texts: Texts to embed (expected items followed by actual items).
        :param model: Embedding model name.
        :param path: Evaluation path, used in log messages.
        :return: One embedding vector per input text, in input order.
        """
        response = embedding(model=model, input=texts)
        try:
            self._record_evaluation_cost(completion_cost(completion_response=response))
        except Exception:  # noqa: BLE001
            logger.warning("embed_cost_tracking_failed at '%s'", path)

        return [d["embedding"] for d in response.data]

    def _align_by_hungarian(
        self,
        sims: list[list[float]],
        candidates_per_e: list[list[int]],
        lo: float,
    ) -> dict[int, int | None]:
        """Globally optimal one-to-one assignment over the cosine matrix (no LLM calls).

        Builds a cost matrix (cost = 1 - cosine) over only the pairs that pass the
        ``required_fields_to_match`` pre-filter and clear the ``lo`` floor; every other pair is
        forbidden with a prohibitively large cost. ``scipy.optimize.linear_sum_assignment`` then
        finds the assignment minimizing total cost (i.e. maximizing total cosine), and any
        forbidden pair it is forced to pick on a rectangular matrix is dropped to unmatched.

        :param sims: Expected-by-actual cosine similarity matrix.
        :param candidates_per_e: Per expected item, the allowed actual indices (pre-filter).
        :param lo: Minimum cosine for a pair to be eligible.
        :return: Mapping of each expected index to a chosen actual index or None.
        """
        n_exp = len(sims)
        n_act = len(sims[0]) if n_exp else 0
        e_assignment: dict[int, int | None] = {i: None for i in range(n_exp)}
        if n_exp == 0 or n_act == 0:
            return e_assignment

        forbid = 1e6
        allowed = [set(c) for c in candidates_per_e]
        cost = np.full((n_exp, n_act), forbid, dtype=float)
        for i in range(n_exp):
            for j in allowed[i]:
                if sims[i][j] >= lo:
                    cost[i][j] = 1.0 - sims[i][j]

        rows, cols = linear_sum_assignment(cost)
        for i, j in zip(rows, cols):
            if cost[i][j] < forbid:  # a real, eligible pair (not a forced forbidden slot)
                e_assignment[int(i)] = int(j)

        return e_assignment

    def _align_by_embedding(
        self,
        exp: list[Any],
        act: list[Any],
        candidates_per_e: list[list[int]],
        path: str,
        m_cfg: "ListMetricConfig",
    ) -> tuple[dict[int, int | None], dict[str, Any]]:
        """Align expected->actual items by optimal one-to-one assignment over cosine similarity.

        One batched embedding call scores every expected item against every actual item by
        cosine similarity (over the match-key rendering of each item), then
        :meth:`_align_by_hungarian` finds the globally optimal assignment, leaving pairs below
        ``align_lo`` unmatched. No LLM aligner calls are made. If the embedding call fails at
        runtime, scoring degrades to all-unmatched (logged) rather than blocking the run.

        :param exp: Expected list of items.
        :param act: Actual list of items.
        :param candidates_per_e: Per expected item, the actual indices passing the
            ``required_fields_to_match`` pre-filter.
        :param path: Current evaluation path.
        :param m_cfg: The list's metric config (supplies the embedding model and align_lo).
        :return: A ``(e_assignment, stats)`` tuple, where ``e_assignment`` maps each expected
            index to a chosen actual index or None, and ``stats`` carries diagnostic counts.
        """
        n_exp, n_act = len(exp), len(act)
        e_assignment: dict[int, int | None] = {i: None for i in range(n_exp)}
        stats: dict[str, Any] = {
            "n_matched": 0, "n_embed_calls": 0,
            "embedding_ok": False, "match_key_fields": None,
        }

        if n_exp == 0 or n_act == 0:
            return e_assignment, stats

        # One batched embedding call -> cosine matrix. On failure, leave everything unmatched
        # rather than blocking the run.
        try:
            fields = self._select_match_key_fields(m_cfg, exp, act, path)
            texts = [_match_key_text(it, fields) for it in exp] + [
                _match_key_text(it, fields) for it in act
            ]
            vecs = self._embed_texts(texts, m_cfg.align_embedding_model, path)
            exp_vecs, act_vecs = vecs[:n_exp], vecs[n_exp:]
            sims = [
                [_cosine(exp_vecs[i], act_vecs[j]) for j in range(n_act)]
                for i in range(n_exp)
            ]
            stats["n_embed_calls"] = 1
            stats["embedding_ok"] = True
            stats["match_key_fields"] = fields
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "Embedding alignment unavailable at '%s'; leaving items unmatched: %s", path, e,
            )
            return e_assignment, stats

        e_assignment = self._align_by_hungarian(sims, candidates_per_e, m_cfg.align_lo)
        stats["n_matched"] = sum(1 for v in e_assignment.values() if v is not None)
        return e_assignment, stats

    def _eval_list_unordered_with_alignment(
        self,
        config: FieldConfig,
        exp: list[Any],
        act: list[Any],
        path: str,
        m_cfg: "ListMetricConfig",
    ) -> EvalResult:
        """Evaluate an unordered list whose items contain LLM-judge leaves, aligning items by
        optimal assignment over embedding cosine similarity, then judging each matched pair.

        Alignment is delegated to :meth:`_align_by_embedding`: one batched embedding call scores
        all expected×actual pairs by cosine similarity and the Hungarian algorithm finds the
        globally optimal one-to-one assignment — no LLM aligner calls. Once alignment is decided,
        each matched pair is evaluated by recursing ``item_logic`` so every leaf (LLM-judge,
        embedding, or local) runs through its own configured metric with its own prompt template.

        :param config: The list's :class:`FieldConfig`.
        :param exp: Expected list of items.
        :param act: Actual (predicted) list of items.
        :param path: Current evaluation path.
        :param m_cfg: The list's metric config.
        :return: An :class:`EvalResult` for the list.
        """
        if not exp and not act:
            return EvalResult(
                path=path, score=1.0, weight=config.weight,
                metric="list_embed_hungarian_f1",
                alignment=[], precision=0.0, recall=0.0,
                tp=0, fp=0, fn=0,
                details={"matched_items": 0, "aligner_used": True},
                is_correct=True,
            )

        if not exp:
            return EvalResult(
                path=path, score=0.0, weight=config.weight,
                metric="list_embed_hungarian_f1",
                alignment=[], precision=0.0, recall=0.0,
                fp=len(act), fn=0, tp=0,
                details={"matched_items": 0, "aligner_used": True},
                is_correct=False,
            )

        item_logic = m_cfg.item_logic
        assert item_logic is not None
        required_fields = m_cfg.required_fields_to_match or []

        # Per-E candidate filtering by required_fields_to_match.  Skipping a candidate
        # here is functionally identical to the LLM returning matched_a_idx=null for it,
        # but cheaper and deterministic.
        candidates_per_e: list[list[int]] = []
        for i, e_item in enumerate(exp):
            if (
                required_fields
                and item_logic
                and item_logic.fields
                and isinstance(e_item, dict)
            ):
                cands: list[int] = []
                for j, a_item in enumerate(act):
                    if not isinstance(a_item, dict):
                        continue
                    ok = True
                    for rf in required_fields:
                        rf_cfg = item_logic.fields.get(rf, FieldConfig())
                        rf_res = self._recurse(
                            rf_cfg, e_item.get(rf), a_item.get(rf),
                            f"{path}[{i}].{rf}",
                        )
                        if not rf_res.is_correct:
                            ok = False
                            break
                    if ok:
                        cands.append(j)
                candidates_per_e.append(cands)
            else:
                candidates_per_e.append(list(range(len(act))))

        # Optimal one-to-one assignment over embedding cosine decides the alignment
        # (no LLM aligner calls; see _align_by_embedding).
        max_workers = min(32, max(1, len(exp)))
        e_assignment, align_stats = self._align_by_embedding(
            exp, act, candidates_per_e, path, m_cfg
        )

        # Run the leaf-judging recursion for every expected item in parallel.  Each
        # _recurse call makes its own LLM-judge / embedding / local-metric calls
        # internally; running 32 of them concurrently lets the API round-trips overlap
        # rather than blocking serially on the main thread.  Output order is preserved
        # by ex.map() so the alignments list stays e_idx-ordered.
        def _judge_one(i: int) -> AlignmentItem:
            a_idx = e_assignment[i]
            if a_idx is not None:
                res = self._recurse(item_logic, exp[i], act[a_idx], f"{path}[{i}]")
                return AlignmentItem(e_idx=i, a_idx=a_idx, score=res.score, result=res)
            res = self._recurse(item_logic, exp[i], None, f"{path}[{i}]")
            return AlignmentItem(e_idx=i, a_idx=-1, score=0.0, result=res)

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            alignments: list[AlignmentItem] = list(ex.map(_judge_one, range(len(exp))))

        matched_e: set[int] = {a.e_idx for a in alignments if a.a_idx >= 0}
        matched_a: set[int] = {a.a_idx for a in alignments if a.a_idx >= 0}

        precision = len(matched_a) / len(act) if act else 0
        recall = len(matched_e) / len(exp) if exp else 0

        soft_tp = sum(a.result.score for a in alignments if a.a_idx >= 0)
        soft_precision = soft_tp / len(act) if act else 0.0
        soft_recall = soft_tp / len(exp) if exp else 0.0
        soft_f1 = (2 * soft_precision * soft_recall) / (soft_precision + soft_recall) if (soft_precision + soft_recall) > 0 else 0.0

        all_correct = (
            len(matched_e) == len(exp)
            and len(matched_a) == len(act)
            and all(a.result.is_correct for a in alignments)
        )

        return EvalResult(
            path=path,
            score=1.0 if all_correct else soft_f1,
            weight=config.weight,
            metric="list_embed_hungarian_f1",
            alignment=alignments,
            precision=precision,
            recall=recall,
            fp=len(act) - len(matched_a),
            fn=len(exp) - len(matched_e),
            tp=len(matched_e),
            details={
                "matched_items": len(matched_e),
                "aligner_used": True,
                "align_method": "embed_hungarian" if align_stats["embedding_ok"] else "embedding_unavailable",
                "embedding_model": m_cfg.align_embedding_model,
                "match_key_fields": align_stats["match_key_fields"],
                "n_matched": align_stats["n_matched"],
                "n_embed_calls": align_stats["n_embed_calls"],
            },
            is_correct=all_correct,
        )
