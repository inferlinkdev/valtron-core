from __future__ import annotations
from typing import Literal, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator
from concurrent.futures import ThreadPoolExecutor
import json
import copy
import logging
import os
import threading
import warnings
from litellm import completion, completion_cost

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
import os

logger = logging.getLogger(__name__)


MAX_LIST_LENGTH_FOR_EXPENSIVE_COMPARE = 4
DEFAULT_LLM_ALIGNER_MODEL = "gpt-4o-mini"


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
            list_keys = ('match_threshold', 'item_logic', 'required_fields_to_match', 'allow_expensive_comparisons_for')
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
        mc = config.metric_config
        if mc and mc.item_logic:
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
        mc = config.metric_config
        if mc and not mc.ordered:
            allowed = set(mc.allow_expensive_comparisons_for or [])
            if mc.item_logic:
                for issue in _scan_item_logic_for_expensive_metrics(
                    mc.item_logic, f"{path}[]", custom_metric_names
                ):
                    if issue["relative_path"] not in allowed:
                        issue["list_path"] = path
                        issues.append(issue)
        # Always recurse into item_logic in case there are nested lists
        if mc and mc.item_logic:
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


class _PerItemAlignment(BaseModel):
    """LLM response for one expected item's alignment.

    Tiny one-field schema designed to be trivially reliable under OpenAI structured-output
    strict mode.  Leaf-level judging is intentionally NOT collapsed into this call; it runs
    afterwards through the regular leaf-judge code path on each matched pair.

    :param matched_a_idx: Index into the candidate actual list, or None if no actual item
        plausibly matches the expected item.
    """
    matched_a_idx: int | None


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
        custom_metrics: dict[str, callable] | None = None,
        custom_aggs: dict[str, callable] | None = None,
    ):
        self._template_vars: dict[str, Any] = {}
        self._evaluation_cost_usd: float = 0.0
        self._evaluation_cost_lock = threading.Lock()

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
        m_cfg = config.metric_config
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
        child_results = {}
        eval_results = []
        for key in exp.keys():
            field_cfg = config.fields.get(key, FieldConfig())
            res = self._recurse(field_cfg, exp.get(key), act.get(key), f"{path}.{key}")
            child_results[key] = res
            eval_results.append(copy.deepcopy(res))

        agg_fn = self.agg_registry.get(config.metric_config.propagation, self._weighted_avg)
        return EvalResult(
            path=path,
            score=agg_fn(eval_results),
            weight=config.weight,
            metric=f"agg:{config.metric_config.propagation}",
            children=child_results,
            is_correct=all(res.is_correct for res in eval_results) and (bool(exp) or not bool(act)),
            precision=self._weighted_avg_field(eval_results, "precision"),
            recall=self._weighted_avg_field(eval_results, "recall"),
        )

    def _eval_list(self, config: FieldConfig, exp: list[Any], act: list[Any], path: str) -> EvalResult:
        m_cfg = config.metric_config
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
                "Unordered list at '%s': using LLM-based alignment because at least one leaf "
                "below uses an LLM-judge metric. This converts O(k^2) judge calls into O(k+1).",
                path,
            )
            return self._eval_list_unordered_with_llm_alignment(config, exp, act, path, m_cfg)

        return self._eval_list_unordered(config, exp, act, path, m_cfg)

    def _eval_list_ordered(self, config: FieldConfig, exp: list[Any], act: list[Any], path: str, m_cfg: "ListMetricConfig") -> EvalResult:
        alignments: list[AlignmentItem] = []
        min_len = min(len(exp), len(act))
        for i in range(min_len):
            res = self._recurse(m_cfg.item_logic, exp[i], act[i], f"{path}[{i}]")
            alignments.append(AlignmentItem(e_idx=i, a_idx=i, score=res.score, result=res))

        matched = sum(1 for a in alignments if a.result.is_correct)

        # Add unmatched expected items so leaf-level fn is counted
        for i in range(min_len, len(exp)):
            res = self._recurse(m_cfg.item_logic, exp[i], None, f"{path}[{i}]")
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
        potential_matches: list[AlignmentItem] = []
        for i, e_item in enumerate(exp):
            for j, a_item in enumerate(act):
                # Pre-filter: if required_fields_to_match is set, check those fields
                # first. If any required field doesn't match, skip this pair — the
                # actual item can't correspond to this expected item (counts as FP).
                if (
                    m_cfg.required_fields_to_match
                    and m_cfg.item_logic
                    and m_cfg.item_logic.fields
                    and isinstance(e_item, dict)
                    and isinstance(a_item, dict)
                ):
                    required_ok = True
                    for rf in m_cfg.required_fields_to_match:
                        rf_config = m_cfg.item_logic.fields.get(rf, FieldConfig())
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

                res = self._recurse(m_cfg.item_logic, e_item, a_item, f"{path}[{i}]")
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
                res = self._recurse(m_cfg.item_logic, exp[i], None, f"{path}[{i}]")
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

    def _llm_align_one_item(
        self,
        e_item: Any,
        act: list[Any],
        candidate_a_indices: list[int],
        path: str,
    ) -> int | None:
        """One LLM call: pick which actual item (if any) corresponds to a given expected
        item.  Returns just ``matched_a_idx`` or None.

        Each call is a single local decision over a trivial one-field schema, so it does
        not suffer the global-consistency failures of one-shot full-list alignment.
        Retries once on schema/parse failure; on second failure returns None (soft fail)
        so the surrounding document is not blocked.

        :param e_item: The expected item being aligned.
        :param act: Full list of actual items (used to read candidate items by index).
        :param candidate_a_indices: Indices into ``act`` that are eligible matches (after
            the ``required_fields_to_match`` pre-filter is applied).
        :param path: Evaluation path used in log messages.
        :return: The chosen index from ``candidate_a_indices``, or None.
        """
        if not candidate_a_indices:
            return None

        model = os.environ.get("VALTRON_ALIGNER_MODEL", DEFAULT_LLM_ALIGNER_MODEL)

        expected_repr = json.dumps(e_item, default=str, ensure_ascii=False, indent=2)
        actuals_repr = "\n".join(
            f"[A{j}]\n{json.dumps(act[j], default=str, ensure_ascii=False, indent=2)}"
            for j in candidate_a_indices
        )
        valid_indices_str = ", ".join(str(j) for j in candidate_a_indices)

        base_prompt = (
            "You are aligning one expected item against a list of candidate actual items, "
            "as part of an automated evaluation pipeline.\n\n"
            "Decide which (if any) of the candidate ACTUAL items corresponds to the EXPECTED "
            "item below.\n"
            f"- matched_a_idx must be one of: [{valid_indices_str}] (the integer after the `A` "
            "tag), or null if no candidate plausibly matches.\n"
            "- Be strict: prefer null over a doubtful match.\n\n"
            f"EXPECTED:\n{expected_repr}\n\n"
            f"ACTUAL CANDIDATES ({len(candidate_a_indices)}):\n{actuals_repr}\n\n"
            "Return JSON only, matching the schema."
        )

        last_err: Exception | None = None
        prompt = base_prompt
        valid_set = set(candidate_a_indices)

        for attempt in range(2):
            try:
                response = completion(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    response_format=_PerItemAlignment,
                )

                # Account for judge spend: each alignment call is a billable LLM
                # request and is the dominant cost of unordered-list scoring.
                # Record it before parsing so failed-parse retries are counted too.
                # Best-effort — a cost-lookup failure must never block scoring.
                try:
                    self._record_evaluation_cost(completion_cost(completion_response=response))
                except Exception:  # noqa: BLE001
                    logger.warning("evaluation_cost_tracking_failed at '%s'", path)

                content = response.choices[0].message.content
                parsed = _PerItemAlignment.model_validate_json(content)

                if parsed.matched_a_idx is not None and parsed.matched_a_idx not in valid_set:
                    raise ValueError(
                        f"matched_a_idx={parsed.matched_a_idx} not in candidate set "
                        f"{candidate_a_indices}"
                    )

                return parsed.matched_a_idx

            except Exception as e:
                last_err = e
                logger.warning(
                    "Per-item aligner attempt %d/2 failed at '%s' (model=%s): %s",
                    attempt + 1, path, model, e,
                )
                prompt = (
                    base_prompt
                    + f"\n\nYour previous response was invalid: {e}\nTry again, strictly "
                    "obeying the schema and the rules above."
                )

        logger.warning(
            "Per-item aligner failed at '%s' after 2 attempts (model=%s); treating as no "
            "match: %s",
            path, model, last_err,
        )
        return None

    def _eval_list_unordered_with_llm_alignment(
        self,
        config: FieldConfig,
        exp: list[Any],
        act: list[Any],
        path: str,
        m_cfg: "ListMetricConfig",
    ) -> EvalResult:
        """Evaluate an unordered list whose items contain LLM-judge leaves, using per-item
        iterative LLM alignment followed by the regular leaf-judge code path.

        For each expected item, issues one LLM call to pick the matching actual item (or
        none) — a tiny one-field decision that gpt-4o-mini handles reliably.  Calls run
        in parallel.  Once alignment is decided, each matched pair is evaluated by
        recursing ``item_logic`` so every leaf (LLM-judge, embedding, or local) runs
        through its own configured metric with its own prompt template.

        Conflict resolution: when multiple expected items claim the same actual item, the
        lowest e_idx wins; the others fall through to unmatched.  (We don't have a
        confidence signal from the alignment call, so the tiebreaker is just deterministic.)

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
                metric="list_llm_aligned_iter_f1",
                alignment=[], precision=0.0, recall=0.0,
                tp=0, fp=0, fn=0,
                details={"matched_items": 0, "aligner_used": True},
                is_correct=True,
            )

        if not exp:
            return EvalResult(
                path=path, score=0.0, weight=config.weight,
                metric="list_llm_aligned_iter_f1",
                alignment=[], precision=0.0, recall=0.0,
                fp=len(act), fn=0, tp=0,
                details={"matched_items": 0, "aligner_used": True},
                is_correct=False,
            )

        item_logic = m_cfg.item_logic
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

        # Fan out one LLM call per expected item — each is a one-field decision, immune
        # to the global-consistency failures of one-shot full-list alignment.
        max_workers = min(32, max(1, len(exp)))
        proposed_a_per_e: dict[int, int | None] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            future_to_idx = {
                ex.submit(
                    self._llm_align_one_item,
                    exp[i], act, candidates_per_e[i], f"{path}[{i}]",
                ): i
                for i in range(len(exp))
            }
            for fut, i in future_to_idx.items():
                proposed_a_per_e[i] = fut.result()

        # Conflict resolution: each actual item can be claimed at most once; lowest e_idx
        # wins.  Losing claimants fall through to unmatched.
        e_assignment: dict[int, int | None] = {i: None for i in range(len(exp))}
        a_taken: set[int] = set()
        for i in range(len(exp)):
            proposed = proposed_a_per_e.get(i)
            if proposed is not None and proposed not in a_taken:
                e_assignment[i] = proposed
                a_taken.add(proposed)

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
            metric="list_llm_aligned_iter_f1",
            alignment=alignments,
            precision=precision,
            recall=recall,
            fp=len(act) - len(matched_a),
            fn=len(exp) - len(matched_e),
            tp=len(matched_e),
            details={
                "matched_items": len(matched_e),
                "aligner_used": True,
                "aligner_model": os.environ.get("VALTRON_ALIGNER_MODEL", DEFAULT_LLM_ALIGNER_MODEL),
                "n_aligner_calls": len(exp),
            },
            is_correct=all_correct,
        )
