from __future__ import annotations
from typing import Literal, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator
import json
import copy

from valtron_core.evaluation.comparison_functions import Comparator, element_compare_uses_third_party

MAX_LIST_LENGTH_FOR_EXPENSIVE_COMPARE = 4

def comparator_metric(expected, actual, params) -> tuple[float, bool]:
    comp = Comparator(
        element_compare=params.get("element_compare", "exact"),
        text_similarity_threshold=params.get("text_similarity_threshold", None),
        text_similarity_metric=params.get("text_similarity_metric", "fuzz_ratio"),
        llm_model=params.get("llm_model", "gpt-4o-mini"),
        embedding_model=params.get("embedding_model", "text-embedding-3-small"),
        embedding_threshold=params.get("embedding_threshold", None),
        case_sensitive=params.get("case_sensitive", False),
        ignore_spaces=params.get("ignore_spaces", False),
    )
    compare_result = comp.compare(expected, actual)

    if isinstance(compare_result, bool):
        return (1.0 if compare_result else 0.0), compare_result

    threshold = params.get("comparison_threshold", None) or params.get("text_similarity_threshold", None)
    if threshold is not None:
        is_correct = compare_result >= threshold
        return compare_result, is_correct

    return compare_result, True


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
    def _route_metric_config(cls, data):
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
# Built-in metric 3rd-party API declarations
# ---------------------------------------------------------------------------
# Every metric in JsonEvaluator.metric_registry MUST be handled in
# _check_builtin_metric_expensive() below.  When you add a new built-in
# metric, add a branch there and declare whether it calls a 3rd-party API.
# Omitting it raises NotImplementedError at pre-flight time.

_BUILTIN_METRIC_NAMES: frozenset[str] = frozenset({"exact", "threshold", "comparator"})


def _check_builtin_metric_expensive(metric_name: str, params: dict) -> tuple[bool, str]:
    """Return ``(is_expensive, description)`` for a built-in metric.

    DEVELOPER NOTE — when adding a new metric to ``JsonEvaluator.metric_registry``:
      1. Add a branch here.
      2. Return ``(True, "<description>")`` if it calls a 3rd-party API,
         ``(False, "")`` if it is entirely local.
    Omitting this raises ``NotImplementedError`` at pre-flight time.
    """
    if metric_name in ("exact", "threshold"):
        return False, ""

    if metric_name == "comparator":
        element_compare = params.get("element_compare", "exact")
        return element_compare_uses_third_party(element_compare, params)

    raise NotImplementedError(
        f"Built-in metric '{metric_name}' has no 3rd-party API declaration.\n"
        "When adding a new metric to JsonEvaluator.metric_registry you MUST:\n"
        "  1. Add a branch in _check_builtin_metric_expensive() in json_eval.py\n"
        "     and return (True, description) if it calls a 3rd-party API, or (False, '') if not.\n"
        "This check exists to prevent accidental n²-cost list evaluations."
    )


def _scan_item_logic_for_expensive_metrics(
    config: "FieldConfig",
    path: str,
    custom_metric_names: frozenset[str],
    relative_path: str = "",
) -> list[dict]:
    """Recursively scan a FieldConfig subtree for expensive metrics.

    ``relative_path`` tracks the position within the current list's item_logic,
    used to populate ``allow_expensive_comparisons_for`` suggestions.  For a list
    of primitives (item_logic is itself a leaf) the relative_path is ``"$item"``.

    Returns a list of issue dicts with keys:
      metric_path, relative_path, type ("builtin" | "custom"), metric, description
    """
    issues: list[dict] = []

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
                "description": (
                    f"custom metric '{mc.metric}' — its implementation is user-defined "
                    "and may call a 3rd-party service"
                ),
            })
        elif mc.metric in _BUILTIN_METRIC_NAMES:
            expensive, desc = _check_builtin_metric_expensive(mc.metric, mc.params)
            if expensive:
                issues.append({
                    "metric_path": path,
                    "relative_path": display_rel,
                    "type": "builtin",
                    "metric": mc.metric,
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
            # Nested list — relative_path resets to "" for the inner list's item_logic
            issues.extend(
                _scan_item_logic_for_expensive_metrics(
                    mc.item_logic, f"{path}[]", custom_metric_names, ""
                )
            )

    return issues


def _find_expensive_lists_recursive(
    config: "FieldConfig",
    path: str,
    custom_metric_names: frozenset[str],
    issues: list,
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


def find_expensive_unordered_list_fields(
    config_dict: dict,
    custom_metric_names: set[str] | None = None,
) -> list[dict]:
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
    issues: list[dict] = []
    config = FieldConfig.model_validate(config_dict)
    _find_expensive_lists_recursive(
        config, "root", frozenset(custom_metric_names or set()), issues
    )
    return issues


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
        # Metric Registry: (expected, actual, params) -> tuple[float, bool]:
        self.metric_registry = {
            "exact": lambda e, a, p: 1.0 if e == a else 0.0,
            "threshold": lambda e, a, p: 1.0 if (a or 0) >= p.get("min", 0) else 0.0,
            "comparator": comparator_metric,
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

    def evaluate(
        self, config_dict: dict | str, expected: dict | str, actual: dict | str
    ) -> EvalResult:
        if isinstance(config_dict, str):
            config_dict = json.loads(config_dict)
        if isinstance(expected, str):
            expected = json.loads(expected)
        if isinstance(actual, str):
            actual = json.loads(actual)

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
        result = metric_fn(exp, act, m_cfg.params)
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


    def _eval_object(self, config: FieldConfig, exp: dict, act: dict, path: str) -> EvalResult:
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

    def _eval_list(self, config: FieldConfig, exp: list, act: list, path: str) -> EvalResult:
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
                    f"explicit opt-in. This causes n^2 API calls per document. "
                    f"Add these paths to allow_expensive_comparisons_for on the list's metric_config if you accept the fact that n^2 comparisons will be made per document. Otherwise, replace the metric(s) with ones that don't invoked 3rd party APIs.\n\n"
                )

        if m_cfg.ordered:
            return self._eval_list_ordered(config, exp, act, path, m_cfg)
        else:
            return self._eval_list_unordered(config, exp, act, path, m_cfg)

    def _eval_list_ordered(self, config: FieldConfig, exp: list, act: list, path: str, m_cfg) -> EvalResult:
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
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return EvalResult(
            path=path,
            score=f1,
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

    def _eval_list_unordered(self, config: FieldConfig, exp: list, act: list, path: str, m_cfg) -> EvalResult:
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
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        all_correct = len(matched_e) == len(exp) and len(matched_a) == len(act) and all(a.result.is_correct for a in alignments)
        return EvalResult(
            path=path,
            score=1.0 if all_correct else f1,
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
