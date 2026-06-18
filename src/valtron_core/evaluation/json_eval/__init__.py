from __future__ import annotations

from valtron_core.evaluation.json_eval.evaluator import JsonEvaluator
from valtron_core.evaluation.json_eval.registries import comparator_metric
from valtron_core.evaluation.json_eval.schema import (
    DEFAULT_ALIGN_EMBEDDING_MODEL,
    DEFAULT_ALIGN_LO,
    DEFAULT_MATCH_KEY_MODEL,
    MATCH_KEY_MAX_CHARS,
    MAX_LIST_LENGTH_FOR_EXPENSIVE_COMPARE,
    AlignmentItem,
    EvalResult,
    ExpensiveListComparisonError,
    FieldConfig,
    LeafMetricConfig,
    ListMetricConfig,
    ObjectMetricConfig,
)
from valtron_core.evaluation.json_eval.validation import (
    collect_field_metric_llm_models,
    find_expensive_unordered_list_fields,
)

__all__ = [
    # main class
    "JsonEvaluator",
    # config models
    "FieldConfig",
    "LeafMetricConfig",
    "ObjectMetricConfig",
    "ListMetricConfig",
    # result models
    "EvalResult",
    "AlignmentItem",
    # exception
    "ExpensiveListComparisonError",
    # public functions
    "comparator_metric",
    "find_expensive_unordered_list_fields",
    "collect_field_metric_llm_models",
    # constants
    "MAX_LIST_LENGTH_FOR_EXPENSIVE_COMPARE",
    "DEFAULT_ALIGN_EMBEDDING_MODEL",
    "DEFAULT_MATCH_KEY_MODEL",
    "DEFAULT_ALIGN_LO",
    "MATCH_KEY_MAX_CHARS",
]
