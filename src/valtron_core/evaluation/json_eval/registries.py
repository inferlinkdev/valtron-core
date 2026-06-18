from __future__ import annotations
from typing import Any
import warnings

from valtron_core.evaluation.comparison_functions import (
    Comparator,
    element_compare_category,
)
from valtron_core.evaluation.comparisons import (
    _exact_compare,
    _embedding_compare,
    _llm_compare,
    _text_similarity_compare,
)
from valtron_core.evaluation.json_eval.schema import EvalResult


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


def _weighted_avg_field(items: list[EvalResult], field: str) -> float:
    total_weight = sum(res.weight for res in items)
    if total_weight == 0:
        return 0.0
    return sum(getattr(res, field) * res.weight for res in items) / total_weight


def _llm_metric(
    expected: Any, actual: Any, params: dict[str, Any]
) -> tuple[float, bool, dict[str, Any]]:
    model = params.get("model", "gpt-4o-mini")
    match, cost_usd = _llm_compare(
        str(actual), str(expected),
        model=model,
        prompt_template=params.get("prompt_template"),
        prompt_extra_vars=params.get("_template_vars"),
    )
    score, is_correct = _score_to_result(match, params)
    return score, is_correct, {"cost": cost_usd, "model": model}


def _embedding_metric(
    expected: Any, actual: Any, params: dict[str, Any]
) -> tuple[float, bool, dict[str, Any]]:
    model = params.get("model", "text-embedding-3-small")
    similarity, cost_usd = _embedding_compare(
        str(actual), str(expected),
        model=model,
        threshold=params.get("threshold"),
    )
    score, is_correct = _score_to_result(similarity, params)
    return score, is_correct, {"cost": cost_usd, "model": model}


DEFAULT_METRIC_REGISTRY: dict[str, Any] = {
    # legacy (deprecated) metrics
    "exact": lambda e, a, p: 1.0 if e == a else 0.0,
    "threshold": lambda e, a, p: 1.0 if (a or 0) >= p.get("min", 0) else 0.0,
    "comparator": comparator_metric,
    # standalone metrics
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
    "llm": _llm_metric,
    "embedding": _embedding_metric,
}

DEFAULT_AGG_REGISTRY: dict[str, Any] = {
    "weighted_avg": lambda items: _weighted_avg_field(items, "score"),
    "min": lambda items: min((res.score for res in items), default=0.0),
    "max": lambda items: max((res.score for res in items), default=0.0),
}
