"""Standalone comparison functions for evaluating LLM predictions against ground truth.

Each function takes a predicted value, an expected value, and keyword-only configuration
params. They are designed to be registered directly in JsonEvaluator.metric_registry.

To add a new comparison method:
  1. Add a function here following the same signature pattern.
  2. In json_eval.py, add a branch in _check_builtin_metric_expensive, add the name to
     _BUILTIN_METRIC_NAMES, and register a lambda in JsonEvaluator.metric_registry.
"""

from __future__ import annotations

import json
import math
from typing import Any, Literal

import litellm
from litellm import completion, completion_cost, embedding
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore[import-untyped]
from nltk.translate.gleu_score import sentence_gleu  # type: ignore[import-untyped]
from pydantic import BaseModel
from rapidfuzz import fuzz

TextSimilarityMetric = Literal["fuzz_ratio", "bleu", "gleu", "cosine"]


class _MatchResult(BaseModel):
    match: bool


def _normalize(value: str, *, case_sensitive: bool = False, ignore_spaces: bool = False) -> str:
    result = value.strip()
    if not case_sensitive:
        result = result.lower()
    if ignore_spaces:
        result = result.replace(" ", "")
    return result


def _cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
    return dot_product / (magnitude1 * magnitude2)


def _exact_compare(
    predicted: str,
    expected: str,
    *,
    case_sensitive: bool = False,
    ignore_spaces: bool = False,
) -> bool:
    """Return True if predicted and expected match after normalization."""
    return _normalize(predicted, case_sensitive=case_sensitive, ignore_spaces=ignore_spaces) == _normalize(
        expected, case_sensitive=case_sensitive, ignore_spaces=ignore_spaces
    )


def _text_similarity_compare(
    predicted: str,
    expected: str,
    *,
    metric: TextSimilarityMetric = "fuzz_ratio",
    threshold: float | None = None,
    case_sensitive: bool = False,
    ignore_spaces: bool = False,
    embedding_model: str = "text-embedding-3-small",
) -> bool | float:
    """Return a similarity score (0.0-1.0) or bool if threshold is set.

    Args:
        predicted: The predicted value.
        expected: The expected ground truth value.
        metric: Which similarity metric to use ("fuzz_ratio", "bleu", "gleu", "cosine").
        threshold: If set, return bool (score >= threshold) instead of raw float.
        case_sensitive: Whether to preserve case when normalizing.
        ignore_spaces: Whether to strip spaces when normalizing.
        embedding_model: LiteLLM model name used when metric="cosine".
    """
    pred_norm = _normalize(predicted, case_sensitive=case_sensitive, ignore_spaces=ignore_spaces)
    exp_norm = _normalize(expected, case_sensitive=case_sensitive, ignore_spaces=ignore_spaces)

    if metric == "fuzz_ratio":
        similarity = fuzz.ratio(pred_norm, exp_norm) / 100.0

    elif metric == "bleu":
        reference = [exp_norm.split()]
        candidate = pred_norm.split()
        smoothing = SmoothingFunction().method1
        similarity = sentence_bleu(reference, candidate, smoothing_function=smoothing)

    elif metric == "gleu":
        reference = [exp_norm.split()]
        candidate = pred_norm.split()
        similarity = float(sentence_gleu(reference, candidate))

    elif metric == "cosine":
        response_predicted = embedding(model=embedding_model, input=[predicted])
        response_expected = embedding(model=embedding_model, input=[expected])
        vec_predicted = response_predicted.data[0]["embedding"]
        vec_expected = response_expected.data[0]["embedding"]
        similarity = _cosine_similarity(vec_predicted, vec_expected)

    else:
        raise ValueError(f"Unknown text_similarity metric: {metric!r}")

    if threshold is None:
        return similarity
    return similarity >= threshold


def _llm_compare(
    predicted: str,
    expected: str,
    *,
    model: str = "gpt-4o-mini",
    prompt_template: str | None = None,
    prompt_extra_vars: dict[str, Any] | None = None,
) -> bool:
    """Return True if an LLM judge considers predicted and expected to match.

    Args:
        predicted: The predicted value.
        expected: The expected ground truth value.
        model: LiteLLM model name to use as the judge.
        prompt_template: Custom prompt with {predicted} and {expected} placeholders.
            Must instruct the model to respond with only "YES" or "NO".
            Supports additional placeholders filled from prompt_extra_vars.
        prompt_extra_vars: Extra placeholder values injected into prompt_template.
    """
    if prompt_template is not None:
        prompt = prompt_template.format(
            predicted=predicted,
            expected=expected,
            **(prompt_extra_vars or {}),
        )
    else:
        prompt = (
            f'Do these two values refer to the same entity or concept? Consider them a match even if one is more specific, has extra qualifiers, or is an abbreviation of the other.\n'
            f"Value 1: {predicted}\n"
            f"Value 2: {expected}\n\n"
            'Respond with only "YES" or "NO".'
        )

    try:
        use_structured = litellm.supports_response_schema(model=model)
    except Exception:
        use_structured = False

    if use_structured:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            response_format=_MatchResult,
        )
    else:
        response = completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )

    if use_structured:
        try:
            result = json.loads(response.choices[0].message.content)
            return bool(result.get("match", False))
        except Exception:
            pass

    answer = response.choices[0].message.content.strip().upper()
    return answer.startswith("YES")


def _embedding_compare(
    predicted: str,
    expected: str,
    *,
    model: str = "text-embedding-3-small",
    threshold: float | None = None,
) -> bool | float:
    """Return embedding cosine similarity (0.0-1.0) or bool if threshold is set.

    Args:
        predicted: The predicted value.
        expected: The expected ground truth value.
        model: LiteLLM embedding model name.
        threshold: If set, return bool (score >= threshold) instead of raw float.
    """
    response_predicted = embedding(model=model, input=[predicted])
    response_expected = embedding(model=model, input=[expected])
    vec_predicted = response_predicted.data[0]["embedding"]
    vec_expected = response_expected.data[0]["embedding"]
    similarity = _cosine_similarity(vec_predicted, vec_expected)
    if threshold is None:
        return similarity
    return similarity >= threshold
