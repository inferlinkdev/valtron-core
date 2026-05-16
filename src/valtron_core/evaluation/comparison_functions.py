"""Comparison functions for evaluating LLM predictions against ground truth.

DEPRECATED: This module is deprecated. Use the standalone functions in
`valtron_core.evaluation.comparisons` instead, and register them directly in
`JsonEvaluator.metric_registry`. This file will be removed in a future release.
"""

import json
import math
import warnings
from typing import Any, Literal

import litellm
from litellm import completion, completion_cost, embedding
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu  # type: ignore[import-untyped]
from nltk.translate.gleu_score import sentence_gleu  # type: ignore[import-untyped]
from pydantic import BaseModel
from rapidfuzz import fuzz

ElementCompareType = Literal["exact", "text_similarity", "llm", "embedding"]
TextSimilarityMetric = Literal["fuzz_ratio", "bleu", "gleu", "cosine"]


<<<<<<< HEAD
MetricCategory = Literal["local", "llm", "embedding"]
=======
def element_compare_uses_third_party(element_compare: str, params: dict[str, str | float]) -> tuple[bool, str]:  # noqa: E501
    warnings.warn(
        "element_compare_uses_third_party is deprecated; use _check_builtin_metric_expensive "
        "in valtron_core.evaluation.json_eval instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    """Return ``(uses_third_party, human_readable_description)`` for a given
    element_compare strategy and its params.
>>>>>>> d0beb09 ([refactor] replace Comparator class with standalone comparison functions)


def element_compare_category(
    element_compare: str, params: dict[str, str | float]
) -> tuple[MetricCategory, str]:
    """Return ``(category, human_readable_description)`` for an element_compare strategy.

    ``category`` is one of ``"local"`` (no external API call), ``"llm"`` (calls a chat-LLM
    judge), or ``"embedding"`` (calls an embedding API).  This is the **single source of
    truth** used by the pre-flight safety check and the auto-LLM-alignment logic in
    json_eval.py to decide how to evaluate unordered lists.

    DEVELOPER NOTE — when adding a new comparison strategy:
      1. Add a branch here and return the correct category and a description.
      2. Add the new value to ``ElementCompareType``.
    Omitting either step will raise ``NotImplementedError`` at pre-flight time, making
    the gap impossible to ship silently.

    :param element_compare: The element comparison strategy name.
    :param params: The metric params dict, used to look up model names for descriptions.
    :return: A ``(category, description)`` tuple.
    """
    if element_compare == "exact":
        return "local", ""

    if element_compare == "text_similarity":
        if params.get("text_similarity_metric") == "cosine":
            model = params.get("embedding_model", "text-embedding-3-small")
            return "embedding", (
                f"text_similarity with cosine metric — calls the embedding API "
                f"(model='{model}')"
            )
        return "local", ""

    if element_compare == "llm":
        model = params.get("llm_model", "gpt-4o-mini")
        return "llm", f"LLM comparison (element_compare='llm', model='{model}')"

    if element_compare == "embedding":
        model = params.get("embedding_model", "text-embedding-3-small")
        return "embedding", f"embedding comparison (element_compare='embedding', model='{model}')"

    raise NotImplementedError(
        f"element_compare='{element_compare}' has no category declaration.\n"
        "When adding a new comparison strategy you MUST:\n"
        "  1. Add a branch in element_compare_category() in comparison_functions.py\n"
        "     and return the correct category ('local' | 'llm' | 'embedding') and a description.\n"
        "  2. Add the value to ElementCompareType.\n"
        "This check exists to prevent accidental n²-cost list evaluations."
    )


def element_compare_uses_third_party(element_compare: str, params: dict[str, str | float]) -> tuple[bool, str]:
    """Return ``(uses_third_party, human_readable_description)`` for an element_compare strategy.

    Thin wrapper over :func:`element_compare_category` preserved for backwards compatibility.

    :param element_compare: The element comparison strategy name.
    :param params: The metric params dict.
    :return: A ``(is_third_party, description)`` tuple.
    """
    category, desc = element_compare_category(element_compare, params)
    return category != "local", desc


class _MatchResult(BaseModel):
    match: bool


class Comparator:
    """Low-level element comparator with pluggable comparison strategies.

    Compares individual string elements and returns a score (float) or boolean.
    Uses LiteLLM for LLM and embedding comparisons, providing unified access to
    100+ models across providers (OpenAI, Anthropic, Google, Cohere, Ollama, etc.).
    """

    def __init__(
        self,
        element_compare: ElementCompareType = "exact",
        text_similarity_threshold: float | None = None,
        text_similarity_metric: TextSimilarityMetric = "fuzz_ratio",
        llm_model: str = "gpt-4o",
        llm_prompt_template: str | None = None,
        llm_prompt_extra_vars: dict[str, Any] | None = None,
        embedding_model: str = "text-embedding-3-small",
        embedding_threshold: float | None = None,
        case_sensitive: bool = False,
        ignore_spaces: bool = False,
    ) -> None:
        """
        Initialize the comparator.

        Args:
            element_compare: Comparison strategy ("exact", "text_similarity", "llm", or "embedding")
            text_similarity_threshold: Threshold for text similarity matching (0.0-1.0). None returns raw score.
            text_similarity_metric: Which metric to use for text similarity ("fuzz_ratio", "bleu", "gleu", "cosine")
            llm_model: LiteLLM model name for LLM comparisons
            llm_prompt_template: Custom prompt template for LLM comparisons. Must contain {predicted}
                and {expected} placeholders and must end with instructions to respond with only "YES"
                or "NO". Supports additional placeholders: {prompt_used} (the prompt sent to the
                evaluated model) and {example_content} / {example_<key>} (document fields). If None,
                uses the default entity-matching prompt.
            llm_prompt_extra_vars: Additional placeholder values injected into llm_prompt_template
                at format time. Keys map directly to template placeholders. Populated automatically
                from document data when called via the evaluation pipeline.
            embedding_model: LiteLLM model name for embeddings
            embedding_threshold: Threshold for embedding matching (0.0-1.0). None returns raw score.
            case_sensitive: Whether comparison should be case sensitive
            ignore_spaces: Whether comparison should ignore spaces
        """
        warnings.warn(
            "Comparator is deprecated; use the standalone functions in "
            "valtron_core.evaluation.comparisons and register them in "
            "JsonEvaluator.metric_registry instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.element_compare = element_compare
        self.text_similarity_threshold = text_similarity_threshold
        self.text_similarity_metric = text_similarity_metric
        self.llm_model = llm_model
        self.llm_prompt_template = llm_prompt_template
        self.llm_prompt_extra_vars: dict[str, Any] = llm_prompt_extra_vars or {}
        self.embedding_model = embedding_model
        self.embedding_threshold = embedding_threshold
        self.case_sensitive = case_sensitive
        self.ignore_spaces = ignore_spaces
        self.total_comparison_cost = 0.0
        self.comparison_count = 0

    def _normalize(self, value: str) -> str:
        """Normalize a string based on comparator settings."""
        result = value.strip()
        if not self.case_sensitive:
            result = result.lower()
        if self.ignore_spaces:
            result = result.replace(" ", "")
        return result

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0

        return dot_product / (magnitude1 * magnitude2)

    def _exact_compare(self, predicted: str, expected: str) -> bool:
        """Compare two strings for exact match after normalization."""
        return self._normalize(predicted) == self._normalize(expected)

    def _text_similarity_compare(self, predicted: str, expected: str) -> bool | float:
        """Compare two strings using text similarity matching."""
        # Normalize strings
        pred_norm = self._normalize(predicted)
        exp_norm = self._normalize(expected)

        if self.text_similarity_metric == "fuzz_ratio":
            # rapidfuzz returns 0-100, normalize to 0-1
            similarity = fuzz.ratio(pred_norm, exp_norm) / 100.0

        elif self.text_similarity_metric == "bleu":
            # BLEU expects tokenized input
            reference = [exp_norm.split()]
            candidate = pred_norm.split()
            # Use smoothing to handle short sentences
            smoothing = SmoothingFunction().method1
            similarity = sentence_bleu(reference, candidate, smoothing_function=smoothing)

        elif self.text_similarity_metric == "gleu":
            # GLEU also expects tokenized input
            reference = [exp_norm.split()]
            candidate = pred_norm.split()
            similarity = sentence_gleu(reference, candidate)

        elif self.text_similarity_metric == "cosine":
            # Use embeddings for cosine similarity
            response_predicted = embedding(
                model=self.embedding_model,
                input=[predicted],  # Use original, not normalized
            )
            response_expected = embedding(
                model=self.embedding_model,
                input=[expected],
            )

            try:
                cost1 = completion_cost(completion_response=response_predicted)
                cost2 = completion_cost(completion_response=response_expected)
                self.total_comparison_cost += cost1 + cost2
            except Exception:
                pass

            vec_predicted = response_predicted.data[0]["embedding"]
            vec_expected = response_expected.data[0]["embedding"]
            similarity = self._cosine_similarity(vec_predicted, vec_expected)

        else:
            raise ValueError(f"Unknown text_similarity_metric: {self.text_similarity_metric}")

        # Apply threshold if set
        if self.text_similarity_threshold is None:
            return similarity
        return similarity >= self.text_similarity_threshold

    def _llm_compare(self, predicted: str, expected: str, context: str | None = None) -> bool:
        """Compare two strings using an LLM as a judge.

        Args:
            predicted: The predicted value
            expected: The expected ground truth value
            context: Optional source document text for additional context
        """
        if self.llm_prompt_template is not None:
            prompt = self.llm_prompt_template.format(
                predicted=predicted,
                expected=expected,
                **self.llm_prompt_extra_vars,
            )
        else:
            context_section = ""
            if context:
                context_section = f"\nSource text: {context}\n"

            prompt = f"""Do these two values refer to the same entity or concept? Consider them a match even if one is more specific, has extra qualifiers, or is an abbreviation of the other.
{context_section}
Value 1: {predicted}
Value 2: {expected}

Respond with only "YES" or "NO"."""

        try:
            use_structured = litellm.supports_response_schema(model=self.llm_model)
        except Exception:
            use_structured = False

        if use_structured:
            response = completion(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format=_MatchResult,
            )
        else:
            response = completion(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=10,
            )

        try:
            cost = completion_cost(completion_response=response)
            self.total_comparison_cost += cost
        except Exception:
            pass

        if use_structured:
            try:
                result = json.loads(response.choices[0].message.content)
                return bool(result.get("match", False))
            except Exception:
                pass

        answer = response.choices[0].message.content.strip().upper()
        return answer.startswith("YES")

    def _embedding_compare(self, predicted: str, expected: str) -> bool | float:
        """Compare two strings using embedding cosine similarity."""
        response_predicted = embedding(
            model=self.embedding_model,
            input=[predicted],
        )
        response_expected = embedding(
            model=self.embedding_model,
            input=[expected],
        )

        try:
            cost1 = completion_cost(completion_response=response_predicted)
            cost2 = completion_cost(completion_response=response_expected)
            self.total_comparison_cost += cost1 + cost2
        except Exception:
            pass

        vec_predicted = response_predicted.data[0]["embedding"]
        vec_expected = response_expected.data[0]["embedding"]

        similarity = self._cosine_similarity(vec_predicted, vec_expected)

        if self.embedding_threshold is None:
            return similarity
        return similarity >= self.embedding_threshold

    def compare(self, predicted: str, expected: str, context: str | None = None) -> bool | float:
        """
        Compare two strings using the configured strategy.

        Args:
            predicted: The predicted value
            expected: The expected ground truth value
            context: Optional source document text for LLM comparison context

        Returns:
            bool if threshold is set, float (similarity score) if threshold is None
        """
        self.comparison_count += 1

        # Cheap short-circuit: when normalised values are identical, every strategy would
        # return a positive match.  Skip whatever expensive backend was configured (LLM
        # round-trip, embedding API calls, BLEU/GLEU computation).  Return type matches
        # what the configured strategy would have produced (bool vs float) so callers
        # cannot tell the difference.
        if self._exact_compare(predicted, expected):
            return 1.0 if self.is_score_mode() else True

        if self.element_compare == "exact":
            return self._exact_compare(predicted, expected)
        elif self.element_compare == "text_similarity":
            return self._text_similarity_compare(predicted, expected)
        elif self.element_compare == "llm":
            return self._llm_compare(predicted, expected, context)
        elif self.element_compare == "embedding":
            return self._embedding_compare(predicted, expected)
        else:
            raise ValueError(f"Unknown element_compare type: {self.element_compare}")

    def is_score_mode(self) -> bool:
        """Check if comparator returns scores (float) instead of booleans."""
        return (
            (self.element_compare == "embedding" and self.embedding_threshold is None) or
            (self.element_compare == "text_similarity" and self.text_similarity_threshold is None)
        )

    def get_stats(self) -> dict[str, int | float]:
        """Get comparison statistics."""
        return {
            "comparison_count": self.comparison_count,
            "total_comparison_cost": self.total_comparison_cost,
        }
