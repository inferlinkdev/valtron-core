"""Comparison functions for evaluating LLM predictions against ground truth.

This module provides two classes:
- **Comparator**: Low-level element comparison (exact, text_similarity, llm, embedding)
- **Grader**: High-level orchestration with list handling and score aggregation

Uses LiteLLM for unified access to multiple AI providers.

Example usage:
    from evaltron_core.comparison_functions import Comparator, Grader

    # Create comparator for element-level comparison
    comparator = Comparator(
        element_compare="embedding",
        embedding_model="text-embedding-3-large",
        embedding_threshold=0.85,  # or None for raw scores
    )

    # Create grader for list/aggregation logic
    grader = Grader(comparator=comparator)

    # Single string comparison
    score = grader.grade_str("NYC", "New York")  # True or 0.92

    # List comparisons - returns list of scores
    scores = grader.grade_list(["a", "b"], ["A", "B"])  # [True, True]

    # List comparisons - order doesn't matter (sort first, then compare)
    scores = grader.grade_list(["b", "a"], ["A", "B"], order_matters=False, aggregation="avg")

    # With aggregation
    avg = grader.grade_list(["a", "b"], ["A", "B"], aggregation="avg")  # 1.0
    passed = grader.grade_list(["a", "b"], ["A", "B"], aggregation="all")  # True

    # JSON grading - returns per-key scores
    predicted = '{"name": "John", "city": "NYC"}'
    expected = '{"name": "John", "city": "New York"}'
    scores = grader.grade_json(predicted, expected)
    # Returns: {"name": True, "city": False}

    # Nested JSON with lists
    predicted = '{"person": {"name": "John"}, "tags": ["dev", "py"]}'
    expected = '{"person": {"name": "John"}, "tags": ["developer", "python"]}'
    scores = grader.grade_json(predicted, expected, aggregation="avg")
    # Returns: {"person": {"name": True}, "tags": 0.5}

    # Use with PromptEvaluator
    from evaltron_core.evaluator import PromptEvaluator
    evaluator = PromptEvaluator()
    result = await evaluator.evaluate(
        eval_input=my_input,
        comparison_fn=grader.grade_str,
    )
"""

import json
import math
from typing import Literal

from litellm import completion, completion_cost, embedding
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.translate.gleu_score import sentence_gleu
from rapidfuzz import fuzz

ElementCompareType = Literal["exact", "text_similarity", "llm", "embedding"]
TextSimilarityMetric = Literal["fuzz_ratio", "bleu", "gleu", "cosine"]
AggregationType = Literal["avg", "all", "any"]

# Maximum list length for expensive comparisons (embedding, llm)
# With N×M comparisons, 10 elements = 100 API calls max
MAX_LIST_LENGTH_FOR_EXPENSIVE_COMPARE = 10


def element_compare_uses_third_party(element_compare: str, params: dict) -> tuple[bool, str]:
    """Return ``(uses_third_party, human_readable_description)`` for a given
    element_compare strategy and its params.

    This is the **single source of truth** for whether a comparison type calls
    a 3rd-party API.  It is used by the pre-flight safety check in the runner
    to detect n²-cost list comparisons before any model is invoked.

    DEVELOPER NOTE — when adding a new comparison strategy:
      1. Add a branch here and return ``(True, "<description>")`` if it calls a
         3rd-party API, or ``(False, "")`` if it runs entirely locally.
      2. Add the new value to ``ElementCompareType``.
    Omitting either step will raise ``NotImplementedError`` at pre-flight time,
    making the gap impossible to ship silently.
    """
    if element_compare == "exact":
        return False, ""

    if element_compare == "text_similarity":
        if params.get("text_similarity_metric") == "cosine":
            model = params.get("embedding_model", "text-embedding-3-small")
            return True, (
                f"text_similarity with cosine metric — calls the embedding API "
                f"(model='{model}')"
            )
        return False, ""

    if element_compare == "llm":
        model = params.get("llm_model", "gpt-4o-mini")
        return True, f"LLM comparison (element_compare='llm', model='{model}')"

    if element_compare == "embedding":
        model = params.get("embedding_model", "text-embedding-3-small")
        return True, f"embedding comparison (element_compare='embedding', model='{model}')"

    raise NotImplementedError(
        f"element_compare='{element_compare}' has no 3rd-party API declaration.\n"
        "When adding a new comparison strategy you MUST:\n"
        "  1. Add a branch in element_compare_uses_third_party() in comparison_functions.py\n"
        "     and return (True, description) if it calls a 3rd-party API, or (False, '') if not.\n"
        "  2. Add the value to ElementCompareType.\n"
        "This check exists to prevent accidental n²-cost list evaluations."
    )


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
            embedding_model: LiteLLM model name for embeddings
            embedding_threshold: Threshold for embedding matching (0.0-1.0). None returns raw score.
            case_sensitive: Whether comparison should be case sensitive
            ignore_spaces: Whether comparison should ignore spaces
        """
        self.element_compare = element_compare
        self.text_similarity_threshold = text_similarity_threshold
        self.text_similarity_metric = text_similarity_metric
        self.llm_model = llm_model
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
        context_section = ""
        if context:
            context_section = f"\nSource text: {context}\n"

        prompt = f"""Do these two values refer to the same entity or concept? Consider them a match even if one is more specific, has extra qualifiers, or is an abbreviation of the other.
{context_section}
Value 1: {predicted}
Value 2: {expected}

Respond with only "YES" or "NO"."""

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

        answer = response.choices[0].message.content.strip().upper()
        return answer == "YES"

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

    def get_stats(self) -> dict:
        """Get comparison statistics."""
        return {
            "comparison_count": self.comparison_count,
            "total_comparison_cost": self.total_comparison_cost,
        }


class Grader:
    """High-level grader that orchestrates comparisons and aggregates scores.

    Uses a Comparator for element-level comparisons and provides methods for
    list comparisons and score aggregation.
    """

    def __init__(
        self,
        comparator: Comparator,
        order_matters: bool = True,
        threshold: float = 0.5,
    ) -> None:
        """
        Initialize the grader.

        Args:
            comparator: Comparator instance for element comparisons
            order_matters: Default setting for whether list element order matters
            threshold: Default threshold for converting scores to booleans in all/any methods
        """
        self.comparator = comparator
        self.order_matters = order_matters
        self.threshold = threshold

    def grade_str(self, predicted: str, expected: str, context: str | None = None) -> bool | float:
        """
        Grade a single string prediction.

        Args:
            predicted: The predicted value
            expected: The expected ground truth value
            context: Optional source document text for LLM comparison context

        Returns:
            bool or float depending on comparator configuration
        """
        return self.comparator.compare(predicted, expected, context)

    def grade_list(
        self,
        predicted: list[str],
        expected: list[str],
        order_matters: bool | None = None,
        aggregation: AggregationType | None = None,
        context: str | None = None,
        threshold: float | None = None,
    ) -> list[bool] | list[float] | float | bool:
        """
        Grade a list of predictions against expected values.

        When order_matters=True: Compares elements 1-1 positionally (L1[i] vs L2[i]).
        When order_matters=False: Computes N×M similarity matrix and uses greedy
        1-to-1 matching (best scores first). Requires aggregation.

        Args:
            predicted: List of predicted values
            expected: List of expected ground truth values
            order_matters: If True, compare L1[i] to L2[i] directly (lists must be same length).
                          If False, use N×M greedy matching (aggregation required).
                          If None, uses the instance default.
            aggregation: How to aggregate results:
                        - None: return list of scores (only valid when order_matters=True)
                        - "avg": return average score (float)
                        - "all": return True if all meet threshold (bool)
                        - "any": return True if any meets threshold (bool)
            threshold: Threshold for "all"/"any" aggregation (default: instance threshold)

        Returns:
            - list[bool] or list[float] when aggregation is None (order_matters=True only)
            - float when aggregation is "avg"
            - bool when aggregation is "all" or "any"

        Raises:
            ValueError: If lists have different lengths when order_matters=True
            ValueError: If aggregation is None when order_matters=False
            ValueError: If list length > 10 for embedding/llm comparison with order_matters=False
        """
        # Use instance defaults
        if order_matters is None:
            order_matters = self.order_matters
        if threshold is None:
            threshold = self.threshold

        # Length check only applies when order matters (1-to-1 positional)
        if len(predicted) != len(expected) and order_matters:
            raise ValueError(
                f"Lists must have same length when order_matters=True: {len(predicted)} != {len(expected)}"
            )

        # Enforce aggregation when order doesn't matter
        if not order_matters and aggregation is None:
            raise ValueError(
                "aggregation is required when order_matters=False "
                "(positional scores are meaningless with greedy matching)"
            )

        # Limit list length for expensive comparison types (N×M can be costly)
        if not order_matters and self.comparator.element_compare in ("embedding", "llm"):
            max_len = max(len(predicted), len(expected))
            if max_len > MAX_LIST_LENGTH_FOR_EXPENSIVE_COMPARE:
                raise ValueError(
                    f"List too long for {self.comparator.element_compare} comparison: "
                    f"{max_len} > {MAX_LIST_LENGTH_FOR_EXPENSIVE_COMPARE}. "
                    f"N×M comparisons would require {len(predicted) * len(expected)} API calls. "
                    f"Use exact or text_similarity comparison for long lists."
                )

        # Get element-wise scores
        if order_matters:
            scores = [self.comparator.compare(p, e, context) for p, e in zip(predicted, expected)]
        else:
            # N×M comparisons with greedy 1-to-1 matching
            all_pairs = []
            for i, p in enumerate(predicted):
                for j, e in enumerate(expected):
                    score = self.comparator.compare(p, e, context)
                    score_num = float(score) if isinstance(score, bool) else score
                    all_pairs.append((score_num, i, j))

            # Sort by score descending (best matches first)
            all_pairs.sort(key=lambda x: x[0], reverse=True)

            # Greedy 1-to-1 assignment
            matched_pred: set[int] = set()
            matched_exp: set[int] = set()
            scores = []

            for score, i, j in all_pairs:
                if i not in matched_pred and j not in matched_exp:
                    scores.append(score)
                    matched_pred.add(i)
                    matched_exp.add(j)

            # Unmatched items get score 0 (for length mismatch)
            max_len = max(len(predicted), len(expected))
            while len(scores) < max_len:
                scores.append(0.0)

        # Return raw scores if no aggregation
        if aggregation is None:
            return scores

        # Convert to numeric for aggregation
        numeric = [float(s) if isinstance(s, bool) else s for s in scores]

        if aggregation == "avg":
            return sum(numeric) / len(numeric) if numeric else 0.0
        elif aggregation == "all":
            return all(s >= threshold for s in numeric)
        elif aggregation == "any":
            return any(s >= threshold for s in numeric)

        return scores

    def grade_json(
        self,
        predicted: str | dict,
        expected: str | dict,
        order_matters: bool | None = None,
        aggregation: AggregationType | None = None,
        threshold: float | None = None,
        context: str | None = None,
    ) -> dict:
        """
        Grade JSON predictions recursively, returning per-key scores.

        Traverses the JSON structure and grades each value based on its type:
        - str: uses grade_str()
        - list: uses grade_list() with the provided parameters
        - dict: recurses into the nested structure

        Args:
            predicted: Predicted JSON (string or already parsed dict)
            expected: Expected JSON (string or already parsed dict)
            order_matters: For list comparisons - whether order matters
            aggregation: For list comparisons - how to aggregate ("avg", "all", "any", or None)
            threshold: For list comparisons with "all"/"any" aggregation

        Returns:
            Dict with same structure as expected, containing scores per key:
            - str values → bool | float
            - list values → list[bool|float] | float | bool (depending on aggregation)
            - dict values → nested dict of scores
        """
        # Parse JSON strings if needed
        if isinstance(predicted, str):
            predicted = json.loads(predicted.strip())
        if isinstance(expected, str):
            expected = json.loads(expected.strip())

        return self._grade_json_recursive(
            predicted, expected, order_matters, aggregation, threshold, context
        )

    def _grade_json_recursive(
        self,
        predicted: dict,
        expected: dict,
        order_matters: bool | None,
        aggregation: AggregationType | None,
        threshold: float | None,
        context: str | None = None,
    ) -> dict:
        """Recursively grade JSON structure."""
        results = {}

        for key, exp_value in expected.items():
            pred_value = predicted.get(key)

            # Handle missing key in predicted
            if pred_value is None:
                if isinstance(exp_value, dict):
                    # Recurse with empty dict
                    results[key] = self._grade_json_recursive(
                        {}, exp_value, order_matters, aggregation, threshold, context
                    )
                elif isinstance(exp_value, list):
                    # Return appropriate failure value for missing list
                    if aggregation in ("all", "any"):
                        results[key] = False
                    elif aggregation == "avg":
                        results[key] = 0.0
                    else:
                        is_score_mode = self.comparator.is_score_mode()
                        results[key] = [0.0 if is_score_mode else False] * len(exp_value)
                else:
                    # String comparison with empty
                    results[key] = self.grade_str("", str(exp_value), context)
                continue

            # Grade based on expected value type
            if isinstance(exp_value, dict):
                # Recurse for nested dict
                pred_dict = pred_value if isinstance(pred_value, dict) else {}
                results[key] = self._grade_json_recursive(
                    pred_dict, exp_value, order_matters, aggregation, threshold, context
                )
            elif isinstance(exp_value, list):
                # Grade list
                pred_list = pred_value if isinstance(pred_value, list) else []
                # Convert all elements to strings for comparison
                pred_strs = [str(x) for x in pred_list]
                exp_strs = [str(x) for x in exp_value]
                try:
                    results[key] = self.grade_list(
                        pred_strs, exp_strs, order_matters, aggregation, context, threshold
                    )
                except ValueError:
                    # Length mismatch - return appropriate failure value
                    if aggregation in ("all", "any"):
                        results[key] = False
                    elif aggregation == "avg":
                        results[key] = 0.0
                    else:
                        is_score_mode = self.comparator.is_score_mode()
                        results[key] = [0.0 if is_score_mode else False] * len(exp_value)
            else:
                # Grade string
                results[key] = self.grade_str(str(pred_value), str(exp_value), context)

        return results

    def get_stats(self) -> dict:
        """Get grader and comparator statistics."""
        return self.comparator.get_stats()
