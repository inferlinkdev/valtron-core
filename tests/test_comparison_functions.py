"""Tests for the deprecated Comparator class in comparison_functions.py.

All Comparator constructions are expected to emit DeprecationWarning.
"""

from unittest.mock import Mock, patch

import pytest

from valtron_core.evaluation.comparison_functions import (
    Comparator,
    element_compare_category,
)


class TestComparatorInitialization:
    def test_default_initialization(self) -> None:
        with pytest.warns(DeprecationWarning, match="Comparator is deprecated"):
            comparator = Comparator()

        assert comparator.element_compare == "exact"
        assert comparator.text_similarity_threshold is None
        assert comparator.text_similarity_metric == "fuzz_ratio"
        assert comparator.llm_model == "gpt-4o"
        assert comparator.embedding_model == "text-embedding-3-small"
        assert comparator.embedding_threshold is None
        assert comparator.case_sensitive is False
        assert comparator.ignore_spaces is False
        assert comparator.total_comparison_cost == 0.0
        assert comparator.comparison_count == 0

    def test_custom_initialization(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(
                element_compare="embedding",
                text_similarity_threshold=0.8,
                text_similarity_metric="bleu",
                llm_model="gpt-4",
                embedding_model="text-embedding-3-large",
                embedding_threshold=0.9,
                case_sensitive=True,
                ignore_spaces=True,
            )

        assert comparator.element_compare == "embedding"
        assert comparator.text_similarity_threshold == 0.8
        assert comparator.text_similarity_metric == "bleu"
        assert comparator.llm_model == "gpt-4"
        assert comparator.embedding_model == "text-embedding-3-large"
        assert comparator.embedding_threshold == 0.9
        assert comparator.case_sensitive is True
        assert comparator.ignore_spaces is True


class TestComparatorNormalization:
    def test_normalize_strips_whitespace(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator()
        assert comparator._normalize("  hello  ") == "hello"
        assert comparator._normalize("\thello\n") == "hello"

    def test_normalize_lowercase_by_default(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator()
        assert comparator._normalize("HELLO") == "hello"
        assert comparator._normalize("HeLLo WoRLd") == "hello world"

    def test_normalize_case_sensitive(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(case_sensitive=True)
        assert comparator._normalize("HELLO") == "HELLO"
        assert comparator._normalize("HeLLo") == "HeLLo"

    def test_normalize_ignore_spaces(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(ignore_spaces=True)
        assert comparator._normalize("hello world") == "helloworld"
        assert comparator._normalize("a b c") == "abc"

    def test_normalize_combined_options(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(case_sensitive=True, ignore_spaces=True)
        assert comparator._normalize("  Hello World  ") == "HelloWorld"

        with pytest.warns(DeprecationWarning):
            comparator2 = Comparator(case_sensitive=False, ignore_spaces=True)
        assert comparator2._normalize("  Hello World  ") == "helloworld"


class TestCosineSimilarity:
    def test_cosine_similarity_identical_vectors(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator()
        vec = [1.0, 2.0, 3.0]
        assert comparator._cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal_vectors(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator()
        assert comparator._cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(0.0)

    def test_cosine_similarity_opposite_vectors(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator()
        assert comparator._cosine_similarity([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]) == pytest.approx(-1.0)

    def test_cosine_similarity_zero_vector(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator()
        assert comparator._cosine_similarity([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == 0.0
        assert comparator._cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0


class TestExactCompare:
    def test_exact_match_same_string(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="exact")
        assert comparator._exact_compare("hello", "hello") is True

    def test_exact_match_case_insensitive(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="exact")
        assert comparator._exact_compare("Hello", "hello") is True
        assert comparator._exact_compare("WORLD", "world") is True

    def test_exact_match_case_sensitive(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="exact", case_sensitive=True)
        assert comparator._exact_compare("Hello", "hello") is False
        assert comparator._exact_compare("Hello", "Hello") is True

    def test_exact_match_whitespace_handling(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="exact")
        assert comparator._exact_compare("  hello  ", "hello") is True
        assert comparator._exact_compare("hello", "  hello  ") is True

    def test_exact_no_match(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="exact")
        assert comparator._exact_compare("hello", "world") is False
        assert comparator._exact_compare("abc", "def") is False


class TestTextSimilarityCompare:
    def test_fuzz_ratio_identical_strings(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="text_similarity")
        result = comparator._text_similarity_compare("hello", "hello")
        assert result == pytest.approx(1.0)

    def test_fuzz_ratio_similar_strings(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="text_similarity")
        result = comparator._text_similarity_compare("hello", "hallo")
        assert 0.7 < result < 1.0

    def test_fuzz_ratio_with_threshold_pass(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="text_similarity", text_similarity_threshold=0.8)
        result = comparator._text_similarity_compare("hello", "hello")
        assert result is True

    def test_fuzz_ratio_with_threshold_fail(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="text_similarity", text_similarity_threshold=0.99)
        result = comparator._text_similarity_compare("hello", "hallo")
        assert result is False

    def test_bleu_score_identical(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="text_similarity", text_similarity_metric="bleu")
        result = comparator._text_similarity_compare(
            "the cat sat on the mat", "the cat sat on the mat"
        )
        assert result == pytest.approx(1.0)

    def test_bleu_score_similar(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="text_similarity", text_similarity_metric="bleu")
        result = comparator._text_similarity_compare(
            "the cat sat on the mat", "the cat is on the mat"
        )
        assert 0.0 < result < 1.0

    def test_gleu_score_identical(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="text_similarity", text_similarity_metric="gleu")
        result = comparator._text_similarity_compare(
            "the cat sat on the mat", "the cat sat on the mat"
        )
        assert result == pytest.approx(1.0)

    def test_gleu_score_similar(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="text_similarity", text_similarity_metric="gleu")
        result = comparator._text_similarity_compare(
            "the cat sat on the mat", "the cat is on the mat"
        )
        assert 0.0 < result < 1.0

    def test_cosine_similarity_metric(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(
                element_compare="text_similarity", text_similarity_metric="cosine"
            )

        mock_response1 = Mock()
        mock_response1.data = [{"embedding": [1.0, 0.0, 0.0]}]
        mock_response2 = Mock()
        mock_response2.data = [{"embedding": [1.0, 0.0, 0.0]}]

        with patch("valtron_core.evaluation.comparison_functions.embedding") as mock_embedding:
            mock_embedding.side_effect = [mock_response1, mock_response2]
            with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                result = comparator._text_similarity_compare("hello", "hello")

        assert result == pytest.approx(1.0)

    def test_unknown_metric_raises_error(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="text_similarity")
        comparator.text_similarity_metric = "invalid_metric"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Unknown text_similarity_metric"):
            comparator._text_similarity_compare("hello", "world")


class TestLLMCompare:
    def _mock_completion(self, content: str) -> Mock:
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = content
        return mock_response

    def test_llm_compare_equivalent_text_fallback(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion("YES")

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    result = comparator._llm_compare("NYC", "New York City")

        assert result is True

    def test_llm_compare_not_equivalent_text_fallback(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion("NO")

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    result = comparator._llm_compare("apple", "orange")

        assert result is False

    def test_llm_compare_equivalent_structured_output(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion('{"match": true}')

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=True):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    result = comparator._llm_compare("NYC", "New York City")

        assert result is True

    def test_llm_compare_not_equivalent_structured_output(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion('{"match": false}')

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=True):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    result = comparator._llm_compare("apple", "orange")

        assert result is False

    def test_llm_compare_structured_output_json_parse_failure_falls_back_to_text(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion("yes.")

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=True):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    result = comparator._llm_compare("a", "a")

        assert result is True

    def test_llm_compare_text_fallback_case_insensitive(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="llm")

        for content in ("yes", "Yes", "YES", "yes because they match", "YES, correct"):
            mock_response = self._mock_completion(content)
            with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
                with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                    with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.0):
                        assert comparator._llm_compare("a", "a") is True, f"failed for {content!r}"

    def test_llm_compare_text_fallback_rejects_no_response(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="llm")

        for content in ("no", "No", "NO", "no they differ"):
            mock_response = self._mock_completion(content)
            with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
                with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                    with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.0):
                        assert comparator._llm_compare("a", "b") is False, f"failed for {content!r}"

    def test_llm_compare_structured_output_passes_response_format(self) -> None:
        from valtron_core.evaluation.comparison_functions import _MatchResult

        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion('{"match": true}')

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=True):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response) as mock_completion:
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    comparator._llm_compare("a", "a")

        _, kwargs = mock_completion.call_args
        assert kwargs["response_format"] is _MatchResult

    def test_llm_compare_text_fallback_passes_max_tokens(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion("YES")

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response) as mock_completion:
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    comparator._llm_compare("a", "a")

        _, kwargs = mock_completion.call_args
        assert kwargs["max_tokens"] == 10

    def test_llm_compare_tracks_cost(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion("YES")

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.005):
                    comparator._llm_compare("a", "a")

        assert comparator.total_comparison_cost == pytest.approx(0.005)

    def test_llm_compare_supports_response_schema_exception_falls_back(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion("YES")

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", side_effect=Exception("unknown model")):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    result = comparator._llm_compare("a", "a")

        assert result is True

    def test_llm_prompt_template_replaces_default_prompt(self) -> None:
        template = "Is '{predicted}' the same as '{expected}'? YES or NO."
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="llm", llm_prompt_template=template)
        mock_response = self._mock_completion("YES")

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response) as mock_completion:
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    result = comparator._llm_compare("NYC", "New York")

        assert result is True
        sent_prompt = mock_completion.call_args[1]["messages"][0]["content"]
        assert sent_prompt == "Is 'NYC' the same as 'New York'? YES or NO."

    def test_llm_prompt_template_with_extra_vars(self) -> None:
        template = "Document: {example_content}\nDoes '{predicted}' equal '{expected}'? YES or NO."
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(
                element_compare="llm",
                llm_prompt_template=template,
                llm_prompt_extra_vars={"example_content": "The capital of France is Paris."},
            )
        mock_response = self._mock_completion("YES")

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response) as mock_completion:
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    comparator._llm_compare("Paris", "Paris")

        sent_prompt = mock_completion.call_args[1]["messages"][0]["content"]
        assert "The capital of France is Paris." in sent_prompt
        assert "Paris" in sent_prompt

    def test_llm_no_template_uses_default_prompt(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion("YES")

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response) as mock_completion:
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    comparator._llm_compare("NYC", "New York")

        sent_prompt = mock_completion.call_args[1]["messages"][0]["content"]
        assert "same entity or concept" in sent_prompt
        assert "Value 1: NYC" in sent_prompt
        assert "Value 2: New York" in sent_prompt

    def test_comparator_default_llm_prompt_template_is_none(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator()
        assert comparator.llm_prompt_template is None

    def test_comparator_default_llm_prompt_extra_vars_is_empty(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator()
        assert comparator.llm_prompt_extra_vars == {}


class TestEmbeddingCompare:
    def test_embedding_compare_similar(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="embedding")

        mock_response1 = Mock()
        mock_response1.data = [{"embedding": [1.0, 0.0, 0.0]}]
        mock_response2 = Mock()
        mock_response2.data = [{"embedding": [1.0, 0.0, 0.0]}]

        with patch("valtron_core.evaluation.comparison_functions.embedding") as mock_embedding:
            mock_embedding.side_effect = [mock_response1, mock_response2]
            with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                result = comparator._embedding_compare("hello", "hello")

        assert result == pytest.approx(1.0)

    def test_embedding_compare_with_threshold(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="embedding", embedding_threshold=0.9)

        mock_response1 = Mock()
        mock_response1.data = [{"embedding": [1.0, 0.0, 0.0]}]
        mock_response2 = Mock()
        mock_response2.data = [{"embedding": [1.0, 0.0, 0.0]}]

        with patch("valtron_core.evaluation.comparison_functions.embedding") as mock_embedding:
            mock_embedding.side_effect = [mock_response1, mock_response2]
            with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                result = comparator._embedding_compare("hello", "hello")

        assert result is True

    def test_embedding_compare_without_threshold_returns_float(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="embedding", embedding_threshold=None)

        mock_response1 = Mock()
        mock_response1.data = [{"embedding": [1.0, 0.5, 0.0]}]
        mock_response2 = Mock()
        mock_response2.data = [{"embedding": [0.5, 1.0, 0.0]}]

        with patch("valtron_core.evaluation.comparison_functions.embedding") as mock_embedding:
            mock_embedding.side_effect = [mock_response1, mock_response2]
            with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                result = comparator._embedding_compare("hello", "world")

        assert isinstance(result, float)
        assert 0.0 < result < 1.0

    def test_embedding_compare_tracks_cost(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="embedding")

        mock_response1 = Mock()
        mock_response1.data = [{"embedding": [1.0, 0.0, 0.0]}]
        mock_response2 = Mock()
        mock_response2.data = [{"embedding": [1.0, 0.0, 0.0]}]

        with patch("valtron_core.evaluation.comparison_functions.embedding") as mock_embedding:
            mock_embedding.side_effect = [mock_response1, mock_response2]
            with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.002):
                comparator._embedding_compare("hello", "hello")

        assert comparator.total_comparison_cost == pytest.approx(0.004)


class TestComparatorCompare:
    def test_compare_dispatches_to_exact(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="exact")
        assert comparator.compare("hello", "hello") is True
        assert comparator.compare("hello", "world") is False

    def test_compare_dispatches_to_text_similarity(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="text_similarity")
        result = comparator.compare("hello", "hello")
        assert result == pytest.approx(1.0)

    def test_compare_dispatches_to_llm(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="llm")

        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "YES"

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    result = comparator.compare("NYC", "New York")

        assert result is True

    def test_compare_dispatches_to_embedding(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="embedding")

        mock_response1 = Mock()
        mock_response1.data = [{"embedding": [1.0, 0.0, 0.0]}]
        mock_response2 = Mock()
        mock_response2.data = [{"embedding": [1.0, 0.0, 0.0]}]

        with patch("valtron_core.evaluation.comparison_functions.embedding") as mock_embedding:
            mock_embedding.side_effect = [mock_response1, mock_response2]
            with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                result = comparator.compare("hello", "hello")

        assert result == pytest.approx(1.0)

    def test_compare_unknown_type_raises_error(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator()
        comparator.element_compare = "invalid_type"  # type: ignore[assignment]

        with pytest.raises(ValueError, match="Unknown element_compare type"):
            comparator.compare("hello", "world")

    def test_compare_increments_count(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="exact")
        assert comparator.comparison_count == 0

        comparator.compare("a", "a")
        assert comparator.comparison_count == 1

        comparator.compare("b", "b")
        assert comparator.comparison_count == 2


class TestComparatorIsScoreMode:
    def test_is_score_mode_embedding_no_threshold(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="embedding", embedding_threshold=None)
        assert comparator.is_score_mode() is True

    def test_is_score_mode_embedding_with_threshold(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="embedding", embedding_threshold=0.9)
        assert comparator.is_score_mode() is False

    def test_is_score_mode_text_similarity_no_threshold(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="text_similarity", text_similarity_threshold=None)
        assert comparator.is_score_mode() is True

    def test_is_score_mode_text_similarity_with_threshold(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="text_similarity", text_similarity_threshold=0.8)
        assert comparator.is_score_mode() is False

    def test_is_score_mode_exact_always_false(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="exact")
        assert comparator.is_score_mode() is False


class TestComparatorStats:
    def test_get_stats_initial(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator()
        stats = comparator.get_stats()

        assert stats["comparison_count"] == 0
        assert stats["total_comparison_cost"] == 0.0

    def test_get_stats_after_comparisons(self) -> None:
        with pytest.warns(DeprecationWarning):
            comparator = Comparator(element_compare="exact")
        comparator.compare("a", "a")
        comparator.compare("b", "b")

        stats = comparator.get_stats()
        assert stats["comparison_count"] == 2

class TestElementCompareCategory:
    """Tests for the element_compare_category function."""

    def test_exact_returns_local(self) -> None:
        cat, desc = element_compare_category("exact", {})
        assert cat == "local"
        assert desc == ""

    def test_text_similarity_fuzz_returns_local(self) -> None:
        cat, desc = element_compare_category("text_similarity", {})
        assert cat == "local"
        assert desc == ""

    def test_text_similarity_cosine_returns_embedding(self) -> None:
        cat, desc = element_compare_category("text_similarity", {"text_similarity_metric": "cosine"})
        assert cat == "embedding"
        assert "embedding" in desc.lower()

    def test_text_similarity_cosine_includes_model_in_desc(self) -> None:
        _, desc = element_compare_category(
            "text_similarity",
            {"text_similarity_metric": "cosine", "embedding_model": "my-model"},
        )
        assert "my-model" in desc

    def test_llm_returns_llm_category(self) -> None:
        cat, desc = element_compare_category("llm", {})
        assert cat == "llm"
        assert "llm" in desc.lower() or "LLM" in desc

    def test_embedding_returns_embedding_category(self) -> None:
        cat, desc = element_compare_category("embedding", {})
        assert cat == "embedding"
        assert "embedding" in desc.lower()

    def test_unknown_strategy_raises_not_implemented(self) -> None:
        with pytest.raises(NotImplementedError, match="no category declaration"):
            element_compare_category("unknown_strategy", {})



@pytest.mark.unit
class TestComparatorShortCircuit:
    """Tests for the early-exit short-circuit when predicted matches expected."""

    def test_matching_values_skip_llm_backend(self) -> None:
        comp = Comparator(element_compare="llm")
        with patch.object(comp, "_llm_compare") as mock_llm:
            result = comp.compare("hello", "hello")
        mock_llm.assert_not_called()
        assert result is True

    def test_matching_values_skip_embedding_backend(self) -> None:
        comp = Comparator(element_compare="embedding")
        with patch.object(comp, "_embedding_compare") as mock_embed:
            comp.compare("hello", "hello")
        mock_embed.assert_not_called()

    def test_matching_values_skip_text_similarity_backend(self) -> None:
        comp = Comparator(element_compare="text_similarity")
        with patch.object(comp, "_text_similarity_compare") as mock_ts:
            comp.compare("hello", "hello")
        mock_ts.assert_not_called()

    def test_non_matching_values_proceed_to_llm_backend(self) -> None:
        comp = Comparator(element_compare="llm")
        with patch.object(comp, "_llm_compare", return_value=True) as mock_llm:
            comp.compare("hello", "world")
        mock_llm.assert_called_once()

    def test_short_circuit_returns_true_in_bool_mode(self) -> None:
        comp = Comparator(element_compare="llm")
        result = comp.compare("same", "same")
        assert result is True

    def test_short_circuit_returns_float_in_score_mode_embedding(self) -> None:
        comp = Comparator(element_compare="embedding", embedding_threshold=None)
        result = comp.compare("same", "same")
        assert result == 1.0

    def test_short_circuit_returns_float_in_score_mode_text_similarity(self) -> None:
        comp = Comparator(
            element_compare="text_similarity",
            text_similarity_metric="cosine",
            text_similarity_threshold=None,
        )
        result = comp.compare("same", "same")
        assert result == 1.0

    def test_short_circuit_uses_normalized_comparison(self) -> None:
        comp = Comparator(element_compare="llm")
        with patch.object(comp, "_llm_compare") as mock_llm:
            result = comp.compare("  HELLO  ", "hello")
        mock_llm.assert_not_called()
        assert result is True

    def test_short_circuit_case_sensitive_mode_no_skip_on_case_diff(self) -> None:
        comp = Comparator(element_compare="llm", case_sensitive=True)
        with patch.object(comp, "_llm_compare", return_value=False) as mock_llm:
            comp.compare("HELLO", "hello")
        mock_llm.assert_called_once()

    def test_short_circuit_increments_comparison_count(self) -> None:
        comp = Comparator(element_compare="llm")
        comp.compare("same", "same")
        assert comp.comparison_count == 1