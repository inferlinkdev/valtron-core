"""Tests for comparison functions module."""

from unittest.mock import Mock, patch

import pytest

from valtron_core.evaluation.comparison_functions import (
    Comparator,
    Grader,
    element_compare_category,
    element_compare_uses_third_party,
)


class TestComparatorInitialization:
    """Tests for Comparator initialization."""

    def test_default_initialization(self) -> None:
        """Test default parameter values."""
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
        """Test custom parameter values are stored."""
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
    """Tests for string normalization."""

    def test_normalize_strips_whitespace(self) -> None:
        """Test that normalization strips leading/trailing whitespace."""
        comparator = Comparator()
        assert comparator._normalize("  hello  ") == "hello"
        assert comparator._normalize("\thello\n") == "hello"

    def test_normalize_lowercase_by_default(self) -> None:
        """Test that normalization lowercases by default."""
        comparator = Comparator()
        assert comparator._normalize("HELLO") == "hello"
        assert comparator._normalize("HeLLo WoRLd") == "hello world"

    def test_normalize_case_sensitive(self) -> None:
        """Test that case_sensitive=True preserves case."""
        comparator = Comparator(case_sensitive=True)
        assert comparator._normalize("HELLO") == "HELLO"
        assert comparator._normalize("HeLLo") == "HeLLo"

    def test_normalize_ignore_spaces(self) -> None:
        """Test that ignore_spaces=True removes spaces."""
        comparator = Comparator(ignore_spaces=True)
        assert comparator._normalize("hello world") == "helloworld"
        assert comparator._normalize("a b c") == "abc"

    def test_normalize_combined_options(self) -> None:
        """Test combined normalization options."""
        comparator = Comparator(case_sensitive=True, ignore_spaces=True)
        assert comparator._normalize("  Hello World  ") == "HelloWorld"

        comparator2 = Comparator(case_sensitive=False, ignore_spaces=True)
        assert comparator2._normalize("  Hello World  ") == "helloworld"


class TestCosineSimilarity:
    """Tests for cosine similarity calculation."""

    def test_cosine_similarity_identical_vectors(self) -> None:
        """Test cosine similarity of identical vectors is 1.0."""
        comparator = Comparator()
        vec = [1.0, 2.0, 3.0]
        assert comparator._cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal_vectors(self) -> None:
        """Test cosine similarity of orthogonal vectors is 0.0."""
        comparator = Comparator()
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        assert comparator._cosine_similarity(vec1, vec2) == pytest.approx(0.0)

    def test_cosine_similarity_opposite_vectors(self) -> None:
        """Test cosine similarity of opposite vectors is -1.0."""
        comparator = Comparator()
        vec1 = [1.0, 2.0, 3.0]
        vec2 = [-1.0, -2.0, -3.0]
        assert comparator._cosine_similarity(vec1, vec2) == pytest.approx(-1.0)

    def test_cosine_similarity_zero_vector(self) -> None:
        """Test cosine similarity with zero vector returns 0.0."""
        comparator = Comparator()
        vec1 = [1.0, 2.0, 3.0]
        vec_zero = [0.0, 0.0, 0.0]
        assert comparator._cosine_similarity(vec1, vec_zero) == 0.0
        assert comparator._cosine_similarity(vec_zero, vec1) == 0.0


class TestExactCompare:
    """Tests for exact string comparison."""

    def test_exact_match_same_string(self) -> None:
        """Test exact match with identical strings."""
        comparator = Comparator(element_compare="exact")
        assert comparator._exact_compare("hello", "hello") is True

    def test_exact_match_case_insensitive(self) -> None:
        """Test exact match is case insensitive by default."""
        comparator = Comparator(element_compare="exact")
        assert comparator._exact_compare("Hello", "hello") is True
        assert comparator._exact_compare("WORLD", "world") is True

    def test_exact_match_case_sensitive(self) -> None:
        """Test exact match with case sensitivity enabled."""
        comparator = Comparator(element_compare="exact", case_sensitive=True)
        assert comparator._exact_compare("Hello", "hello") is False
        assert comparator._exact_compare("Hello", "Hello") is True

    def test_exact_match_whitespace_handling(self) -> None:
        """Test exact match handles whitespace correctly."""
        comparator = Comparator(element_compare="exact")
        assert comparator._exact_compare("  hello  ", "hello") is True
        assert comparator._exact_compare("hello", "  hello  ") is True

    def test_exact_no_match(self) -> None:
        """Test exact match returns False for different strings."""
        comparator = Comparator(element_compare="exact")
        assert comparator._exact_compare("hello", "world") is False
        assert comparator._exact_compare("abc", "def") is False


class TestTextSimilarityCompare:
    """Tests for text similarity comparison."""

    def test_fuzz_ratio_identical_strings(self) -> None:
        """Test fuzz_ratio returns 1.0 for identical strings."""
        comparator = Comparator(element_compare="text_similarity")
        result = comparator._text_similarity_compare("hello", "hello")
        assert result == pytest.approx(1.0)

    def test_fuzz_ratio_similar_strings(self) -> None:
        """Test fuzz_ratio returns high score for similar strings."""
        comparator = Comparator(element_compare="text_similarity")
        result = comparator._text_similarity_compare("hello", "hallo")
        assert 0.7 < result < 1.0

    def test_fuzz_ratio_with_threshold_pass(self) -> None:
        """Test fuzz_ratio with threshold returns True when above threshold."""
        comparator = Comparator(
            element_compare="text_similarity",
            text_similarity_threshold=0.8,
        )
        result = comparator._text_similarity_compare("hello", "hello")
        assert result is True

    def test_fuzz_ratio_with_threshold_fail(self) -> None:
        """Test fuzz_ratio with threshold returns False when below threshold."""
        comparator = Comparator(
            element_compare="text_similarity",
            text_similarity_threshold=0.99,
        )
        result = comparator._text_similarity_compare("hello", "hallo")
        assert result is False

    def test_bleu_score_identical(self) -> None:
        """Test BLEU score for identical strings."""
        comparator = Comparator(
            element_compare="text_similarity",
            text_similarity_metric="bleu",
        )
        result = comparator._text_similarity_compare(
            "the cat sat on the mat",
            "the cat sat on the mat",
        )
        assert result == pytest.approx(1.0)

    def test_bleu_score_similar(self) -> None:
        """Test BLEU score for similar strings."""
        comparator = Comparator(
            element_compare="text_similarity",
            text_similarity_metric="bleu",
        )
        result = comparator._text_similarity_compare(
            "the cat sat on the mat",
            "the cat is on the mat",
        )
        assert 0.0 < result < 1.0

    def test_gleu_score_identical(self) -> None:
        """Test GLEU score for identical strings."""
        comparator = Comparator(
            element_compare="text_similarity",
            text_similarity_metric="gleu",
        )
        result = comparator._text_similarity_compare(
            "the cat sat on the mat",
            "the cat sat on the mat",
        )
        assert result == pytest.approx(1.0)

    def test_gleu_score_similar(self) -> None:
        """Test GLEU score for similar strings."""
        comparator = Comparator(
            element_compare="text_similarity",
            text_similarity_metric="gleu",
        )
        result = comparator._text_similarity_compare(
            "the cat sat on the mat",
            "the cat is on the mat",
        )
        assert 0.0 < result < 1.0

    def test_cosine_similarity_metric(self) -> None:
        """Test cosine similarity metric with mocked embeddings."""
        comparator = Comparator(
            element_compare="text_similarity",
            text_similarity_metric="cosine",
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
        """Test that unknown metric raises ValueError."""
        comparator = Comparator(element_compare="text_similarity")
        comparator.text_similarity_metric = "invalid_metric"

        with pytest.raises(ValueError, match="Unknown text_similarity_metric"):
            comparator._text_similarity_compare("hello", "world")


class TestLLMCompare:
    """Tests for LLM-based comparison."""

    def _mock_completion(self, content: str) -> Mock:
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = content
        return mock_response

    def test_llm_compare_equivalent_text_fallback(self) -> None:
        """Returns True for YES when model does not support structured output."""
        comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion("YES")

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    result = comparator._llm_compare("NYC", "New York City")

        assert result is True

    def test_llm_compare_not_equivalent_text_fallback(self) -> None:
        """Returns False for NO when model does not support structured output."""
        comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion("NO")

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    result = comparator._llm_compare("apple", "orange")

        assert result is False

    def test_llm_compare_equivalent_structured_output(self) -> None:
        """Returns True when model returns structured JSON {match: true}."""
        comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion('{"match": true}')

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=True):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    result = comparator._llm_compare("NYC", "New York City")

        assert result is True

    def test_llm_compare_not_equivalent_structured_output(self) -> None:
        """Returns False when model returns structured JSON {match: false}."""
        comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion('{"match": false}')

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=True):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    result = comparator._llm_compare("apple", "orange")

        assert result is False

    def test_llm_compare_structured_output_json_parse_failure_falls_back_to_text(self) -> None:
        """Falls back to normalised text check if JSON parsing fails."""
        comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion("yes.")

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=True):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    result = comparator._llm_compare("a", "a")

        assert result is True

    def test_llm_compare_text_fallback_case_insensitive(self) -> None:
        """Text fallback accepts yes/YES/Yes and verbose yes responses."""
        comparator = Comparator(element_compare="llm")

        for content in ("yes", "Yes", "YES", "yes because they match", "YES, correct"):
            mock_response = self._mock_completion(content)
            with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
                with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                    with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.0):
                        assert comparator._llm_compare("a", "a") is True, f"failed for {content!r}"

    def test_llm_compare_text_fallback_rejects_no_response(self) -> None:
        """Text fallback returns False for responses starting with NO."""
        comparator = Comparator(element_compare="llm")

        for content in ("no", "No", "NO", "no they differ"):
            mock_response = self._mock_completion(content)
            with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
                with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                    with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.0):
                        assert comparator._llm_compare("a", "b") is False, f"failed for {content!r}"

    def test_llm_compare_structured_output_passes_response_format(self) -> None:
        """Structured output path passes _MatchResult as response_format."""
        from valtron_core.evaluation.comparison_functions import _MatchResult

        comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion('{"match": true}')

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=True):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response) as mock_completion:
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    comparator._llm_compare("a", "a")

        _, kwargs = mock_completion.call_args
        assert kwargs["response_format"] is _MatchResult

    def test_llm_compare_text_fallback_passes_max_tokens(self) -> None:
        """Text fallback path passes max_tokens=10."""
        comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion("YES")

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response) as mock_completion:
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    comparator._llm_compare("a", "a")

        _, kwargs = mock_completion.call_args
        assert kwargs["max_tokens"] == 10

    def test_llm_compare_tracks_cost(self) -> None:
        """LLM compare tracks cost regardless of which path is used."""
        comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion("YES")

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.005):
                    comparator._llm_compare("a", "a")

        assert comparator.total_comparison_cost == pytest.approx(0.005)

    def test_llm_compare_supports_response_schema_exception_falls_back(self) -> None:
        """If supports_response_schema raises, falls back to text parsing."""
        comparator = Comparator(element_compare="llm")
        mock_response = self._mock_completion("YES")

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", side_effect=Exception("unknown model")):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response):
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    result = comparator._llm_compare("a", "a")

        assert result is True

    def test_llm_prompt_template_replaces_default_prompt(self) -> None:
        """Custom template is sent to the LLM instead of the default prompt."""
        template = "Is '{predicted}' the same as '{expected}'? YES or NO."
        comparator = Comparator(
            element_compare="llm",
            llm_prompt_template=template,
        )
        mock_response = self._mock_completion("YES")

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response) as mock_completion:
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    result = comparator._llm_compare("NYC", "New York")

        assert result is True
        sent_prompt = mock_completion.call_args[1]["messages"][0]["content"]
        assert sent_prompt == "Is 'NYC' the same as 'New York'? YES or NO."

    def test_llm_prompt_template_with_extra_vars(self) -> None:
        """Extra vars are interpolated into the custom template."""
        template = "Document: {example_content}\nDoes '{predicted}' equal '{expected}'? YES or NO."
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

    def test_llm_prompt_template_with_prompt_used_var(self) -> None:
        """The {prompt_used} placeholder is interpolated when present in extra vars."""
        template = "Prompt: {prompt_used}\nDoes '{predicted}' equal '{expected}'? YES or NO."
        comparator = Comparator(
            element_compare="llm",
            llm_prompt_template=template,
            llm_prompt_extra_vars={"prompt_used": "Extract the city name."},
        )
        mock_response = self._mock_completion("YES")

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response) as mock_completion:
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    comparator._llm_compare("Paris", "Paris")

        sent_prompt = mock_completion.call_args[1]["messages"][0]["content"]
        assert "Extract the city name." in sent_prompt

    def test_llm_prompt_template_with_dict_content_vars(self) -> None:
        """Per-key example vars ({example_key1}, {example_key2}) are interpolated."""
        template = "Key1: {example_key1}, Key2: {example_key2}\n'{predicted}' vs '{expected}'? YES or NO."
        comparator = Comparator(
            element_compare="llm",
            llm_prompt_template=template,
            llm_prompt_extra_vars={"example_key1": "alpha", "example_key2": "beta"},
        )
        mock_response = self._mock_completion("YES")

        with patch("valtron_core.evaluation.comparison_functions.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparison_functions.completion", return_value=mock_response) as mock_completion:
                with patch("valtron_core.evaluation.comparison_functions.completion_cost", return_value=0.001):
                    comparator._llm_compare("x", "x")

        sent_prompt = mock_completion.call_args[1]["messages"][0]["content"]
        assert "Key1: alpha" in sent_prompt
        assert "Key2: beta" in sent_prompt

    def test_llm_no_template_uses_default_prompt(self) -> None:
        """When no template is set, the default entity-matching prompt is used."""
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
        """llm_prompt_template defaults to None."""
        comparator = Comparator()
        assert comparator.llm_prompt_template is None

    def test_comparator_default_llm_prompt_extra_vars_is_empty(self) -> None:
        """llm_prompt_extra_vars defaults to an empty dict."""
        comparator = Comparator()
        assert comparator.llm_prompt_extra_vars == {}


class TestEmbeddingCompare:
    """Tests for embedding-based comparison."""

    def test_embedding_compare_similar(self) -> None:
        """Test embedding compare with similar vectors."""
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
        """Test embedding compare with threshold returns boolean."""
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
        """Test embedding compare without threshold returns float."""
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
        """Test embedding compare tracks cost."""
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
    """Tests for the main compare method."""

    def test_compare_dispatches_to_exact(self) -> None:
        """Test compare dispatches to exact comparison."""
        comparator = Comparator(element_compare="exact")
        assert comparator.compare("hello", "hello") is True
        assert comparator.compare("hello", "world") is False

    def test_compare_dispatches_to_text_similarity(self) -> None:
        """Test compare dispatches to text similarity comparison."""
        comparator = Comparator(element_compare="text_similarity")
        result = comparator.compare("hello", "hello")
        assert result == pytest.approx(1.0)

    def test_compare_dispatches_to_llm(self) -> None:
        """Test compare dispatches to LLM comparison."""
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
        """Test compare dispatches to embedding comparison."""
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
        """Test compare raises error for unknown type."""
        comparator = Comparator()
        comparator.element_compare = "invalid_type"

        with pytest.raises(ValueError, match="Unknown element_compare type"):
            comparator.compare("hello", "world")

    def test_compare_increments_count(self) -> None:
        """Test compare increments comparison count."""
        comparator = Comparator(element_compare="exact")
        assert comparator.comparison_count == 0

        comparator.compare("a", "a")
        assert comparator.comparison_count == 1

        comparator.compare("b", "b")
        assert comparator.comparison_count == 2


class TestComparatorIsScoreMode:
    """Tests for is_score_mode method."""

    def test_is_score_mode_embedding_no_threshold(self) -> None:
        """Test is_score_mode returns True for embedding without threshold."""
        comparator = Comparator(element_compare="embedding", embedding_threshold=None)
        assert comparator.is_score_mode() is True

    def test_is_score_mode_embedding_with_threshold(self) -> None:
        """Test is_score_mode returns False for embedding with threshold."""
        comparator = Comparator(element_compare="embedding", embedding_threshold=0.9)
        assert comparator.is_score_mode() is False

    def test_is_score_mode_text_similarity_no_threshold(self) -> None:
        """Test is_score_mode returns True for text_similarity without threshold."""
        comparator = Comparator(
            element_compare="text_similarity",
            text_similarity_threshold=None,
        )
        assert comparator.is_score_mode() is True

    def test_is_score_mode_text_similarity_with_threshold(self) -> None:
        """Test is_score_mode returns False for text_similarity with threshold."""
        comparator = Comparator(
            element_compare="text_similarity",
            text_similarity_threshold=0.8,
        )
        assert comparator.is_score_mode() is False

    def test_is_score_mode_exact_always_false(self) -> None:
        """Test is_score_mode returns False for exact comparison."""
        comparator = Comparator(element_compare="exact")
        assert comparator.is_score_mode() is False


class TestComparatorStats:
    """Tests for statistics tracking."""

    def test_get_stats_initial(self) -> None:
        """Test get_stats returns initial values."""
        comparator = Comparator()
        stats = comparator.get_stats()

        assert stats["comparison_count"] == 0
        assert stats["total_comparison_cost"] == 0.0

    def test_get_stats_after_comparisons(self) -> None:
        """Test get_stats after some comparisons."""
        comparator = Comparator(element_compare="exact")
        comparator.compare("a", "a")
        comparator.compare("b", "b")

        stats = comparator.get_stats()
        assert stats["comparison_count"] == 2


class TestGraderInitialization:
    """Tests for Grader initialization."""

    def test_grader_initialization(self) -> None:
        """Test grader stores comparator and settings."""
        comparator = Comparator()
        grader = Grader(comparator=comparator, order_matters=False, threshold=0.7)

        assert grader.comparator is comparator
        assert grader.order_matters is False
        assert grader.threshold == 0.7

    def test_grader_default_values(self) -> None:
        """Test grader default values."""
        comparator = Comparator()
        grader = Grader(comparator=comparator)

        assert grader.order_matters is True
        assert grader.threshold == 0.5


class TestGraderGradeStr:
    """Tests for string grading."""

    def test_grade_str_delegates_to_comparator(self) -> None:
        """Test grade_str delegates to comparator."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        assert grader.grade_str("hello", "hello") is True
        assert grader.grade_str("hello", "world") is False


class TestGraderGradeList:
    """Tests for list grading."""

    def test_grade_list_same_length(self) -> None:
        """Test grade_list with same length lists."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        result = grader.grade_list(["a", "b", "c"], ["a", "b", "c"])
        assert result == [True, True, True]

    def test_grade_list_different_length_raises_error_when_order_matters(self) -> None:
        """Test grade_list raises error for different length lists when order_matters=True."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        with pytest.raises(ValueError, match="Lists must have same length when order_matters=True"):
            grader.grade_list(["a", "b"], ["a", "b", "c"], order_matters=True)

    def test_grade_list_different_length_allowed_when_order_not_matters(self) -> None:
        """Test grade_list allows different length lists when order_matters=False."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        # 3 predicted vs 2 expected: 2 matches (a, b) + 1 unmatched (c) = 0
        result = grader.grade_list(["a", "b", "c"], ["a", "b"], order_matters=False, aggregation="avg")
        assert result == pytest.approx(2 / 3)  # (1 + 1 + 0) / 3

    def test_grade_list_greedy_matching(self) -> None:
        """Test grade_list uses greedy N×M matching when order_matters=False."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        # Items are in different order, but greedy matching should find optimal pairs
        # predicted: ["c", "a", "b"] vs expected: ["b", "c", "a"]
        # Greedy matching: a↔a, b↔b, c↔c (all match)
        result = grader.grade_list(["c", "a", "b"], ["b", "c", "a"], order_matters=False, aggregation="avg")
        assert result == 1.0

    def test_grade_list_greedy_matching_partial(self) -> None:
        """Test grade_list greedy matching with partial matches."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        # predicted: ["a", "x", "b"] vs expected: ["b", "y", "a"]
        # Greedy matching: a↔a (1.0), b↔b (1.0), x↔y (0.0)
        result = grader.grade_list(["a", "x", "b"], ["b", "y", "a"], order_matters=False, aggregation="avg")
        assert result == pytest.approx(2 / 3)

    def test_grade_list_order_matters_true(self) -> None:
        """Test grade_list with order_matters=True."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        result = grader.grade_list(["a", "b"], ["b", "a"], order_matters=True)
        assert result == [False, False]

    def test_grade_list_order_matters_false_requires_aggregation(self) -> None:
        """Test grade_list raises ValueError when order_matters=False without aggregation."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        with pytest.raises(ValueError, match="aggregation is required when order_matters=False"):
            grader.grade_list(["b", "a"], ["a", "b"], order_matters=False)

    def test_grade_list_order_matters_false_with_avg(self) -> None:
        """Test grade_list with order_matters=False and avg aggregation."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        result = grader.grade_list(["b", "a"], ["a", "b"], order_matters=False, aggregation="avg")
        assert result == 1.0

    def test_grade_list_order_matters_false_with_all(self) -> None:
        """Test grade_list with order_matters=False and all aggregation."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        result = grader.grade_list(["b", "a"], ["a", "b"], order_matters=False, aggregation="all")
        assert result is True

    def test_grade_list_order_matters_false_with_any(self) -> None:
        """Test grade_list with order_matters=False and any aggregation."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        result = grader.grade_list(["b", "a", "x"], ["a", "b", "c"], order_matters=False, aggregation="any")
        assert result is True

    def test_grade_list_aggregation_avg(self) -> None:
        """Test grade_list with avg aggregation."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        result = grader.grade_list(["a", "b", "c"], ["a", "b", "x"], aggregation="avg")
        assert result == pytest.approx(2 / 3)

    def test_grade_list_aggregation_all_pass(self) -> None:
        """Test grade_list with all aggregation - all pass."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        result = grader.grade_list(["a", "b"], ["a", "b"], aggregation="all")
        assert result is True

    def test_grade_list_aggregation_all_fail(self) -> None:
        """Test grade_list with all aggregation - one fails."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        result = grader.grade_list(["a", "x"], ["a", "b"], aggregation="all")
        assert result is False

    def test_grade_list_aggregation_any_pass(self) -> None:
        """Test grade_list with any aggregation - one passes."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        result = grader.grade_list(["a", "x"], ["a", "b"], aggregation="any")
        assert result is True

    def test_grade_list_aggregation_any_fail(self) -> None:
        """Test grade_list with any aggregation - all fail."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        result = grader.grade_list(["x", "y"], ["a", "b"], aggregation="any")
        assert result is False

    def test_grade_list_empty_lists(self) -> None:
        """Test grade_list with empty lists."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        result = grader.grade_list([], [])
        assert result == []

    def test_grade_list_uses_instance_defaults_raises_without_aggregation(self) -> None:
        """Test grade_list raises ValueError when instance default order_matters=False without aggregation."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator, order_matters=False)

        # Should raise ValueError because order_matters=False requires aggregation
        with pytest.raises(ValueError, match="aggregation is required when order_matters=False"):
            grader.grade_list(["b", "a"], ["a", "b"])

    def test_grade_list_uses_instance_defaults_with_aggregation(self) -> None:
        """Test grade_list uses instance default for order_matters with aggregation."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator, order_matters=False)

        # Should use order_matters=False by default, works with aggregation
        result = grader.grade_list(["b", "a"], ["a", "b"], aggregation="avg")
        assert result == 1.0

    def test_grade_list_embedding_max_length_exceeded(self) -> None:
        """Test grade_list raises error when list exceeds max length for embedding."""
        comparator = Comparator(element_compare="embedding")
        grader = Grader(comparator=comparator)

        # 11 elements exceeds the limit of 10
        long_list = [f"item{i}" for i in range(11)]
        with pytest.raises(ValueError, match="List too long for embedding comparison"):
            grader.grade_list(long_list, long_list, order_matters=False, aggregation="avg")

    def test_grade_list_llm_max_length_exceeded(self) -> None:
        """Test grade_list raises error when list exceeds max length for llm."""
        comparator = Comparator(element_compare="llm")
        grader = Grader(comparator=comparator)

        # 11 elements exceeds the limit of 10
        long_list = [f"item{i}" for i in range(11)]
        with pytest.raises(ValueError, match="List too long for llm comparison"):
            grader.grade_list(long_list, long_list, order_matters=False, aggregation="avg")

    def test_grade_list_exact_no_length_limit(self) -> None:
        """Test grade_list has no length limit for exact comparison."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        # 20 elements should work fine for exact comparison
        long_list = [f"item{i}" for i in range(20)]
        result = grader.grade_list(long_list, long_list, order_matters=False, aggregation="avg")
        assert result == 1.0

    def test_grade_list_text_similarity_no_length_limit(self) -> None:
        """Test grade_list has no length limit for text_similarity comparison."""
        comparator = Comparator(element_compare="text_similarity")
        grader = Grader(comparator=comparator)

        # 20 elements should work fine for text_similarity comparison
        long_list = [f"item{i}" for i in range(20)]
        result = grader.grade_list(long_list, long_list, order_matters=False, aggregation="avg")
        assert result == 1.0


class TestGraderGradeJSON:
    """Tests for JSON grading."""

    def test_grade_json_string_values(self) -> None:
        """Test grade_json with string values."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        predicted = {"name": "John", "city": "NYC"}
        expected = {"name": "John", "city": "NYC"}

        result = grader.grade_json(predicted, expected)
        assert result == {"name": True, "city": True}

    def test_grade_json_from_string_input(self) -> None:
        """Test grade_json with JSON string input."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        predicted = '{"name": "John"}'
        expected = '{"name": "John"}'

        result = grader.grade_json(predicted, expected)
        assert result == {"name": True}

    def test_grade_json_nested_dict(self) -> None:
        """Test grade_json with nested dictionaries."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        predicted = {"person": {"name": "John", "age": "30"}}
        expected = {"person": {"name": "John", "age": "30"}}

        result = grader.grade_json(predicted, expected)
        assert result == {"person": {"name": True, "age": True}}

    def test_grade_json_with_list(self) -> None:
        """Test grade_json with list values."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        predicted = {"tags": ["a", "b", "c"]}
        expected = {"tags": ["a", "b", "c"]}

        result = grader.grade_json(predicted, expected)
        assert result == {"tags": [True, True, True]}

    def test_grade_json_with_list_aggregation(self) -> None:
        """Test grade_json with list aggregation."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        predicted = {"tags": ["a", "b", "x"]}
        expected = {"tags": ["a", "b", "c"]}

        result = grader.grade_json(predicted, expected, aggregation="avg")
        assert result == {"tags": pytest.approx(2 / 3)}

    def test_grade_json_missing_key_in_predicted(self) -> None:
        """Test grade_json when key is missing in predicted."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        predicted = {"name": "John"}
        expected = {"name": "John", "city": "NYC"}

        result = grader.grade_json(predicted, expected)
        assert result["name"] is True
        assert result["city"] is False  # Empty string vs "NYC"

    def test_grade_json_missing_key_with_list(self) -> None:
        """Test grade_json when missing key is a list."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        predicted = {}
        expected = {"tags": ["a", "b"]}

        result = grader.grade_json(predicted, expected)
        assert result == {"tags": [False, False]}

    def test_grade_json_list_length_mismatch(self) -> None:
        """Test grade_json with list length mismatch."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        predicted = {"tags": ["a"]}
        expected = {"tags": ["a", "b", "c"]}

        result = grader.grade_json(predicted, expected)
        # Length mismatch returns failure values
        assert result == {"tags": [False, False, False]}

    def test_grade_json_complex_nested_structure(self) -> None:
        """Test grade_json with complex nested structure."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        predicted = {
            "person": {
                "name": "John",
                "address": {"city": "NYC", "zip": "10001"},
            },
            "tags": ["dev", "python"],
        }
        expected = {
            "person": {
                "name": "John",
                "address": {"city": "NYC", "zip": "10001"},
            },
            "tags": ["dev", "python"],
        }

        result = grader.grade_json(predicted, expected)
        assert result == {
            "person": {
                "name": True,
                "address": {"city": True, "zip": True},
            },
            "tags": [True, True],
        }


class TestGraderStats:
    """Tests for grader statistics."""

    def test_grader_get_stats_delegates_to_comparator(self) -> None:
        """Test grader get_stats delegates to comparator."""
        comparator = Comparator(element_compare="exact")
        grader = Grader(comparator=comparator)

        grader.grade_str("a", "a")
        grader.grade_str("b", "b")

        stats = grader.get_stats()
        assert stats["comparison_count"] == 2


class TestElementCompareCategory:
    """Tests for the element_compare_category function and its backwards-compat wrapper."""

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

    def test_uses_third_party_wrapper_false_for_local(self) -> None:
        uses_tp, _ = element_compare_uses_third_party("exact", {})
        assert uses_tp is False

    def test_uses_third_party_wrapper_true_for_llm(self) -> None:
        uses_tp, _ = element_compare_uses_third_party("llm", {})
        assert uses_tp is True

    def test_uses_third_party_wrapper_true_for_embedding(self) -> None:
        uses_tp, _ = element_compare_uses_third_party("embedding", {})
        assert uses_tp is True

    def test_wrapper_consistent_with_category_for_all_strategies(self) -> None:
        for strategy, params in [
            ("exact", {}),
            ("text_similarity", {}),
            ("text_similarity", {"text_similarity_metric": "cosine"}),
            ("llm", {}),
            ("embedding", {}),
        ]:
            cat, _ = element_compare_category(strategy, params)
            uses_tp, _ = element_compare_uses_third_party(strategy, params)
            assert uses_tp == (cat != "local"), (
                f"Wrapper inconsistent with category for strategy={strategy!r}: "
                f"category={cat!r}, uses_third_party={uses_tp}"
            )


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
