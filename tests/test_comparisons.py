"""Tests for the standalone comparison functions in comparisons.py."""

from unittest.mock import Mock, patch

import pytest

from valtron_core.evaluation.comparisons import (
    _cosine_similarity,
    _embedding_compare,
    _exact_compare,
    _llm_compare,
    _normalize,
    _text_similarity_compare,
)


class TestNormalize:
    def test_strips_whitespace(self) -> None:
        assert _normalize("  hello  ") == "hello"
        assert _normalize("\thello\n") == "hello"

    def test_lowercase_by_default(self) -> None:
        assert _normalize("HELLO") == "hello"
        assert _normalize("HeLLo WoRLd") == "hello world"

    def test_case_sensitive_preserves_case(self) -> None:
        assert _normalize("HELLO", case_sensitive=True) == "HELLO"
        assert _normalize("HeLLo", case_sensitive=True) == "HeLLo"

    def test_ignore_spaces_removes_spaces(self) -> None:
        assert _normalize("hello world", ignore_spaces=True) == "helloworld"
        assert _normalize("a b c", ignore_spaces=True) == "abc"

    def test_combined_options(self) -> None:
        assert _normalize("  Hello World  ", case_sensitive=True, ignore_spaces=True) == "HelloWorld"
        assert _normalize("  Hello World  ", case_sensitive=False, ignore_spaces=True) == "helloworld"


class TestCosineSimilarity:
    def test_identical_vectors(self) -> None:
        vec = [1.0, 2.0, 3.0]
        assert _cosine_similarity(vec, vec) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        assert _cosine_similarity([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        assert _cosine_similarity([1.0, 2.0, 3.0], [-1.0, -2.0, -3.0]) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        assert _cosine_similarity([1.0, 2.0, 3.0], [0.0, 0.0, 0.0]) == 0.0
        assert _cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0]) == 0.0


class TestExactCompare:
    def test_identical_strings(self) -> None:
        assert _exact_compare("hello", "hello") is True

    def test_case_insensitive_by_default(self) -> None:
        assert _exact_compare("Hello", "hello") is True
        assert _exact_compare("WORLD", "world") is True

    def test_case_sensitive(self) -> None:
        assert _exact_compare("Hello", "hello", case_sensitive=True) is False
        assert _exact_compare("Hello", "Hello", case_sensitive=True) is True

    def test_strips_whitespace(self) -> None:
        assert _exact_compare("  hello  ", "hello") is True
        assert _exact_compare("hello", "  hello  ") is True

    def test_ignore_spaces(self) -> None:
        assert _exact_compare("hello world", "helloworld", ignore_spaces=True) is True

    def test_no_match(self) -> None:
        assert _exact_compare("hello", "world") is False
        assert _exact_compare("abc", "def") is False


class TestTextSimilarityCompare:
    def test_fuzz_ratio_identical(self) -> None:
        result = _text_similarity_compare("hello", "hello")
        assert result == pytest.approx(1.0)

    def test_fuzz_ratio_similar(self) -> None:
        result = _text_similarity_compare("hello", "hallo")
        assert isinstance(result, float)
        assert 0.7 < result < 1.0

    def test_fuzz_ratio_threshold_pass(self) -> None:
        result = _text_similarity_compare("hello", "hello", threshold=0.8)
        assert result is True

    def test_fuzz_ratio_threshold_fail(self) -> None:
        result = _text_similarity_compare("hello", "hallo", threshold=0.99)
        assert result is False

    def test_bleu_identical(self) -> None:
        result = _text_similarity_compare(
            "the cat sat on the mat", "the cat sat on the mat", metric="bleu"
        )
        assert result == pytest.approx(1.0)

    def test_bleu_similar(self) -> None:
        result = _text_similarity_compare(
            "the cat sat on the mat", "the cat is on the mat", metric="bleu"
        )
        assert isinstance(result, float)
        assert 0.0 < result < 1.0

    def test_gleu_identical(self) -> None:
        result = _text_similarity_compare(
            "the cat sat on the mat", "the cat sat on the mat", metric="gleu"
        )
        assert result == pytest.approx(1.0)

    def test_gleu_similar(self) -> None:
        result = _text_similarity_compare(
            "the cat sat on the mat", "the cat is on the mat", metric="gleu"
        )
        assert isinstance(result, float)
        assert 0.0 < result < 1.0

    def test_cosine_metric_with_mock_embeddings(self) -> None:
        mock_resp1 = Mock()
        mock_resp1.data = [{"embedding": [1.0, 0.0, 0.0]}]
        mock_resp2 = Mock()
        mock_resp2.data = [{"embedding": [1.0, 0.0, 0.0]}]

        with patch("valtron_core.evaluation.comparisons.embedding") as mock_embedding:
            mock_embedding.side_effect = [mock_resp1, mock_resp2]
            result = _text_similarity_compare("hello", "hello", metric="cosine")

        assert result == pytest.approx(1.0)

    def test_unknown_metric_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown text_similarity metric"):
            _text_similarity_compare("hello", "world", metric="invalid_metric")  # type: ignore[arg-type]

    def test_no_threshold_returns_float(self) -> None:
        result = _text_similarity_compare("hello", "hallo")
        assert isinstance(result, float)

    def test_threshold_returns_bool(self) -> None:
        result = _text_similarity_compare("hello", "hello", threshold=0.5)
        assert isinstance(result, bool)

    def test_case_sensitive_param(self) -> None:
        lower = _text_similarity_compare("HELLO", "hello", case_sensitive=True)
        insensitive = _text_similarity_compare("HELLO", "hello", case_sensitive=False)
        assert isinstance(lower, float) and isinstance(insensitive, float)
        assert insensitive > lower


class TestLLMCompare:
    def _mock_completion(self, content: str) -> Mock:
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = content
        return mock_response

    def test_yes_text_fallback(self) -> None:
        mock_response = self._mock_completion("YES")
        with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparisons.completion", return_value=mock_response):
                result = _llm_compare("NYC", "New York City")
        assert result is True

    def test_no_text_fallback(self) -> None:
        mock_response = self._mock_completion("NO")
        with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparisons.completion", return_value=mock_response):
                result = _llm_compare("apple", "orange")
        assert result is False

    def test_structured_output_true(self) -> None:
        mock_response = self._mock_completion('{"match": true}')
        with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=True):
            with patch("valtron_core.evaluation.comparisons.completion", return_value=mock_response):
                result = _llm_compare("NYC", "New York City")
        assert result is True

    def test_structured_output_false(self) -> None:
        mock_response = self._mock_completion('{"match": false}')
        with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=True):
            with patch("valtron_core.evaluation.comparisons.completion", return_value=mock_response):
                result = _llm_compare("apple", "orange")
        assert result is False

    def test_structured_json_parse_failure_falls_back_to_text(self) -> None:
        mock_response = self._mock_completion("yes.")
        with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=True):
            with patch("valtron_core.evaluation.comparisons.completion", return_value=mock_response):
                result = _llm_compare("a", "a")
        assert result is True

    def test_text_fallback_case_insensitive(self) -> None:
        for content in ("yes", "Yes", "YES", "yes because they match"):
            mock_response = self._mock_completion(content)
            with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=False):
                with patch("valtron_core.evaluation.comparisons.completion", return_value=mock_response):
                    assert _llm_compare("a", "a") is True, f"failed for {content!r}"

    def test_text_fallback_rejects_no(self) -> None:
        for content in ("no", "No", "NO", "no they differ"):
            mock_response = self._mock_completion(content)
            with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=False):
                with patch("valtron_core.evaluation.comparisons.completion", return_value=mock_response):
                    assert _llm_compare("a", "b") is False, f"failed for {content!r}"

    def test_structured_path_passes_response_format(self) -> None:
        from valtron_core.evaluation.comparisons import _MatchResult

        mock_response = self._mock_completion('{"match": true}')
        with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=True):
            with patch("valtron_core.evaluation.comparisons.completion", return_value=mock_response) as mock_comp:
                _llm_compare("a", "a")
        _, kwargs = mock_comp.call_args
        assert kwargs["response_format"] is _MatchResult

    def test_text_fallback_passes_max_tokens(self) -> None:
        mock_response = self._mock_completion("YES")
        with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparisons.completion", return_value=mock_response) as mock_comp:
                _llm_compare("a", "a")
        _, kwargs = mock_comp.call_args
        assert kwargs["max_tokens"] == 10

    def test_supports_response_schema_exception_falls_back(self) -> None:
        mock_response = self._mock_completion("YES")
        with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", side_effect=Exception("err")):
            with patch("valtron_core.evaluation.comparisons.completion", return_value=mock_response):
                result = _llm_compare("a", "a")
        assert result is True

    def test_custom_prompt_template(self) -> None:
        template = "Is '{predicted}' the same as '{expected}'? YES or NO."
        mock_response = self._mock_completion("YES")
        with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparisons.completion", return_value=mock_response) as mock_comp:
                _llm_compare("NYC", "New York", prompt_template=template)
        sent = mock_comp.call_args[1]["messages"][0]["content"]
        assert sent == "Is 'NYC' the same as 'New York'? YES or NO."

    def test_prompt_extra_vars_interpolated(self) -> None:
        template = "Doc: {example_content}\n'{predicted}' vs '{expected}'? YES or NO."
        mock_response = self._mock_completion("YES")
        with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparisons.completion", return_value=mock_response) as mock_comp:
                _llm_compare(
                    "Paris", "Paris",
                    prompt_template=template,
                    prompt_extra_vars={"example_content": "France"},
                )
        sent = mock_comp.call_args[1]["messages"][0]["content"]
        assert "France" in sent

    def test_default_prompt_contains_entity_matching_text(self) -> None:
        mock_response = self._mock_completion("YES")
        with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparisons.completion", return_value=mock_response) as mock_comp:
                _llm_compare("NYC", "New York")
        sent = mock_comp.call_args[1]["messages"][0]["content"]
        assert "same entity or concept" in sent
        assert "Value 1: NYC" in sent
        assert "Value 2: New York" in sent

    def test_model_param_forwarded(self) -> None:
        mock_response = self._mock_completion("YES")
        with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparisons.completion", return_value=mock_response) as mock_comp:
                _llm_compare("a", "a", model="gpt-4")
        assert mock_comp.call_args[1]["model"] == "gpt-4"


class TestEmbeddingCompare:
    def test_identical_vectors_returns_one(self) -> None:
        mock_resp1 = Mock()
        mock_resp1.data = [{"embedding": [1.0, 0.0, 0.0]}]
        mock_resp2 = Mock()
        mock_resp2.data = [{"embedding": [1.0, 0.0, 0.0]}]

        with patch("valtron_core.evaluation.comparisons.embedding") as mock_emb:
            mock_emb.side_effect = [mock_resp1, mock_resp2]
            result = _embedding_compare("hello", "hello")

        assert result == pytest.approx(1.0)

    def test_threshold_pass(self) -> None:
        mock_resp1 = Mock()
        mock_resp1.data = [{"embedding": [1.0, 0.0, 0.0]}]
        mock_resp2 = Mock()
        mock_resp2.data = [{"embedding": [1.0, 0.0, 0.0]}]

        with patch("valtron_core.evaluation.comparisons.embedding") as mock_emb:
            mock_emb.side_effect = [mock_resp1, mock_resp2]
            result = _embedding_compare("hello", "hello", threshold=0.9)

        assert result is True

    def test_no_threshold_returns_float(self) -> None:
        mock_resp1 = Mock()
        mock_resp1.data = [{"embedding": [1.0, 0.5, 0.0]}]
        mock_resp2 = Mock()
        mock_resp2.data = [{"embedding": [0.5, 1.0, 0.0]}]

        with patch("valtron_core.evaluation.comparisons.embedding") as mock_emb:
            mock_emb.side_effect = [mock_resp1, mock_resp2]
            result = _embedding_compare("hello", "world")

        assert isinstance(result, float)
        assert 0.0 < result < 1.0

    def test_model_param_forwarded(self) -> None:
        mock_resp = Mock()
        mock_resp.data = [{"embedding": [1.0, 0.0]}]

        with patch("valtron_core.evaluation.comparisons.embedding") as mock_emb:
            mock_emb.side_effect = [mock_resp, mock_resp]
            _embedding_compare("a", "b", model="text-embedding-3-large")

        assert mock_emb.call_args_list[0][1]["model"] == "text-embedding-3-large"
