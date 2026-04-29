"""Tests for evaluation components."""

from unittest.mock import AsyncMock, patch

import pytest
from litellm.utils import ModelResponse

from valtron_core.evaluator import PromptEvaluator
from valtron_core.models import Document, EvaluationInput, Label


class TestPromptEvaluator:
    """Tests for PromptEvaluator."""

    def test_format_prompt(self, mock_env_vars: dict[str, str]) -> None:
        """Test prompt formatting."""
        evaluator = PromptEvaluator()

        template = "Classify this text: {content}"
        doc = Document(id="1", content="This is great!")

        result = evaluator._format_prompt(template, doc)
        assert result == "Classify this text: This is great!"

    def test_normalize_value(self, mock_env_vars: dict[str, str]) -> None:
        """Test value normalization."""
        evaluator = PromptEvaluator()

        assert evaluator._normalize_value("  Positive  ") == "positive"
        assert evaluator._normalize_value("NEGATIVE") == "negative"

    def test_compare_values_default(self, mock_env_vars: dict[str, str]) -> None:
        """Test default value comparison."""
        evaluator = PromptEvaluator()

        assert evaluator._compare_values("positive", "Positive") is True
        assert evaluator._compare_values("positive", "negative") is False
        assert evaluator._compare_values("  yes  ", "YES") is True

    def test_compare_values_custom(self, mock_env_vars: dict[str, str]) -> None:
        """Test custom comparison function."""
        evaluator = PromptEvaluator()

        def custom_compare(pred: str, exp: str, context: str | None = None) -> bool:
            return pred.startswith(exp)

        assert evaluator._compare_values("positive sentiment", "positive", custom_compare) is True
        assert evaluator._compare_values("negative", "positive", custom_compare) is False

    @pytest.mark.asyncio
    async def test_evaluate_single_success(
        self,
        mock_env_vars: dict[str, str],
        mock_model_response: ModelResponse,
    ) -> None:
        """Test successful single evaluation."""
        evaluator = PromptEvaluator()

        doc = Document(id="1", content="This is amazing!")
        label = Label(document_id="1", value="positive")
        template = "Classify: {content}"

        # Mock the response to return "positive"
        mock_model_response.choices[0].message.content = "positive"

        with patch("valtron_core.client.acompletion", new=AsyncMock(return_value=mock_model_response)):
            result = await evaluator.evaluate_single(
                document=doc,
                label=label,
                prompt_template=template,
                model="gpt-3.5-turbo",
            )

            assert result.document_id == "1"
            assert result.predicted_value == "positive"
            assert result.expected_value == "positive"
            assert result.is_correct is True
            assert result.response_time > 0

    @pytest.mark.asyncio
    async def test_evaluate_single_incorrect(
        self,
        mock_env_vars: dict[str, str],
        mock_model_response: ModelResponse,
    ) -> None:
        """Test evaluation with incorrect prediction."""
        evaluator = PromptEvaluator()

        doc = Document(id="1", content="This is terrible!")
        label = Label(document_id="1", value="negative")
        template = "Classify: {content}"

        # Mock the response to return "positive" (incorrect)
        mock_model_response.choices[0].message.content = "positive"

        with patch("valtron_core.client.acompletion", new=AsyncMock(return_value=mock_model_response)):
            result = await evaluator.evaluate_single(
                document=doc,
                label=label,
                prompt_template=template,
                model="gpt-3.5-turbo",
            )

            assert result.is_correct is False
            assert result.predicted_value == "positive"
            assert result.expected_value == "negative"

    @pytest.mark.asyncio
    async def test_evaluate_single_error(
        self,
        mock_env_vars: dict[str, str],
    ) -> None:
        """Test evaluation with API error."""
        evaluator = PromptEvaluator()

        doc = Document(id="1", content="Test")
        label = Label(document_id="1", value="positive")
        template = "Classify: {content}"

        with patch("valtron_core.client.acompletion", new=AsyncMock(side_effect=Exception("API Error"))):
            result = await evaluator.evaluate_single(
                document=doc,
                label=label,
                prompt_template=template,
                model="gpt-3.5-turbo",
            )

            assert result.is_correct is False
            assert "ERROR" in result.predicted_value
            assert result.metadata.get("error") == "API Error"

    @pytest.mark.asyncio
    async def test_evaluate_full(
        self,
        mock_env_vars: dict[str, str],
        mock_model_response: ModelResponse,
    ) -> None:
        """Test full evaluation of multiple documents."""
        evaluator = PromptEvaluator()

        documents = [
            Document(id="1", content="Great!"),
            Document(id="2", content="Terrible!"),
            Document(id="3", content="Okay."),
        ]

        labels = [
            Label(document_id="1", value="positive"),
            Label(document_id="2", value="negative"),
            Label(document_id="3", value="neutral"),
        ]

        # Mock responses: 2 correct, 1 incorrect
        responses = [
            "positive",  # Correct
            "positive",  # Incorrect (should be negative)
            "neutral",   # Correct
        ]

        call_count = 0

        async def mock_acompletion(*args, **kwargs):
            nonlocal call_count
            response = mock_model_response
            response.choices[0].message.content = responses[call_count]
            call_count += 1
            return response

        eval_input = EvaluationInput(
            documents=documents,
            labels=labels,
            prompt_template="Classify: {content}",
            model="gpt-3.5-turbo",
        )

        with patch("valtron_core.client.acompletion", new=mock_acompletion):
            result = await evaluator.evaluate(eval_input)

            assert result.status == "completed"
            assert len(result.predictions) == 3
            assert result.metrics is not None
            assert result.metrics.total_documents == 3
            assert result.metrics.correct_predictions == 2
            assert result.metrics.accuracy == pytest.approx(2/3)            