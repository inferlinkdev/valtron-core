"""Tests for custom model parameter input functionality."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from valtron_core.client import LLMClient
from valtron_core.evaluator import PromptEvaluator
from valtron_core.models import Document, Label, EvaluationInput
from valtron_core.runner import EvaluationRunner


@pytest.fixture
def sample_documents():
    """Sample documents for testing."""
    return [
        Document(id="1", content="This is great!"),
        Document(id="2", content="This is terrible."),
    ]


@pytest.fixture
def sample_labels():
    """Sample labels for testing."""
    return [
        Label(document_id="1", value="positive"),
        Label(document_id="2", value="negative"),
    ]


@pytest.fixture
def mock_response():
    """Mock LiteLLM response."""
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = "positive"
    return response


class TestCustomModelParams:
    """Test suite for custom model parameter input."""

    @pytest.mark.asyncio
    async def test_client_complete_with_string_model(self, mock_response):
        """Test that client.complete works with string model names."""
        client = LLMClient()

        with patch("valtron_core.client.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            messages = [{"role": "user", "content": "Hello"}]
            result = await client.complete(
                model="gpt-3.5-turbo",
                messages=messages,
                temperature=0.7
            )

            # Verify acompletion was called with string model
            mock_acompletion.assert_called_once()
            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["model"] == "gpt-3.5-turbo"
            assert call_kwargs["messages"] == messages
            assert call_kwargs["temperature"] == 0.7

    @pytest.mark.asyncio
    async def test_client_complete_with_dict_model(self, mock_response):
        """Test that client.complete works with dict model definitions."""
        client = LLMClient()

        with patch("valtron_core.client.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            messages = [{"role": "user", "content": "Hello"}]
            model_config = {
                "model": "gpt-4",
                "api_base": "https://custom.endpoint.com/v1",
                "api_key": "custom-key"
            }

            result = await client.complete(
                model=model_config,
                messages=messages,
                temperature=0.5
            )

            # Verify acompletion was called with merged parameters
            mock_acompletion.assert_called_once()
            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["model"] == "gpt-4"
            assert call_kwargs["api_base"] == "https://custom.endpoint.com/v1"
            assert call_kwargs["api_key"] == "custom-key"
            assert call_kwargs["messages"] == messages
            assert call_kwargs["temperature"] == 0.5

    @pytest.mark.asyncio
    async def test_client_complete_dict_overrides_temperature(self, mock_response):
        """Test that dict model parameters can override function parameters."""
        client = LLMClient()

        with patch("valtron_core.client.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            messages = [{"role": "user", "content": "Hello"}]
            model_config = {
                "model": "gpt-4",
                "temperature": 0.0,  # Override temperature in dict
            }

            result = await client.complete(
                model=model_config,
                messages=messages,
                temperature=0.7  # This should be overridden
            )

            # Verify temperature from dict takes precedence
            mock_acompletion.assert_called_once()
            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["temperature"] == 0.0

    @pytest.mark.asyncio
    async def test_client_complete_with_fallback_mixed_types(self, mock_response):
        """Test complete_with_fallback with mixed string and dict models."""
        client = LLMClient()

        with patch("asyncio.sleep", new=AsyncMock()):
            with patch("valtron_core.client.acompletion", new_callable=AsyncMock) as mock_acompletion:
                # First model exhausts all retries, second succeeds
                n_retries = client.config.optimization.max_retries
                mock_acompletion.side_effect = (
                    [Exception("First model failed")] * (n_retries + 1) + [mock_response]
                )

                messages = [{"role": "user", "content": "Hello"}]
                models = [
                    "gpt-3.5-turbo",
                    {"model": "gpt-4", "api_key": "custom-key"}
                ]

                result = await client.complete_with_fallback(
                    models=models,
                    messages=messages
                )

                # First model made n_retries+1 acompletion calls; second model made 1
                total_acompletion_calls = n_retries + 2
                assert mock_acompletion.call_count == total_acompletion_calls

                # All calls for the first model used the string model name
                first_call_kwargs = mock_acompletion.call_args_list[0].kwargs
                assert first_call_kwargs["model"] == "gpt-3.5-turbo"

                # Last call was the dict model
                last_call_kwargs = mock_acompletion.call_args_list[-1].kwargs
                assert last_call_kwargs["model"] == "gpt-4"
                assert last_call_kwargs["api_key"] == "custom-key"

    @pytest.mark.asyncio
    async def test_evaluator_evaluate_single_with_string_model(
        self, sample_documents, sample_labels, mock_response
    ):
        """Test evaluator.evaluate_single with string model."""
        client = LLMClient()
        evaluator = PromptEvaluator(client=client)

        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_response

            result = await evaluator.evaluate_single(
                document=sample_documents[0],
                label=sample_labels[0],
                prompt_template="Classify: {content}",
                model="gpt-3.5-turbo",
                temperature=0.0
            )

            # Verify result
            assert result.document_id == "1"
            assert result.model == "gpt-3.5-turbo"
            assert result.predicted_value == "positive"

            # Verify client was called correctly
            mock_complete.assert_called_once()
            call_kwargs = mock_complete.call_args.kwargs
            assert call_kwargs["model"] == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_evaluator_evaluate_single_with_dict_model(
        self, sample_documents, sample_labels, mock_response
    ):
        """Test evaluator.evaluate_single with dict model definition."""
        client = LLMClient()
        evaluator = PromptEvaluator(client=client)

        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_response

            model_config = {
                "model": "gpt-4",
                "api_base": "https://custom.endpoint.com/v1"
            }

            result = await evaluator.evaluate_single(
                document=sample_documents[0],
                label=sample_labels[0],
                prompt_template="Classify: {content}",
                model=model_config,
                temperature=0.0
            )

            # Verify result uses extracted model name
            assert result.document_id == "1"
            assert result.model == "gpt-4"
            assert result.predicted_value == "positive"

            # Verify client was called with full dict
            mock_complete.assert_called_once()
            call_kwargs = mock_complete.call_args.kwargs
            assert call_kwargs["model"] == model_config

    @pytest.mark.asyncio
    async def test_runner_evaluate_from_file_with_mixed_model_types(
        self, mock_response, tmp_path
    ):
        """Test runner.evaluate_from_file with mixed string and dict models."""
        data = [
            {"id": "1", "content": "This is great!", "label": "positive"},
            {"id": "2", "content": "This is terrible.", "label": "negative"},
        ]
        data_file = tmp_path / "data.json"
        data_file.write_text(json.dumps(data))

        runner = EvaluationRunner()

        with patch("valtron_core.client.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            models = [
                "gpt-3.5-turbo",
                {"model": "gpt-4", "api_key": "custom-key"},
                {"model": "claude-3-opus-20240229", "temperature": 0.0},
            ]

            results = await runner.evaluate_from_file(
                data_file=data_file,
                prompt_template="Classify: {content}",
                models=models,
                max_concurrent=1,
            )

            assert len(results) == 3
            assert results[0].model == "gpt-3.5-turbo"
            assert results[1].model == "gpt-4"
            assert results[2].model == "claude-3-opus-20240229"

    @pytest.mark.asyncio
    async def test_evaluation_input_accepts_string_model(self, sample_documents, sample_labels):
        """Test that EvaluationInput accepts string model."""
        eval_input = EvaluationInput(
            documents=sample_documents,
            labels=sample_labels,
            prompt_template="Classify: {content}",
            model="gpt-3.5-turbo"
        )

        assert eval_input.model == "gpt-3.5-turbo"
        assert isinstance(eval_input.model, str)

    @pytest.mark.asyncio
    async def test_evaluation_input_accepts_dict_model(self, sample_documents, sample_labels):
        """Test that EvaluationInput accepts dict model."""
        model_config = {
            "model": "gpt-4",
            "api_base": "https://custom.endpoint.com/v1",
            "api_key": "custom-key"
        }

        eval_input = EvaluationInput(
            documents=sample_documents,
            labels=sample_labels,
            prompt_template="Classify: {content}",
            model=model_config
        )

        assert eval_input.model == model_config
        assert isinstance(eval_input.model, dict)
        assert eval_input.model["model"] == "gpt-4"

    @pytest.mark.asyncio
    async def test_error_handling_with_dict_model(
        self, sample_documents, sample_labels
    ):
        """Test that errors are properly handled with dict models."""
        client = LLMClient()
        evaluator = PromptEvaluator(client=client)

        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.side_effect = Exception("API Error")

            model_config = {"model": "gpt-4", "api_key": "invalid"}

            result = await evaluator.evaluate_single(
                document=sample_documents[0],
                label=sample_labels[0],
                prompt_template="Classify: {content}",
                model=model_config,
                temperature=0.0
            )

            # Verify error is captured with correct model name
            assert result.is_correct is False
            assert result.model == "gpt-4"
            assert "ERROR:" in result.predicted_value
            assert "API Error" in result.metadata["error"]

    @pytest.mark.asyncio
    async def test_dict_model_without_model_key(self, mock_response):
        """Test handling of dict model without 'model' key."""
        client = LLMClient()

        with patch("valtron_core.client.acompletion", new_callable=AsyncMock) as mock_acompletion:
            mock_acompletion.return_value = mock_response

            # Dict without 'model' key should use 'unknown' for logging
            model_config = {
                "api_base": "https://custom.endpoint.com/v1",
                "api_key": "custom-key"
            }

            messages = [{"role": "user", "content": "Hello"}]

            # This should work, but model name for logging will be 'unknown'
            result = await client.complete(
                model=model_config,
                messages=messages
            )

            mock_acompletion.assert_called_once()

    @pytest.mark.asyncio
    async def test_evaluation_result_model_name_extraction(
        self, sample_documents, sample_labels, mock_response
    ):
        """Test that EvaluationResult correctly extracts model name from dict."""
        client = LLMClient()
        evaluator = PromptEvaluator(client=client)

        with patch.object(client, "complete", new_callable=AsyncMock) as mock_complete:
            mock_complete.return_value = mock_response

            model_config = {
                "model": "gpt-4-custom",
                "api_base": "https://custom.endpoint.com/v1"
            }

            eval_input = EvaluationInput(
                documents=sample_documents,
                labels=sample_labels,
                prompt_template="Classify: {content}",
                model=model_config
            )

            result = await evaluator.evaluate(
                eval_input=eval_input,
                max_concurrent=1
            )

            # Verify EvaluationResult has string model name, not dict
            assert isinstance(result.model, str)
            assert result.model == "gpt-4-custom"
