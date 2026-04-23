"""Tests for LLM client module."""

from typing import Any
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest
from litellm.utils import ModelResponse

from valtron_core.client import LLMClient


class TestLLMClientInitialization:
    """Tests for LLMClient initialization."""

    def test_client_initialization(self, mock_env_vars: dict[str, str]) -> None:
        """Test basic client initialization."""
        client = LLMClient()

        assert client.config is not None
        assert client._call_count == 0
        assert client._total_cost == 0.0

    def test_stats_initialization(self, mock_env_vars: dict[str, str]) -> None:
        """Test stats are initialized correctly."""
        client = LLMClient()
        stats = client.get_stats()

        assert stats["total_calls"] == 0
        assert stats["total_cost"] == 0.0


class TestLLMClientCompletion:
    """Tests for LLMClient completion methods."""

    @pytest.mark.asyncio
    async def test_complete_success(
        self,
        mock_env_vars: dict[str, str],
        sample_messages: list[dict[str, str]],
        mock_model_response: ModelResponse,
    ) -> None:
        """Test successful async completion."""
        client = LLMClient()

        with patch("valtron_core.client.acompletion", new=AsyncMock(return_value=mock_model_response)):
            response = await client.complete(
                model="gpt-3.5-turbo",
                messages=sample_messages,
                temperature=0.7,
            )

            assert response is not None
            assert response.choices[0].message.content == "The capital of France is Paris."

    @pytest.mark.asyncio
    async def test_complete_increments_call_count(
        self,
        mock_env_vars: dict[str, str],
        sample_messages: list[dict[str, str]],
        mock_model_response: ModelResponse,
    ) -> None:
        """Test that call count is incremented."""
        client = LLMClient()

        with patch("valtron_core.client.acompletion", new=AsyncMock(return_value=mock_model_response)):
            await client.complete(model="gpt-3.5-turbo", messages=sample_messages)
            await client.complete(model="gpt-3.5-turbo", messages=sample_messages)

            stats = client.get_stats()
            assert stats["total_calls"] == 2

    @pytest.mark.asyncio
    async def test_complete_tracks_cost(
        self,
        mock_env_vars: dict[str, str],
        sample_messages: list[dict[str, str]],
        mock_model_response: ModelResponse,
    ) -> None:
        """Test that costs are tracked."""
        client = LLMClient()
        mock_model_response = {
            "choices":[{"message": {"role": "assistant", "content": "LiteLLM is awesome"}}],
            "usage":{"completion_tokens": 100},
            "model":"gpt-3.5-turbo"
        }



        with patch("valtron_core.client.acompletion", new=AsyncMock(return_value=mock_model_response)):
            await client.complete(model="gpt-3.5-turbo", messages=sample_messages)
            await client.complete(model="gpt-3.5-turbo", messages=sample_messages)

            stats = client.get_stats()
            assert stats["total_cost"] != 0

    @pytest.mark.asyncio
    async def test_complete_with_max_tokens(
        self,
        mock_env_vars: dict[str, str],
        sample_messages: list[dict[str, str]],
        mock_model_response: ModelResponse,
    ) -> None:
        """Test completion with max_tokens parameter."""
        client = LLMClient()

        with patch(
            "valtron_core.client.acompletion", new=AsyncMock(return_value=mock_model_response)
        ) as mock_acompletion:
            await client.complete(
                model="gpt-3.5-turbo",
                messages=sample_messages,
                max_tokens=100,
            )

            mock_acompletion.assert_called_once()
            call_kwargs = mock_acompletion.call_args.kwargs
            assert call_kwargs["max_tokens"] == 100

    @pytest.mark.asyncio
    async def test_complete_error_handling(
        self,
        mock_env_vars: dict[str, str],
        sample_messages: list[dict[str, str]],
    ) -> None:
        """Test error handling in completion."""
        client = LLMClient()

        with patch("asyncio.sleep", new=AsyncMock()):
            with patch("valtron_core.client.acompletion", new=AsyncMock(side_effect=Exception("API Error"))):
                with pytest.raises(Exception, match="API Error"):
                    await client.complete(model="gpt-3.5-turbo", messages=sample_messages)


class TestLLMClientSyncCompletion:
    """Tests for synchronous completion."""

    def test_complete_sync_success(
        self,
        mock_env_vars: dict[str, str],
        sample_messages: list[dict[str, str]],
        mock_model_response: ModelResponse,
    ) -> None:
        """Test successful synchronous completion."""
        client = LLMClient()

        with patch("valtron_core.client.completion", return_value=mock_model_response):
            response = client.complete_sync(
                model="gpt-3.5-turbo",
                messages=sample_messages,
            )

            assert response is not None
            assert response.choices[0].message.content == "The capital of France is Paris."

    def test_complete_sync_increments_call_count(
        self,
        mock_env_vars: dict[str, str],
        sample_messages: list[dict[str, str]],
        mock_model_response: ModelResponse,
    ) -> None:
        """Test that sync calls increment call count."""
        client = LLMClient()

        with patch("valtron_core.client.completion", return_value=mock_model_response):
            client.complete_sync(model="gpt-3.5-turbo", messages=sample_messages)
            client.complete_sync(model="gpt-3.5-turbo", messages=sample_messages)

            stats = client.get_stats()
            assert stats["total_calls"] == 2


class TestLLMClientFallback:
    """Tests for fallback functionality."""

    @pytest.mark.asyncio
    async def test_fallback_first_model_succeeds(
        self,
        mock_env_vars: dict[str, str],
        sample_messages: list[dict[str, str]],
        mock_model_response: ModelResponse,
    ) -> None:
        """Test fallback when first model succeeds."""
        client = LLMClient()

        with patch("valtron_core.client.acompletion", new=AsyncMock(return_value=mock_model_response)):
            response = await client.complete_with_fallback(
                models=["gpt-4", "gpt-3.5-turbo"],
                messages=sample_messages,
            )

            assert response is not None
            # Should only make one call since first model succeeds
            stats = client.get_stats()
            assert stats["total_calls"] == 1

    @pytest.mark.asyncio
    async def test_fallback_to_second_model(
        self,
        mock_env_vars: dict[str, str],
        sample_messages: list[dict[str, str]],
        mock_model_response: ModelResponse,
    ) -> None:
        """Test fallback when first model fails all retries."""
        client = LLMClient()

        # First model exhausts all retries (max_retries=3 → 4 attempts), then second succeeds
        n_retries = client.config.optimization.max_retries
        mock_acompletion = AsyncMock(
            side_effect=[Exception("Rate limit")] * (n_retries + 1) + [mock_model_response]
        )

        with patch("asyncio.sleep", new=AsyncMock()):
            with patch("valtron_core.client.acompletion", new=mock_acompletion):
                response = await client.complete_with_fallback(
                    models=["gpt-4", "gpt-3.5-turbo"],
                    messages=sample_messages,
                )

                assert response is not None
                # Should make two complete() calls (one per model)
                stats = client.get_stats()
                assert stats["total_calls"] == 2

    @pytest.mark.asyncio
    async def test_fallback_all_models_fail(
        self,
        mock_env_vars: dict[str, str],
        sample_messages: list[dict[str, str]],
    ) -> None:
        """Test fallback when all models fail."""
        client = LLMClient()

        with patch("asyncio.sleep", new=AsyncMock()):
            with patch(
                "valtron_core.client.acompletion", new=AsyncMock(side_effect=Exception("All failed"))
            ):
                with pytest.raises(Exception, match="All models failed"):
                    await client.complete_with_fallback(
                        models=["gpt-4", "gpt-3.5-turbo", "claude-3-opus"],
                        messages=sample_messages,
                    )

            # Should have tried all three models
            stats = client.get_stats()
            assert stats["total_calls"] == 3


class TestLLMClientStats:
    """Tests for statistics tracking."""

    def test_get_stats(self, mock_env_vars: dict[str, str]) -> None:
        """Test getting statistics."""
        client = LLMClient()
        stats = client.get_stats()

        assert "total_calls" in stats
        assert "total_cost" in stats
        assert isinstance(stats["total_calls"], int)
        assert isinstance(stats["total_cost"], float)

    def test_reset_stats(
        self,
        mock_env_vars: dict[str, str],
        sample_messages: list[dict[str, str]],
        mock_model_response: ModelResponse,
    ) -> None:
        """Test resetting statistics."""
        client = LLMClient()

        # Make some calls
        with patch("valtron_core.client.completion", return_value=mock_model_response):
            client.complete_sync(model="gpt-3.5-turbo", messages=sample_messages)
            client.complete_sync(model="gpt-3.5-turbo", messages=sample_messages)

        # Verify stats are non-zero
        stats = client.get_stats()
        assert stats["total_calls"] == 2

        # Reset
        client.reset_stats()

        # Verify stats are zero
        stats = client.get_stats()
        assert stats["total_calls"] == 0
        assert stats["total_cost"] == 0.0


class TestLLMClientStreaming:
    """Tests for streaming responses."""

    @pytest.mark.asyncio
    async def test_streaming_response(
        self,
        mock_env_vars: dict[str, str],
        sample_messages: list[dict[str, str]],
    ) -> None:
        """Test streaming response handling."""
        client = LLMClient()

        # Create a mock async generator
        async def mock_stream() -> Any:
            chunks = ["Hello", " ", "World"]
            for chunk in chunks:
                response = Mock(spec=ModelResponse)
                response.choices = [Mock()]
                response.choices[0].delta = Mock()
                response.choices[0].delta.content = chunk
                yield response

        with patch("valtron_core.client.acompletion", return_value=mock_stream()):
            response = await client.complete(
                model="gpt-3.5-turbo",
                messages=sample_messages,
                stream=True,
            )

            # Collect streamed chunks
            chunks = []
            async for chunk in response:
                chunks.append(chunk.choices[0].delta.content)

            assert chunks == ["Hello", " ", "World"]
