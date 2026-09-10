"""Tests for the Generator decorators in evaluation/stages/generate.py."""

from unittest.mock import AsyncMock, patch

import pytest

from valtron_core.evaluation.stages.generate import HallucinationFilterGenerator, RawPrediction
from valtron_core.models import Document


class _FakeGenerator:
    """A minimal stand-in Generator, for testing decorators in isolation."""

    def __init__(self, raw: RawPrediction) -> None:
        self._raw = raw
        self.calls: list[dict] = []

    async def generate(self, document, prompt_template, model, **kwargs):
        self.calls.append(
            {"document": document, "prompt_template": prompt_template, "model": model, **kwargs}
        )
        return self._raw


class TestHallucinationFilterGenerator:
    @pytest.mark.asyncio
    async def test_filters_inner_generators_output(self):
        document = Document(id="d1", content="Alice was here.")
        inner_raw = RawPrediction(
            document_id="d1", predicted_value='{"name": "Alice"}', response_time=1.0
        )
        inner = _FakeGenerator(inner_raw)

        with patch(
            "valtron_core.decompose.filter_hallucinated_values",
            new=AsyncMock(return_value='{"name": "Alice", "filtered": true}'),
        ) as mock_filter:
            wrapped = HallucinationFilterGenerator(inner, model="test-model")
            result = await wrapped.generate(document, "prompt: {content}", "test-model")

        mock_filter.assert_awaited_once_with(
            '{"name": "Alice"}', document.content, "test-model", wrapped._client
        )
        assert result.predicted_value == '{"name": "Alice", "filtered": true}'
        assert result is inner_raw  # filtered in place, same RawPrediction object
        assert inner.calls[0]["prompt_template"] == "prompt: {content}"
        assert inner.calls[0]["model"] == "test-model"

    @pytest.mark.asyncio
    async def test_skips_filtering_when_generation_failed(self):
        """A failed generation is returned untouched: nothing to filter."""
        document = Document(id="d1", content="Alice was here.")
        failed_raw = RawPrediction(
            document_id="d1",
            predicted_value="ERROR: boom",
            error="boom",
            response_time=0.5,
        )
        inner = _FakeGenerator(failed_raw)

        with patch(
            "valtron_core.decompose.filter_hallucinated_values",
            new=AsyncMock(),
        ) as mock_filter:
            wrapped = HallucinationFilterGenerator(inner, model="test-model")
            result = await wrapped.generate(document, "prompt: {content}", "test-model")

        mock_filter.assert_not_awaited()
        assert result is failed_raw
        assert result.predicted_value == "ERROR: boom"
