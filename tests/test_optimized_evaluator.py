"""Tests for optimized evaluator module."""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from valtron_core.optimized_evaluator import (
    OptimizedPromptEvaluator,
    compare_original_vs_decomposed,
)
from valtron_core.models import Document, Label, EvaluationResult


class TestOptimizedPromptEvaluatorInit:
    """Tests for OptimizedPromptEvaluator initialization."""

    def test_init(self):
        """Test basic initialization."""
        mock_chain_evaluator = Mock()
        mock_decomposer = Mock()

        evaluator = OptimizedPromptEvaluator(
            chain_evaluator=mock_chain_evaluator,
            decomposer=mock_decomposer,
        )

        assert evaluator.chain_evaluator is mock_chain_evaluator
        assert evaluator.decomposer is mock_decomposer


class TestEvaluateWithDecomposition:
    """Tests for evaluate_with_decomposition method."""

    @pytest.mark.asyncio
    async def test_evaluate_with_decomposition_success(self):
        """Test successful evaluation with decomposition."""
        mock_chain_evaluator = Mock()
        mock_decomposer = Mock()

        # Mock decomposer
        mock_decomposer.optimize = AsyncMock(return_value={
            "sub_prompts": ["Step 1", "Step 2"],
            "strategy": "chain",
        })
        mock_decomposer.create_chained_prompts = Mock(return_value=[
            "First step: {content}",
            "Second step: {previous_output}",
        ])

        # Mock chain evaluator
        mock_chain_evaluator.execute_chain = AsyncMock(return_value=(
            "positive",  # final output
            ["intermediate result 1"],  # intermediate outputs
        ))

        evaluator = OptimizedPromptEvaluator(
            chain_evaluator=mock_chain_evaluator,
            decomposer=mock_decomposer,
        )

        documents = [Document(id="doc-1", content="Great product!")]
        labels = [Label(document_id="doc-1", value="positive")]

        result, decomposition = await evaluator.evaluate_with_decomposition(
            documents=documents,
            labels=labels,
            original_prompt="Classify sentiment: {content}",
            model="gpt-4o-mini",
            temperature=0.0,
        )

        assert isinstance(result, EvaluationResult)
        assert result.status == "completed"
        assert len(result.predictions) == 1
        assert result.predictions[0].is_correct is True
        assert "decomposed" in result.model

    @pytest.mark.asyncio
    async def test_evaluate_with_decomposition_stores_intermediates(self):
        """Test that intermediate outputs are stored in metadata."""
        mock_chain_evaluator = Mock()
        mock_decomposer = Mock()

        mock_decomposer.optimize = AsyncMock(return_value={"sub_prompts": ["Step 1"]})
        mock_decomposer.create_chained_prompts = Mock(return_value=["Prompt 1"])

        intermediate_outputs = ["First step result", "Second step result"]
        mock_chain_evaluator.execute_chain = AsyncMock(return_value=(
            "positive",
            intermediate_outputs,
        ))

        evaluator = OptimizedPromptEvaluator(
            chain_evaluator=mock_chain_evaluator,
            decomposer=mock_decomposer,
        )

        documents = [Document(id="doc-1", content="Test")]
        labels = [Label(document_id="doc-1", value="positive")]

        result, _ = await evaluator.evaluate_with_decomposition(
            documents=documents,
            labels=labels,
            original_prompt="Classify: {content}",
            model="gpt-4o-mini",
        )

        assert result.predictions[0].metadata["intermediate_outputs"] == intermediate_outputs

    @pytest.mark.asyncio
    async def test_evaluate_with_decomposition_missing_labels(self):
        """Test that documents without labels are skipped."""
        mock_chain_evaluator = Mock()
        mock_decomposer = Mock()

        mock_decomposer.optimize = AsyncMock(return_value={"sub_prompts": ["Step 1"]})
        mock_decomposer.create_chained_prompts = Mock(return_value=["Prompt 1"])
        mock_chain_evaluator.execute_chain = AsyncMock(return_value=("positive", []))

        evaluator = OptimizedPromptEvaluator(
            chain_evaluator=mock_chain_evaluator,
            decomposer=mock_decomposer,
        )

        documents = [
            Document(id="doc-1", content="Test 1"),
            Document(id="doc-2", content="Test 2"),
        ]
        labels = [Label(document_id="doc-1", value="positive")]  # No label for doc-2

        result, _ = await evaluator.evaluate_with_decomposition(
            documents=documents,
            labels=labels,
            original_prompt="Classify: {content}",
            model="gpt-4o-mini",
        )

        assert len(result.predictions) == 1
        assert result.predictions[0].document_id == "doc-1"

    @pytest.mark.asyncio
    async def test_evaluate_with_decomposition_custom_comparison_fn(self):
        """Test evaluation with custom comparison function."""
        mock_chain_evaluator = Mock()
        mock_decomposer = Mock()

        mock_decomposer.optimize = AsyncMock(return_value={"sub_prompts": ["Step 1"]})
        mock_decomposer.create_chained_prompts = Mock(return_value=["Prompt 1"])
        mock_chain_evaluator.execute_chain = AsyncMock(return_value=("POSITIVE", []))

        evaluator = OptimizedPromptEvaluator(
            chain_evaluator=mock_chain_evaluator,
            decomposer=mock_decomposer,
        )

        documents = [Document(id="doc-1", content="Test")]
        labels = [Label(document_id="doc-1", value="positive")]

        # Custom comparison that always returns True
        def custom_compare(predicted, expected):
            return True

        result, _ = await evaluator.evaluate_with_decomposition(
            documents=documents,
            labels=labels,
            original_prompt="Classify: {content}",
            model="gpt-4o-mini",
            comparison_fn=custom_compare,
        )

        assert result.predictions[0].is_correct is True

    @pytest.mark.asyncio
    async def test_evaluate_with_decomposition_handles_error(self):
        """Test handling of errors during chain execution."""
        mock_chain_evaluator = Mock()
        mock_decomposer = Mock()

        mock_decomposer.optimize = AsyncMock(return_value={"sub_prompts": ["Step 1"]})
        mock_decomposer.create_chained_prompts = Mock(return_value=["Prompt 1"])
        mock_chain_evaluator.execute_chain = AsyncMock(
            side_effect=RuntimeError("Chain execution failed")
        )

        evaluator = OptimizedPromptEvaluator(
            chain_evaluator=mock_chain_evaluator,
            decomposer=mock_decomposer,
        )

        documents = [Document(id="doc-1", content="Test")]
        labels = [Label(document_id="doc-1", value="positive")]

        result, _ = await evaluator.evaluate_with_decomposition(
            documents=documents,
            labels=labels,
            original_prompt="Classify: {content}",
            model="gpt-4o-mini",
        )

        assert len(result.predictions) == 1
        assert result.predictions[0].is_correct is False
        assert "ERROR:" in result.predictions[0].predicted_value

    @pytest.mark.asyncio
    async def test_evaluate_with_decomposition_computes_metrics(self):
        """Test that metrics are computed after evaluation."""
        mock_chain_evaluator = Mock()
        mock_decomposer = Mock()

        mock_decomposer.optimize = AsyncMock(return_value={"sub_prompts": ["Step 1"]})
        mock_decomposer.create_chained_prompts = Mock(return_value=["Prompt 1"])
        mock_chain_evaluator.execute_chain = AsyncMock(return_value=("positive", []))

        evaluator = OptimizedPromptEvaluator(
            chain_evaluator=mock_chain_evaluator,
            decomposer=mock_decomposer,
        )

        documents = [Document(id="doc-1", content="Test")]
        labels = [Label(document_id="doc-1", value="positive")]

        result, _ = await evaluator.evaluate_with_decomposition(
            documents=documents,
            labels=labels,
            original_prompt="Classify: {content}",
            model="gpt-4o-mini",
        )

        assert result.metrics is not None
        assert result.metrics.accuracy == 1.0


class TestCompareOriginalVsDecomposed:
    """Tests for compare_original_vs_decomposed function.

    Note: These tests verify the function signature and basic behavior.
    Full integration tests would require mocking many internal components.
    """

    def test_function_exists(self):
        """Test that compare_original_vs_decomposed function exists with correct signature."""
        import inspect
        sig = inspect.signature(compare_original_vs_decomposed)
        params = list(sig.parameters.keys())

        assert "documents" in params
        assert "labels" in params
        assert "original_prompt" in params
        assert "model" in params

    def test_function_is_async(self):
        """Test that the function is async."""
        import asyncio
        assert asyncio.iscoroutinefunction(compare_original_vs_decomposed)
