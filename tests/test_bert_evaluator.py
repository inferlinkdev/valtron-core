"""Tests for BERT evaluator module."""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch

from valtron_core.bert_evaluator import BERTEvaluator, create_bert_model_for_comparison
from valtron_core.models import Document, Label, EvaluationInput, PredictionResult


class TestBERTEvaluatorInit:
    """Tests for BERTEvaluator initialization."""

    def test_init_with_trainer(self):
        """Test initialization with a trainer."""
        mock_trainer = Mock()
        evaluator = BERTEvaluator(trainer=mock_trainer)

        assert evaluator.trainer is mock_trainer


class TestEvaluateSingle:
    """Tests for BERTEvaluator.evaluate_single method."""

    @pytest.mark.asyncio
    async def test_evaluate_single_correct_prediction(self):
        """Test evaluating a single document with correct prediction."""
        mock_trainer = Mock()
        mock_trainer.predict_single = Mock(return_value="positive")

        evaluator = BERTEvaluator(trainer=mock_trainer)

        document = Document(id="doc-1", content="Great product!")
        label = Label(document_id="doc-1", value="positive")

        result = await evaluator.evaluate_single(document, label)

        assert isinstance(result, PredictionResult)
        assert result.document_id == "doc-1"
        assert result.predicted_value == "positive"
        assert result.expected_value == "positive"
        assert result.is_correct is True
        assert result.cost == 0.0
        assert result.model == "bert-local"
        mock_trainer.predict_single.assert_called_once_with("Great product!")

    @pytest.mark.asyncio
    async def test_evaluate_single_incorrect_prediction(self):
        """Test evaluating a single document with incorrect prediction."""
        mock_trainer = Mock()
        mock_trainer.predict_single = Mock(return_value="negative")

        evaluator = BERTEvaluator(trainer=mock_trainer)

        document = Document(id="doc-1", content="Great product!")
        label = Label(document_id="doc-1", value="positive")

        result = await evaluator.evaluate_single(document, label)

        assert result.is_correct is False
        assert result.predicted_value == "negative"
        assert result.expected_value == "positive"

    @pytest.mark.asyncio
    async def test_evaluate_single_case_insensitive(self):
        """Test that comparison is case insensitive."""
        mock_trainer = Mock()
        mock_trainer.predict_single = Mock(return_value="POSITIVE")

        evaluator = BERTEvaluator(trainer=mock_trainer)

        document = Document(id="doc-1", content="Great product!")
        label = Label(document_id="doc-1", value="positive")

        result = await evaluator.evaluate_single(document, label)

        assert result.is_correct is True

    @pytest.mark.asyncio
    async def test_evaluate_single_strips_whitespace(self):
        """Test that comparison strips whitespace."""
        mock_trainer = Mock()
        mock_trainer.predict_single = Mock(return_value="  positive  ")

        evaluator = BERTEvaluator(trainer=mock_trainer)

        document = Document(id="doc-1", content="Great product!")
        label = Label(document_id="doc-1", value="positive")

        result = await evaluator.evaluate_single(document, label)

        assert result.is_correct is True

    @pytest.mark.asyncio
    async def test_evaluate_single_prediction_error(self):
        """Test handling of prediction errors."""
        mock_trainer = Mock()
        mock_trainer.predict_single = Mock(side_effect=RuntimeError("Model error"))

        evaluator = BERTEvaluator(trainer=mock_trainer)

        document = Document(id="doc-1", content="Great product!")
        label = Label(document_id="doc-1", value="positive")

        result = await evaluator.evaluate_single(document, label)

        assert result.is_correct is False
        assert "ERROR:" in result.predicted_value
        assert "Model error" in result.predicted_value
        assert result.metadata.get("error") == "Model error"

    @pytest.mark.asyncio
    async def test_evaluate_single_response_time_tracked(self):
        """Test that response time is tracked."""
        mock_trainer = Mock()
        mock_trainer.predict_single = Mock(return_value="positive")

        evaluator = BERTEvaluator(trainer=mock_trainer)

        document = Document(id="doc-1", content="Great product!")
        label = Label(document_id="doc-1", value="positive")

        result = await evaluator.evaluate_single(document, label)

        assert result.response_time >= 0
        assert result.response_time < 1  # Should be fast for mocked call


class TestEvaluate:
    """Tests for BERTEvaluator.evaluate method."""

    @pytest.mark.asyncio
    async def test_evaluate_batch_all_correct(self):
        """Test evaluating a batch with all correct predictions."""
        mock_trainer = Mock()
        mock_trainer.predict_single = Mock(side_effect=["positive", "negative"])

        evaluator = BERTEvaluator(trainer=mock_trainer)

        documents = [
            Document(id="doc-1", content="Great!"),
            Document(id="doc-2", content="Terrible!"),
        ]
        labels = [
            Label(document_id="doc-1", value="positive"),
            Label(document_id="doc-2", value="negative"),
        ]

        eval_input = EvaluationInput(
            documents=documents,
            labels=labels,
            prompt_template="Test prompt",
            model="bert-local",
        )

        result = await evaluator.evaluate(eval_input)

        assert len(result.predictions) == 2
        assert all(p.is_correct for p in result.predictions)
        assert result.status == "completed"
        assert result.metrics is not None
        assert result.metrics.accuracy == 1.0

    @pytest.mark.asyncio
    async def test_evaluate_batch_mixed_results(self):
        """Test evaluating a batch with mixed results."""
        mock_trainer = Mock()
        mock_trainer.predict_single = Mock(side_effect=["positive", "positive"])  # Second is wrong

        evaluator = BERTEvaluator(trainer=mock_trainer)

        documents = [
            Document(id="doc-1", content="Great!"),
            Document(id="doc-2", content="Terrible!"),
        ]
        labels = [
            Label(document_id="doc-1", value="positive"),
            Label(document_id="doc-2", value="negative"),
        ]

        eval_input = EvaluationInput(
            documents=documents,
            labels=labels,
            prompt_template="Test prompt",
            model="bert-local",
        )

        result = await evaluator.evaluate(eval_input)

        assert len(result.predictions) == 2
        assert result.predictions[0].is_correct is True
        assert result.predictions[1].is_correct is False
        assert result.metrics.accuracy == 0.5

    @pytest.mark.asyncio
    async def test_evaluate_missing_labels_skipped(self):
        """Test that documents without labels are skipped."""
        mock_trainer = Mock()
        mock_trainer.predict_single = Mock(return_value="positive")

        evaluator = BERTEvaluator(trainer=mock_trainer)

        documents = [
            Document(id="doc-1", content="Great!"),
            Document(id="doc-2", content="Missing label doc"),
        ]
        labels = [
            Label(document_id="doc-1", value="positive"),
            # No label for doc-2
        ]

        eval_input = EvaluationInput(
            documents=documents,
            labels=labels,
            prompt_template="Test prompt",
            model="bert-local",
        )

        result = await evaluator.evaluate(eval_input)

        assert len(result.predictions) == 1
        assert result.predictions[0].document_id == "doc-1"

    @pytest.mark.asyncio
    async def test_evaluate_creates_run_id(self):
        """Test that a run ID is created."""
        mock_trainer = Mock()
        mock_trainer.predict_single = Mock(return_value="positive")

        evaluator = BERTEvaluator(trainer=mock_trainer)

        documents = [Document(id="doc-1", content="Great!")]
        labels = [Label(document_id="doc-1", value="positive")]

        eval_input = EvaluationInput(
            documents=documents,
            labels=labels,
            prompt_template="Test prompt",
            model="bert-local",
        )

        result = await evaluator.evaluate(eval_input)

        assert result.run_id is not None
        assert len(result.run_id) > 0

    @pytest.mark.asyncio
    async def test_evaluate_sets_timestamps(self):
        """Test that timestamps are set."""
        mock_trainer = Mock()
        mock_trainer.predict_single = Mock(return_value="positive")

        evaluator = BERTEvaluator(trainer=mock_trainer)

        documents = [Document(id="doc-1", content="Great!")]
        labels = [Label(document_id="doc-1", value="positive")]

        eval_input = EvaluationInput(
            documents=documents,
            labels=labels,
            prompt_template="Test prompt",
            model="bert-local",
        )

        result = await evaluator.evaluate(eval_input)

        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.completed_at >= result.started_at

    @pytest.mark.asyncio
    async def test_evaluate_cost_is_zero(self):
        """Test that BERT evaluation cost is zero."""
        mock_trainer = Mock()
        mock_trainer.predict_single = Mock(return_value="positive")

        evaluator = BERTEvaluator(trainer=mock_trainer)

        documents = [Document(id="doc-1", content="Great!")]
        labels = [Label(document_id="doc-1", value="positive")]

        eval_input = EvaluationInput(
            documents=documents,
            labels=labels,
            prompt_template="Test prompt",
            model="bert-local",
        )

        result = await evaluator.evaluate(eval_input)

        assert result.metrics.total_cost == 0.0


class TestCreateBertModelForComparison:
    """Tests for create_bert_model_for_comparison function."""

    def test_create_bert_model_returns_trainer(self):
        """Test that the function returns a trained BERTTrainer."""
        documents = [
            Document(id="doc-1", content="Great product!"),
            Document(id="doc-2", content="Terrible experience."),
        ]
        labels = [
            Label(document_id="doc-1", value="positive"),
            Label(document_id="doc-2", value="negative"),
        ]

        with patch("valtron_core.bert_evaluator.BERTTrainer") as MockTrainer:
            mock_trainer = Mock()
            mock_trainer.prepare_data = Mock(return_value=(Mock(), Mock()))
            mock_trainer.train = Mock(return_value={
                "eval_accuracy": 0.95,
                "model_dir": "./bert_models"
            })
            MockTrainer.return_value = mock_trainer

            result = create_bert_model_for_comparison(
                documents=documents,
                labels=labels,
                model_name="bert-base-uncased",
                output_dir="./test_models",
                num_epochs=1,
                batch_size=4,
            )

            assert result is mock_trainer
            MockTrainer.assert_called_once_with(
                model_name="bert-base-uncased",
                output_dir="./test_models",
            )
            mock_trainer.prepare_data.assert_called_once_with(documents, labels)
            mock_trainer.train.assert_called_once()
