"""Tests for the TransformerModelWrapper class."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from valtron_core.transformer_wrapper import TransformerModelWrapper


class TestTransformerModelWrapper:
    """Tests for TransformerModelWrapper class."""

    def test_init_loads_model(self, tmp_path, mock_transformer_classifier):
        """Test that initialization loads the model."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        with patch(
            "valtron_core.transformer_wrapper.TransformerClassifier"
        ) as mock_trainer_class:
            mock_trainer_class.return_value = mock_transformer_classifier

            wrapper = TransformerModelWrapper(
                model_path=model_path,
                model_name="test-transformer",
            )

            mock_transformer_classifier.load_model.assert_called_once_with(model_path)
            assert wrapper.model_name == "test-transformer"

    def test_predict_returns_label(self, tmp_path, mock_transformer_classifier):
        """Test that predict returns the plain label string from the classifier."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        mock_transformer_classifier.predict_single = Mock(return_value="yes")

        with patch(
            "valtron_core.transformer_wrapper.TransformerClassifier"
        ) as mock_trainer_class:
            mock_trainer_class.return_value = mock_transformer_classifier

            wrapper = TransformerModelWrapper(model_path=model_path)

            result = wrapper.predict("Are these locations the same?")

            assert result == "yes"

    def test_predict_returns_different_labels(self, tmp_path, mock_transformer_classifier):
        """Test that predict returns whatever label string the classifier produces."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        mock_transformer_classifier.predict_single = Mock(return_value="no")

        with patch(
            "valtron_core.transformer_wrapper.TransformerClassifier"
        ) as mock_trainer_class:
            mock_trainer_class.return_value = mock_transformer_classifier

            wrapper = TransformerModelWrapper(model_path=model_path)

            result = wrapper.predict("Are these locations different?")

            assert result == "no"

    def test_batch_predict(self, tmp_path, mock_transformer_classifier):
        """Test batch prediction."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        mock_transformer_classifier.predict = Mock(return_value=["yes", "no", "yes"])

        with patch(
            "valtron_core.transformer_wrapper.TransformerClassifier"
        ) as mock_trainer_class:
            mock_trainer_class.return_value = mock_transformer_classifier

            wrapper = TransformerModelWrapper(model_path=model_path)

            results = wrapper.batch_predict([
                "Location 1",
                "Location 2",
                "Location 3",
            ])

            assert len(results) == 3
            assert results[0] == "yes"
            assert results[1] == "no"
            assert results[2] == "yes"

    def test_get_stats(self, tmp_path, mock_transformer_classifier):
        """Test getting wrapper statistics."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        with patch(
            "valtron_core.transformer_wrapper.TransformerClassifier"
        ) as mock_trainer_class:
            mock_trainer_class.return_value = mock_transformer_classifier

            wrapper = TransformerModelWrapper(
                model_path=model_path,
                model_name="my-transformer",
            )

            # Make a prediction to increment count
            wrapper.predict("test")

            stats = wrapper.get_stats()

            assert stats["model_name"] == "my-transformer"
            assert stats["model_path"] == str(model_path)
            assert stats["prediction_count"] == 1
            assert stats["total_cost"] == 0.0

    def test_reset_stats(self, tmp_path, mock_transformer_classifier):
        """Test resetting statistics."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        with patch(
            "valtron_core.transformer_wrapper.TransformerClassifier"
        ) as mock_trainer_class:
            mock_trainer_class.return_value = mock_transformer_classifier

            wrapper = TransformerModelWrapper(model_path=model_path)

            # Make some predictions
            wrapper.predict("test1")
            wrapper.predict("test2")
            wrapper.predict("test3")

            assert wrapper.prediction_count == 3

            # Reset
            wrapper.reset_stats()

            assert wrapper.prediction_count == 0
            assert wrapper.total_cost == 0.0

    def test_prediction_count_increments(self, tmp_path, mock_transformer_classifier):
        """Test that prediction count increments correctly."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        with patch(
            "valtron_core.transformer_wrapper.TransformerClassifier"
        ) as mock_trainer_class:
            mock_trainer_class.return_value = mock_transformer_classifier

            wrapper = TransformerModelWrapper(model_path=model_path)

            assert wrapper.prediction_count == 0

            wrapper.predict("test1")
            assert wrapper.prediction_count == 1

            wrapper.predict("test2")
            assert wrapper.prediction_count == 2

            # Batch predict should add batch size
            mock_transformer_classifier.predict = Mock(return_value=["yes", "no", "yes"])
            wrapper.batch_predict(["a", "b", "c"])
            assert wrapper.prediction_count == 5

    def test_cost_always_zero(self, tmp_path, mock_transformer_classifier):
        """Test that cost is always zero for transformer inference."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        with patch(
            "valtron_core.transformer_wrapper.TransformerClassifier"
        ) as mock_trainer_class:
            mock_trainer_class.return_value = mock_transformer_classifier

            wrapper = TransformerModelWrapper(model_path=model_path)

            # Make many predictions
            for _ in range(100):
                wrapper.predict("test")

            assert wrapper.total_cost == 0.0

            stats = wrapper.get_stats()
            assert stats["total_cost"] == 0.0
            assert stats.get("cost_per_prediction", 0.0) == 0.0

    def test_init_with_string_path(self, tmp_path, mock_transformer_classifier):
        """Test initialization with string path instead of Path object."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        with patch(
            "valtron_core.transformer_wrapper.TransformerClassifier"
        ) as mock_trainer_class:
            mock_trainer_class.return_value = mock_transformer_classifier

            # Pass as string
            wrapper = TransformerModelWrapper(
                model_path=str(model_path),
                model_name="test",
            )

            assert wrapper.model_path == Path(model_path)

    def test_default_model_name(self, tmp_path, mock_transformer_classifier):
        """Test default model name."""
        model_path = tmp_path / "model"
        model_path.mkdir()

        with patch(
            "valtron_core.transformer_wrapper.TransformerClassifier"
        ) as mock_trainer_class:
            mock_trainer_class.return_value = mock_transformer_classifier

            wrapper = TransformerModelWrapper(model_path=model_path)

            assert wrapper.model_name == "transformer"
