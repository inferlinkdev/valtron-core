"""Wrapper for integrating trained transformers with the recipe framework."""

from pathlib import Path
from typing import Any

import structlog

from valtron_core.transformer_classifier import TransformerClassifier

logger = structlog.get_logger()


class TransformerModelWrapper:
    """
    Wrapper for using trained transformer models in the recipe evaluation framework.

    Expects a model directory produced by TransformerClassifier.train() (or the
    train_transformer utility). The directory must contain a label_mapping.json
    so that predictions are returned as the original label strings rather than
    integer class indices.
    """

    def __init__(self, model_path: str | Path, model_name: str = "transformer"):
        """
        Initialize the transformer wrapper.

        Args:
            model_path: Path to the trained model directory (the ``final_model``
                        subdir produced by TransformerClassifier.train()).
            model_name: Display name for the model (used in logging and reports).
        """
        self.model_path = Path(model_path)
        self.model_name = model_name

        logger.info("loading_transformer_model", path=str(model_path), name=model_name)
        # Pass output_dir=model_path so TransformerClassifier does not create a
        # stray ./transformer_models/ directory when loading an existing model.
        self._classifier = TransformerClassifier(output_dir=self.model_path)
        self._classifier.load_model(self.model_path)
        logger.info("transformer_model_loaded", name=model_name)

        self.prediction_count = 0
        self.total_cost = 0.0  # Transformers are free!

    def predict(self, document: str) -> str:
        """
        Classify a single document.

        Args:
            document: Text to classify.

        Returns:
            Predicted label string (e.g. "positive", "yes", "category_a").
        """
        label = self._classifier.predict_single(document)
        self.prediction_count += 1
        return label

    def batch_predict(self, documents: list[str]) -> list[str]:
        """
        Classify multiple documents.

        Args:
            documents: List of texts to classify.

        Returns:
            List of predicted label strings.
        """
        labels = self._classifier.predict(documents)
        self.prediction_count += len(documents)
        return labels

    def get_stats(self) -> dict[str, Any]:
        """Return prediction statistics."""
        return {
            "model_name": self.model_name,
            "model_path": str(self.model_path),
            "prediction_count": self.prediction_count,
            "total_cost": self.total_cost,
            "cost_per_prediction": 0.0,
        }

    def reset_stats(self) -> None:
        """Reset prediction statistics."""
        self.prediction_count = 0
        self.total_cost = 0.0
