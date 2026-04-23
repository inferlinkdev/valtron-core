"""BERT model training for classification tasks.

.. deprecated::
    Use :class:`evaltron_core.transformer_classifier.TransformerClassifier` instead.
    ``BERTTrainer`` will be removed in a future release.
"""

import warnings

warnings.warn(
    "evaltron_core.bert_trainer.BERTTrainer is deprecated. "
    "Use evaltron_core.transformer_classifier.TransformerClassifier instead.",
    DeprecationWarning,
    stacklevel=2,
)

import json
from pathlib import Path
from typing import Any

import structlog
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from valtron_core.models import Document, Label

logger = structlog.get_logger()


class BERTTrainer:
    """Train and manage BERT-based classification models."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        output_dir: str | Path = "./bert_models",
    ) -> None:
        """
        Initialize BERT trainer.

        Args:
            model_name: Pretrained model name from HuggingFace
            output_dir: Directory to save trained models
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.tokenizer: Any = None
        self.model: Any = None
        self.label_to_id: dict[str, int] = {}
        self.id_to_label: dict[int, str] = {}

    def prepare_data(
        self,
        documents: list[Document],
        labels: list[Label],
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[Dataset, Dataset]:
        """
        Prepare data for training.

        Args:
            documents: List of documents
            labels: List of labels
            test_size: Fraction of data to use for testing
            random_state: Random seed for reproducibility

        Returns:
            Tuple of (train_dataset, test_dataset)
        """
        # Create label mapping
        label_map = {label.document_id: label.value for label in labels}

        # Extract texts and labels
        texts = []
        label_values = []

        for doc in documents:
            if doc.id in label_map:
                texts.append(doc.content)
                label_values.append(label_map[doc.id])

        # Create unique label mappings
        unique_labels = sorted(set(label_values))
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

        # Convert labels to integers
        label_ids = [self.label_to_id[label] for label in label_values]

        # Split data
        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, label_ids, test_size=test_size, random_state=random_state, stratify=label_ids
        )

        # Create datasets
        train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
        test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

        logger.info(
            "data_prepared",
            num_train=len(train_dataset),
            num_test=len(test_dataset),
            num_labels=len(unique_labels),
        )

        return train_dataset, test_dataset

    def _tokenize_function(self, examples: dict[str, Any]) -> dict[str, Any]:
        """Tokenize examples."""
        return self.tokenizer(examples["text"], padding="max_length", truncation=True)

    def train(
        self,
        train_dataset: Dataset,
        test_dataset: Dataset,
        num_epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        warmup_steps: int = 500,
        weight_decay: float = 0.01,
        save_steps: int = 500,
        eval_steps: int = 500,
    ) -> dict[str, Any]:
        """
        Train BERT model.

        Args:
            train_dataset: Training dataset
            test_dataset: Test dataset
            num_epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            warmup_steps: Number of warmup steps
            weight_decay: Weight decay for optimizer
            save_steps: Save checkpoint every N steps
            eval_steps: Evaluate every N steps

        Returns:
            Training metrics and results
        """
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_to_id),
            id2label=self.id_to_label,
            label2id=self.label_to_id,
        )

        # Tokenize datasets
        train_dataset = train_dataset.map(self._tokenize_function, batched=True)
        test_dataset = test_dataset.map(self._tokenize_function, batched=True)

        # Set format for PyTorch
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            logging_dir=str(self.output_dir / "logs"),
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_steps=save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
        )

        # Compute metrics function
        def compute_metrics(eval_pred: Any) -> dict[str, float]:
            predictions, labels = eval_pred
            predictions = predictions.argmax(-1)
            accuracy = accuracy_score(labels, predictions)
            return {"accuracy": accuracy}

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        # Train
        logger.info("training_started", epochs=num_epochs, batch_size=batch_size)
        train_result = trainer.train()

        # Evaluate
        eval_result = trainer.evaluate()

        logger.info("training_completed", train_loss=train_result.training_loss, **eval_result)

        # Save model
        final_model_dir = self.output_dir / "final_model"
        trainer.save_model(str(final_model_dir))
        self.tokenizer.save_pretrained(str(final_model_dir))

        # Save label mappings
        label_map_path = final_model_dir / "label_mapping.json"
        with open(label_map_path, "w") as f:
            json.dump(
                {"label_to_id": self.label_to_id, "id_to_label": self.id_to_label},
                f,
                indent=2,
            )

        return {
            "train_loss": train_result.training_loss,
            "eval_accuracy": eval_result["eval_accuracy"],
            "eval_loss": eval_result["eval_loss"],
            "model_dir": str(final_model_dir),
        }

    def load_model(self, model_dir: str | Path) -> None:
        """
        Load a trained model.

        Args:
            model_dir: Directory containing saved model
        """
        model_dir = Path(model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

        # Load label mappings
        label_map_path = model_dir / "label_mapping.json"
        if label_map_path.exists():
            with open(label_map_path, "r") as f:
                mappings = json.load(f)
                self.label_to_id = mappings["label_to_id"]
                # Convert string keys back to integers for id_to_label
                self.id_to_label = {int(k): v for k, v in mappings["id_to_label"].items()}

        logger.info("model_loaded", model_dir=str(model_dir))

    def predict(self, texts: list[str]) -> list[str]:
        """
        Make predictions on new texts.

        Args:
            texts: List of texts to classify

        Returns:
            List of predicted labels
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Train or load a model first.")

        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

        # Predict
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(-1).tolist()

        # Convert to labels
        predicted_labels = [self.id_to_label[pred] for pred in predictions]

        return predicted_labels

    def predict_single(self, text: str) -> str:
        """
        Predict label for a single text.

        Args:
            text: Text to classify

        Returns:
            Predicted label
        """
        return self.predict([text])[0]

    def evaluate_on_documents(
        self,
        documents: list[Document],
        labels: list[Label],
    ) -> dict[str, Any]:
        """
        Evaluate model on a set of documents.

        Args:
            documents: List of documents
            labels: List of expected labels

        Returns:
            Evaluation metrics
        """
        # Create label mapping
        label_map = {label.document_id: label.value for label in labels}

        # Get predictions
        texts = [doc.content for doc in documents if doc.id in label_map]
        expected_labels = [label_map[doc.id] for doc in documents if doc.id in label_map]

        predicted_labels = self.predict(texts)

        # Calculate metrics
        accuracy = accuracy_score(expected_labels, predicted_labels)
        report = classification_report(expected_labels, predicted_labels, output_dict=True)

        return {
            "accuracy": accuracy,
            "classification_report": report,
            "num_samples": len(texts),
        }
