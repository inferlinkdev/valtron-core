"""Transformer classifier for sequence classification tasks."""

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


class TransformerClassifier:
    """Train and run any HuggingFace sequence classification model.

    Supports arbitrary label sets — label mappings are inferred from training data
    and saved alongside the model so inference works without re-training.

    Typical usage:

        # Training
        classifier = TransformerClassifier(model_name="distilbert-base-uncased", output_dir="./my_model")
        train_ds, test_ds = classifier.prepare_data(documents, labels)
        results = classifier.train(train_ds, test_ds)
        # model saved to ./my_model/final_model/

        # Inference (load saved model)
        classifier = TransformerClassifier(output_dir="./my_model/final_model")
        classifier.load_model("./my_model/final_model")
        label = classifier.predict_single("some text")
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        output_dir: str | Path = "./transformer_models",
    ) -> None:
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
        """Convert documents and labels into train/test datasets.

        Infers the label set from the provided labels and stores the
        label_to_id / id_to_label mappings for use during training and inference.

        Args:
            documents: Input documents.
            labels: Corresponding ground-truth labels.
            test_size: Fraction of data held out for evaluation.
            random_state: Random seed for reproducibility.

        Returns:
            (train_dataset, test_dataset)
        """
        label_map = {label.document_id: label.value for label in labels}

        texts: list[str] = []
        label_values: list[str] = []

        for doc in documents:
            if doc.id in label_map:
                texts.append(doc.content)
                label_values.append(label_map[doc.id])

        unique_labels = sorted(set(label_values))
        self.label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id_to_label = {idx: label for label, idx in self.label_to_id.items()}

        label_ids = [self.label_to_id[label] for label in label_values]

        train_texts, test_texts, train_labels, test_labels = train_test_split(
            texts, label_ids, test_size=test_size, random_state=random_state, stratify=label_ids
        )

        train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
        test_dataset = Dataset.from_dict({"text": test_texts, "label": test_labels})

        logger.info(
            "data_prepared",
            num_train=len(train_dataset),
            num_test=len(test_dataset),
            num_labels=len(unique_labels),
            labels=unique_labels,
        )

        return train_dataset, test_dataset

    def _tokenize_function(self, examples: dict[str, Any]) -> dict[str, Any]:
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
        """Fine-tune the model on the prepared datasets.

        Saves the model, tokenizer, and label_mapping.json to
        ``output_dir/final_model/``.

        Args:
            train_dataset: Training split from prepare_data().
            test_dataset: Test/eval split from prepare_data().
            num_epochs: Number of training epochs.
            batch_size: Per-device batch size.
            learning_rate: AdamW learning rate.
            warmup_steps: Linear warmup steps.
            weight_decay: L2 regularisation weight.
            save_steps: Checkpoint save interval (steps).
            eval_steps: Evaluation interval (steps).

        Returns:
            Dict with keys: model_dir, train_loss, eval_accuracy, eval_loss.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label_to_id),
            id2label=self.id_to_label,
            label2id=self.label_to_id,
        )

        train_dataset = train_dataset.map(self._tokenize_function, batched=True)
        test_dataset = test_dataset.map(self._tokenize_function, batched=True)

        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

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

        def compute_metrics(eval_pred: Any) -> dict[str, float]:
            predictions, labels = eval_pred
            predictions = predictions.argmax(-1)
            return {"accuracy": accuracy_score(labels, predictions)}

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=compute_metrics,
        )

        logger.info("training_started", model=self.model_name, epochs=num_epochs, batch_size=batch_size)
        train_result = trainer.train()
        eval_result = trainer.evaluate()

        logger.info(
            "training_completed",
            train_loss=train_result.training_loss,
            eval_accuracy=eval_result.get("eval_accuracy"),
        )

        final_model_dir = self.output_dir / "final_model"
        trainer.save_model(str(final_model_dir))
        self.tokenizer.save_pretrained(str(final_model_dir))

        label_map_path = final_model_dir / "label_mapping.json"
        with open(label_map_path, "w") as f:
            json.dump(
                {"label_to_id": self.label_to_id, "id_to_label": self.id_to_label},
                f,
                indent=2,
            )

        logger.info("model_saved", model_dir=str(final_model_dir))

        return {
            "train_loss": train_result.training_loss,
            "eval_accuracy": eval_result["eval_accuracy"],
            "eval_loss": eval_result["eval_loss"],
            "model_dir": str(final_model_dir),
        }

    def load_model(self, model_dir: str | Path) -> None:
        """Load a previously trained model from disk.

        Args:
            model_dir: Directory produced by train() (contains label_mapping.json).
        """
        model_dir = Path(model_dir)

        self.tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_dir))

        label_map_path = model_dir / "label_mapping.json"
        if label_map_path.exists():
            with open(label_map_path) as f:
                mappings = json.load(f)
                self.label_to_id = mappings["label_to_id"]
                self.id_to_label = {int(k): v for k, v in mappings["id_to_label"].items()}

        logger.info("model_loaded", model_dir=str(model_dir))

    def predict(self, texts: list[str]) -> list[str]:
        """Classify a batch of texts.

        Args:
            texts: Input strings.

        Returns:
            Predicted label strings (e.g. ["positive", "negative"]).
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("No model loaded. Call train() or load_model() first.")

        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = outputs.logits.argmax(-1).tolist()

        return [self.id_to_label[pred] for pred in predictions]

    def predict_single(self, text: str) -> str:
        """Classify a single text.

        Args:
            text: Input string.

        Returns:
            Predicted label string.
        """
        return self.predict([text])[0]

    def evaluate_on_documents(
        self,
        documents: list[Document],
        labels: list[Label],
    ) -> dict[str, Any]:
        """Evaluate the loaded model against ground-truth labels.

        Args:
            documents: Input documents.
            labels: Ground-truth labels.

        Returns:
            Dict with accuracy, classification_report, num_samples.
        """
        label_map = {label.document_id: label.value for label in labels}

        texts = [doc.content for doc in documents if doc.id in label_map]
        expected = [label_map[doc.id] for doc in documents if doc.id in label_map]
        predicted = self.predict(texts)

        return {
            "accuracy": accuracy_score(expected, predicted),
            "classification_report": classification_report(expected, predicted, output_dict=True),
            "num_samples": len(texts),
        }
