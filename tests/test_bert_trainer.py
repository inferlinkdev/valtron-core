"""Tests for BERT trainer."""

import tempfile
from pathlib import Path

from nltk import test
import pytest

from valtron_core.bert_trainer import BERTTrainer
from valtron_core.models import Document, Label


@pytest.fixture
def sample_training_data() -> tuple[list[Document], list[Label]]:
    """Create sample training data."""
    documents = [
        Document(id="1", content="This is great!"),
        Document(id="2", content="This is terrible!"),
        Document(id="3", content="This is okay."),
        Document(id="4", content="I love this!"),
        Document(id="5", content="I hate this!"),
        Document(id="6", content="It's fine."),
        Document(id="7", content="It's bad."),
        Document(id="8", content="It's amazing."),
        Document(id="9", content="It's very bad."),
    ]

    labels = [
        Label(document_id="1", value="positive"),
        Label(document_id="2", value="negative"),
        Label(document_id="3", value="neutral"),
        Label(document_id="4", value="positive"),
        Label(document_id="5", value="negative"),
        Label(document_id="6", value="neutral"),
        Label(document_id="7", value="positive"),
        Label(document_id="8", value="positive"),
        Label(document_id="9", value="negative"),
    ]

    return documents, labels


class TestBERTTrainer:
    """Tests for BERTTrainer."""

    def test_initialization(self) -> None:
        """Test trainer initialization."""
        trainer = BERTTrainer(model_name="bert-base-uncased")
        assert trainer.model_name == "bert-base-uncased"
        assert trainer.tokenizer is None
        assert trainer.model is None

    def test_prepare_data(self, sample_training_data: tuple[list[Document], list[Label]]) -> None:
        """Test data preparation."""
        documents, labels = sample_training_data
        trainer = BERTTrainer()

        train_dataset, test_dataset = trainer.prepare_data(
            documents=documents,
            labels=labels,
            test_size=0.33,
            random_state=42,
        )

        # Check datasets
        assert len(train_dataset) > 0
        assert len(test_dataset) > 0
        assert len(train_dataset) + len(test_dataset) == len(documents)

        # Check label mapping
        assert len(trainer.label_to_id) == 3  # positive, negative, neutral
        assert len(trainer.id_to_label) == 3
        assert "positive" in trainer.label_to_id
        assert "negative" in trainer.label_to_id
        assert "neutral" in trainer.label_to_id

    @pytest.mark.slow
    def test_train_small_model(
        self, sample_training_data: tuple[list[Document], list[Label]]
    ) -> None:
        """Test training a small model (marked as slow test)."""
        documents, labels = sample_training_data

        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = BERTTrainer(
                model_name="prajjwal1/bert-tiny",  # Use tiny BERT for faster testing
                output_dir=tmpdir,
            )

            train_dataset, test_dataset = trainer.prepare_data(
                documents=documents,
                labels=labels,
                test_size=0.33,
            )

            # Train for just 1 epoch to test
            results = trainer.train(
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                num_epochs=1,
                batch_size=1,
            )

            # Check results
            assert "train_loss" in results
            assert "eval_accuracy" in results
            assert "model_dir" in results
            assert Path(results["model_dir"]).exists()

    def test_label_mapping_consistency(
        self, sample_training_data: tuple[list[Document], list[Label]]
    ) -> None:
        """Test that label mappings are consistent."""
        documents, labels = sample_training_data
        trainer = BERTTrainer()

        trainer.prepare_data(documents=documents, labels=labels, test_size=0.33)

        # Check bidirectional mapping
        for label, idx in trainer.label_to_id.items():
            assert trainer.id_to_label[idx] == label

        for idx, label in trainer.id_to_label.items():
            assert trainer.label_to_id[label] == idx
