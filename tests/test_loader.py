"""Tests for document loader."""

import json
import tempfile
from pathlib import Path

import pytest

from valtron_core.loader import DocumentLoader
from valtron_core.models import Document, EvaluationResult, Label


class TestDocumentLoader:
    """Tests for DocumentLoader."""

    def test_load_documents_from_json(self) -> None:
        """Test loading documents from JSON."""
        loader = DocumentLoader()

        data = [
            {"id": "1", "content": "Text 1", "metadata": {"source": "test"}},
            {"id": "2", "content": "Text 2"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            documents = loader.load_documents_from_json(temp_path)

            assert len(documents) == 2
            assert documents[0].id == "1"
            assert documents[0].content == "Text 1"
            assert documents[0].metadata == {"source": "test"}
            assert documents[1].id == "2"
            assert documents[1].content == "Text 2"
        finally:
            Path(temp_path).unlink()

    def test_load_documents_from_csv(self) -> None:
        """Test loading documents from CSV."""
        loader = DocumentLoader()

        csv_content = """id,content,source
1,Text 1,test
2,Text 2,prod"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            documents = loader.load_documents_from_csv(temp_path)

            assert len(documents) == 2
            assert documents[0].id == "1"
            assert documents[0].content == "Text 1"
            assert documents[0].metadata["source"] == "test"
        finally:
            Path(temp_path).unlink()

    def test_load_labels_from_json(self) -> None:
        """Test loading labels from JSON."""
        loader = DocumentLoader()

        data = [
            {"document_id": "1", "value": "positive"},
            {"document_id": "2", "value": "negative", "metadata": {"confidence": 0.9}},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            labels = loader.load_labels_from_json(temp_path)

            assert len(labels) == 2
            assert labels[0].document_id == "1"
            assert labels[0].value == "positive"
            assert labels[1].metadata.get("confidence") == 0.9
        finally:
            Path(temp_path).unlink()

    def test_load_labels_from_csv(self) -> None:
        """Test loading labels from CSV."""
        loader = DocumentLoader()

        csv_content = """document_id,label,confidence
1,positive,0.95
2,negative,0.85"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            labels = loader.load_labels_from_csv(temp_path)

            assert len(labels) == 2
            assert labels[0].document_id == "1"
            assert labels[0].value == "positive"
            assert labels[0].metadata["confidence"] == "0.95"
        finally:
            Path(temp_path).unlink()

    def test_load_combined_from_json(self) -> None:
        """Test loading combined documents and labels from JSON."""
        loader = DocumentLoader()

        data = [
            {"id": "1", "content": "Text 1", "label": "positive"},
            {"id": "2", "content": "Text 2", "label": "negative"},
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            temp_path = f.name

        try:
            documents, labels = loader.load_combined_from_json(temp_path)

            assert len(documents) == 2
            assert len(labels) == 2
            assert documents[0].id == "1"
            assert labels[0].document_id == "1"
            assert labels[0].value == "positive"
        finally:
            Path(temp_path).unlink()

    def test_load_combined_from_csv(self) -> None:
        """Test loading combined documents and labels from CSV."""
        loader = DocumentLoader()

        csv_content = """id,content,label
1,Text 1,positive
2,Text 2,negative"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write(csv_content)
            temp_path = f.name

        try:
            documents, labels = loader.load_combined_from_csv(temp_path)

            assert len(documents) == 2
            assert len(labels) == 2
            assert documents[0].id == "1"
            assert documents[0].content == "Text 1"
            assert labels[0].value == "positive"
        finally:
            Path(temp_path).unlink()

    def test_save_results_to_json(self) -> None:
        """Test saving results to JSON."""
        loader = DocumentLoader()

        # Create a simple result
        result = EvaluationResult(
            run_id="test-123",
            prompt_template="Test prompt",
            model="gpt-3.5-turbo",
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            loader.save_results_to_json(result, temp_path)

            # Verify file was created and can be read
            with open(temp_path, "r") as f:
                data = json.load(f)

            assert data["run_id"] == "test-123"
            assert data["model"] == "gpt-3.5-turbo"
        finally:
            Path(temp_path).unlink()
