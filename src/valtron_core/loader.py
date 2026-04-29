"""Document and label loading utilities."""

import csv
import json
from pathlib import Path
from typing import Any

from valtron_core.evaluation.json_eval import EvalResult
from valtron_core.models import Document, EvaluationResult, Label, PredictionResult


class DocumentLoader:
    """Utility for loading documents and labels from various formats."""

    @staticmethod
    def load_documents_from_json(file_path: str | Path) -> list[Document]:
        """
        Load documents from a JSON file.

        Expected format:
        [
            {"id": "doc1", "content": "text here", "metadata": {...}},
            {"id": "doc2", "content": "more text", "metadata": {...}}
        ]

        Args:
            file_path: Path to JSON file

        Returns:
            List of Document objects
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents: list[Document] = []
        for item in data:
            # Coerce common fields to expected types (pydantic may be strict)
            doc_id = str(item.get("id", ""))
            content_raw = item.get("content", "")
            content: str | dict[str, str] = (
                {str(k): str(v) for k, v in content_raw.items()}
                if isinstance(content_raw, dict)
                else str(content_raw)
            )
            metadata = item.get("metadata", {}) or {}

            documents.append(Document(id=doc_id, content=content, metadata=metadata))

        return documents

    @staticmethod
    def load_documents_from_csv(
        file_path: str | Path,
        id_column: str = "id",
        content_column: str = "content",
    ) -> list[Document]:
        """
        Load documents from a CSV file.

        Args:
            file_path: Path to CSV file
            id_column: Name of the ID column
            content_column: Name of the content column

        Returns:
            List of Document objects
        """
        documents = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                doc_id = row.pop(id_column)
                content = row.pop(content_column)

                # Remaining columns become metadata
                metadata = {k: v for k, v in row.items() if v}

                documents.append(
                    Document(id=doc_id, content=content, metadata=metadata)
                )

        return documents

    @staticmethod
    def load_labels_from_json(file_path: str | Path) -> list[Label]:
        """
        Load labels from a JSON file.

        Expected format:
        [
            {"document_id": "doc1", "value": "positive", "metadata": {...}},
            {"document_id": "doc2", "value": "negative", "metadata": {...}}
        ]

        Args:
            file_path: Path to JSON file

        Returns:
            List of Label objects
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        labels: list[Label] = []
        for item in data:
            document_id = str(item.get("document_id", ""))
            value = str(item.get("value", ""))
            metadata = item.get("metadata", {}) or {}

            labels.append(Label(document_id=document_id, value=value, metadata=metadata))

        return labels

    @staticmethod
    def load_labels_from_csv(
        file_path: str | Path,
        document_id_column: str = "document_id",
        value_column: str = "label",
    ) -> list[Label]:
        """
        Load labels from a CSV file.

        Args:
            file_path: Path to CSV file
            document_id_column: Name of the document ID column
            value_column: Name of the label value column

        Returns:
            List of Label objects
        """
        labels = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                doc_id = row.pop(document_id_column)
                value = row.pop(value_column)

                # Remaining columns become metadata
                metadata = {k: v for k, v in row.items() if v}

                labels.append(
                    Label(document_id=doc_id, value=value, metadata=metadata)
                )

        return labels

    @staticmethod
    def load_combined_from_json(file_path: str | Path) -> tuple[list[Document], list[Label]]:
        """
        Load documents and labels from a single JSON file.

        Expected format:
        [
            {
                "id": "doc1",
                "content": "text here",
                "label": "positive",
                "metadata": {...}
            }
        ]

        Args:
            file_path: Path to JSON file

        Returns:
            Tuple of (documents, labels)
        """
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        documents: list[Document] = []
        labels: list[Label] = []

        for item in data:
            # Coerce to expected types: id and textual fields -> str
            doc_id = str(item.get("id", ""))
            content_raw = item.get("content", "")
            content: str | dict[str, str] = (
                {str(k): str(v) for k, v in content_raw.items()}
                if isinstance(content_raw, dict)
                else str(content_raw)
            )
            label_raw = item.get("label", "")
            label_value = json.dumps(label_raw) if isinstance(label_raw, (dict, list)) else str(label_raw)
            metadata = item.get("metadata", {}) or {}
            attachments = item.get("attachments", []) or []

            documents.append(
                Document(id=doc_id, content=content, metadata=metadata, attachments=attachments)
            )
            labels.append(
                Label(document_id=doc_id, value=label_value, metadata=metadata)
            )

        return documents, labels

    @staticmethod
    def load_combined_from_csv(
        file_path: str | Path,
        id_column: str = "id",
        content_column: str = "content",
        label_column: str = "label",
    ) -> tuple[list[Document], list[Label]]:
        """
        Load documents and labels from a single CSV file.

        Args:
            file_path: Path to CSV file
            id_column: Name of the ID column
            content_column: Name of the content column
            label_column: Name of the label column

        Returns:
            Tuple of (documents, labels)
        """
        documents = []
        labels = []

        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            for row in reader:
                doc_id = row.pop(id_column)
                content = row.pop(content_column)
                label_value = row.pop(label_column)

                # Remaining columns become metadata
                metadata = {k: v for k, v in row.items() if v}

                documents.append(
                    Document(id=doc_id, content=content, metadata=metadata)
                )
                labels.append(
                    Label(document_id=doc_id, value=label_value, metadata=metadata)
                )

        return documents, labels

    @staticmethod
    def save_results_to_json(
        results: Any,
        file_path: str | Path,
        indent: int = 2,
    ) -> None:
        """
        Save evaluation results to a JSON file.

        Args:
            results: Pydantic model or dict to save
            file_path: Path to output JSON file
            indent: JSON indentation level
        """
        with open(file_path, "w", encoding="utf-8") as f:
            if hasattr(results, "model_dump"):
                data = results.model_dump(mode="json")
            else:
                data = results
            json.dump(data, f, indent=indent, default=str)

    def load_results_from_dir(self, dir_path: str | Path) -> list[EvaluationResult]:
        """
        Load saved result JSON files from a directory and return a list of
        `EvaluationResult` objects. Supports two formats:

        - Full `EvaluationResult` JSON dumps (saved via `save_results`),
        - Simplified per-model prediction lists (array of dicts with keys
          `id`, `content`, `label`, `predicted_label`, etc.) saved by
          `compare_models_from_files`.

        Args:
            dir_path: Directory containing JSON result files

        Returns:
            List of `EvaluationResult` objects
        """
        dir_path = Path(dir_path)
        results: list[EvaluationResult] = []

        if not dir_path.exists() or not dir_path.is_dir():
            return results

        for file in sorted(dir_path.glob("*.json")):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # If file contains a dict with run_id or metrics, try to parse
                if isinstance(data, dict) and ("run_id" in data or "predictions" in data or "metrics" in data):
                    # Pydantic can construct from dict
                    try:
                        er = EvaluationResult(**data)
                    except Exception:
                        # Try model_validate if available (pydantic v2)
                        try:
                            er = EvaluationResult.model_validate(data)
                        except Exception:
                            # Skip file if parsing fails
                            continue

                    # Ensure metrics exist
                    if not er.metrics and er.predictions:
                        er.compute_metrics()

                    results.append(er)

                # If file contains a list, assume simplified predictions
                elif isinstance(data, list):
                    # Try to infer model and run_id from filename: name_runid.json
                    stem = file.stem
                    model_name = stem
                    run_id = "" 
                    if "_" in stem:
                        model_name, run_id = stem.rsplit("_", 1)

                    preds: list[PredictionResult] = []
                    for item in data:
                        doc_id = str(item.get("id", ""))
                        predicted = str(item.get("predicted_label", ""))
                        expected = str(item.get("label", ""))
                        # Compute correctness from label vs predicted_label instead of trusting saved data
                        def _normalize(v: str) -> str:
                            return (v or "").strip().lower()

                        is_correct = item.get("is_correct", False) or _normalize(predicted) == _normalize(expected)
                        response_time = float(item.get("response_time", 0.0) or 0.0)
                        cost = float(item.get("cost", 0.0) or 0.0)
                        example_score = item.get("example_score", None)

                        # Load field metrics if available
                        field_metrics = None
                        if "field_metrics" in item and item["field_metrics"]:
                            # Try to parse field metrics as EvalResult object
                            try:
                                field_metrics = EvalResult(**item["field_metrics"])
                            except Exception:
                                try:
                                    field_metrics = EvalResult.model_validate(item["field_metrics"])
                                except Exception:
                                    field_metrics = None

                        # Build metadata with human_match and content if present
                        metadata = {}
                        # Store document content in metadata for report generation
                        if "content" in item:
                            metadata["content"] = str(item["content"])
                        # Store attachments in metadata for report generation
                        if "attachments" in item and item["attachments"]:
                            metadata["attachments"] = item["attachments"]

                        preds.append(
                            PredictionResult(
                                document_id=doc_id,
                                predicted_value=predicted,
                                expected_value=expected,
                                is_correct=is_correct,
                                response_time=response_time,
                                cost=cost,
                                example_score=example_score,
                                model=model_name,
                                metadata=metadata,
                                field_metrics=field_metrics
                            )
                        )

                    # Build EvaluationResult
                    er = EvaluationResult(
                        run_id=run_id or file.stem,
                        predictions=preds,
                        metrics=None,
                        prompt_template="",
                        model=model_name,
                        status="completed",
                    )

                    # Compute metrics from predictions
                    try:
                        er.compute_metrics()
                    except Exception:
                        # If there are no predictions, skip
                        continue

                    results.append(er)

            except Exception:
                # Ignore files we can't read/parse
                continue

        return results