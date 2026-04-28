"""Report generation with LLM-powered recommendations."""

import base64
from datetime import datetime
from pathlib import Path
import re
from typing import Any

from jinja2 import Environment, FileSystemLoader
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend - must be before pyplot import
import matplotlib.pyplot as plt
from valtron_core.attachments import _EXT_MIME, _MAGIC, detect_mime_hint
from valtron_core.client import LLMClient
from valtron_core.models import EvaluationResult

TEMPLATES_DIR = Path(__file__).parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))


def _check_weasyprint_available() -> None:
    try:
        from weasyprint import HTML as _HTML  # noqa: F401
    except (ImportError, OSError) as e:
        raise ImportError(
            "PDF generation requires WeasyPrint system dependencies. "
            "See the installation guide: https://doc.courtbouillon.org/weasyprint/stable/first_steps.html"
        ) from e


class ReportGenerator:
    """Generate evaluation reports with visualizations and recommendations."""

    def __init__(self, client: LLMClient | None = None) -> None:
        """
        Initialize report generator.

        Args:
            client: Optional LLMClient for generating recommendations
        """
        self.client = client or LLMClient()

    def generate_recommendation(
        self,
        results: list[EvaluationResult],
        use_case: str = "general purpose",
    ) -> str:
        """
        Generate LLM-powered recommendation for best model.

        Args:
            results: List of evaluation results
            use_case: Description of the use case

        Returns:
            Recommendation text
        """
        # Prepare data for the LLM
        metrics_summary = []
        for result in results:
            if not result.metrics:
                continue

            metrics_summary.append(
                f"- {result.model}: "
                f"Accuracy={result.metrics.accuracy:.2%}, "
                f"Total Cost=${result.metrics.total_cost:.6f}, "
                f"Avg Cost/Doc=${result.metrics.average_cost_per_document:.6f}, "
                f"Total Time={result.metrics.total_time:.2f}s, "
                f"Avg Time/Doc={result.metrics.average_time_per_document:.2f}s"
            )

        prompt = f"""You are an AI model selection expert. Analyze the following evaluation results and provide a recommendation.

Use Case: {use_case}

Evaluation Results:
{chr(10).join(metrics_summary)}

PRIMARY GOAL: Find the model with the highest accuracy for the lowest cost. Calculate the accuracy-to-cost ratio (accuracy / total_cost) to identify the best value.

Based on these metrics, provide:
1. A clear recommendation for the model with the best accuracy-to-cost ratio
2. Brief justification showing the accuracy-to-cost calculation
3. Secondary recommendation if speed is a critical factor
4. Warning if the highest accuracy model is significantly more expensive

Keep your response concise and actionable (3-4 paragraphs maximum)."""

        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.client.complete_sync(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"Could not generate recommendation: {str(e)}"

    def _encode_image(self, image_path: Path) -> str:
        """Encode image to base64 for embedding in HTML."""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode()

    @staticmethod
    def _format_metric_display(metric: str, params: dict) -> str:
        """Return a human-readable string for a metric name + params.

        Used in the PDF report to show how each field is scored.
        """
        _METRIC_NAMES: dict[str, str] = {
            "exact":              "Exact Match",
            "exact_match":        "Exact Match",
            "text_similarity":    "Text Similarity",
            "llm":                "LLM Judge",
            "embedding":          "Embedding Similarity",
            "list_greedy_f1":     "List — Greedy F1",
            "list_ordered_f1":    "List — Ordered F1",
            "aggregated":         "Aggregated (weighted sub-fields)",
        }

        if metric.startswith("agg:"):
            name = "Aggregated (weighted sub-fields)"
        else:
            name = _METRIC_NAMES.get(metric, metric.replace("_", " ").title())

        if not params:
            return name

        # Format params as "key: value" pairs, skipping None/empty values
        _PARAM_LABELS: dict[str, str] = {
            "text_similarity_metric":    "similarity metric",
            "text_similarity_threshold": "threshold",
            "llm_model":                 "model",
            "embedding_model":           "model",
        }
        parts = []
        for k, v in params.items():
            if v is None:
                continue
            label = _PARAM_LABELS.get(k, k.replace("_", " "))
            parts.append(f"{label}: {v}")

        return f"{name} ({', '.join(parts)})" if parts else name

    def _prepare_field_metrics_data(
        self, results: list[EvaluationResult], field_names: list[str]
    ) -> dict[str, Any]:
        """
        Prepare field metrics data for visualization.

        Args:
            results: List of evaluation results
            field_names: List of field names to include

        Returns:
            Dictionary with field metrics organized by field name
        """
        field_metrics = {}

        for field_name in field_names:
            models = []
            precision = []
            recall = []
            f1_score = []
            comparison_methods = []
            comparison_configs = []

            for result in results:
                if result.metrics and result.metrics.aggregated_field_metrics:
                    field_metric = result.metrics.aggregated_field_metrics.get(field_name)
                    if field_metric:
                        models.append(result.model)
                        precision.append(round(field_metric.precision * 100, 2))
                        recall.append(round(field_metric.recall * 100, 2))
                        if field_metric.precision is not None and field_metric.recall is not None:
                            f1 = 2 * (field_metric.precision * field_metric.recall) / (field_metric.precision + field_metric.recall) if (field_metric.precision + field_metric.recall) > 0 else 0
                            f1_score.append(round(f1 * 100, 2))
                        else:
                            f1_score.append(0.0)

                        comparison_methods.append(field_metric.metric)

                        # Convert config to dict for JSON serialization
                        config_dict = None
                        if field_metric.params:
                            config_dict = field_metric.params
                        comparison_configs.append(config_dict)

            field_metrics[field_name] = {
                "models": models,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "comparison_methods": comparison_methods,
                "comparison_configs": comparison_configs,
            }

        return field_metrics

    def _compute_field_max_values(
        self, results: list[EvaluationResult], all_field_names: list[str]
    ) -> dict[str, dict[str, float]]:
        """
        Compute maximum precision, recall, F1 score, and score for each field across all models.

        Args:
            results: List of evaluation results
            all_field_names: List of field names to process

        Returns:
            Dictionary mapping field_name -> {max_precision, max_recall, max_f1}
        """
        field_max_values = {}

        for field_name in all_field_names:
            max_precision = -1.0
            max_recall = -1.0
            max_f1 = -1.0

            for result in results:
                if result.metrics and result.metrics.aggregated_field_metrics:
                    field_metric = result.metrics.aggregated_field_metrics.get(field_name)
                    if field_metric:
                        if field_metric.precision > max_precision:
                            max_precision = field_metric.precision
                        if field_metric.recall > max_recall:
                            max_recall = field_metric.recall
                        if field_metric.precision is not None and field_metric.recall is not None:
                            f1 = 2 * (field_metric.precision * field_metric.recall) / (field_metric.precision + field_metric.recall) if (field_metric.precision + field_metric.recall) > 0 else 0
                            if f1 > max_f1:
                                max_f1 = f1

            field_max_values[field_name] = {
                "max_precision": max_precision,
                "max_recall": max_recall,
                "max_f1": max_f1,
            }

        return field_max_values

    def _build_field_metrics_tree(
        self, results: list[EvaluationResult], all_field_names: list[str]
    ) -> dict[str, Any]:
        """
        Build a hierarchical tree from flat dot-separated field paths.

        Produces a JSON-serializable tree where each node contains:
        - field_key: the local segment name (e.g. "people")
        - full_path: the full dot path (e.g. "entities.people")
        - has_metrics: whether this path exists in aggregated_field_metrics
        - is_array: whether the aggregated metric uses a list-based metric
        - children: ordered dict of child nodes
        - method: the common comparison method across models (if any)

        Returns:
            Nested dict representing the tree structure
        """
        tree: dict[str, Any] = {}
        # "[*]" is the special root-list key — it's not a real tree node
        tree_field_names = [p for p in all_field_names if p != "[*]"]

        for field_path in sorted(tree_field_names):
            parts = field_path.split(".")
            current = tree
            for i, part in enumerate(parts):
                if part not in current:
                    partial_path = ".".join(parts[: i + 1])
                    current[part] = {
                        "field_key": part,
                        "full_path": partial_path,
                        "has_metrics": partial_path in all_field_names,
                        "is_array": False,
                        "method": None,
                        "children": {},
                    }
                current = current[part]["children"]

        # Detect array fields and common methods from results
        for field_path in tree_field_names:
            methods = set()
            is_array = False
            for result in results:
                if result.metrics and result.metrics.aggregated_field_metrics:
                    fm = result.metrics.aggregated_field_metrics.get(field_path)
                    if fm:
                        methods.add(fm.metric)
                        if fm.metric in ("list_greedy_f1", "list_ordered_f1"):
                            is_array = True

            # Walk into tree to set values
            parts = field_path.split(".")
            node = tree
            for part in parts:
                node = node[part]
                if part == parts[-1]:
                    node["is_array"] = is_array
                    node["method"] = list(methods)[0] if len(methods) == 1 else None
                else:
                    node = node["children"]

        return tree

    def _compute_performance_best_values(
        self, results: list[EvaluationResult]
    ) -> dict[str, float]:
        """
        Compute the best values for performance metrics across all models.
        Best = highest accuracy, lowest cost, lowest time.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary with best values for each metric
        """
        best_accuracy = -1.0
        best_total_cost = float('inf')
        best_avg_cost = float('inf')
        best_total_time = float('inf')
        best_avg_time = float('inf')

        for result in results:
            if result.metrics:
                if result.metrics.accuracy > best_accuracy:
                    best_accuracy = result.metrics.accuracy
                if result.metrics.total_cost < best_total_cost:
                    best_total_cost = result.metrics.total_cost
                if result.metrics.average_cost_per_document < best_avg_cost:
                    best_avg_cost = result.metrics.average_cost_per_document
                if result.metrics.total_time < best_total_time:
                    best_total_time = result.metrics.total_time
                if result.metrics.average_time_per_document < best_avg_time:
                    best_avg_time = result.metrics.average_time_per_document

        return {
            "best_accuracy": best_accuracy,
            "best_total_cost": best_total_cost,
            "best_avg_cost": best_avg_cost,
            "best_total_time": best_total_time,
            "best_avg_time": best_avg_time,
        }

    def _compute_performance_ranks(
        self, results: list[EvaluationResult]
    ) -> dict[str, dict]:
        """
        Compute dense performance ranks based on average_example_score (descending).
        Ties receive the same rank. Each entry includes the delta in percentage
        points from the top score, so near-ties are visually distinguishable.

        Returns:
            Dict mapping model name to {"rank": int, "delta_pct": float}
        """
        model_scores = [
            (r.model, r.metrics.average_example_score)
            for r in results
            if r.metrics
        ]
        if not model_scores:
            return {}

        sorted_models = sorted(model_scores, key=lambda x: x[1], reverse=True)
        best_score = sorted_models[0][1]

        ranks: dict[str, dict] = {}
        current_rank = 1
        for i, (model, score) in enumerate(sorted_models):
            if i > 0 and score < sorted_models[i - 1][1]:
                current_rank += 1
            ranks[model] = {
                "rank": current_rank,
                "delta_pct": round((score - best_score) * 100, 2),
            }

        return ranks

    def _prepare_chart_data(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """
        Prepare data for ECharts visualizations.

        Args:
            results: List of evaluation results

        Returns:
            Dictionary with chart data for models
        """
        models = []
        accuracy = []
        avg_time = []
        total_time = []
        avg_cost = []
        total_cost = []

        # Raw per-document data for histogram generation
        all_doc_data = {}  # model -> list of {id, cost, time, score}
        num_documents = 0

        for result in results:
            if result.metrics:
                models.append(result.model)
                accuracy.append(round(result.metrics.accuracy * 100, 2))
                avg_time.append(round(result.metrics.average_time_per_document, 3))
                total_time.append(round(result.metrics.total_time, 2))
                avg_cost.append(round(result.metrics.average_cost_per_document, 6))
                total_cost.append(round(result.metrics.total_cost, 6))

                # Collect per-document data from predictions with document IDs
                doc_data = []  # List of (doc_id, cost, time, score)
                for i, pred in enumerate(result.predictions):
                    doc_id = pred.document_id if pred.document_id else f"Doc {i + 1}"
                    # Use example_score if available (continuous 0-1), otherwise fall back to binary is_correct
                    if pred.example_score is not None:
                        score = round(pred.example_score * 100, 1)  # Convert to percentage
                    else:
                        score = 100 if pred.is_correct else 0
                    doc_data.append({
                        'id': doc_id,
                        'cost': round(pred.cost, 6),
                        'time': round(pred.response_time, 3),
                        'score': score
                    })

                # Track number of documents
                if len(result.predictions) > num_documents:
                    num_documents = len(result.predictions)

                all_doc_data[result.model] = doc_data

        # Prepare histogram data for cost and time
        histogram_data = self._prepare_histogram_data(all_doc_data, models)

        return {
            "models": models,
            "accuracy": accuracy,
            "avg_time": avg_time,
            "total_time": total_time,
            "avg_cost": avg_cost,
            "total_cost": total_cost,
            "histogram_cost": histogram_data["cost"],
            "histogram_time": histogram_data["time"],
            "histogram_score": histogram_data["score"],
        }

    def _prepare_histogram_data(
        self, all_doc_data: dict[str, list[dict]], models: list[str], num_bins: int = 10
    ) -> dict[str, Any]:
        """
        Prepare histogram bin data for cost, time, and score distributions.

        Args:
            all_doc_data: Dictionary mapping model name to list of document data
            models: List of model names
            num_bins: Number of histogram bins

        Returns:
            Dictionary with histogram data for cost, time, and score
        """
        import math

        result = {"cost": {}, "time": {}, "score": {}}

        # Collect all values across all models to determine global bin ranges
        all_costs = []
        all_times = []
        all_scores = []

        for model, docs in all_doc_data.items():
            for doc in docs:
                all_costs.append(doc['cost'])
                all_times.append(doc['time'])
                all_scores.append(doc['score'])

        if not all_costs:
            return result

        def nice_number(x: float, round_up: bool = True) -> float:
            """Round a number to a nice value (1, 2, 5, 10, etc.)"""
            if x == 0:
                return 0
            sign = 1 if x >= 0 else -1
            x = abs(x)
            exponent = math.floor(math.log10(x))
            fraction = x / (10 ** exponent)

            if round_up:
                if fraction <= 1:
                    nice_fraction = 1
                elif fraction <= 2:
                    nice_fraction = 2
                elif fraction <= 5:
                    nice_fraction = 5
                else:
                    nice_fraction = 10
            else:
                if fraction < 1.5:
                    nice_fraction = 1
                elif fraction < 3:
                    nice_fraction = 2
                elif fraction < 7:
                    nice_fraction = 5
                else:
                    nice_fraction = 10

            return sign * nice_fraction * (10 ** exponent)

        def create_nice_bins(values: list[float], target_bins: int) -> list[float]:
            """Create bins with nice round numbers."""
            min_val = min(values)
            max_val = max(values)

            if min_val == max_val:
                # All values are the same
                return [min_val - 0.5, max_val + 0.5]

            # Calculate a nice bin width
            data_range = max_val - min_val
            raw_bin_width = data_range / target_bins
            bin_width = nice_number(raw_bin_width, round_up=True)

            # Round min down and max up to bin width multiples
            nice_min = math.floor(min_val / bin_width) * bin_width
            nice_max = math.ceil(max_val / bin_width) * bin_width

            # Generate bins
            bins = []
            current = nice_min
            while current <= nice_max + bin_width / 2:  # Small tolerance for float comparison
                bins.append(round(current, 10))  # Round to avoid float precision issues
                current += bin_width

            # Ensure we have at least 2 bin edges
            if len(bins) < 2:
                bins = [nice_min, nice_max]

            return bins

        cost_bins = create_nice_bins(all_costs, num_bins)
        time_bins = create_nice_bins(all_times, num_bins)
        score_bins = create_nice_bins(all_scores, num_bins)

        # Create bin labels with appropriate decimal places
        def get_decimal_places(bins: list[float]) -> int:
            """Determine appropriate decimal places based on bin width."""
            if len(bins) < 2:
                return 2
            bin_width = bins[1] - bins[0]
            if bin_width == 0:
                return 2
            # Calculate decimals needed to show meaningful differences
            if bin_width >= 1:
                return 0
            elif bin_width >= 0.1:
                return 1
            elif bin_width >= 0.01:
                return 2
            elif bin_width >= 0.001:
                return 3
            elif bin_width >= 0.0001:
                return 4
            elif bin_width >= 0.00001:
                return 5
            else:
                return 6

        def create_bin_labels(bins: list[float], prefix: str = "", suffix: str = "") -> list[str]:
            decimals = get_decimal_places(bins)
            labels = []
            for i in range(len(bins) - 1):
                labels.append(f"{prefix}{bins[i]:.{decimals}f}{suffix} - {prefix}{bins[i+1]:.{decimals}f}{suffix}")
            return labels

        result["cost"]["bin_labels"] = create_bin_labels(cost_bins, prefix="$")
        result["time"]["bin_labels"] = create_bin_labels(time_bins, suffix="s")
        result["score"]["bin_labels"] = create_bin_labels(score_bins, suffix="%")

        result["cost"]["bins"] = cost_bins
        result["time"]["bins"] = time_bins
        result["score"]["bins"] = score_bins

        # Prepare raw data for CSV export (document -> model values)
        # Get all unique document IDs
        all_doc_ids = set()
        for model, docs in all_doc_data.items():
            for doc in docs:
                all_doc_ids.add(doc['id'])
        all_doc_ids = sorted(all_doc_ids)

        # Build raw data tables
        raw_cost_data = []
        raw_time_data = []
        raw_score_data = []

        for doc_id in all_doc_ids:
            cost_row = {"document": doc_id}
            time_row = {"document": doc_id}
            score_row = {"document": doc_id}

            for model in models:
                docs = all_doc_data.get(model, [])
                doc_entry = next((d for d in docs if d['id'] == doc_id), None)
                if doc_entry:
                    cost_row[model] = doc_entry['cost']
                    time_row[model] = doc_entry['time']
                    score_row[model] = doc_entry['score']
                else:
                    cost_row[model] = None
                    time_row[model] = None
                    score_row[model] = None

            raw_cost_data.append(cost_row)
            raw_time_data.append(time_row)
            raw_score_data.append(score_row)

        result["cost"]["raw_data"] = raw_cost_data
        result["time"]["raw_data"] = raw_time_data
        result["score"]["raw_data"] = raw_score_data

        # For each model, compute histogram counts
        for model in models:
            docs = all_doc_data.get(model, [])

            cost_counts = [0] * (len(cost_bins) - 1)
            time_counts = [0] * (len(time_bins) - 1)
            score_counts = [0] * (len(score_bins) - 1)

            for doc in docs:
                cost_bin_idx = self._find_bin_index(doc['cost'], cost_bins)
                if cost_bin_idx is not None:
                    cost_counts[cost_bin_idx] += 1

                time_bin_idx = self._find_bin_index(doc['time'], time_bins)
                if time_bin_idx is not None:
                    time_counts[time_bin_idx] += 1

                score_bin_idx = self._find_bin_index(doc['score'], score_bins)
                if score_bin_idx is not None:
                    score_counts[score_bin_idx] += 1

            if "models" not in result["cost"]:
                result["cost"]["models"] = {}
                result["time"]["models"] = {}
                result["score"]["models"] = {}

            result["cost"]["models"][model] = {"counts": cost_counts}
            result["time"]["models"][model] = {"counts": time_counts}
            result["score"]["models"][model] = {"counts": score_counts}

        return result

    def _find_bin_index(self, value: float, bins: list[float]) -> int | None:
        """Find the index of the bin that contains the given value."""
        for i in range(len(bins) - 1):
            if i == len(bins) - 2:
                # Last bin includes the upper bound
                if bins[i] <= value <= bins[i + 1]:
                    return i
            else:
                if bins[i] <= value < bins[i + 1]:
                    return i
        return None

    @staticmethod
    def _normalize_attachment(s: str) -> dict[str, Any]:
        """
        Convert a string attachment (URL, local path, or data URI) to the structured
        dict ``{url, mime_type, type, data}`` expected by the detailed analysis template.
        Local files are embedded as base64; URLs and data URIs are passed through as ``url``.
        """
        # Data URI — already carries mime and encoded data
        if s.startswith("data:"):
            mime_type = detect_mime_hint(s)
            return {"url": s, "mime_type": mime_type, "type": mime_type, "data": None}

        # HTTP/HTTPS URL — pass through; type detected from extension
        if s.startswith(("http://", "https://")):
            mime_type = detect_mime_hint(s)
            return {"url": s, "mime_type": mime_type, "type": mime_type, "data": None}

        # Local file — embed as base64 so the browser can render it inline
        try:
            raw = Path(s).read_bytes()
            mime_type = detect_mime_hint(s)
            if not mime_type:
                for magic, mime in _MAGIC:
                    if raw[: len(magic)] == magic:
                        mime_type = mime
                        break
            if not mime_type:
                mime_type = "application/octet-stream"
            b64 = base64.b64encode(raw).decode()
            return {"url": None, "mime_type": mime_type, "type": mime_type, "data": b64}
        except Exception:
            return {"url": s, "mime_type": "", "type": "", "data": None}

    def _prepare_detailed_analysis_data(
        self,
        results: list[EvaluationResult],
        documents: list[Any] | None = None
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """
        Prepare data for detailed input/output analysis page.

        Args:
            results: List of evaluation results
            documents: Optional list of original Document objects

        Returns:
            Tuple of (document data list, metadata dict)
        """
        if not results:
            return [], {}

        # Create maps for document content, metadata, and attachments if documents are provided
        doc_content_map = {}
        doc_metadata_map = {}
        doc_attachments_map = {}
        if documents:
            for doc in documents:
                doc_content_map[doc.id] = doc.content
                if hasattr(doc, 'metadata') and doc.metadata:
                    doc_metadata_map[f"doc{len(doc_content_map)}"] = doc.metadata
                if hasattr(doc, 'attachments') and doc.attachments:
                    doc_attachments_map[doc.id] = doc.attachments

        # Get unique document IDs from first result
        documents_map: dict[str, dict[str, Any]] = {}

        # Collect all predictions grouped by document
        for result in results:
            for prediction in result.predictions:
                doc_id = prediction.document_id

                if doc_id not in documents_map:
                    # Get content from doc_content_map, or from prediction metadata, or show document ID
                    content = doc_content_map.get(doc_id)
                    if not content and hasattr(prediction, 'metadata') and prediction.metadata:
                        content = prediction.metadata.get('content')
                    if not content:
                        content = f"Document ID: {doc_id}"

                    # Get attachments from doc_attachments_map or from prediction metadata
                    raw_attachments = doc_attachments_map.get(doc_id)
                    if not raw_attachments and hasattr(prediction, 'metadata') and prediction.metadata:
                        raw_attachments = prediction.metadata.get('attachments', [])

                    # Normalize string attachments to structured dicts for the template
                    attachments = [
                        self._normalize_attachment(a) if isinstance(a, str) else a
                        for a in (raw_attachments or [])
                    ]

                    # Initialize document entry
                    documents_map[doc_id] = {
                        "document_id": doc_id,
                        "content": content,
                        "attachments": attachments,
                        "expected_value": prediction.expected_value,
                        "metadata": doc_metadata_map.get(doc_id),
                        "model_results": [],
                    }

                # Add this model's prediction
                # Convert Pydantic models to dicts for JSON serialization
                field_metrics_model_dump = prediction.field_metrics.model_dump() if prediction.field_metrics else None

                documents_map[doc_id]["model_results"].append({
                    "model": result.model,
                    "predicted_value": prediction.predicted_value,
                    "is_correct": prediction.is_correct,
                    "response_time": prediction.response_time,
                    "cost": prediction.cost,
                    "example_score": prediction.example_score,
                    "field_metrics": field_metrics_model_dump
                })

        # Create metadata lookup by doc position
        metadata_lookup = {}
        for idx, doc_id in enumerate(documents_map.keys(), 1):
            if documents:
                for doc in documents:
                    if doc.id == doc_id and hasattr(doc, 'metadata') and doc.metadata:
                        metadata_lookup[f"doc{idx}"] = doc.metadata
                        break

        return list(documents_map.values()), metadata_lookup

    def _extract_recommended_model(self, recommendation: str, results: list[EvaluationResult]) -> str | None:
        """
        Extract the recommended model name from the AI recommendation text.

        Args:
            recommendation: The AI-generated recommendation text
            results: List of evaluation results with model names

        Returns:
            The recommended model name, or None if not found
        """
        if not recommendation:
            return None

        # Get all model names from results
        model_names = [result.model for result in results]

        # Search for model names in the recommendation text
        # Look for the first mentioned model name as the primary recommendation
        recommendation_lower = recommendation.lower()

        for model_name in model_names:
            # Check if model name appears in the recommendation
            # Look for common patterns like "recommend X" or "best choice is X"
            if model_name.lower() in recommendation_lower:
                # Return the first model mentioned (likely the primary recommendation)
                return model_name

        # If no model name found, return None
        return None

    def _create_html_template(self):
        """Create Jinja2 HTML template for report."""
        return _jinja_env.get_template("evaluation_report.jinja2.html")

    def _create_pdf_template(self):
        """Create Jinja2 HTML template optimized for PDF generation via weasyprint."""
        return _jinja_env.get_template("pdf_report.jinja2.html")


    def _create_detailed_analysis_template(self):
        """Create Jinja2 HTML template for detailed analysis page."""
        return _jinja_env.get_template("detailed_analysis.jinja2.html")


    def _generate_charts(
        self,
        results: list[EvaluationResult],
        output_dir: Path,
    ) -> list[Path]:
        """
        Generate matplotlib charts for the PDF report.

        Uses pastel colors on a clean white background with consistent
        model-to-color mapping across all charts.

        Args:
            results: List of evaluation results
            output_dir: Directory to save chart images

        Returns:
            List of paths to generated chart images
        """
        chart_paths = []

        # Extract data
        models = []
        accuracies = []
        costs = []
        times = []

        for result in results:
            if result.metrics:
                models.append(result.model)
                accuracies.append(result.metrics.accuracy * 100)
                costs.append(result.metrics.total_cost)
                times.append(result.metrics.total_time)

        if not models:
            return chart_paths

        # Pastel color palette - consistent model-to-color mapping
        pastel_colors = [
            '#a8d5e5',  # Pastel Blue
            '#b5e6b5',  # Pastel Green
            '#d4b5e6',  # Pastel Purple
            '#f5d5a8',  # Pastel Orange
            '#f5b5c5',  # Pastel Pink
            '#b5d5f5',  # Pastel Sky Blue
            '#e5e5b5',  # Pastel Yellow
            '#c5e5e5',  # Pastel Cyan
        ]

        # Assign consistent colors to models (same color per model across all charts)
        colors = [pastel_colors[i % len(pastel_colors)] for i in range(len(models))]

        # Bar width - thinner bars
        bar_width = 0.5

        def setup_clean_theme(ax, fig):
            """Apply clean white theme with no gridlines."""
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            # Remove top and right spines for cleaner look
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#cccccc')
            ax.spines['left'].set_color('#cccccc')
            ax.tick_params(colors='#333333', which='both')
            ax.xaxis.label.set_color('#333333')
            ax.yaxis.label.set_color('#333333')
            ax.title.set_color('#333333')
            # No gridlines
            ax.grid(False)

        # Chart 1: Accuracy Comparison
        fig, ax = plt.subplots(figsize=(6, 3.5))
        setup_clean_theme(ax, fig)
        bars = ax.bar(range(len(models)), accuracies, width=bar_width, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_title('Quality: Accuracy by Model', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 105)
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, color='#333333')
        plt.tight_layout()
        accuracy_path = output_dir / 'chart_accuracy.png'
        plt.savefig(accuracy_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        chart_paths.append(accuracy_path)

        # Chart 2: Cost Comparison
        fig, ax = plt.subplots(figsize=(6, 3.5))
        setup_clean_theme(ax, fig)
        bars = ax.bar(range(len(models)), costs, width=bar_width, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Total Cost ($)', fontsize=10)
        ax.set_title('Cost: Average Cost per Document', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        for bar, cost in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'${cost:.4f}', ha='center', va='bottom', fontsize=9, color='#333333')
        plt.tight_layout()
        cost_path = output_dir / 'chart_cost.png'
        plt.savefig(cost_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        chart_paths.append(cost_path)

        # Chart 3: Time Comparison
        fig, ax = plt.subplots(figsize=(6, 3.5))
        setup_clean_theme(ax, fig)
        bars = ax.bar(range(len(models)), times, width=bar_width, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Total Time (s)', fontsize=10)
        ax.set_title('Speed: Average Time per Document', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        for bar, time in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                   f'{time:.2f}s', ha='center', va='bottom', fontsize=9, color='#333333')
        plt.tight_layout()
        time_path = output_dir / 'chart_time.png'
        plt.savefig(time_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        chart_paths.append(time_path)

        return chart_paths

    def generate_pdf_report(
        self,
        results: list[EvaluationResult],
        output_path: str | Path,
        recommendation: str | None = None,
        original_prompt: str | None = None,
        prompt_optimizations: dict[str, list[str]] | None = None,
        model_override_prompts: dict[str, str] | None = None,
    ) -> Path:
        """
        Generate a PDF report using weasyprint from HTML.

        Args:
            results: List of evaluation results
            output_path: Path to save the PDF report (without extension)
            recommendation: Optional AI-generated recommendation text
            original_prompt: Optional original prompt template
            prompt_optimizations: Optional dict mapping model names to applied manipulations

        Returns:
            Path to the generated PDF
        """
        _check_weasyprint_available()
        from weasyprint import HTML

        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        # Sort results by total cost for consistent ordering
        results = sorted(
            results,
            key=lambda r: r.metrics.total_cost if r.metrics else float('inf')
        )

        # Generate static charts for PDF
        chart_paths = self._generate_charts(results, output_dir)

        # Encode charts as base64 for embedding in HTML
        chart_images = {}
        for chart_path in chart_paths:
            chart_name = chart_path.stem  # e.g., "accuracy_chart"
            with open(chart_path, "rb") as f:
                chart_images[chart_name] = base64.b64encode(f.read()).decode("utf-8")

        # Prepare data for template (similar to generate_html_report)
        num_models = len(results)
        num_documents = results[0].metrics.total_documents if results and results[0].metrics else 0

        # Prepare chart data
        chart_data = self._prepare_chart_data(results)

        # Check for field metrics
        all_field_names = set()
        has_field_metrics = False
        for result in results:
            if result.metrics and result.metrics.aggregated_field_metrics:
                has_field_metrics = True
                all_field_names.update(result.metrics.aggregated_field_metrics.keys())
        all_field_names = sorted(all_field_names) if all_field_names else []

        # Prepare field metrics data
        field_metrics_data = {}
        field_max_values = {}
        if has_field_metrics:
            for field_name in all_field_names:
                field_metrics_data[field_name] = []
                max_precision = max_recall = max_f1 = 0
                for result in results:
                    if result.metrics and result.metrics.aggregated_field_metrics:
                        fm = result.metrics.aggregated_field_metrics.get(field_name)
                        if fm:
                            f1 = (2 * fm.precision * fm.recall / (fm.precision + fm.recall)) if (fm.precision + fm.recall) > 0 else 0.0
                            max_precision = max(max_precision, fm.precision)
                            max_recall = max(max_recall, fm.recall)
                            max_f1 = max(max_f1, f1)
                            field_metrics_data[field_name].append({
                                "model": result.model,
                                "precision": fm.precision,
                                "recall": fm.recall,
                                "f1_score": f1,
                                "metric_display": self._format_metric_display(
                                    fm.metric, fm.params or {}
                                ),
                            })
                field_max_values[field_name] = {
                    "precision": max_precision,
                    "recall": max_recall,
                    "f1_score": max_f1,
                }

        # Compute best values for highlighting
        performance_best = self._compute_performance_best_values(results)

        # Compute performance ranks (used when has_field_metrics)
        performance_ranks = self._compute_performance_ranks(results)

        # Check for optimizations
        has_optimizations = prompt_optimizations is not None and any(
            len(m) > 0 for m in prompt_optimizations.values()
        )

        # Create PDF-specific template
        template = self._create_pdf_template()

        html_content = template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            num_models=num_models,
            num_documents=num_documents,
            results=results,
            recommendation=recommendation,
            has_field_metrics=has_field_metrics,
            all_field_names=all_field_names,
            prompt_optimizations=prompt_optimizations or {},
            has_optimizations=has_optimizations,
            model_override_prompts=model_override_prompts or {},
            original_prompt=original_prompt,
            chart_data=chart_data,
            field_metrics_data=field_metrics_data,
            field_max_values=field_max_values,
            performance_best=performance_best,
            performance_ranks=performance_ranks,
            chart_images=chart_images,
        )

        # Convert HTML to PDF using weasyprint
        pdf_path = output_dir / f"{output_path.stem}.pdf"
        HTML(string=html_content).write_pdf(pdf_path)

        return pdf_path

    def generate_html_report(
        self,
        results: list[EvaluationResult],
        output_path: str | Path,
        use_case: str = "general purpose",
        include_recommendation: bool = True,
        prompt_optimizations: dict[str, list[str]] | None = None,
        model_prompts: dict[str, str] | None = None,
        model_override_prompts: dict[str, str] | None = None,
        original_prompt: str | None = None,
        documents: list[Any] | None = None,
        field_config: dict[str, Any] | None = None,
    ) -> tuple[Path, str | None]:
        """
        Generate an HTML report with visualizations and recommendations.

        Args:
            results: List of evaluation results
            output_path: Path to save the HTML report
            use_case: Description of use case for recommendation
            include_recommendation: Whether to include LLM recommendation
            prompt_optimizations: Optional dict mapping model names to list of applied manipulations (e.g., ["explanation", "few_shot"])
            model_prompts: Optional dict mapping model names to the actual prompts used
            original_prompt: Optional original prompt template (before any optimizations)
            documents: Optional list of original Document objects for detailed analysis

        Returns:
            Tuple of (Path to generated report, recommendation text or None)
        """
        output_path = Path(output_path)

        # Sort results by total cost (ascending) for consistent ordering across all charts and tables
        results = sorted(
            results,
            key=lambda r: r.metrics.total_cost if r.metrics else float('inf')
        )

        # Generate recommendation
        recommendation = None
        recommended_model = None
        if include_recommendation:
            recommendation = self.generate_recommendation(results, use_case)
            # Extract recommended model from the recommendation text
            recommended_model = self._extract_recommended_model(recommendation, results)

        # Get number of documents from first result
        num_documents = 0
        # Use original prompt if provided, otherwise fall back to first result's prompt
        prompt_template = original_prompt if original_prompt else ""
        if not prompt_template and results and results[0].metrics:
            num_documents = results[0].metrics.total_documents
            prompt_template = results[0].prompt_template
        elif results and results[0].metrics:
            num_documents = results[0].metrics.total_documents

        # Collect all field names from results
        all_field_names = set()
        has_field_metrics = False
        for result in results:
            if result.metrics and result.metrics.aggregated_field_metrics:
                has_field_metrics = True
                all_field_names.update(result.metrics.aggregated_field_metrics.keys())

        # Sort field names for consistent display
        all_field_names = sorted(all_field_names)

        # Prepare field metrics data for graphs
        field_metrics_data = self._prepare_field_metrics_data(results, all_field_names)

        # Compute max values for each field (for bolding best values in tables)
        field_max_values = self._compute_field_max_values(results, all_field_names)

        # Build hierarchical tree for JSON tree view
        field_metrics_tree = self._build_field_metrics_tree(results, all_field_names)

        # Compute best values for performance metrics table
        performance_best = self._compute_performance_best_values(results)

        # Compute performance ranks (used when has_field_metrics)
        performance_ranks = self._compute_performance_ranks(results)

        # Check if any model used optimization (has non-empty manipulation list)
        has_optimizations = prompt_optimizations is not None and any(
            len(manipulations) > 0 for manipulations in prompt_optimizations.values()
        )
        has_overrides = bool(model_override_prompts)

        # Prepare chart data for ECharts
        chart_data = self._prepare_chart_data(results)

        # Prepare detailed analysis data
        documents_data, document_metadata = self._prepare_detailed_analysis_data(results, documents)

        field_config_json = field_config if field_config else None

        # Render main report template
        template = self._create_html_template()
        html_content = template.render(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            num_models=len(results),
            num_documents=num_documents,
            prompt_template=prompt_template,
            results=results,
            recommended_model=recommended_model,
            recommendation=recommendation,
            has_field_metrics=has_field_metrics,
            all_field_names=all_field_names,
            prompt_optimizations=prompt_optimizations or {},
            has_optimizations=has_optimizations,
            model_prompts=model_prompts or {},
            model_override_prompts=model_override_prompts or {},
            has_overrides=has_overrides,
            chart_data=chart_data,
            field_metrics_data=field_metrics_data,
            field_max_values=field_max_values,
            field_metrics_tree=field_metrics_tree,
            root_list_field_key="[*]" if "[*]" in field_metrics_data else None,
            performance_best=performance_best,
            performance_ranks=performance_ranks,
            field_config_json=field_config_json,
        )

        # Write main report to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding="utf-8")

        #Generate detailed analysis page
        analysis_path = output_path.parent / "detailed_analysis.html"
        analysis_template = self._create_detailed_analysis_template()
        analysis_content = analysis_template.render(
            documents_data=documents_data,
            document_metadata=document_metadata
        )
        analysis_path.write_text(analysis_content, encoding="utf-8")

        return output_path, recommendation
