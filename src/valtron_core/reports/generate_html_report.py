"""HTML report generation."""

import base64
from datetime import datetime
from pathlib import Path
from typing import Any

from valtron_core.attachments import _MAGIC, detect_mime_hint
from valtron_core.client import LLMClient
from valtron_core.models import EvaluationResult
from valtron_core.reports._base import _ReportBase, _jinja_env


class HtmlReportGenerator(_ReportBase):
    """Generate interactive HTML evaluation reports."""

    def generate_recommendation(
        self,
        results: list[EvaluationResult],
        use_case: str = "general purpose",
    ) -> str:
        """Generate LLM-powered recommendation for best model."""
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

    def _prepare_field_metrics_data(
        self, results: list[EvaluationResult], field_names: list[str]
    ) -> dict[str, Any]:
        """Prepare field metrics data for visualization."""
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
        """Compute maximum precision, recall, F1 score for each field across all models."""
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
        """Build a hierarchical tree from flat dot-separated field paths."""
        tree: dict[str, Any] = {}
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

    @staticmethod
    def _normalize_attachment(s: str) -> dict[str, Any]:
        """Convert a string attachment to the structured dict expected by the template."""
        if s.startswith("data:"):
            mime_type = detect_mime_hint(s)
            return {"url": s, "mime_type": mime_type, "type": mime_type, "data": None}

        if s.startswith(("http://", "https://")):
            mime_type = detect_mime_hint(s)
            return {"url": s, "mime_type": mime_type, "type": mime_type, "data": None}

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
        """Prepare data for detailed input/output analysis page."""
        if not results:
            return [], {}

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

        documents_map: dict[str, dict[str, Any]] = {}

        for result in results:
            for prediction in result.predictions:
                doc_id = prediction.document_id

                if doc_id not in documents_map:
                    content = doc_content_map.get(doc_id)
                    if not content and hasattr(prediction, 'metadata') and prediction.metadata:
                        content = prediction.metadata.get('content')
                    if not content:
                        content = f"Document ID: {doc_id}"

                    raw_attachments = doc_attachments_map.get(doc_id)
                    if not raw_attachments and hasattr(prediction, 'metadata') and prediction.metadata:
                        raw_attachments = prediction.metadata.get('attachments', [])

                    attachments = [
                        self._normalize_attachment(a) if isinstance(a, str) else a
                        for a in (raw_attachments or [])
                    ]

                    documents_map[doc_id] = {
                        "document_id": doc_id,
                        "content": content,
                        "attachments": attachments,
                        "expected_value": prediction.expected_value,
                        "metadata": doc_metadata_map.get(doc_id),
                        "model_results": [],
                    }

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

        metadata_lookup = {}
        for idx, doc_id in enumerate(documents_map.keys(), 1):
            if documents:
                for doc in documents:
                    if doc.id == doc_id and hasattr(doc, 'metadata') and doc.metadata:
                        metadata_lookup[f"doc{idx}"] = doc.metadata
                        break

        return list(documents_map.values()), metadata_lookup

    def _extract_recommended_model(self, recommendation: str, results: list[EvaluationResult]) -> str | None:
        """Extract the recommended model name from the AI recommendation text."""
        if not recommendation:
            return None

        model_names = [result.model for result in results]
        recommendation_lower = recommendation.lower()

        for model_name in model_names:
            if model_name.lower() in recommendation_lower:
                return model_name

        return None

    def _create_html_template(self):
        """Load Jinja2 HTML template for report."""
        return _jinja_env.get_template("evaluation_report.jinja2.html")

    def _create_detailed_analysis_template(self):
        """Load Jinja2 HTML template for detailed analysis page."""
        return _jinja_env.get_template("detailed_analysis.jinja2.html")

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

        Returns:
            Tuple of (Path to generated report, recommendation text or None)
        """
        output_path = Path(output_path)

        results = sorted(
            results,
            key=lambda r: r.metrics.total_cost if r.metrics else float('inf')
        )

        recommendation = None
        recommended_model = None
        if include_recommendation:
            recommendation = self.generate_recommendation(results, use_case)
            recommended_model = self._extract_recommended_model(recommendation, results)

        num_documents = 0
        prompt_template = original_prompt if original_prompt else ""
        if not prompt_template and results and results[0].metrics:
            num_documents = results[0].metrics.total_documents
            prompt_template = results[0].prompt_template
        elif results and results[0].metrics:
            num_documents = results[0].metrics.total_documents

        all_field_names = set()
        has_field_metrics = False
        for result in results:
            if result.metrics and result.metrics.aggregated_field_metrics:
                has_field_metrics = True
                all_field_names.update(result.metrics.aggregated_field_metrics.keys())

        all_field_names = sorted(all_field_names)

        field_metrics_data = self._prepare_field_metrics_data(results, all_field_names)
        field_max_values = self._compute_field_max_values(results, all_field_names)
        field_metrics_tree = self._build_field_metrics_tree(results, all_field_names)
        performance_best = self._compute_performance_best_values(results)
        performance_ranks = self._compute_performance_ranks(results)

        has_optimizations = prompt_optimizations is not None and any(
            len(manipulations) > 0 for manipulations in prompt_optimizations.values()
        )
        has_overrides = bool(model_override_prompts)

        chart_data = self._prepare_chart_data(results)
        documents_data, document_metadata = self._prepare_detailed_analysis_data(results, documents)

        field_config_json = field_config if field_config else None

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

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html_content, encoding="utf-8")

        analysis_path = output_path.parent / "detailed_analysis.html"
        analysis_template = self._create_detailed_analysis_template()
        analysis_content = analysis_template.render(
            documents_data=documents_data,
            document_metadata=document_metadata
        )
        analysis_path.write_text(analysis_content, encoding="utf-8")

        return output_path, recommendation
