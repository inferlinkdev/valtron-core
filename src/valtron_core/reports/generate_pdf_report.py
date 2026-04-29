"""PDF report generation."""

import base64
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend - must be before pyplot import
import matplotlib.pyplot as plt

from valtron_core.client import LLMClient
from valtron_core.models import EvaluationResult
from valtron_core.reports._base import _ReportBase, _jinja_env


def _check_weasyprint_available() -> None:
    try:
        from weasyprint import HTML as _HTML  # noqa: F401
    except (ImportError, OSError) as e:
        raise ImportError(
            "PDF generation requires WeasyPrint system dependencies. "
            "See the installation guide: https://doc.courtbouillon.org/weasyprint/stable/first_steps.html"
        ) from e


class PdfReportGenerator(_ReportBase):
    """Generate PDF evaluation reports."""

    @staticmethod
    def _format_metric_display(metric: str, params: dict) -> str:
        """Return a human-readable string for a metric name + params."""
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

    def _create_pdf_template(self):
        """Load Jinja2 HTML template optimized for PDF generation via weasyprint."""
        return _jinja_env.get_template("pdf_report.jinja2.html")

    def _generate_charts(
        self,
        results: list[EvaluationResult],
        output_dir: Path,
    ) -> list[Path]:
        """Generate matplotlib charts for the PDF report."""
        chart_paths = []

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

        colors = [pastel_colors[i % len(pastel_colors)] for i in range(len(models))]
        bar_width = 0.5

        def setup_clean_theme(ax, fig):
            fig.patch.set_facecolor('white')
            ax.set_facecolor('white')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_color('#cccccc')
            ax.spines['left'].set_color('#cccccc')
            ax.tick_params(colors='#333333', which='both')
            ax.xaxis.label.set_color('#333333')
            ax.yaxis.label.set_color('#333333')
            ax.title.set_color('#333333')
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

        Returns:
            Path to the generated PDF
        """
        _check_weasyprint_available()
        from weasyprint import HTML

        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        results = sorted(
            results,
            key=lambda r: r.metrics.total_cost if r.metrics else float('inf')
        )

        chart_paths = self._generate_charts(results, output_dir)

        chart_images = {}
        for chart_path in chart_paths:
            chart_name = chart_path.stem
            with open(chart_path, "rb") as f:
                chart_images[chart_name] = base64.b64encode(f.read()).decode("utf-8")
            chart_path.unlink()

        num_models = len(results)
        num_documents = results[0].metrics.total_documents if results and results[0].metrics else 0

        chart_data = self._prepare_chart_data(results)

        all_field_names = set()
        has_field_metrics = False
        for result in results:
            if result.metrics and result.metrics.aggregated_field_metrics:
                has_field_metrics = True
                all_field_names.update(result.metrics.aggregated_field_metrics.keys())
        all_field_names = sorted(all_field_names) if all_field_names else []

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

        performance_best = self._compute_performance_best_values(results)
        performance_ranks = self._compute_performance_ranks(results)

        has_optimizations = prompt_optimizations is not None and any(
            len(m) > 0 for m in prompt_optimizations.values()
        )

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

        pdf_path = output_dir / f"{output_path.stem}.pdf"
        HTML(string=html_content).write_pdf(pdf_path)

        return pdf_path
