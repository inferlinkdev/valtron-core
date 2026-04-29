"""PDF report generation using ReportLab Platypus."""

import tempfile
import xml.sax.saxutils as _xml
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use('Agg')  # non-interactive backend - must be before pyplot import
import matplotlib.pyplot as plt

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from valtron_core.client import LLMClient
from valtron_core.models import EvaluationResult
from valtron_core.reports._base import _ReportBase

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

_CONTENT_WIDTH = 7.0 * inch   # 8.5 in page − 2 × 0.75 in margins

# ---------------------------------------------------------------------------
# Colour palette (mirrors the CSS in the old pdf_report.jinja2.html)
# ---------------------------------------------------------------------------

C_BLUE      = colors.HexColor('#2563eb')
C_GRAY_BG   = colors.HexColor('#f8f9fa')
C_BORDER    = colors.HexColor('#dee2e6')
C_DARK      = colors.HexColor('#1a1a1a')
C_MEDIUM    = colors.HexColor('#6b7280')
C_LIGHT     = colors.HexColor('#9ca3af')
C_SUBHEAD   = colors.HexColor('#4b5563')
C_TH_BG     = colors.HexColor('#f3f4f6')
C_BEST_BG   = colors.HexColor('#eff6ff')
C_BEST_TEXT = C_BLUE
C_PURPLE_BG = colors.HexColor('#f5f3ff')
C_PURPLE    = colors.HexColor('#7c3aed')
C_APP_BODY  = colors.HexColor('#374151')

# ---------------------------------------------------------------------------
# Paragraph styles
# ---------------------------------------------------------------------------

_STYLES: dict[str, ParagraphStyle] = {
    "h1":         ParagraphStyle("rl_h1",  fontName="Helvetica-Bold",    fontSize=18, textColor=C_BLUE,    leading=22),
    "h2":         ParagraphStyle("rl_h2",  fontName="Helvetica-Bold",    fontSize=13, textColor=C_BLUE,    leading=17, spaceAfter=4),
    "h3":         ParagraphStyle("rl_h3",  fontName="Helvetica-Bold",    fontSize=11, textColor=C_SUBHEAD, leading=14, spaceBefore=10, spaceAfter=6),
    "h4":         ParagraphStyle("rl_h4",  fontName="Helvetica-Bold",    fontSize=10, textColor=C_DARK,    leading=13, spaceBefore=8,  spaceAfter=2),
    "body":       ParagraphStyle("rl_bod", fontName="Helvetica",         fontSize=10, textColor=C_DARK,    leading=14, spaceAfter=6),
    "small":      ParagraphStyle("rl_sm",  fontName="Helvetica",         fontSize=8.5,textColor=C_MEDIUM,  leading=12),
    "timestamp":  ParagraphStyle("rl_ts",  fontName="Helvetica",         fontSize=8,  textColor=C_LIGHT,   leading=11),
    "note":       ParagraphStyle("rl_nt",  fontName="Helvetica-Oblique", fontSize=8,  textColor=C_MEDIUM,  leading=11),
    "code":       ParagraphStyle("rl_cd",  fontName="Courier",           fontSize=8,  textColor=C_DARK,    leading=11),
    "field_note": ParagraphStyle("rl_fn",  fontName="Helvetica-Oblique", fontSize=8.5,textColor=C_SUBHEAD, leading=11),
    "app_body":   ParagraphStyle("rl_ab",  fontName="Helvetica",         fontSize=9.5,textColor=C_APP_BODY,leading=13),
    "app_subhead":ParagraphStyle("rl_as",  fontName="Helvetica-Bold",    fontSize=10.5,textColor=C_BLUE,   leading=14, spaceBefore=14, spaceAfter=6),
    "rec":        ParagraphStyle("rl_rec", fontName="Helvetica",         fontSize=10, textColor=C_DARK,    leading=15),
    "td_cell":    ParagraphStyle("rl_tdc", fontName="Helvetica",         fontSize=9,  textColor=C_DARK,    leading=11),
}


class PdfReportGenerator(_ReportBase):
    """Generate PDF evaluation reports using ReportLab Platypus."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_pdf_report(
        self,
        results: list[EvaluationResult],
        output_path: str | Path,
        recommendation: str | None = None,
        original_prompt: str | None = None,
        prompt_optimizations: dict[str, list[str]] | None = None,
        model_override_prompts: dict[str, str] | None = None,
    ) -> Path:
        """Generate a PDF report using ReportLab.

        Returns:
            Path to the generated PDF.
        """
        output_path = Path(output_path)
        output_dir = output_path.parent
        output_dir.mkdir(parents=True, exist_ok=True)

        results = sorted(
            results,
            key=lambda r: r.metrics.total_cost if r.metrics else float('inf'),
        )

        # Collect field names
        all_field_names: list[str] = sorted({
            f
            for r in results
            if r.metrics and r.metrics.aggregated_field_metrics
            for f in r.metrics.aggregated_field_metrics
        })
        has_field_metrics = bool(all_field_names)

        # Per-field metrics
        field_metrics_data: dict[str, list[dict]] = {}
        field_max_values: dict[str, dict] = {}
        if has_field_metrics:
            for field_name in all_field_names:
                field_metrics_data[field_name] = []
                max_p = max_r = max_f1 = 0.0
                for result in results:
                    if result.metrics and result.metrics.aggregated_field_metrics:
                        fm = result.metrics.aggregated_field_metrics.get(field_name)
                        if fm:
                            f1 = (
                                (2 * fm.precision * fm.recall / (fm.precision + fm.recall))
                                if (fm.precision + fm.recall) > 0 else 0.0
                            )
                            max_p  = max(max_p,  fm.precision)
                            max_r  = max(max_r,  fm.recall)
                            max_f1 = max(max_f1, f1)
                            field_metrics_data[field_name].append({
                                "model":          result.model,
                                "precision":      fm.precision,
                                "recall":         fm.recall,
                                "f1_score":       f1,
                                "metric_display": self._format_metric_display(fm.metric, fm.params or {}),
                            })
                field_max_values[field_name] = {
                    "precision": max_p,
                    "recall":    max_r,
                    "f1_score":  max_f1,
                }

        performance_best  = self._compute_performance_best_values(results)
        performance_ranks = self._compute_performance_ranks(results)
        has_optimizations = (
            prompt_optimizations is not None
            and any(len(v) > 0 for v in prompt_optimizations.values())
        )

        num_models    = len(results)
        num_documents = (results[0].metrics.total_documents
                         if results and results[0].metrics else 0)
        timestamp     = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Charts are written to a temp dir; the context keeps them alive
        # for the duration of doc.build() since Image() reads files at build time.
        with tempfile.TemporaryDirectory() as tmpdir:
            chart_paths = self._generate_charts(results, Path(tmpdir))

            data: dict[str, Any] = {
                "timestamp":          timestamp,
                "num_models":         num_models,
                "num_documents":      num_documents,
                "results":            results,
                "recommendation":     recommendation,
                "has_field_metrics":  has_field_metrics,
                "all_field_names":    all_field_names,
                "prompt_optimizations":   prompt_optimizations or {},
                "has_optimizations":  has_optimizations,
                "model_override_prompts": model_override_prompts or {},
                "original_prompt":    original_prompt,
                "field_metrics_data": field_metrics_data,
                "field_max_values":   field_max_values,
                "performance_best":   performance_best,
                "performance_ranks":  performance_ranks,
                "chart_paths":        chart_paths,
            }

            story = self._build_pdf_story(data)

            pdf_path = output_dir / f"{output_path.stem}.pdf"
            doc = SimpleDocTemplate(
                str(pdf_path),
                pagesize=LETTER,
                leftMargin=0.75 * inch,
                rightMargin=0.75 * inch,
                topMargin=0.75 * inch,
                bottomMargin=0.75 * inch,
            )
            doc.build(story)

        return pdf_path

    # ------------------------------------------------------------------
    # Metric formatting (shared with field metrics tables)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_metric_display(metric: str, params: dict) -> str:
        """Return a human-readable label for a metric name + params."""
        _METRIC_NAMES: dict[str, str] = {
            "exact":           "Exact Match",
            "exact_match":     "Exact Match",
            "text_similarity": "Text Similarity",
            "llm":             "LLM Judge",
            "embedding":       "Embedding Similarity",
            "list_greedy_f1":  "List - Greedy F1",
            "list_ordered_f1": "List - Ordered F1",
            "aggregated":      "Aggregated (weighted sub-fields)",
        }
        name = (
            "Aggregated (weighted sub-fields)" if metric.startswith("agg:")
            else _METRIC_NAMES.get(metric, metric.replace("_", " ").title())
        )
        if not params:
            return name
        _PARAM_LABELS: dict[str, str] = {
            "text_similarity_metric":    "similarity metric",
            "text_similarity_threshold": "threshold",
            "llm_model":                 "model",
            "embedding_model":           "model",
        }
        parts = [
            f"{_PARAM_LABELS.get(k, k.replace('_', ' '))}: {v}"
            for k, v in params.items() if v is not None
        ]
        return f"{name} ({', '.join(parts)})" if parts else name

    # ------------------------------------------------------------------
    # Chart generation
    # ------------------------------------------------------------------

    def _generate_charts(
        self,
        results: list[EvaluationResult],
        output_dir: Path,
    ) -> list[Path]:
        """Generate matplotlib bar charts and save them to output_dir."""
        chart_paths: list[Path] = []

        models:     list[str]   = []
        accuracies: list[float] = []
        costs:      list[float] = []
        times:      list[float] = []

        for result in results:
            if result.metrics:
                models.append(result.model)
                accuracies.append(result.metrics.accuracy * 100)
                costs.append(result.metrics.total_cost)
                times.append(result.metrics.total_time)

        if not models:
            return chart_paths

        pastel_colors = [
            '#a8d5e5', '#b5e6b5', '#d4b5e6', '#f5d5a8',
            '#f5b5c5', '#b5d5f5', '#e5e5b5', '#c5e5e5',
        ]
        bar_colors = [pastel_colors[i % len(pastel_colors)] for i in range(len(models))]
        bar_width  = 0.5

        def _clean_theme(ax: Any, fig: Any) -> None:
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

        # Accuracy
        fig, ax = plt.subplots(figsize=(6, 3.5))
        _clean_theme(ax, fig)
        bars = ax.bar(range(len(models)), accuracies, width=bar_width,
                      color=bar_colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Accuracy (%)', fontsize=10)
        ax.set_title('Quality: Accuracy by Model', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        ax.set_ylim(0, 105)
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, color='#333333')
        plt.tight_layout()
        acc_path = output_dir / 'chart_accuracy.png'
        plt.savefig(acc_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        chart_paths.append(acc_path)

        # Cost
        fig, ax = plt.subplots(figsize=(6, 3.5))
        _clean_theme(ax, fig)
        bars = ax.bar(range(len(models)), costs, width=bar_width,
                      color=bar_colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Total Cost ($)', fontsize=10)
        ax.set_title('Cost: Average Cost per Document', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        for bar, cost in zip(bars, costs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'${cost:.4f}', ha='center', va='bottom', fontsize=9, color='#333333')
        plt.tight_layout()
        cost_path = output_dir / 'chart_cost.png'
        plt.savefig(cost_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        chart_paths.append(cost_path)

        # Time
        fig, ax = plt.subplots(figsize=(6, 3.5))
        _clean_theme(ax, fig)
        bars = ax.bar(range(len(models)), times, width=bar_width,
                      color=bar_colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Model', fontsize=10)
        ax.set_ylabel('Total Time (s)', fontsize=10)
        ax.set_title('Speed: Average Time per Document', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
        for bar, t in zip(bars, times):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f'{t:.2f}s', ha='center', va='bottom', fontsize=9, color='#333333')
        plt.tight_layout()
        time_path = output_dir / 'chart_time.png'
        plt.savefig(time_path, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        chart_paths.append(time_path)

        return chart_paths

    # ------------------------------------------------------------------
    # Story builder - top-level orchestrator
    # ------------------------------------------------------------------

    def _build_pdf_story(self, data: dict) -> list:
        story: list = []
        story.extend(self._build_header(data))
        story.extend(self._build_summary(data))
        story.extend(self._build_perf_section(data))

        if data["has_field_metrics"] and len(data["all_field_names"]) > 1:
            story.append(Spacer(1, 0.1 * inch))
            story.extend(self._build_field_metrics(data))

        if data["recommendation"]:
            story.append(Spacer(1, 0.1 * inch))
            story.extend(self._build_recommendation(data))

        story.append(Spacer(1, 0.1 * inch))
        story.append(PageBreak())
        story.extend(self._build_visual_analysis(data))
        story.extend(self._build_appendix_a(data))

        has_field_metrics = data["has_field_metrics"]
        all_field_names  = data["all_field_names"]

        use_rank = has_field_metrics and len(all_field_names) > 1

        if use_rank:
            story.extend(self._build_appendix_b(data))
        return story

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_header(self, data: dict) -> list:
        rows = [
            [Paragraph("Valtron Evaluation Report", _STYLES["h1"])],
            [Paragraph("LLM Model Comparison Analysis", _STYLES["small"])],
            [Paragraph(f"Generated: {data['timestamp']}", _STYLES["timestamp"])],
        ]
        t = Table(rows, colWidths=[_CONTENT_WIDTH])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), C_GRAY_BG),
            ("BOX",           (0, 0), (-1, -1), 0.5, C_BORDER),
            ("LEFTPADDING",   (0, 0), (-1, -1), 20),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 20),
            ("TOPPADDING",    (0, 0), (0,  0),  16),
            ("BOTTOMPADDING", (0, -1), (-1, -1), 16),
            ("TOPPADDING",    (0, 1), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -2), 2),
        ]))
        return [t, Spacer(1, 0.15 * inch)]

    def _build_summary(self, data: dict) -> list:
        elems: list = []
        elems.extend(self._section_heading("Summary"))
        elems.append(Paragraph(
            f"<b>Models Evaluated:</b> {data['num_models']}", _STYLES["body"]
        ))
        elems.append(Paragraph(
            f"<b>Documents Tested:</b> {data['num_documents']}", _STYLES["body"]
        ))
        if data["original_prompt"]:
            elems.append(Paragraph("Prompt Template", _STYLES["h3"]))
            elems.append(self._make_code_box(data["original_prompt"]))
        return elems

    def _build_perf_section(self, data: dict) -> list:
        elems: list = []
        elems.append(Spacer(1, 0.1 * inch))
        elems.extend(self._section_heading("Performance Metrics"))
        elems.extend(self._build_perf_table(data))
        return elems

    def _build_perf_table(self, data: dict) -> list:
        results          = data["results"]
        performance_best = data["performance_best"]
        performance_ranks = data["performance_ranks"]
        has_field_metrics = data["has_field_metrics"]
        all_field_names  = data["all_field_names"]
        override_prompts = data["model_override_prompts"]
        prompt_opts      = data["prompt_optimizations"]
        has_opts         = data["has_optimizations"]

        use_rank   = has_field_metrics and len(all_field_names) > 1
        use_single = has_field_metrics and len(all_field_names) == 1

        rank_or_acc_header = "Performance Rank†" if use_rank else "Accuracy"
        headers = ["Model", "Prompt*", rank_or_acc_header,
                   "Total Cost", "Avg Cost/Doc", "Total Time", "Avg Time/Doc"]
        if has_opts:
            headers.append("Optimizations")

        # Column widths summing to _CONTENT_WIDTH (7.0 in).
        # Sized so header text fits on one line at 8pt bold Helvetica.
        col_w = [1.3*inch, 0.70*inch, 0.95*inch, 0.95*inch, 1.05*inch, 0.95*inch, 1.1*inch]
        if has_opts:
            # Add Optimizations column; shrink Model + Accuracy to stay at 7.0 in
            col_w = [1.0*inch, 0.65*inch, 0.80*inch, 0.85*inch, 0.9*inch, 0.85*inch, 0.9*inch, 1.05*inch]

        base_style: list = [
            ("BACKGROUND",   (0, 0), (-1, 0),  C_TH_BG),
            ("GRID",         (0, 0), (-1, -1), 0.5, C_BORDER),
            ("LEFTPADDING",  (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING",   (0, 0), (-1, -1), 5),
            ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
            ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ]

        # Paragraph-based cells so long content wraps instead of overflowing.
        def _th(text: str) -> Paragraph:
            return Paragraph(f"<b>{_xml.escape(text)}</b>", _STYLES["td_cell"])

        def _td(text: str, is_best: bool = False) -> Paragraph:
            safe = _xml.escape(str(text))
            if is_best:
                return Paragraph(
                    f'<b><font color="#2563eb">{safe}</font></b>', _STYLES["td_cell"]
                )
            return Paragraph(safe, _STYLES["td_cell"])

        table_data = [[_th(h) for h in headers]]
        for row_idx, result in enumerate(results):
            if not result.metrics:
                continue
            r = row_idx + 1
            m_display    = result.model.split('/')[-1] if '/' in result.model else result.model
            prompt_label = "Overridden" if override_prompts.get(result.model) else "Base"
            has_rate     = bool(result.llm_config and result.llm_config.get("cost_rate") is not None)

            # Accuracy/rank cell
            if use_rank:
                ri = performance_ranks.get(result.model, {"rank": "-", "delta_pct": 0.0})
                acc_cell  = f"#{ri['rank']}"
                if ri.get("delta_pct", 0) < 0:
                    acc_cell += f" ({ri['delta_pct']:.2f}%)"
                is_best_acc = ri.get("rank") == 1
            elif use_single:
                sf  = all_field_names[0]
                fm  = (result.metrics.aggregated_field_metrics or {}).get(sf)
                acc_cell    = f"{fm.precision * 100:.2f}%" if fm else "N/A"
                is_best_acc = fm is not None and fm.precision >= data["field_max_values"][sf]["precision"]
            else:
                acc_cell    = f"{result.metrics.accuracy * 100:.2f}%"
                is_best_acc = result.metrics.accuracy == performance_best["best_accuracy"]

            suffix = "*" if has_rate else ""
            best_total_cost = result.metrics.total_cost == performance_best["best_total_cost"]
            best_avg_cost   = result.metrics.average_cost_per_document == performance_best["best_avg_cost"]
            best_total_time = result.metrics.total_time == performance_best["best_total_time"]
            best_avg_time   = result.metrics.average_time_per_document == performance_best["best_avg_time"]

            row = [
                _td(m_display),
                _td(prompt_label),
                _td(acc_cell, is_best_acc),
                _td(f"${result.metrics.total_cost:.6f}{suffix}", best_total_cost),
                _td(f"${result.metrics.average_cost_per_document:.6f}{suffix}", best_avg_cost),
                _td(f"{result.metrics.total_time:.2f}s", best_total_time),
                _td(f"{result.metrics.average_time_per_document:.3f}s", best_avg_time),
            ]
            if has_opts:
                opts = prompt_opts.get(result.model, [])
                row.append(_td(", ".join(opts) if opts else "None"))
            table_data.append(row)

            # Best-value background highlighting (BACKGROUND still applies to Paragraph cells)
            highlights = {
                2: is_best_acc,
                3: best_total_cost,
                4: best_avg_cost,
                5: best_total_time,
                6: best_avg_time,
            }
            for col, is_best in highlights.items():
                if is_best:
                    base_style += [("BACKGROUND", (col, r), (col, r), C_BEST_BG)]

        t = Table(table_data, colWidths=col_w, repeatRows=1)
        t.setStyle(TableStyle(base_style))

        elems: list = [t, Spacer(1, 4)]
        elems.append(Paragraph("* See Appendix A for prompt references.", _STYLES["note"]))
        if use_rank:
            elems.append(Paragraph("† See Appendix B for metric definitions.", _STYLES["note"]))

        for result in results:
            if result.llm_config and result.llm_config.get("cost_rate") is not None:
                mn   = result.model.split('/')[-1] if '/' in result.model else result.model
                rate = result.llm_config["cost_rate"]
                unit = result.llm_config.get("cost_rate_time_unit", "1hr")
                elems.append(Paragraph(
                    f"* Cost for {_xml.escape(mn)} was estimated using a custom rate of "
                    f"${rate:.2f} per {_xml.escape(str(unit))}.",
                    _STYLES["note"],
                ))
        return elems

    def _build_field_metrics(self, data: dict) -> list:
        elems: list = []
        elems.extend(self._section_heading("Per-Field Metrics"))
        elems.append(Paragraph(
            "Detailed accuracy metrics for individual fields in structured outputs. "
            "Precision and Recall for each field are determined by its configured scoring "
            "function - see Appendix B for how fields are scored.",
            _STYLES["body"],
        ))

        col_w = [1.75 * inch, 1.75 * inch, 1.75 * inch, 1.75 * inch]

        for field_name in data["all_field_names"]:
            rows      = data["field_metrics_data"].get(field_name, [])
            maxv      = data["field_max_values"].get(field_name, {})
            seen      = list(dict.fromkeys(r["metric_display"] for r in rows))
            is_agg    = bool(seen and seen[0].startswith("Aggregated"))

            p_header  = "Average Precision of Subfields" if is_agg else "Precision"
            r_header  = "Average Recall of Subfields"   if is_agg else "Recall"
            f1_header = "Average F1 of Subfields"       if is_agg else "F1 Score"

            block: list = [
                Paragraph(f"Field: {_xml.escape(field_name)}", _STYLES["h3"]),
                Paragraph(
                    f"Scored using: {_xml.escape(' / '.join(seen))}", _STYLES["field_note"]
                ),
            ]

            style_cmds: list = [
                ("BACKGROUND",   (0, 0), (-1, 0),  C_TH_BG),
                ("FONTNAME",     (0, 0), (-1, 0),  "Helvetica-Bold"),
                ("FONTSIZE",     (0, 0), (-1, 0),  8),
                ("TEXTCOLOR",    (0, 0), (-1, 0),  C_DARK),
                ("FONTSIZE",     (0, 1), (-1, -1), 9),
                ("GRID",         (0, 0), (-1, -1), 0.5, C_BORDER),
                ("LEFTPADDING",  (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING",   (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
            ]
            table_data = [["Model", p_header, r_header, f1_header]]

            for ri, fm in enumerate(rows):
                r   = ri + 1
                mn  = fm["model"].split('/')[-1] if '/' in fm["model"] else fm["model"]
                table_data.append([
                    mn,
                    f"{fm['precision'] * 100:.2f}%",
                    f"{fm['recall'] * 100:.2f}%",
                    f"{fm['f1_score'] * 100:.2f}%",
                ])
                if fm["precision"] >= maxv.get("precision", float("inf")):
                    style_cmds += [("BACKGROUND", (1, r), (1, r), C_BEST_BG),
                                   ("TEXTCOLOR",  (1, r), (1, r), C_BEST_TEXT),
                                   ("FONTNAME",   (1, r), (1, r), "Helvetica-Bold")]
                if fm["recall"] >= maxv.get("recall", float("inf")):
                    style_cmds += [("BACKGROUND", (2, r), (2, r), C_BEST_BG),
                                   ("TEXTCOLOR",  (2, r), (2, r), C_BEST_TEXT),
                                   ("FONTNAME",   (2, r), (2, r), "Helvetica-Bold")]
                if fm["f1_score"] >= maxv.get("f1_score", float("inf")):
                    style_cmds += [("BACKGROUND", (3, r), (3, r), C_BEST_BG),
                                   ("TEXTCOLOR",  (3, r), (3, r), C_BEST_TEXT),
                                   ("FONTNAME",   (3, r), (3, r), "Helvetica-Bold")]

            ft = Table(table_data, colWidths=col_w, repeatRows=1)
            ft.setStyle(TableStyle(style_cmds))
            block.append(ft)
            elems.append(KeepTogether(block))
            elems.append(Spacer(1, 0.1 * inch))

        elems.append(Paragraph(
            "* Precision, Recall, and F1 Score - see B for definitions.",
            _STYLES["note"],
        ))
        return elems

    def _build_recommendation(self, data: dict) -> list:
        rec = data["recommendation"]
        if not rec:
            return []
        elems: list = []
        elems.extend(self._section_heading("AI-Powered Recommendation"))
        inner = Paragraph(
            _xml.escape(rec).replace("\n", "<br/>"), _STYLES["rec"]
        )
        box = Table([[inner]], colWidths=[_CONTENT_WIDTH - 0.3 * inch])
        box.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, 0), C_PURPLE_BG),
            ("LINEBEFORE",    (0, 0), (0, 0), 4, C_PURPLE),
            ("LEFTPADDING",   (0, 0), (0, 0), 14),
            ("RIGHTPADDING",  (0, 0), (0, 0), 12),
            ("TOPPADDING",    (0, 0), (0, 0), 12),
            ("BOTTOMPADDING", (0, 0), (0, 0), 12),
        ]))
        elems.append(box)
        return elems

    def _build_visual_analysis(self, data: dict) -> list:
        chart_paths   = data["chart_paths"]
        has_fm        = data["has_field_metrics"]
        field_names   = data["all_field_names"]
        show_accuracy = not has_fm or len(field_names) == 1

        # chart_paths order from _generate_charts: accuracy, cost, time
        chart_config = [
            (0, "Quality: Accuracy by Model",   show_accuracy),
            (1, "Cost: Total Cost by Model",     True),
            (2, "Speed: Total Time by Model",    True),
        ]

        elems: list = []
        elems.extend(self._section_heading("Visual Analysis"))

        for idx, label, should_show in chart_config:
            if not should_show or idx >= len(chart_paths):
                continue
            path = chart_paths[idx]
            if not path.exists():
                continue
            img = Image(str(path), width=5.0 * inch, height=3.0 * inch)
            elems.append(KeepTogether([
                Paragraph(label, _STYLES["h3"]),
                img,
                Spacer(1, 0.2 * inch),
            ]))

        return elems

    def _build_appendix_b(self, data: dict) -> list:
        use_rank = data["has_field_metrics"] and len(data["all_field_names"]) > 1

        def _subhead(text: str) -> list:
            return [
                Paragraph(text, _STYLES["app_subhead"]),
                HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=6),
            ]

        def _entry(title: str, tech_term: str | None, body: str) -> list:
            heading = f"<b>{_xml.escape(title)}</b>"
            if tech_term:
                heading += (
                    f' <font color="#6b7280" size="8">({_xml.escape(tech_term)})</font>'
                )
            return [
                Paragraph(heading, _STYLES["h4"]),
                Paragraph(_xml.escape(body), _STYLES["app_body"]),
                Spacer(1, 10),
            ]

        elems: list = [PageBreak()]
        elems.extend(self._section_heading("Appendix B: Metric Definitions"))

        elems.extend(_subhead("Performance Metrics"))
        if use_rank:
            elems.extend(_entry(
                "Performance Rank", None,
                "Models are ranked by their mean per-document score - a continuous 0–100% "
                "value that averages each field's score across all documents. Unlike binary "
                "accuracy (all-or-nothing per document), this gives partial credit for "
                "partially correct results, making it a more sensitive indicator of true "
                "model quality. The delta shown in parentheses is the percentage-point gap "
                "from the top-scoring model.",
            ))
        else:
            elems.extend(_entry(
                "Accuracy", None,
                "The percentage of documents where the model's output matched the expected "
                "output correctly, averaged across all evaluated examples. Higher is better.",
            ))

        elems.extend(_subhead("Per-Field Metrics"))
        elems.append(Paragraph(
            "These columns appear in the Per-Field Metrics tables. Precision, Recall, and "
            "F1 Score are computed using micro aggregation - predictions are pooled across "
            "all examples before dividing, so fields with more examples contribute "
            "proportionally more. The scoring function used for each field is shown beneath "
            "its name in the tables above.",
            _STYLES["app_body"],
        ))
        elems.append(Spacer(1, 10))
        elems.extend(_entry(
            "Precision", "micro",
            "Out of everything the model predicted for this field, how much of it was "
            "correct, as determined by the field's configured scoring function. For list "
            "fields, longer lists have more influence on this number than shorter ones.",
        ))
        elems.extend(_entry(
            "Recall", "micro",
            "Out of all the values that were expected for this field, how many did the "
            "model get right, as determined by the field's configured scoring function. "
            "For list fields, longer lists have more influence on this number than shorter ones.",
        ))
        elems.extend(_entry(
            "F1 Score", "micro-F1",
            "A single score that balances Precision and Recall by taking their harmonic "
            "mean - a high F1 requires both good precision and good recall. Can differ "
            "from Precision and Recall individually when list lengths or sub-field counts "
            "vary between examples.",
        ))

        elems.extend(_subhead("How Fields Are Scored"))
        elems.append(Paragraph(
            "Precision and Recall for each field are determined by its configured scoring "
            "function - for example, exact match, fuzzy similarity, or other comparators. "
            "The specific function is shown beneath each field name in the Per-Field "
            "Metrics tables.",
            _STYLES["app_body"],
        ))
        elems.append(Spacer(1, 10))
        elems.extend(_entry(
            "When the field holds a single value", None,
            "The predicted value is compared directly to the expected value using the "
            "field's configured scoring function (e.g. Exact Match, Text Similarity, LLM "
            "Judge). Precision, Recall, and F1 all agree for standalone value fields.",
        ))
        elems.extend(_entry(
            "When the field holds a list of items", None,
            "Each predicted item is matched to the closest available expected item "
            "(greedy matching), or compared position-by-position (ordered matching). "
            "Unmatched expected items count against Recall; unmatched predicted items "
            "count against Precision.",
        ))
        elems.extend(_entry(
            "When the field holds an object or dictionary", None,
            "The weighted average of each sub-field's aggregated Precision, Recall, and "
            "F1 is reported - so the object-level figures always match the average of the "
            "sub-field values shown. Sub-fields with a higher configured weight contribute "
            "more to these averages.",
        ))
        elems.extend(_entry(
            "When a value field is nested inside a list object", None,
            "Precision and Recall reflect how well the predicted object matched the "
            "expected object it was paired with. Extra predicted items that had no "
            "matching expected item are penalized at the parent list field level - not "
            "at the individual sub-field level.",
        ))
        return elems

    def _build_appendix_a(self, data: dict) -> list:
        elems: list = [PageBreak()]
        elems.extend(self._section_heading("Appendix A: Prompts"))

        elems.append(Paragraph("<b>Base Prompt</b>", _STYLES["h4"]))
        if data["original_prompt"]:
            elems.append(self._make_code_box(data["original_prompt"]))
        else:
            elems.append(Paragraph("No base prompt recorded.", _STYLES["body"]))
        elems.append(Spacer(1, 0.1 * inch))

        for result in data["results"]:
            override = data["model_override_prompts"].get(result.model)
            if override:
                mn = result.model.split('/')[-1] if '/' in result.model else result.model
                elems.append(Paragraph(
                    f"<b>Override Prompt - {_xml.escape(mn)}</b>", _STYLES["h4"]
                ))
                elems.append(self._make_code_box(override))
                elems.append(Spacer(1, 0.08 * inch))

        return elems

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _section_heading(self, text: str) -> list:
        return [
            Paragraph(text, _STYLES["h2"]),
            HRFlowable(width="100%", thickness=0.5, color=C_BORDER, spaceAfter=8),
        ]

    def _make_code_box(self, text: str) -> Table:
        inner = Paragraph(
            _xml.escape(text).replace("\n", "<br/>"),
            _STYLES["code"],
        )
        t = Table([[inner]], colWidths=[_CONTENT_WIDTH - 0.3 * inch])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (0, 0), C_GRAY_BG),
            ("BOX",           (0, 0), (0, 0), 0.5, C_BORDER),
            ("LEFTPADDING",   (0, 0), (0, 0), 10),
            ("RIGHTPADDING",  (0, 0), (0, 0), 10),
            ("TOPPADDING",    (0, 0), (0, 0), 8),
            ("BOTTOMPADDING", (0, 0), (0, 0), 8),
        ]))
        return t
