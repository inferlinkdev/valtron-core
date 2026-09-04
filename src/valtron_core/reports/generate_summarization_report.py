"""Report generation for the summarization recipe.

The classification/extraction report is built around accuracy: correct/incorrect
badges, an accuracy column, an accuracy-to-cost recommendation. None of that
applies to a task with no ground truth, so this is a separate generator and a
separate template rather than a widening of those. It answers the questions a
summarization run actually raises:

* Which model won, and by how much -- the tiered ranking.
* *Why* it won. A score is a blend of four axes, and a zero from a failed
  faithfulness gate means something entirely different from a zero from thin
  coverage, so the axes are shown alongside every score rather than behind it.
* What the run spent, split into generating summaries, judging them, and the
  per-document work every candidate shared.
* What the judge actually decided, per document: which facts it deemed
  must-convey, which of them each summary carried, and which checklist items
  each satisfied.

Nothing here mutates the shared report layer. ``_ReportBase`` supplies the
Jinja environment and the cost/latency chart data, both of which are already
free of any correctness assumption.
"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import structlog
from reportlab.lib import colors  # type: ignore[import-untyped]
from reportlab.lib.pagesizes import letter  # type: ignore[import-untyped]
from reportlab.lib.styles import (  # type: ignore[import-untyped]
    ParagraphStyle,
    getSampleStyleSheet,
)
from reportlab.lib.units import inch  # type: ignore[import-untyped]
from reportlab.platypus import (  # type: ignore[import-untyped]
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from valtron_core.models import EvaluationResult
from valtron_core.reports._base import TEMPLATES_DIR, _jinja_env, _ReportBase

if TYPE_CHECKING:
    from valtron_core.evaluation.summarization import SummarizationRanking

logger = structlog.get_logger()

AXIS_LABELS = {
    "correctness": "Faithfulness",
    "salient_coverage": "Salient coverage",
    "salient_precision": "Salient precision",
    "requirements_met": "Requirements met",
}


def _percent(value: float | None) -> str:
    """Format an axis for display, distinguishing undefined from zero."""
    return "—" if value is None else f"{value * 100:.1f}%"


class SummarizationReportGenerator(_ReportBase):
    """Builds the HTML and PDF reports for a ``SummarizationExperiment`` run."""

    # -------------------------------------------------------------------------
    # Recommendation
    # -------------------------------------------------------------------------

    def generate_recommendation(
        self,
        ranking: "SummarizationRanking",
        use_case: str = "summarization",
        recommendation_model: str = "gpt-4o",
    ) -> str | None:
        """Ask a model to read the ranking and say what it means.

        Deliberately not the classification recommendation, which reasons about
        an accuracy-to-cost ratio and would raise on our ``accuracy=None``. The
        trade-off worth narrating here is different anyway: coverage against
        precision, and whether anything failed the faithfulness gate.
        """
        if not ranking.scores:
            return None

        lines = []
        for entry in ranking.scores:
            axes = ", ".join(
                f"{AXIS_LABELS[name]}={_percent(value)}" for name, value in entry.axes().items()
            )
            lines.append(
                f"- {entry.model}: score={entry.score:.4f}, {axes}, "
                f"scored on {entry.documents_scored} document(s)"
            )

        gate = ranking.parameters.get("gate", 0.5)
        prompt = f"""You are advising on which model to use for a summarization task.

Use case: {use_case}

Each model was scored with no reference summaries. A judge decomposed every
document into atomic facts and marked which ones a good summary must convey.
The axes mean:

- Faithfulness: share of the summary's claims supported by the document. This is
  a GATE, not a term: below {gate:.0%} the model scores zero outright.
- Salient coverage: share of the must-convey facts the summary actually conveys.
- Salient precision: share of the summary's claims that land on must-convey
  material. Coverage and precision are combined as a harmonic mean, so padding a
  summary cannot buy a better score.
- Requirements met: share of an optional per-class checklist satisfied. Shown as
  "—" when no checklist was supplied.

Results, best first:
{chr(10).join(lines)}

Provide, as Markdown, in three short paragraphs at most:
1. Which model to pick and why, in terms of the axes rather than the score alone.
2. The trade-off between the top models -- is the leader better because it covers
   more, or because it pads less? Note any model that failed the faithfulness gate.
3. A caveat if the ranking rests on few documents, since this method is meant to
   be read over a corpus rather than trusted per document.
"""
        try:
            response = self.client.complete_sync(
                model=recommendation_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            content = response.choices[0].message.content
            return str(content).strip() if content else None
        except Exception as error:
            logger.warning("summarization_recommendation_failed", error=str(error))
            return None

    # -------------------------------------------------------------------------
    # Context building
    # -------------------------------------------------------------------------

    @staticmethod
    def _axis_chart(ranking: "SummarizationRanking") -> dict[str, Any]:
        """Grouped-bar data: one series per axis, one category per model.

        A bar rather than a radar, because the axes are not commensurable -- a
        radar's enclosed area would imply they trade off evenly, and the
        faithfulness gate means they emphatically do not.
        """
        models = [entry.model for entry in ranking.scores]
        return {
            "models": models,
            "series": [
                {
                    "name": label,
                    "data": [
                        None if (v := getattr(entry, name)) is None else round(v * 100, 1)
                        for entry in ranking.scores
                    ],
                }
                for name, label in AXIS_LABELS.items()
            ],
            "scores": [round(entry.score, 4) for entry in ranking.scores],
        }

    @staticmethod
    def _tier_of(ranking: "SummarizationRanking", model: str) -> int:
        for index, tier in enumerate(ranking.tiers):
            if model in tier:
                return index + 1
        return 0

    def _document_rows(self, results: list[EvaluationResult]) -> list[dict[str, Any]]:
        """One row per document, carrying every model's showing on it.

        The judge's own verdicts are the point of this section: a ranking is
        arguable only if a reader can see which facts it turned on.
        """
        by_document: dict[str, dict[str, Any]] = {}
        for result in results:
            for prediction in result.predictions:
                row = by_document.setdefault(
                    prediction.document_id,
                    {
                        "id": prediction.document_id,
                        "salient_facts": prediction.metadata.get("salient_facts", []),
                        "document_facts": prediction.metadata.get("document_facts", []),
                        "requirements": prediction.metadata.get("requirements", []),
                        "models": [],
                    },
                )
                # Salience is a property of the document, but only reaches us
                # through a prediction; the first model to report it wins, and a
                # later one that failed has none to offer.
                if not row["salient_facts"]:
                    row["salient_facts"] = prediction.metadata.get("salient_facts", [])
                scores = prediction.task_scores or {}
                covered = prediction.metadata.get("coverage_verdicts", {})
                salient = prediction.metadata.get("salient_facts", [])
                facts = prediction.metadata.get("document_facts", [])
                salient_ids = [f"d{i}" for i, text in enumerate(facts) if text in salient]
                row["models"].append(
                    {
                        "model": prediction.model,
                        "summary": prediction.predicted_value,
                        "error": prediction.error,
                        "axes": {name: _percent(scores.get(name)) for name in AXIS_LABELS},
                        "hits": sum(1 for fid in salient_ids if covered.get(fid)),
                        "salient_total": len(salient_ids),
                        "requirement_verdicts": prediction.metadata.get("requirement_verdicts", {}),
                        "cost": prediction.llm_cost + prediction.evaluation_cost,
                        "seconds": prediction.response_time,
                    }
                )
        return sorted(by_document.values(), key=lambda row: str(row["id"]))

    def _context(
        self,
        results: list[EvaluationResult],
        ranking: "SummarizationRanking",
        *,
        use_case: str,
        original_prompt: str | None,
        model_prompts: dict[str, str] | None,
        recommendation: str | None,
    ) -> dict[str, Any]:
        totals = {
            key: ranking.usage[key] for key in ("generation", "judge_per_candidate", "judge_shared")
        }
        total_cost = sum(part["cost_usd"] for part in totals.values())
        total_tokens = sum(part["total_tokens"] for part in totals.values())
        return {
            "use_case": use_case,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "num_models": len(ranking.scores),
            "num_documents": max((len(r.predictions) for r in results), default=0),
            "judge_model": ranking.usage.get("judge_model", ""),
            "parameters": ranking.parameters,
            "tiers": ranking.tiers,
            "scores": [
                {
                    "model": entry.model,
                    "tier": self._tier_of(ranking, entry.model),
                    "score": entry.score,
                    "axes": {name: _percent(value) for name, value in entry.axes().items()},
                    "gated": entry.correctness is not None
                    and entry.correctness < ranking.parameters.get("gate", 0.5),
                    "documents_scored": entry.documents_scored,
                }
                for entry in ranking.scores
            ],
            "axis_labels": AXIS_LABELS,
            "chart": self._axis_chart(ranking),
            "cost_chart": self._prepare_chart_data(results),
            "usage": totals,
            "total_cost": total_cost,
            "total_tokens": total_tokens,
            "documents": self._document_rows(results),
            "original_prompt": original_prompt or "",
            "model_prompts": model_prompts or {},
            "recommendation": recommendation,
        }

    # -------------------------------------------------------------------------
    # HTML
    # -------------------------------------------------------------------------

    def generate_html_report(
        self,
        results: list[EvaluationResult],
        ranking: "SummarizationRanking",
        output_path: str | Path,
        *,
        use_case: str = "summarization",
        include_recommendation: bool = True,
        recommendation_model: str = "gpt-4o",
        original_prompt: str | None = None,
        model_prompts: dict[str, str] | None = None,
    ) -> tuple[Path, str | None]:
        """Write ``html_report/summarization_report.html``; return it and the recommendation."""
        destination = Path(output_path) / "html_report"
        destination.mkdir(parents=True, exist_ok=True)

        recommendation = (
            self.generate_recommendation(ranking, use_case, recommendation_model)
            if include_recommendation
            else None
        )
        context = self._context(
            results,
            ranking,
            use_case=use_case,
            original_prompt=original_prompt,
            model_prompts=model_prompts,
            recommendation=recommendation,
        )
        template = _jinja_env.get_template("summarization_report.jinja2.html")
        report_path = destination / "summarization_report.html"
        report_path.write_text(template.render(**context), encoding="utf-8")

        favicon = TEMPLATES_DIR / "favicon.svg"
        if favicon.exists():
            (destination / "favicon.svg").write_text(favicon.read_text(), encoding="utf-8")

        logger.info("summarization_html_report_written", path=str(report_path))
        return report_path, recommendation

    # -------------------------------------------------------------------------
    # PDF
    # -------------------------------------------------------------------------

    def generate_pdf_report(
        self,
        results: list[EvaluationResult],
        ranking: "SummarizationRanking",
        output_path: str | Path,
        *,
        use_case: str = "summarization",
        recommendation: str | None = None,
    ) -> Path:
        """Write ``summarization_report.pdf`` covering the same ground as the HTML."""
        destination = Path(output_path)
        destination.mkdir(parents=True, exist_ok=True)
        pdf_path = destination / "summarization_report.pdf"

        styles = getSampleStyleSheet()
        heading = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            spaceBefore=18,
            spaceAfter=8,
            textColor=colors.HexColor("#1f2937"),
        )
        note = ParagraphStyle(
            "Note", parent=styles["BodyText"], fontSize=8.5, textColor=colors.HexColor("#6b7280")
        )

        story: list[Any] = [
            Paragraph("Summarization Evaluation", styles["Title"]),
            Paragraph(
                f"{use_case} &mdash; {len(ranking.scores)} models over "
                f"{max((len(r.predictions) for r in results), default=0)} documents, "
                f"judged by {ranking.usage.get('judge_model', 'n/a')}",
                styles["Normal"],
            ),
            Spacer(1, 0.25 * inch),
            Paragraph("Ranking", heading),
            self._ranking_table(ranking),
            Spacer(1, 0.1 * inch),
            Paragraph(
                "Faithfulness is a gate, not a term: below "
                f"{ranking.parameters.get('gate', 0.5):.0%} a model scores zero however well "
                "it covers the document. Coverage and precision are combined as a harmonic "
                "mean, so padding a summary cannot buy a better score. An em dash means the "
                "axis was undefined, which is not the same as zero.",
                note,
            ),
            Paragraph("Cost", heading),
            self._cost_table(ranking),
            Spacer(1, 0.1 * inch),
            Paragraph(
                "The shared judge cost is the per-document fact extraction and salience "
                "pass, which every candidate reuses; it is divided evenly between them in "
                "the per-model figures.",
                note,
            ),
        ]

        if recommendation:
            story.extend(
                [
                    Paragraph("Recommendation", heading),
                    Paragraph(recommendation.replace("\n", "<br/>"), styles["BodyText"]),
                ]
            )

        rows = self._document_rows(results)
        if rows:
            story.append(PageBreak())
            story.append(Paragraph("Per-document detail", heading))
            for row in rows:
                story.append(self._document_table(row))
                story.append(Spacer(1, 0.18 * inch))

        SimpleDocTemplate(
            str(pdf_path),
            pagesize=letter,
            title="Summarization Evaluation",
            leftMargin=0.75 * inch,
            rightMargin=0.75 * inch,
        ).build(story)
        logger.info("summarization_pdf_report_written", path=str(pdf_path))
        return pdf_path

    @staticmethod
    def _styled(data: list[list[Any]], widths: list[float]) -> Table:
        table = Table(data, colWidths=widths, hAlign="LEFT")
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f2937")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#d1d5db")),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                    (
                        "ROWBACKGROUNDS",
                        (0, 1),
                        (-1, -1),
                        [colors.white, colors.HexColor("#f9fafb")],
                    ),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        return table

    def _ranking_table(self, ranking: "SummarizationRanking") -> Table:
        header = ["Tier", "Model", "Score"] + list(AXIS_LABELS.values()) + ["Docs"]
        data: list[list[Any]] = [header]
        for entry in ranking.scores:
            data.append(
                [
                    str(self._tier_of(ranking, entry.model)),
                    entry.model,
                    f"{entry.score:.4f}",
                    *[_percent(value) for value in entry.axes().values()],
                    str(entry.documents_scored),
                ]
            )
        return self._styled(
            data, [0.4 * inch, 1.5 * inch, 0.6 * inch] + [0.85 * inch] * 4 + [0.4 * inch]
        )

    def _cost_table(self, ranking: "SummarizationRanking") -> Table:
        data: list[list[Any]] = [["What", "Calls", "Prompt tokens", "Completion tokens", "Cost"]]
        labels = {
            "generation": "Generating summaries",
            "judge_per_candidate": "Judging each summary",
            "judge_shared": "Shared per-document work",
        }
        total_cost = 0.0
        for key, label in labels.items():
            part = ranking.usage[key]
            total_cost += part["cost_usd"]
            data.append(
                [
                    label,
                    str(part["calls"]),
                    f"{part['prompt_tokens']:,}",
                    f"{part['completion_tokens']:,}",
                    f"${part['cost_usd']:.6f}",
                ]
            )
        data.append(["Total", "", "", "", f"${total_cost:.6f}"])
        return self._styled(data, [2.0 * inch, 0.7 * inch, 1.1 * inch, 1.3 * inch, 1.0 * inch])

    def _document_table(self, row: dict[str, Any]) -> Table:
        data: list[list[Any]] = [[f"Document {row['id']}", "Score axes", "Salient hits", "Summary"]]
        for entry in row["models"]:
            axes = " / ".join(entry["axes"][name] for name in AXIS_LABELS)
            summary = entry["error"] or str(entry["summary"])
            data.append(
                [
                    entry["model"],
                    axes,
                    f"{entry['hits']}/{entry['salient_total']}",
                    Paragraph(
                        summary[:400] + ("…" if len(summary) > 400 else ""),
                        ParagraphStyle("Cell", fontSize=7.5, leading=9.5),
                    ),
                ]
            )
        return self._styled(data, [1.2 * inch, 1.9 * inch, 0.8 * inch, 3.1 * inch])
