"""Concrete ``ReportWriter``s: one per (generator, format) pair that exists today.

Each of these wraps an existing generator unchanged and exposes it through
``ReportWriter``'s uniform ``write(results, output_path, **context)`` seam,
translating that call into whatever the wrapped generator's own method
actually needs. None of the four generator methods wrapped here (
``HtmlReportGenerator.generate_html_report``,
``PdfReportGenerator.generate_pdf_report``,
``SummarizationReportGenerator.generate_html_report``,
``SummarizationReportGenerator.generate_pdf_report``) are touched or
reimplemented; today's HTML/PDF bytes are unchanged.

Not yet used anywhere: no recipe or ``EvaluationRunner`` constructs one of
these yet. That wiring, and the registry that would pick one by format name,
is a separate, later commit; this one only proves the seam exists and each
writer really does reproduce its wrapped generator's output.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from valtron_core.client import LLMClient
from valtron_core.models import EvaluationResult
from valtron_core.reports.generate_html_report import HtmlReportGenerator
from valtron_core.reports.generate_pdf_report import PdfReportGenerator
from valtron_core.reports.generate_summarization_report import SummarizationReportGenerator

if TYPE_CHECKING:
    from valtron_core.evaluation.summarization import SummarizationRanking

__all__ = [
    "HtmlReportWriter",
    "PdfReportWriter",
    "SummarizationHtmlReportWriter",
    "SummarizationPdfReportWriter",
]


class HtmlReportWriter:
    """``ReportWriter`` over ``HtmlReportGenerator``, for classification/extraction."""

    def __init__(self, client: "LLMClient | None" = None) -> None:
        self._generator = HtmlReportGenerator(client=client)

    def write(
        self,
        results: "list[EvaluationResult]",
        output_path: "str | Path",
        **context: Any,
    ) -> "tuple[Path, str | None]":
        """Forwards every keyword straight to ``generate_html_report``; see its docstring."""
        return self._generator.generate_html_report(results, output_path, **context)


class PdfReportWriter:
    """``ReportWriter`` over ``PdfReportGenerator``, for classification/extraction.

    ``generate_pdf_report`` never generates its own recommendation (unlike
    the HTML generator); pass one in via ``context["recommendation"]`` if the
    caller already computed one, same as today's callers do, or omit it for
    a report with no recommendation section.
    """

    def __init__(self, client: "LLMClient | None" = None) -> None:
        self._generator = PdfReportGenerator(client=client)

    def write(
        self,
        results: "list[EvaluationResult]",
        output_path: "str | Path",
        **context: Any,
    ) -> "tuple[Path, str | None]":
        recommendation = context.get("recommendation")
        path = self._generator.generate_pdf_report(results, output_path, **context)
        return path, recommendation


class SummarizationHtmlReportWriter:
    """``ReportWriter`` over ``SummarizationReportGenerator``'s HTML output.

    Requires ``context["ranking"]``: unlike classification/extraction,
    summarization has no ground truth to score against, so the corpus-level
    ranking (not any per-prediction ``is_correct``) is what the report is
    actually built from.
    """

    def __init__(self, client: "LLMClient | None" = None) -> None:
        self._generator = SummarizationReportGenerator(client=client)

    def write(
        self,
        results: "list[EvaluationResult]",
        output_path: "str | Path",
        **context: Any,
    ) -> "tuple[Path, str | None]":
        ranking: "SummarizationRanking" = context.pop("ranking")
        return self._generator.generate_html_report(results, ranking, output_path, **context)


class SummarizationPdfReportWriter:
    """``ReportWriter`` over ``SummarizationReportGenerator``'s PDF output.

    Requires ``context["ranking"]``, same as ``SummarizationHtmlReportWriter``.
    Like ``PdfReportWriter``, never generates its own recommendation.
    """

    def __init__(self, client: "LLMClient | None" = None) -> None:
        self._generator = SummarizationReportGenerator(client=client)

    def write(
        self,
        results: "list[EvaluationResult]",
        output_path: "str | Path",
        **context: Any,
    ) -> "tuple[Path, str | None]":
        ranking: "SummarizationRanking" = context.pop("ranking")
        recommendation = context.get("recommendation")
        path = self._generator.generate_pdf_report(results, ranking, output_path, **context)
        return path, recommendation
