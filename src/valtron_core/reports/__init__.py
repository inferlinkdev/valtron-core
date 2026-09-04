"""Report generation package."""

from valtron_core.reports._base import _ReportBase, TEMPLATES_DIR, _jinja_env
from valtron_core.reports.generate_html_report import HtmlReportGenerator
from valtron_core.reports.generate_pdf_report import PdfReportGenerator
from valtron_core.reports.generate_summarization_report import SummarizationReportGenerator


class ReportGenerator(HtmlReportGenerator, PdfReportGenerator):
    """Unified generator with both HTML and PDF generation."""


__all__ = [
    "ReportGenerator",
    "HtmlReportGenerator",
    "PdfReportGenerator",
    "SummarizationReportGenerator",
    "_ReportBase",
    "TEMPLATES_DIR",
    "_jinja_env",
]
