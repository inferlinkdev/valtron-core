"""Each ``ReportWriter`` reproduces its wrapped generator's output byte-for-byte.

Not "the writer runs without crashing": the actual claim this commit makes
(``ARCHITECTURE_PROPOSAL.md``: "adapt the three existing report generators,
byte-identical output") is checked directly, by generating the same report
twice, once through the wrapped generator's own method and once through the
writer, and diffing the results.
"""

import re
from datetime import datetime

import pytest

from valtron_core.evaluation.summarization import SummarizationRanking, SummarizationScore
from valtron_core.models import EvaluationResult
from valtron_core.reports import generate_html_report, generate_pdf_report
from valtron_core.reports import generate_summarization_report as generate_summarization_report_mod
from valtron_core.reports.generate_html_report import HtmlReportGenerator
from valtron_core.reports.generate_pdf_report import PdfReportGenerator
from valtron_core.reports.generate_summarization_report import SummarizationReportGenerator
from valtron_core.reports.writers import (
    HtmlReportWriter,
    PdfReportWriter,
    SummarizationHtmlReportWriter,
    SummarizationPdfReportWriter,
)

_PDF_ID_RE = re.compile(rb"/ID \n\[<[0-9a-f]+><[0-9a-f]+>\]")
_PDF_DATE_RE = re.compile(rb"/(CreationDate|ModDate) \(D:\d+[+-]\d\d'\d\d'\)")


class _FrozenDatetime(datetime):
    """Every report generator stamps ``datetime.now()`` into its own output.

    Two back-to-back renders of the same report land in different wall-clock
    seconds often enough to make a byte-for-byte comparison flaky (the PDF
    generator embeds second-precision; a shifted timestamp string shifts the
    PDF's own byte offsets too, cascading through its xref table). Freezing
    this removes that source of flakiness entirely rather than papering over
    it with a wider text-based comparison.
    """

    @classmethod
    def now(cls, tz=None):
        return datetime(2026, 1, 1, 12, 0, 0)


@pytest.fixture(autouse=True)
def _freeze_report_timestamps(monkeypatch):
    for module in (generate_html_report, generate_pdf_report, generate_summarization_report_mod):
        monkeypatch.setattr(module, "datetime", _FrozenDatetime)


def _normalize_pdf(data: bytes) -> bytes:
    """Strip ReportLab's own per-render metadata before comparing two PDFs.

    ReportLab stamps every PDF with a fresh document ID digest and its own
    ``/CreationDate``/``/ModDate``, both from a clock this module's
    ``datetime.now()`` freeze never touches (ReportLab reads the wall clock
    internally, not via ``generate_pdf_report.py``'s own import), even when
    the visible content is otherwise identical. Two back-to-back renders of
    the exact same report never compare equal byte-for-byte without
    stripping all three.
    """
    data = _PDF_ID_RE.sub(b"/ID [normalized]", data)
    data = _PDF_DATE_RE.sub(b"/Date [normalized]", data)
    return data


def _zero_usage_part() -> dict:
    """One accumulator's worth of zero usage, same shape as summarization's own _usage_dict()."""
    return {
        "calls": 0,
        "cache_hits": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "by_model": {},
    }


def _ranking(results: "list[EvaluationResult]") -> SummarizationRanking:
    return SummarizationRanking(
        tiers=[[r.model] for r in results],
        scores=[
            SummarizationScore(
                model=r.model,
                score=1.0,
                correctness=1.0,
                salient_coverage=1.0,
                salient_precision=1.0,
                requirements_met=None,
                documents_scored=len(r.predictions),
            )
            for r in results
        ],
        parameters={"gate": 0.5, "beta": 1.0, "requirement_weight": 0.2, "tier_gap": 0.05},
        usage={
            "judge_model": "judge",
            "generation": _zero_usage_part(),
            "judge_per_candidate": _zero_usage_part(),
            "judge_shared": _zero_usage_part(),
        },
    )


class TestHtmlReportWriter:
    def test_matches_generator_directly(self, mock_llm_client, sample_evaluation_results, tmp_path):
        direct_dir = tmp_path / "direct"
        writer_dir = tmp_path / "writer"

        direct_path, direct_recommendation = HtmlReportGenerator(
            client=mock_llm_client
        ).generate_html_report(sample_evaluation_results, direct_dir, include_recommendation=False)
        writer_path, writer_recommendation = HtmlReportWriter(client=mock_llm_client).write(
            sample_evaluation_results, writer_dir, include_recommendation=False
        )

        assert writer_recommendation == direct_recommendation
        assert writer_path.read_text() == direct_path.read_text()


class TestPdfReportWriter:
    def test_matches_generator_directly(self, mock_llm_client, sample_evaluation_results, tmp_path):
        direct_dir = tmp_path / "direct"
        writer_dir = tmp_path / "writer"

        direct_path = PdfReportGenerator(client=mock_llm_client).generate_pdf_report(
            sample_evaluation_results, direct_dir, recommendation="Use gpt-4o."
        )
        writer_path, writer_recommendation = PdfReportWriter(client=mock_llm_client).write(
            sample_evaluation_results, writer_dir, recommendation="Use gpt-4o."
        )

        assert writer_recommendation == "Use gpt-4o."
        assert _normalize_pdf(writer_path.read_bytes()) == _normalize_pdf(direct_path.read_bytes())

    def test_write_returns_recommendation_untouched_when_omitted(
        self, mock_llm_client, sample_evaluation_results, tmp_path
    ):
        _, recommendation = PdfReportWriter(client=mock_llm_client).write(
            sample_evaluation_results, tmp_path
        )

        assert recommendation is None


class TestSummarizationHtmlReportWriter:
    def test_matches_generator_directly(self, mock_llm_client, sample_evaluation_results, tmp_path):
        ranking = _ranking(sample_evaluation_results)
        direct_dir = tmp_path / "direct"
        writer_dir = tmp_path / "writer"

        direct_path, direct_recommendation = SummarizationReportGenerator(
            client=mock_llm_client
        ).generate_html_report(
            sample_evaluation_results, ranking, direct_dir, include_recommendation=False
        )
        writer_path, writer_recommendation = SummarizationHtmlReportWriter(
            client=mock_llm_client
        ).write(
            sample_evaluation_results,
            writer_dir,
            ranking=ranking,
            include_recommendation=False,
        )

        assert writer_recommendation == direct_recommendation
        assert writer_path.read_text() == direct_path.read_text()


class TestSummarizationPdfReportWriter:
    def test_matches_generator_directly(self, mock_llm_client, sample_evaluation_results, tmp_path):
        ranking = _ranking(sample_evaluation_results)
        direct_dir = tmp_path / "direct"
        writer_dir = tmp_path / "writer"

        direct_path = SummarizationReportGenerator(client=mock_llm_client).generate_pdf_report(
            sample_evaluation_results, ranking, direct_dir, recommendation="Use gpt-4o."
        )
        writer_path, writer_recommendation = SummarizationPdfReportWriter(
            client=mock_llm_client
        ).write(
            sample_evaluation_results,
            writer_dir,
            ranking=ranking,
            recommendation="Use gpt-4o.",
        )

        assert writer_recommendation == "Use gpt-4o."
        assert _normalize_pdf(writer_path.read_bytes()) == _normalize_pdf(direct_path.read_bytes())
