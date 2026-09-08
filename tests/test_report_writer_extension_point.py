"""A new report format is one new class and a registration, nothing else.

``MarkdownReportWriter`` here is not a real product feature and is not meant
to ship; it exists only to prove, from outside ``valtron_core.reports``
entirely, that ``ModelEval``'s ``ReportWriter``/``self._report_writers``
extension point actually holds up: a caller can add a whole new report
format to a live recipe instance without editing any of the three existing
generator files, ``reports/writers.py``, or ``ModelEval``/``SummarizationExperiment``
themselves. If a real Markdown report is ever wanted, copy this and refine
it; nothing here is that.

Deliberately self-contained rather than importing fixtures from
``tests/summarization/test_report.py``: this file exists to prove a point
about the extension mechanism, not to extend that file's own coverage, so
it builds its own (smaller) evaluated experiment instead of depending on
another test module's internals.
"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from valtron_core.client import LLMClient
from valtron_core.evaluation.summarization import SummarizationExperiment
from valtron_core.models import EvaluationResult
from valtron_core.summarization import SALIENCE_SUMMARY_PROMPT
from tests.summarization.fakes import FakeJudge, FakeSummarizer
from tests.summarization.test_experiment import _install


@pytest.fixture
def no_recommendation_call(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Stand in for the recommendation call; no test here should hit a real model."""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = "Pick `m1`."
    complete_sync = Mock(return_value=response)
    monkeypatch.setattr(LLMClient, "complete_sync", complete_sync)
    return complete_sync


async def _evaluated(monkeypatch: pytest.MonkeyPatch, output_dir: Path) -> SummarizationExperiment:
    candidates = {"m1": FakeSummarizer("m1", "KEY alpha")}
    _install(monkeypatch, candidates, FakeJudge())
    config = {
        "models": [{"name": "m1"}],
        "prompt": SALIENCE_SUMMARY_PROMPT,
        "judge_model": "judge",
        "requirements": ["alpha"],
        "output_dir": str(output_dir),
    }
    experiment = SummarizationExperiment(
        config=config, data=[{"id": "d1", "content": "KEY alpha. minor beta"}]
    )
    await experiment.aevaluate()
    return experiment


class MarkdownReportWriter:
    """Deliberately minimal ``ReportWriter``: one heading, one line per model.

    Reads ``ranking`` out of ``**context`` the same way ``SummarizationHtmlReportWriter``/
    ``SummarizationPdfReportWriter`` do, since it's registered against a
    ``SummarizationExperiment`` in the tests below; a writer for a
    ground-truth recipe would read ``context`` differently, or not need it at
    all, which is the whole point of ``ReportWriter.write()`` taking
    ``**context`` rather than a fixed shape.
    """

    def write(
        self,
        results: "list[EvaluationResult]",
        output_path: "str | Path",
        **context: Any,
    ) -> "tuple[Path, str | None]":
        ranking = context["ranking"]
        lines = ["# Summarization ranking", ""]
        for tier_rank, tier in enumerate(ranking.tiers, start=1):
            for model in tier:
                lines.append(f"{tier_rank}. {model}")
        path = Path(output_path) / "report.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")
        return path, None


class TestExtensionPoint:
    async def test_a_registered_writer_produces_a_real_report(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_recommendation_call: Mock
    ) -> None:
        experiment = await _evaluated(monkeypatch, tmp_path)
        experiment._report_writers["markdown"] = MarkdownReportWriter()

        path = experiment.save_report("markdown", tmp_path)

        assert path.exists()
        content = path.read_text()
        assert content.startswith("# Summarization ranking")
        assert "m1" in content

    async def test_existing_formats_are_unaffected_by_the_new_one(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_recommendation_call: Mock
    ) -> None:
        experiment = await _evaluated(monkeypatch, tmp_path)
        experiment._report_writers["markdown"] = MarkdownReportWriter()

        html_path = experiment.save_html_report(tmp_path)
        pdf_path = experiment.save_pdf_report(tmp_path)
        markdown_path = experiment.save_report("markdown", tmp_path)

        assert html_path.exists() and html_path.suffix == ".html"
        assert pdf_path.exists() and pdf_path.suffix == ".pdf"
        assert markdown_path.exists() and markdown_path.suffix == ".md"

    async def test_an_unregistered_format_still_raises_not_implemented(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_recommendation_call: Mock
    ) -> None:
        experiment = await _evaluated(monkeypatch, tmp_path)

        with pytest.raises(NotImplementedError):
            experiment.save_report("markdown", tmp_path)
