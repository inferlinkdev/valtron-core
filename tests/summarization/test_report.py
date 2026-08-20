"""The summarization HTML and PDF reports.

``ModelEval`` refuses to build a report because its own assumes a correctness
notion; this recipe overrides that refusal with a report built for a task that
has none. These tests check the overriding actually happened, that the output
carries the things a reader needs -- the ranking, the axes behind it, the cost
split, the judge's per-document verdicts -- and that it survives the two states
a run can be in: freshly evaluated, and reloaded from disk.
"""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

from valtron_core.client import LLMClient
from valtron_core.evaluation.summarization import SummarizationExperiment
from valtron_core.reports.generate_summarization_report import SummarizationReportGenerator
from valtron_core.summarization import SALIENCE_SUMMARY_PROMPT
from valtron_core.summarization.model import Model
from tests.summarization.fakes import FakeSummarizer, FakeJudge
from tests.summarization.test_experiment import _install

DOCUMENT = "KEY alpha. KEY beta. minor gamma"
GOOD = "KEY alpha. KEY beta"
PADDED = "KEY alpha. unrelated noise"
REQUIREMENTS = ["alpha", "beta"]

RECOMMENDATION = "## Pick `good`\n\nIt covers **both** must-convey facts."


@pytest.fixture
def no_recommendation_call(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Stand in for the recommendation call; the report must not hit a real model."""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = RECOMMENDATION
    complete_sync = Mock(return_value=response)
    monkeypatch.setattr(LLMClient, "complete_sync", complete_sync)
    return complete_sync


async def _evaluated(
    monkeypatch: pytest.MonkeyPatch, output_dir: Path, **overrides: Any
) -> SummarizationExperiment:
    summaries = {"good": GOOD, "padded": PADDED}
    candidates: dict[str, Model] = {
        name: FakeSummarizer(name, text) for name, text in summaries.items()
    }
    _install(monkeypatch, candidates, FakeJudge())
    config: dict[str, Any] = {
        "models": [{"name": name} for name in summaries],
        "prompt": SALIENCE_SUMMARY_PROMPT,
        "judge_model": "judge",
        "requirements": REQUIREMENTS,
        "output_dir": str(output_dir),
    }
    config.update(overrides)
    experiment = SummarizationExperiment(config=config, data=[{"id": "d1", "content": DOCUMENT}])
    await experiment.aevaluate()
    return experiment


class TestHtmlReport:
    """What lands in the HTML."""

    async def test_writes_the_report(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_recommendation_call: Mock
    ) -> None:
        experiment = await _evaluated(monkeypatch, tmp_path)
        path = experiment.save_html_report()
        assert path.exists()
        assert path.name == "summarization_report.html"

    async def test_carries_the_ranking_and_the_axes_behind_it(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_recommendation_call: Mock
    ) -> None:
        experiment = await _evaluated(monkeypatch, tmp_path)
        html = experiment.save_html_report().read_text()
        assert "good" in html and "padded" in html
        assert "1.0000" in html  # good's score
        assert "Faithfulness" in html and "Salient coverage" in html
        # The winner's coverage: both must-convey facts.
        assert "100.0%" in html

    async def test_shows_the_cost_split_rather_than_only_a_total(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_recommendation_call: Mock
    ) -> None:
        experiment = await _evaluated(monkeypatch, tmp_path)
        html = experiment.save_html_report().read_text()
        assert "Generating summaries" in html
        assert "Judging each summary" in html
        assert "Shared per-document work" in html

    async def test_shows_the_judges_per_document_verdicts(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_recommendation_call: Mock
    ) -> None:
        # A ranking is arguable only if a reader can see what it turned on.
        experiment = await _evaluated(monkeypatch, tmp_path)
        html = experiment.save_html_report().read_text()
        assert "must-convey fact" in html
        assert "KEY alpha" in html
        assert "Salient hits" in html

    async def test_embeds_the_recommendation(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_recommendation_call: Mock
    ) -> None:
        experiment = await _evaluated(monkeypatch, tmp_path)
        html = experiment.save_html_report().read_text()
        assert "Pick `good`" in html
        assert no_recommendation_call.call_count == 1

    async def test_the_recommendation_prompt_talks_about_axes_not_accuracy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_recommendation_call: Mock
    ) -> None:
        # The classification recommendation reasons about an accuracy-to-cost
        # ratio and would raise outright on our accuracy=None.
        experiment = await _evaluated(monkeypatch, tmp_path)
        experiment.save_html_report()
        prompt = no_recommendation_call.call_args.kwargs["messages"][0]["content"]
        assert "Faithfulness" in prompt and "Salient coverage" in prompt
        assert "accuracy" not in prompt.lower()

    async def test_shows_the_prompt_each_candidate_saw(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_recommendation_call: Mock
    ) -> None:
        experiment = await _evaluated(monkeypatch, tmp_path)
        html = experiment.save_html_report().read_text()
        assert "Your summary must satisfy these requirements:" in html


class TestPdfReport:
    """The PDF covers the same ground."""

    async def test_writes_a_pdf(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_recommendation_call: Mock
    ) -> None:
        experiment = await _evaluated(monkeypatch, tmp_path)
        path = experiment.save_pdf_report()
        assert path.exists()
        assert path.read_bytes()[:4] == b"%PDF"

    async def test_reuses_the_html_recommendation_rather_than_paying_twice(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_recommendation_call: Mock
    ) -> None:
        experiment = await _evaluated(monkeypatch, tmp_path)
        experiment.save_html_report()
        experiment.save_pdf_report()
        assert no_recommendation_call.call_count == 1


class TestRunEndToEnd:
    """``arun()``, which the base class drives from ``output_formats``."""

    async def test_produces_both_formats(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_recommendation_call: Mock
    ) -> None:
        summaries = {"good": GOOD, "padded": PADDED}
        candidates: dict[str, Model] = {
            name: FakeSummarizer(name, text) for name, text in summaries.items()
        }
        _install(monkeypatch, candidates, FakeJudge())
        experiment = SummarizationExperiment(
            config={
                "models": [{"name": name} for name in summaries],
                "prompt": SALIENCE_SUMMARY_PROMPT,
                "judge_model": "judge",
                "requirements": REQUIREMENTS,
                "output_dir": str(tmp_path),
                "output_formats": ["html", "pdf"],
            },
            data=[{"id": "d1", "content": DOCUMENT}],
        )
        report = await experiment.arun()

        assert report.name == "summarization_report.html"
        assert (tmp_path / "metadata.json").exists()
        assert (tmp_path / "summarization_report.pdf").exists()


class TestReloadedRun:
    """A run read back from disk has predictions but has never been ranked."""

    async def test_reports_from_a_reloaded_run(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, no_recommendation_call: Mock
    ) -> None:
        experiment = await _evaluated(monkeypatch, tmp_path)
        experiment.save_experiment_results()

        reloaded = SummarizationExperiment.load_experiment_results(tmp_path)
        assert isinstance(reloaded, SummarizationExperiment)
        # compute_task_statistics only runs inside aevaluate(), so the report has
        # to rebuild the ranking from the persisted task_scores.
        html = reloaded.save_html_report(tmp_path / "again").read_text()
        assert "good" in html and "1.0000" in html


class TestGeneratorDirectly:
    """Pieces worth pinning without going through the recipe."""

    def test_an_undefined_axis_is_a_dash_not_a_zero(self) -> None:
        from valtron_core.reports.generate_summarization_report import _percent

        assert _percent(None) == "—"
        assert _percent(0.0) == "0.0%"

    def test_no_scores_means_no_recommendation(self) -> None:
        from valtron_core.evaluation.summarization import SummarizationRanking

        generator = SummarizationReportGenerator(client=Mock(spec=LLMClient))
        empty = SummarizationRanking(tiers=[], scores=[], parameters={}, usage={})
        assert generator.generate_recommendation(empty) is None
