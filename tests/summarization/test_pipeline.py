"""The evaluation flow: what it asks, what it keeps, and who pays for it.

:mod:`valtron_core.summarization.api` already covers the axes end to end. What is tested here is
what the pipeline returns *besides* the axes -- the per-fact verdicts behind
each one, and the split of spend between generating a summary, grading it, and
the per-document work every candidate shares. A host application builds its
per-document cost and its detail views out of exactly those, so they are part of
the contract rather than incidental.

The fixture is the one used throughout: ``KEY alpha. KEY beta. minor gamma``,
three document facts of which the two ``KEY`` ones are salient.
"""

from __future__ import annotations

import pytest

from valtron_core.summarization.judge import Judge
from valtron_core.summarization.model import Model
from valtron_core.summarization.pipeline import (
    evaluate_candidate,
    evaluate_document,
    extract_document_facts,
)
from valtron_core.summarization.prompts import DocumentSaliencePrompt, FactExtractionPrompt
from valtron_core.summarization.text import Doc, Requirement
from tests.summarization.fakes import (
    FAKE_CALL_COST,
    FailingSummarizer,
    FakeJudge,
    FakeSummarizer,
)

DOC = Doc("KEY alpha. KEY beta. minor gamma")
REQUIREMENTS = [Requirement("alpha"), Requirement("beta")]
GOOD = "KEY alpha. KEY beta"
PADDED = "KEY alpha. unrelated noise"


class TestExtractDocumentFacts:
    """The per-document work every candidate shares."""

    async def test_separates_the_salient_facts_from_the_rest(self) -> None:
        shared = await extract_document_facts(DOC, Judge(FakeJudge()))
        assert [fact.text for fact in shared.facts] == ["KEY alpha", "KEY beta", "minor gamma"]
        assert [fact.text for fact in shared.salient] == ["KEY alpha", "KEY beta"]
        assert shared.salience == {"d0": True, "d1": True, "d2": False}

    async def test_charges_extraction_and_salience_to_the_shared_bucket(self) -> None:
        # Two calls: decompose the document, then mark which of its facts matter.
        shared = await extract_document_facts(DOC, Judge(FakeJudge()))
        assert shared.usage.calls == 2
        assert shared.usage.cost_usd == pytest.approx(2 * FAKE_CALL_COST)


class TestEvaluateCandidate:
    """One candidate on one document: the axes, and the verdicts behind them."""

    async def _run(self, summary: str, requirements: list[Requirement] | None = None):
        judge = Judge(FakeJudge())
        shared = await extract_document_facts(DOC, judge)
        model = FakeSummarizer("candidate", summary)
        return await evaluate_candidate(
            DOC, model, judge, shared, REQUIREMENTS if requirements is None else requirements
        )

    async def test_derives_the_four_axes(self) -> None:
        result = await self._run(PADDED)
        assert result.axes.correctness == pytest.approx(0.5)
        assert result.axes.salient_coverage == pytest.approx(0.5)
        assert result.axes.salient_precision == pytest.approx(0.5)
        assert result.axes.requirements_met == pytest.approx(0.5)

    async def test_keeps_the_verdict_behind_every_axis(self) -> None:
        # The axes rank a model; the verdicts are what lets someone argue with the
        # ranking, and the judge has already returned them either way.
        result = await self._run(PADDED)
        assert result.faithful_verdicts == {"g0": True, "g1": False}
        assert result.coverage_verdicts == {"d0": True, "d1": False, "d2": False}
        assert result.precision_verdicts == {"g0": True, "g1": False}
        assert result.requirement_verdicts == {"r0": True, "r1": False}

    async def test_keeps_the_summary_and_its_facts(self) -> None:
        result = await self._run(GOOD)
        assert result.summary.text == GOOD
        assert [fact.text for fact in result.summary_facts] == ["KEY alpha", "KEY beta"]
        assert result.model == "candidate"

    async def test_separates_generation_spend_from_judging_spend(self) -> None:
        # One generation call; five judge calls -- decompose the summary, three
        # entailment passes, and the checklist.
        result = await self._run(GOOD)
        assert result.generation_usage.calls == 1
        assert result.judge_usage.calls == 5
        assert result.generation_usage.cost_usd == pytest.approx(FAKE_CALL_COST)
        assert result.judge_usage.cost_usd == pytest.approx(5 * FAKE_CALL_COST)

    async def test_the_shared_work_is_charged_to_neither(self) -> None:
        # Document facts and salience belong to the document, not to whichever
        # candidate happened to need them first.
        result = await self._run(GOOD)
        assert result.generation_usage.calls + result.judge_usage.calls == 6

    async def test_an_empty_checklist_costs_no_judge_call(self) -> None:
        result = await self._run(GOOD, requirements=[])
        assert result.axes.requirements_met is None
        assert result.requirement_verdicts == {}
        assert result.judge_usage.calls == 4

    async def test_records_how_long_it_took(self) -> None:
        result = await self._run(GOOD)
        assert result.generation_seconds >= 0.0
        assert result.seconds >= result.generation_seconds


class TestEvaluateDocument:
    """Every candidate on one document, sharing the per-document work."""

    async def test_decomposes_the_document_once_for_all_candidates(self) -> None:
        judge_model = FakeJudge()
        candidates: dict[str, Model] = {
            "good": FakeSummarizer("good", GOOD),
            "padded": FakeSummarizer("padded", PADDED),
        }
        await evaluate_document(DOC, Judge(judge_model), candidates, REQUIREMENTS)
        assert judge_model.count(DocumentSaliencePrompt) == 1
        # One extraction for the document, one per distinct summary.
        assert judge_model.count(FactExtractionPrompt) == 3

    async def test_records_a_failing_candidate_instead_of_propagating(self) -> None:
        candidates: dict[str, Model] = {
            "good": FakeSummarizer("good", GOOD),
            "broken": FailingSummarizer("broken"),
        }
        result = await evaluate_document(DOC, Judge(FakeJudge()), candidates, REQUIREMENTS)
        assert set(result.candidates) == {"good"}
        assert "always fails" in result.failures["broken"]

    async def test_exposes_the_shared_work_alongside_the_candidates(self) -> None:
        candidates: dict[str, Model] = {"good": FakeSummarizer("good", GOOD)}
        result = await evaluate_document(DOC, Judge(FakeJudge()), candidates, REQUIREMENTS)
        assert [fact.text for fact in result.shared.salient] == ["KEY alpha", "KEY beta"]
        assert result.shared.usage.calls == 2
        assert result.failures == {}
