"""Unit tests for :mod:`valtron_core.summarization.judge` -- the four judge operations."""

from __future__ import annotations

import asyncio

import pytest

from valtron_core.summarization.judge import MAX_CLAIMS_PER_CALL, Judge
from valtron_core.summarization.prompts import (
    DocumentSaliencePrompt,
    FactExtractionPrompt,
    FactMatchPrompt,
    RequirementScoringPrompt,
)
from valtron_core.summarization.text import Fact, FactSource, Requirement, Summary
from tests.summarization.fakes import FakeJudge, OmittingJudge


def _facts(*texts: str, source: FactSource = FactSource.DOCUMENT) -> list[Fact]:
    prefix = "d" if source is FactSource.DOCUMENT else "g"
    return [Fact(id=f"{prefix}{i}", source=source, text=text) for i, text in enumerate(texts)]


class TestFacts:
    """Extraction, and the per-run memoization around it."""

    async def test_extracts_facts_with_source_prefixed_ids(self) -> None:
        judge = Judge(FakeJudge())
        facts = await judge.facts("Alpha. Beta", FactSource.DOCUMENT)
        assert [(f.id, f.text) for f in facts] == [("d0", "Alpha"), ("d1", "Beta")]

    async def test_generated_facts_get_a_different_prefix(self) -> None:
        # Ids must not collide across sources, since verdicts are keyed by id.
        judge = Judge(FakeJudge())
        facts = await judge.facts("Alpha. Beta", FactSource.GENERATED)
        assert [f.id for f in facts] == ["g0", "g1"]

    async def test_the_same_text_is_extracted_only_once(self) -> None:
        model = FakeJudge()
        judge = Judge(model)
        await judge.facts("Alpha. Beta", FactSource.DOCUMENT)
        await judge.facts("Alpha. Beta", FactSource.DOCUMENT)
        assert model.count(FactExtractionPrompt) == 1

    async def test_concurrent_callers_await_one_extraction(self) -> None:
        # The point of storing the task before the first await: racing callers
        # must not each start their own extraction.
        model = FakeJudge()
        judge = Judge(model)
        await asyncio.gather(*(judge.facts("Alpha. Beta", FactSource.DOCUMENT) for _ in range(8)))
        assert model.count(FactExtractionPrompt) == 1

    async def test_the_same_text_under_a_different_source_is_extracted_separately(self) -> None:
        model = FakeJudge()
        judge = Judge(model)
        await judge.facts("Alpha", FactSource.DOCUMENT)
        await judge.facts("Alpha", FactSource.GENERATED)
        assert model.count(FactExtractionPrompt) == 2


class TestFractionSupported:
    """The shared matching call behind three of the four axes."""

    async def test_returns_the_fraction_and_the_per_claim_verdicts(self) -> None:
        judge = Judge(FakeJudge())
        claims = _facts("alpha", "nowhere", source=FactSource.GENERATED)
        references = _facts("alpha and more")
        fraction, verdicts = await judge.fraction_supported(claims, references)
        assert fraction == pytest.approx(0.5)
        assert verdicts == {"g0": True, "g1": False}

    async def test_no_claims_gives_an_undefined_fraction_and_no_call(self) -> None:
        model = FakeJudge()
        judge = Judge(model)
        fraction, verdicts = await judge.fraction_supported([], _facts("alpha"))
        assert fraction is None
        assert verdicts == {}
        assert model.count(FactMatchPrompt) == 0

    async def test_a_long_claim_list_is_split_across_calls(self) -> None:
        # Omissions scale with how many verdicts one response must carry, so long
        # lists are chunked rather than asked all at once.
        model = FakeJudge()
        judge = Judge(model)
        claims = _facts(*[f"claim{i}" for i in range(MAX_CLAIMS_PER_CALL * 2 + 1)])
        await judge.fraction_supported(claims, _facts("nothing"))
        assert model.count(FactMatchPrompt) == 3

    async def test_a_short_claim_list_is_still_one_call(self) -> None:
        model = FakeJudge()
        judge = Judge(model)
        await judge.fraction_supported(_facts(*[f"c{i}" for i in range(MAX_CLAIMS_PER_CALL)]), [])
        assert model.count(FactMatchPrompt) == 1

    async def test_an_omitted_verdict_is_re_asked_then_raises(self) -> None:
        # Never guess: an unjudged claim must fail loudly rather than quietly
        # distorting the score.
        judge = Judge(OmittingJudge(FactMatchPrompt, "verdicts"))
        with pytest.raises(ValueError, match="no verdict for fact"):
            await judge.fraction_supported(_facts("a", "b"), _facts("a"))


class TestMarkSalient:
    """The reference-free importance oracle."""

    async def test_marks_the_must_convey_facts(self) -> None:
        judge = Judge(FakeJudge())
        facts = _facts("KEY finding", "minor detail")
        assert await judge.mark_salient(facts) == {"d0": True, "d1": False}

    async def test_no_facts_makes_no_call(self) -> None:
        model = FakeJudge()
        judge = Judge(model)
        assert await judge.mark_salient([]) == {}
        assert model.count(DocumentSaliencePrompt) == 0

    async def test_a_long_fact_list_is_chunked(self) -> None:
        # Safe precisely because the salience question is absolute per fact, not
        # a ranking of the facts against each other.
        model = FakeJudge()
        judge = Judge(model)
        facts = _facts(*[f"fact{i}" for i in range(MAX_CLAIMS_PER_CALL + 1)])
        await judge.mark_salient(facts)
        assert model.count(DocumentSaliencePrompt) == 2

    async def test_an_omitted_salience_verdict_raises(self) -> None:
        judge = Judge(OmittingJudge(DocumentSaliencePrompt, "saliences"))
        with pytest.raises(ValueError, match="no salience verdict"):
            await judge.mark_salient(_facts("a", "b"))


class TestRequirementsMet:
    """The optional checklist."""

    async def test_scores_the_fraction_satisfied(self) -> None:
        judge = Judge(FakeJudge())
        met, _ = await judge.requirements_met(
            Summary("the director was named"),
            [Requirement("director"), Requirement("running time")],
        )
        assert met == pytest.approx(0.5)

    async def test_reports_which_requirement_was_missed(self) -> None:
        # The fraction says half the checklist went unmet; only the verdicts say
        # which half, and that is the part a reviewer argues with.
        judge = Judge(FakeJudge())
        _, verdicts = await judge.requirements_met(
            Summary("the director was named"),
            [Requirement("director"), Requirement("running time")],
        )
        assert verdicts == {"r0": True, "r1": False}

    async def test_no_requirements_is_undefined_not_zero(self) -> None:
        # Undefined and zero must stay distinct: "no checklist" is not "failed
        # every item", and the score treats them very differently.
        judge = Judge(FakeJudge())
        assert await judge.requirements_met(Summary("anything"), []) == (None, {})

    async def test_an_omitted_requirement_verdict_raises(self) -> None:
        judge = Judge(OmittingJudge(RequirementScoringPrompt, "verdicts"))
        with pytest.raises(ValueError, match="no verdict for requirement"):
            await judge.requirements_met(Summary("x"), [Requirement("a"), Requirement("b")])
