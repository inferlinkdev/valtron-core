"""End-to-end tests for ``SummarizationExperiment``, fully offline.

Two kinds of fixture, deliberately. The synthetic documents (``KEY alpha. KEY
beta. minor gamma``) let every expected score be worked out by hand from the
fakes' stated rules, so an assertion that moves does so for a followable reason.
The billsum documents are real Congressional bill text, and exist to check the
recipe survives contact with real prose -- long, oddly formatted, full of the
punctuation a sentence splitter trips over -- rather than only with sentences
invented to suit it.

Models are faked at the ``ClientModel`` seam. The seam below that, where
``ClientModel`` meets ``LLMClient``, has its own tests in
``test_client_model.py``; mocking both at once would test neither well.
"""

import json
from pathlib import Path
from typing import Any
from unittest.mock import ANY, patch

import pytest

from valtron_core.evaluation.summarization import (
    JUDGE_LABEL,
    SummarizationExperiment,
)
from valtron_core.summarization import SALIENCE_SUMMARY_PROMPT
from valtron_core.summarization.model import Model
from valtron_core.summarization.prompts import (
    DocumentSaliencePrompt,
    FactExtractionPrompt,
    FactMatchPrompt,
    Prompt,
    RequirementScoringPrompt,
)
from tests.summarization.fakes import FakeJudge, FakeSummarizer, FailingSummarizer

DATA = Path(__file__).parent / "data" / "billsum"

DOCUMENT = "KEY alpha. KEY beta. minor gamma"
REQUIREMENTS = ["alpha", "beta"]
GOOD = "KEY alpha. KEY beta"
PADDED = "KEY alpha. unrelated noise"

# The checklist authored for the billsum document class.
BILLSUM_REQUIREMENTS = [
    "Name the bill by its short title, with its year if the bill gives one.",
    "State what the bill does, using its operative action: amends, directs, "
    "requires, authorizes, or establishes.",
]


class PositionalJudge(FakeJudge):
    """A judge that marks the first two facts must-convey, whatever the text.

    The shared fake decides salience by looking for a ``KEY`` marker, which real
    documents obviously do not carry. For the billsum fixtures salience has to
    come from somewhere, and position is the one rule that is arbitrary in a way
    no assertion here depends on.
    """

    async def run_structured[T](self, prompt: Prompt, schema: type[T], *, usage: Any = None) -> T:
        result = await super().run_structured(prompt, schema, usage=usage)
        if isinstance(prompt, DocumentSaliencePrompt):
            data = result.model_dump()  # type: ignore[attr-defined]
            for index, item in enumerate(data["saliences"]):
                item["required"] = index < 2
            return schema.model_validate(data)  # type: ignore[attr-defined,no-any-return]
        return result


def _install(
    monkeypatch: pytest.MonkeyPatch,
    candidates: dict[str, Model],
    judge: FakeJudge | None = None,
) -> FakeJudge:
    """Hand the recipe fakes wherever it would build a ``ClientModel``."""
    judge = judge or FakeJudge()
    registry: dict[str, Model] = {JUDGE_LABEL: judge, **candidates}

    def build(model: Any, *, client: Any, name: str | None = None, **kwargs: Any) -> Model:
        key = name or (model if isinstance(model, str) else model.get("model"))
        return registry[str(key)]

    monkeypatch.setattr("valtron_core.evaluation.summarization.ClientModel", build)
    return judge


def _config(summaries: dict[str, str], **overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "models": [{"name": name} for name in summaries],
        "prompt": SALIENCE_SUMMARY_PROMPT,
        "judge_model": "judge",
        "requirements": REQUIREMENTS,
    }
    config.update(overrides)
    return config


def _experiment(
    monkeypatch: pytest.MonkeyPatch,
    summaries: dict[str, str],
    *,
    data: list[dict[str, Any]] | None = None,
    judge: FakeJudge | None = None,
    **overrides: Any,
) -> SummarizationExperiment:
    candidates: dict[str, Model] = {
        name: FakeSummarizer(name, text) for name, text in summaries.items()
    }
    _install(monkeypatch, candidates, judge)
    return SummarizationExperiment(
        config=_config(summaries, **overrides),
        data=data if data is not None else [{"id": "d1", "content": DOCUMENT}],
    )


class TestRanking:
    """The corpus-level outcome, which is the point of the recipe."""

    async def test_ranks_a_good_summary_above_a_padded_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # good:   coverage 2/2, precision 2/2, reqs 2/2 -> 0.4*1.0 + 0.6*1.0 = 1.0
        # padded: coverage 1/2, precision 1/2, reqs 1/2 -> 0.4*0.5 + 0.6*0.5 = 0.5
        experiment = _experiment(monkeypatch, {"good": GOOD, "padded": PADDED})
        await experiment.aevaluate()

        assert experiment.ranking.tiers == [["good"], ["padded"]]
        assert experiment.ranking.best == ["good"]
        assert experiment.ranking.scores[0].score == pytest.approx(1.0)
        assert experiment.ranking.scores[1].score == pytest.approx(0.5)

    async def test_reports_the_axes_behind_each_score(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        experiment = _experiment(monkeypatch, {"padded": PADDED})
        await experiment.aevaluate()

        (padded,) = experiment.ranking.scores
        assert padded.correctness == pytest.approx(0.5)
        assert padded.salient_coverage == pytest.approx(0.5)
        assert padded.salient_precision == pytest.approx(0.5)
        assert padded.requirements_met == pytest.approx(0.5)
        assert padded.documents_scored == 1

    async def test_records_the_scheme_parameters_used(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Without these the score is not reproducible from the saved run.
        experiment = _experiment(monkeypatch, {"good": GOOD}, gate=0.4, beta=2.0)
        await experiment.aevaluate()
        assert experiment.ranking.parameters["gate"] == 0.4
        assert experiment.ranking.parameters["beta"] == 2.0

    async def test_task_statistics_carry_the_same_ranking(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        experiment = _experiment(monkeypatch, {"good": GOOD, "padded": PADDED})
        await experiment.aevaluate()
        statistics = experiment._task_statistics or {}
        assert statistics["tiers"] == [["good"], ["padded"]]
        assert statistics["scheme"] == "salience-f+reqs"
        assert statistics["models"]["good"]["axes"]["salient_coverage"] == pytest.approx(1.0)

    async def test_the_ranking_is_unavailable_before_evaluating(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        experiment = _experiment(monkeypatch, {"good": GOOD})
        with pytest.raises(RuntimeError, match="Call evaluate\\(\\)"):
            _ = experiment.ranking


class TestPredictions:
    """What each document contributes, and the shape the base class expects."""

    async def test_the_summary_is_the_predicted_value(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        experiment = _experiment(monkeypatch, {"good": GOOD})
        await experiment.aevaluate()
        (prediction,) = experiment.get_traces()
        assert prediction.predicted_value == GOOD

    async def test_leaves_the_ground_truth_fields_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # There is no reference summary and no binary notion of correct. Faking
        # one would be worse than an empty field.
        experiment = _experiment(monkeypatch, {"good": GOOD})
        await experiment.aevaluate()
        (prediction,) = experiment.get_traces()
        assert prediction.expected_value is None
        assert prediction.is_correct is None
        assert prediction.example_score is None

    async def test_the_axes_ride_in_task_scores(self, monkeypatch: pytest.MonkeyPatch) -> None:
        experiment = _experiment(monkeypatch, {"padded": PADDED})
        await experiment.aevaluate()
        (prediction,) = experiment.get_traces()
        assert prediction.task_scores == {
            "correctness": pytest.approx(0.5),
            "salient_coverage": pytest.approx(0.5),
            "salient_precision": pytest.approx(0.5),
            "requirements_met": pytest.approx(0.5),
        }

    async def test_an_undefined_axis_is_omitted_rather_than_zeroed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # No must-convey facts means there is no recall to measure. Recording a
        # zero would read as total failure instead of "not applicable".
        experiment = _experiment(
            monkeypatch,
            {"any": "minor gamma"},
            data=[{"id": "d1", "content": "minor gamma. minor delta"}],
            requirements=[],
        )
        await experiment.aevaluate()
        (prediction,) = experiment.get_traces()
        assert "salient_coverage" not in (prediction.task_scores or {})
        assert "requirements_met" not in (prediction.task_scores or {})

    async def test_aggregated_task_scores_give_the_corpus_axes_for_free(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The reason the axes go in task_scores: the base class averages them
        # per key across documents, which is exactly the aggregation the scheme
        # wants -- and it lands in the saved per-model file at no extra cost.
        experiment = _experiment(
            monkeypatch,
            {"partial": "KEY alpha"},
            data=[
                {"id": "d1", "content": DOCUMENT},
                {"id": "d2", "content": "KEY delta. minor epsilon"},
            ],
            requirements=[],
        )
        await experiment.aevaluate()
        (result,) = experiment.results or []
        assert result.metrics is not None
        aggregated = result.metrics.aggregated_task_scores or {}
        # doc 1: covers 1 of 2 salient facts; doc 2: 0 of 1. Mean 0.25.
        assert aggregated["salient_coverage"] == pytest.approx(0.25)

    async def test_carries_the_content_the_resume_hash_needs(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # _run_evaluations hashes metadata["content"] to decide what a resumed
        # run may skip. Omit it and partial-result caching silently does nothing.
        experiment = _experiment(monkeypatch, {"good": GOOD})
        await experiment.aevaluate()
        (prediction,) = experiment.get_traces()
        assert prediction.metadata["content"] == DOCUMENT

    async def test_keeps_the_verdicts_behind_the_axes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        experiment = _experiment(monkeypatch, {"padded": PADDED})
        await experiment.aevaluate()
        (prediction,) = experiment.get_traces()
        assert prediction.metadata["salient_facts"] == ["KEY alpha", "KEY beta"]
        assert prediction.metadata["faithful_verdicts"] == {"g0": True, "g1": False}
        assert prediction.metadata["requirement_verdicts"] == {"r0": True, "r1": False}


class TestCost:
    """Where spend lands, which is what makes two models comparable."""

    async def test_generation_and_judging_are_charged_separately(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        experiment = _experiment(monkeypatch, {"good": GOOD})
        await experiment.aevaluate()
        (prediction,) = experiment.get_traces()
        # One generation call; five judge calls plus this model's whole share of
        # the two shared ones, since it is the only candidate.
        assert prediction.llm_cost == pytest.approx(0.001)
        assert prediction.evaluation_cost == pytest.approx(7 * 0.001)

    async def test_the_shared_judge_cost_is_split_across_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Two candidates share one document's fact extraction and salience pass,
        # so each carries half of it: 5 own judge calls + 2/2 shared.
        experiment = _experiment(monkeypatch, {"good": GOOD, "padded": PADDED})
        await experiment.aevaluate()
        for prediction in experiment.get_traces():
            assert prediction.evaluation_cost == pytest.approx(6 * 0.001)

    async def test_reports_the_split_rather_than_only_the_total(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        experiment = _experiment(monkeypatch, {"good": GOOD, "padded": PADDED})
        await experiment.aevaluate()
        usage = experiment.ranking.usage
        assert usage["generation"]["calls"] == 2
        assert usage["judge_per_candidate"]["calls"] == 10
        assert usage["judge_shared"]["calls"] == 2
        assert usage["judge_shared"]["total_tokens"] == 2 * 15
        assert usage["judge_model"] == "judge"


class TestSharedWork:
    """The economics: per-document cost must not grow with the model field."""

    async def test_decomposes_each_document_once_for_all_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        names = {f"m{index}": GOOD for index in range(4)}
        judge = _install(monkeypatch, {name: FakeSummarizer(name, GOOD) for name in names})
        experiment = SummarizationExperiment(
            config=_config(names), data=[{"id": "d1", "content": DOCUMENT}]
        )
        await experiment.aevaluate()
        assert judge.count(DocumentSaliencePrompt) == 1

    async def test_a_repeat_evaluate_does_not_re_extract_known_documents(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``add_models()`` followed by another ``evaluate()`` must not re-pay for
        salience marking on documents already in ``self._document_facts``.

        ``_before_evaluation()`` runs on every ``aevaluate()`` call with no
        skip-if-known check of its own; the guard lives one level up. Without it,
        document fact extraction would be free the second time (the judge
        memoizes ``facts()``) but salience marking -- which is not memoized --
        would be paid for twice. ``DocumentSaliencePrompt`` only ever comes from
        phase 1 (unlike ``FactExtractionPrompt``, which phase 3 also issues per
        candidate summary), so it is the unambiguous signal here.
        """
        judge = _install(
            monkeypatch,
            {
                "first": FakeSummarizer("first", GOOD),
                "second": FakeSummarizer("second", GOOD),
            },
        )
        experiment = SummarizationExperiment(
            config=_config({"first": GOOD}), data=[{"id": "d1", "content": DOCUMENT}]
        )
        await experiment.aevaluate()
        assert judge.count(DocumentSaliencePrompt) == 1

        experiment.add_models(["second"])
        await experiment.aevaluate()
        assert judge.count(DocumentSaliencePrompt) == 1


class TestFailures:
    """One bad response must not void a run."""

    async def test_a_failing_candidate_is_recorded_not_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _install(
            monkeypatch,
            {"good": FakeSummarizer("good", GOOD), "broken": FailingSummarizer("broken")},
        )
        experiment = SummarizationExperiment(
            config=_config({"good": GOOD, "broken": ""}),
            data=[{"id": "d1", "content": DOCUMENT}],
        )
        await experiment.aevaluate()

        assert experiment.ranking.best == ["good"]
        broken = next(s for s in experiment.ranking.scores if s.model == "broken")
        assert broken.documents_scored == 0
        assert broken.score == 0.0
        failed = next(p for p in experiment.get_traces() if p.model == "broken")
        assert failed.error is not None and "always fails" in failed.error
        assert failed.task_scores is None


class TestValidation:
    """What the recipe refuses, and why."""

    async def test_structured_content_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        experiment = _experiment(
            monkeypatch, {"good": GOOD}, data=[{"id": "d1", "content": {"a": "x"}}]
        )
        with pytest.raises(ValueError, match="structured content"):
            await experiment.aevaluate()

    async def test_a_prompt_without_a_content_placeholder_is_rejected(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        experiment = _experiment(monkeypatch, {"good": GOOD}, prompt="Summarize {thing}.")
        with pytest.raises(ValueError, match=r"\{content\} placeholder"):
            await experiment.aevaluate()

    async def test_an_empty_document_is_rejected(self, monkeypatch: pytest.MonkeyPatch) -> None:
        experiment = _experiment(monkeypatch, {"good": GOOD}, data=[{"id": "d1", "content": "  "}])
        with pytest.raises(ValueError, match="no content to summarize"):
            await experiment.aevaluate()


class TestPrompt:
    """What the candidates are actually asked."""

    async def test_the_checklist_is_rendered_into_the_prompt(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Substituted in _prepare_model_prompts, so the prompt that gets saved and
        # displayed is the one the candidate saw.
        experiment = _experiment(monkeypatch, {"good": GOOD})
        await experiment.aevaluate()
        prompt = (experiment._model_prompts or {})["good"]
        assert "Your summary must satisfy these requirements:\n- alpha\n- beta" in prompt
        assert "{requirements}" not in prompt
        assert "{content}" in prompt

    async def test_no_checklist_leaves_no_stray_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        experiment = _experiment(monkeypatch, {"good": GOOD}, requirements=[])
        await experiment.aevaluate()
        prompt = (experiment._model_prompts or {})["good"]
        assert "requirements" not in prompt.lower()


class TestPersistence:
    """The run directory, written by the base class without modification."""

    async def test_writes_a_run_directory(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        experiment = _experiment(monkeypatch, {"good": GOOD, "padded": PADDED})
        await experiment.aevaluate()
        run_dir = experiment.save_experiment_results(tmp_path)

        assert (run_dir / "metadata.json").exists()
        saved = json.loads((run_dir / "models" / "good.json").read_text())
        assert saved["predictions"][0]["predicted_value"] == GOOD
        assert saved["predictions"][0]["expected_value"] is None
        # The corpus-level axes land on disk via aggregated_task_scores.
        assert saved["metrics"]["aggregated_task_scores"]["salient_coverage"] == pytest.approx(1.0)

    async def test_the_axes_survive_a_save_and_reload(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # task_scores is the only signal this task has, so losing it on the way
        # to disk would leave a reloaded run unrankable. Regression: the writer
        # used to drop a field the reader already looked for.
        experiment = _experiment(monkeypatch, {"padded": PADDED})
        await experiment.aevaluate()
        experiment.save_experiment_results(tmp_path)

        reloaded = SummarizationExperiment.load_experiment_results(tmp_path)
        (prediction,) = reloaded.get_traces()
        assert prediction.task_scores == {
            "correctness": pytest.approx(0.5),
            "salient_coverage": pytest.approx(0.5),
            "salient_precision": pytest.approx(0.5),
            "requirements_met": pytest.approx(0.5),
        }

    async def test_a_resumed_prediction_keeps_its_axes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # The same asymmetry on the resume path: a staged prediction reused
        # after a crash must carry its score, or the resumed documents quietly
        # contribute nothing to the aggregate.
        from valtron_core.partial_results import PartialResultStore, compute_prediction_hash

        experiment = _experiment(monkeypatch, {"good": GOOD})
        await experiment.aevaluate()
        (prediction,) = experiment.get_traces()

        store = PartialResultStore(tmp_path)
        digest = compute_prediction_hash(DOCUMENT, "prompt", {"model": "good"})
        store.record("good", prediction, digest)
        cached = store.get_valid_cached("good", {"d1": DOCUMENT}, "prompt", {"model": "good"})

        assert cached["d1"].task_scores == prediction.task_scores


class TestReevaluate:
    """Reweighting the scheme and regrading against a new judge or checklist.

    Three tiers, cheapest first: a scheme-only reweight is pure arithmetic and
    makes no judge calls at all; a requirements-only change reruns just the
    requirements axis; a judge_model change reruns everything, since a fresh
    judge has its own opinions about salience from the ground up.
    """

    async def test_raises_if_no_results(self, monkeypatch: pytest.MonkeyPatch) -> None:
        experiment = _experiment(monkeypatch, {"good": GOOD})
        with pytest.raises(ValueError, match="No results to reevaluate"):
            await experiment.areevaluate(gate=0.6)

    async def test_reweight_changes_the_score_with_no_new_judge_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        judge = _install(monkeypatch, {"padded": FakeSummarizer("padded", PADDED)})
        experiment = SummarizationExperiment(
            config=_config({"padded": PADDED}), data=[{"id": "d1", "content": DOCUMENT}]
        )
        await experiment.aevaluate()
        assert experiment.ranking.scores[0].score == pytest.approx(0.5)
        calls_before = len(judge.prompts)

        # padded's correctness (0.5) clears the default gate (0.5) but not a
        # stricter one.
        await experiment.areevaluate(gate=0.6)

        assert len(judge.prompts) == calls_before
        assert experiment.ranking.scores[0].score == pytest.approx(0.0)
        assert experiment.ranking.parameters["gate"] == 0.6
        assert (experiment._task_statistics or {})["parameters"]["gate"] == 0.6

    async def test_requirements_only_change_reruns_just_that_axis(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        judge = _install(monkeypatch, {"padded": FakeSummarizer("padded", PADDED)})
        experiment = SummarizationExperiment(
            config=_config({"padded": PADDED}), data=[{"id": "d1", "content": DOCUMENT}]
        )
        await experiment.aevaluate()
        fact_calls = judge.count(FactExtractionPrompt)
        salience_calls = judge.count(DocumentSaliencePrompt)
        match_calls = judge.count(FactMatchPrompt)

        await experiment.areevaluate(requirements=["alpha"])

        assert judge.count(FactExtractionPrompt) == fact_calls
        assert judge.count(DocumentSaliencePrompt) == salience_calls
        assert judge.count(FactMatchPrompt) == match_calls
        assert judge.count(RequirementScoringPrompt) >= 1

        (prediction,) = experiment.get_traces()
        # "alpha" alone is satisfied by "KEY alpha. unrelated noise".
        assert prediction.task_scores["requirements_met"] == pytest.approx(1.0)
        # The other three axes are untouched from the original run.
        assert prediction.task_scores["correctness"] == pytest.approx(0.5)
        assert prediction.task_scores["salient_coverage"] == pytest.approx(0.5)
        assert prediction.task_scores["salient_precision"] == pytest.approx(0.5)

    async def test_a_new_judge_model_reruns_facts_and_all_axes(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        judge = _install(monkeypatch, {"good": FakeSummarizer("good", GOOD)})
        experiment = SummarizationExperiment(
            config=_config({"good": GOOD}), data=[{"id": "d1", "content": DOCUMENT}]
        )
        await experiment.aevaluate()
        facts_before = judge.count(FactExtractionPrompt)
        salience_before = judge.count(DocumentSaliencePrompt)
        assert salience_before == 1

        await experiment.areevaluate(judge_model="a-different-judge-model")

        # A fresh Judge has no memoized facts, so a full regrade re-extracts the
        # document facts and re-marks salience from scratch, even against the
        # same fake answers: one more document-facts call, one more summary-facts
        # call.
        assert judge.count(FactExtractionPrompt) == facts_before + 2
        assert judge.count(DocumentSaliencePrompt) == salience_before + 1

        (prediction,) = experiment.get_traces()
        assert prediction.task_scores == {
            "correctness": pytest.approx(1.0),
            "salient_coverage": pytest.approx(1.0),
            "salient_precision": pytest.approx(1.0),
            "requirements_met": pytest.approx(1.0),
        }
        # 2 shared calls (facts + salience) plus 5 of its own, same formula as a
        # fresh run with one candidate.
        assert prediction.evaluation_cost == pytest.approx(7 * 0.001)

    async def test_output_dir_persists_the_regraded_scores(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        experiment = _experiment(monkeypatch, {"padded": PADDED})
        await experiment.aevaluate()

        run_dir = await experiment.areevaluate(requirements=["alpha"], output_dir=tmp_path)
        assert run_dir is not None
        saved = json.loads((run_dir / "models" / "padded.json").read_text())
        assert saved["predictions"][0]["task_scores"]["requirements_met"] == pytest.approx(1.0)

        reloaded = SummarizationExperiment.load_experiment_results(run_dir)
        (prediction,) = reloaded.get_traces()
        assert prediction.task_scores["requirements_met"] == pytest.approx(1.0)

    async def test_reusing_a_run_dir_warns_instead_of_overwriting_metadata(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        experiment = _experiment(monkeypatch, {"padded": PADDED})
        await experiment.aevaluate()
        run_dir = experiment.save_experiment_results(tmp_path)

        with patch("valtron_core.evaluation.summarization.logger.warning") as mock_warning:
            await experiment.areevaluate(gate=0.6, output_dir=run_dir)

        mock_warning.assert_any_call(
            "reevaluate_metadata_not_overwritten",
            output_dir=str(run_dir),
            detail=ANY,
        )

    def test_the_sync_wrapper_works_outside_a_running_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A plain (non-async) test, deliberately: reevaluate() wraps areevaluate()
        # in asyncio.run(), the same pattern evaluate()/aevaluate() already uses, and
        # that only works when nothing is already running the event loop.
        experiment = _experiment(monkeypatch, {"padded": PADDED})
        experiment.evaluate()

        experiment.reevaluate(gate=0.6)

        assert experiment.ranking.scores[0].score == pytest.approx(0.0)


class TestRealDocuments:
    """Against actual Congressional bill text, not sentences invented to suit us."""

    @staticmethod
    def _data() -> list[dict[str, Any]]:
        return [
            {"id": path.stem, "content": path.read_text()} for path in sorted(DATA.glob("*.txt"))
        ]

    async def test_runs_end_to_end_over_real_prose(self, monkeypatch: pytest.MonkeyPatch) -> None:
        documents = self._data()
        assert len(documents) == 3

        # Summaries lifted verbatim from each bill, so the fakes' substring rules
        # produce faithful-but-partial summaries rather than degenerate ones.
        opening = documents[0]["content"].split(". ")[0]
        experiment = _experiment(
            monkeypatch,
            {"verbatim": opening, "unrelated": "Something the bills never say"},
            data=documents,
            judge=PositionalJudge(),
            requirements=BILLSUM_REQUIREMENTS,
        )
        await experiment.aevaluate()

        assert set(experiment.ranking.ranked_models) == {"verbatim", "unrelated"}
        assert len(experiment.get_traces()) == 6  # 2 models x 3 documents
        for score_entry in experiment.ranking.scores:
            assert score_entry.documents_scored == 3

    async def test_the_faithful_candidate_outranks_the_fabricating_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The faithfulness gate is the load-bearing part: a summary supported by
        # nothing in the document must not place above one that is.
        documents = self._data()
        opening = documents[0]["content"].split(". ")[0]
        experiment = _experiment(
            monkeypatch,
            {"verbatim": opening, "unrelated": "Something the bills never say"},
            data=documents,
            judge=PositionalJudge(),
            requirements=BILLSUM_REQUIREMENTS,
        )
        await experiment.aevaluate()

        fabricating = next(s for s in experiment.ranking.scores if s.model == "unrelated")
        assert fabricating.correctness == pytest.approx(0.0)
        assert fabricating.score == 0.0
        assert experiment.ranking.best == ["verbatim"]
