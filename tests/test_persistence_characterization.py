"""Characterization tests for the run-directory persistence format.

Locks in today's exact on-disk shape (written by ``save_run_dir`` /
``save_single_model_result`` in ``runner.py``) and documents the known divergence in
how the four existing readers default ``is_correct``/``example_score`` when those
keys are absent from a stored prediction:

- ``ModelEval._result_from_model_data`` defaults both to ``None``.
- ``ReferencedEval.load_experiment_results`` defaults them to ``False``/``0.0``.
- ``EvaluationRunner._load_results_from_run_dir`` defaults them to ``False``/``0.0``.
- ``utilities.aggregate_reports.load_results_from_run_dir`` defaults them to ``False``/``0.0``.

This is the safety net for the persistence-consolidation work described in
``ARCHITECTURE_PROPOSAL.md`` (commit "unify run-directory read/write into
evaluation/persistence.py", and the later commit that intentionally reconciles this
divergence). Any accidental change to the writer's shape or to a reader's
defaulting behavior should show up as a failure here first, before it shows up as a
silent difference between a fresh run and a reloaded one.
"""

import json

from valtron_core.evaluation.model_eval import ModelEval
from valtron_core.evaluation.referenced_eval import ReferencedEval
from valtron_core.models import EvaluationMetrics, EvaluationResult, PredictionResult
from valtron_core.runner import save_run_dir


def _sample_result(*, with_scoring_keys: bool) -> EvaluationResult:
    """One model's EvaluationResult, with or without is_correct/example_score set."""
    return EvaluationResult(
        run_id="run-1",
        model="gpt-4o-mini",
        prompt_template="Classify: {content}",
        status="completed",
        predictions=[
            PredictionResult(
                document_id="d1",
                predicted_value="positive",
                expected_value="positive",
                is_correct=True if with_scoring_keys else None,
                example_score=1.0 if with_scoring_keys else None,
                response_time=1.0,
                original_cost=0.0005,
                llm_cost=0.0005,
                model="gpt-4o-mini",
            ),
        ],
        metrics=EvaluationMetrics(
            total_documents=1,
            correct_predictions=1 if with_scoring_keys else None,
            accuracy=1.0 if with_scoring_keys else None,
            average_example_score=1.0 if with_scoring_keys else None,
            total_cost=0.0005,
            total_time=1.0,
            average_cost_per_document=0.0005,
            average_time_per_document=1.0,
            model="gpt-4o-mini",
        ),
    )


def _write_run_dir(tmp_path, *, with_scoring_keys: bool):
    """Write a run directory via the real writer, optionally stripping the two keys."""
    run_dir = tmp_path / "run"
    result = _sample_result(with_scoring_keys=with_scoring_keys)
    documents = [{"id": "d1", "content": "Good product", "label": "positive"}]
    save_run_dir(
        run_dir,
        [result],
        documents,
        use_case="model evaluation",
        original_prompt="Classify: {content}",
    )

    if not with_scoring_keys:
        # Simulate data written before is_correct/example_score existed, or a
        # partial/corrupted write: the keys are absent entirely, not merely null.
        model_file = run_dir / "models" / "gpt-4o-mini.json"
        with open(model_file) as f:
            model_data = json.load(f)
        for prediction in model_data["predictions"]:
            del prediction["is_correct"]
            del prediction["example_score"]
        with open(model_file, "w") as f:
            json.dump(model_data, f)

    return run_dir


class TestWriterShapeIsStable:
    """Pins today's exact on-disk shape so a future rewrite can diff against it."""

    def test_metadata_json_keys(self, tmp_path):
        run_dir = _write_run_dir(tmp_path, with_scoring_keys=True)
        with open(run_dir / "metadata.json") as f:
            meta = json.load(f)

        assert set(meta.keys()) == {
            "timestamp",
            "use_case",
            "original_prompt",
            "field_metrics_config",
            "response_format_schema",
            "task_config",
            "documents",
            "total_cost",
            "cost",
        }
        assert "format_version" not in meta

    def test_model_file_keys(self, tmp_path):
        run_dir = _write_run_dir(tmp_path, with_scoring_keys=True)
        with open(run_dir / "models" / "gpt-4o-mini.json") as f:
            model_data = json.load(f)

        assert set(model_data.keys()) == {
            "run_id",
            "model",
            "started_at",
            "completed_at",
            "status",
            "prompt_template",
            "prompt_manipulations",
            "override_prompt",
            "llm_config",
            "metrics",
            "predictions",
        }
        prediction = model_data["predictions"][0]
        assert set(prediction.keys()) == {
            "document_id",
            "predicted_value",
            "expected_value",
            "original_cost",
            "llm_cost",
            "evaluation_cost",
            "response_time",
            "is_correct",
            "example_score",
            "task_scores",
            "error",
        }


class TestReadersAgreeWhenKeysArePresent:
    """When a prediction has is_correct/example_score, all four readers agree."""

    def test_model_eval_result_from_model_data(self, tmp_path):
        run_dir = _write_run_dir(tmp_path, with_scoring_keys=True)
        model_file = run_dir / "models" / "gpt-4o-mini.json"
        md = ModelEval._model_data_from_file(model_file)
        result = ModelEval._result_from_model_data(md, label_map={"d1": "positive"})

        assert result.predictions[0].is_correct is True
        assert result.predictions[0].example_score == 1.0

    def test_referenced_eval_load_experiment_results(self, tmp_path):
        run_dir = _write_run_dir(tmp_path, with_scoring_keys=True)
        loaded = ReferencedEval.load_experiment_results(run_dir)

        assert loaded.results[0].predictions[0].is_correct is True
        assert loaded.results[0].predictions[0].example_score == 1.0

    def test_evaluation_runner_load_results_from_run_dir(self, mock_llm_client, tmp_path):
        from valtron_core.runner import EvaluationRunner

        run_dir = _write_run_dir(tmp_path, with_scoring_keys=True)
        runner = EvaluationRunner(client=mock_llm_client)
        results, _metadata = runner._load_results_from_run_dir(run_dir)

        assert results[0].predictions[0].is_correct is True
        assert results[0].predictions[0].example_score == 1.0

    def test_aggregate_reports_load_results_from_run_dir(self, tmp_path):
        from valtron_core.utilities.aggregate_reports import load_results_from_run_dir

        run_dir = _write_run_dir(tmp_path, with_scoring_keys=True)
        results, _metadata = load_results_from_run_dir(run_dir)

        assert results[0].predictions[0].is_correct is True
        assert results[0].predictions[0].example_score == 1.0


class TestReadersDivergeWhenKeysAreMissing:
    """Known, pre-existing divergence: the four loaders disagree on defaults.

    This is not desired behavior, it is today's actual behavior, kept passing on
    purpose so the divergence is visible and testable rather than silent. The
    architecture proposal's persistence-consolidation commit reconciles this to a
    single rule (None, meaning "not scored") in its own dedicated, explicitly
    tested commit, not silently alongside the deduplication itself.
    """

    def test_model_eval_defaults_to_none(self, tmp_path):
        run_dir = _write_run_dir(tmp_path, with_scoring_keys=False)
        model_file = run_dir / "models" / "gpt-4o-mini.json"
        md = ModelEval._model_data_from_file(model_file)
        result = ModelEval._result_from_model_data(md, label_map={"d1": "positive"})

        assert result.predictions[0].is_correct is None
        assert result.predictions[0].example_score is None

    def test_referenced_eval_defaults_to_false_and_zero(self, tmp_path):
        run_dir = _write_run_dir(tmp_path, with_scoring_keys=False)
        loaded = ReferencedEval.load_experiment_results(run_dir)

        assert loaded.results[0].predictions[0].is_correct is False
        assert loaded.results[0].predictions[0].example_score == 0.0

    def test_evaluation_runner_defaults_to_false_and_zero(self, mock_llm_client, tmp_path):
        from valtron_core.runner import EvaluationRunner

        run_dir = _write_run_dir(tmp_path, with_scoring_keys=False)
        runner = EvaluationRunner(client=mock_llm_client)
        results, _metadata = runner._load_results_from_run_dir(run_dir)

        assert results[0].predictions[0].is_correct is False
        assert results[0].predictions[0].example_score == 0.0

    def test_aggregate_reports_defaults_to_false_and_zero(self, tmp_path):
        from valtron_core.utilities.aggregate_reports import load_results_from_run_dir

        run_dir = _write_run_dir(tmp_path, with_scoring_keys=False)
        results, _metadata = load_results_from_run_dir(run_dir)

        assert results[0].predictions[0].is_correct is False
        assert results[0].predictions[0].example_score == 0.0
