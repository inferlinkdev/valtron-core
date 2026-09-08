"""Characterization tests for the run-directory persistence format.

Locks in today's exact on-disk shape (written by ``save_run_dir`` /
``save_single_model_result`` in ``runner.py``) and the reconciled rule its four
readers now share for defaulting ``is_correct``/``example_score`` when those
keys are absent from a stored prediction: ``None``, meaning "not scored".

That rule used to diverge: ``ModelEval._result_from_model_data`` always
defaulted to ``None``, while ``ReferencedEval.load_experiment_results``,
``EvaluationRunner._load_results_from_run_dir``, and
``utilities.aggregate_reports.load_results_from_run_dir`` all defaulted to
``False``/``0.0`` instead, a value that misleadingly reads as "scored, and
wrong" for a prediction that was never scored at all. The persistence-
consolidation commit collapsed all four readers into
``RunDirectoryCodec.build_prediction`` while deliberately preserving that
divergence behind a ``legacy_defaults`` flag; a later, separately tested
commit removed the flag and reconciled every reader to ``None``, which is
what ``TestReadersAgreeOnMissingScoreDefaults`` below now locks in. See
``ARCHITECTURE_PROPOSAL.md`` for both commits.

Also locks in ``format_version``: every run directory written from the
reconciliation commit onward records ``metadata.json["format_version"]``, and
a run directory written before that field existed (no key at all) is read
back as version 1 and reconstructed with the same reconciled rule, not the
old per-reader behavior the file might have been written under.

Any accidental change to the writer's shape or to a reader's defaulting
behavior should show up as a failure here first, before it shows up as a
silent difference between a fresh run and a reloaded one.
"""

import json

from valtron_core.evaluation.model_eval import ModelEval
from valtron_core.evaluation.persistence import FORMAT_VERSION, RunDirectoryCodec
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


def _strip_format_version(run_dir):
    """Simulate a run directory written before ``format_version`` existed at all."""
    metadata_path = run_dir / "metadata.json"
    with open(metadata_path) as f:
        meta = json.load(f)
    del meta["format_version"]
    with open(metadata_path, "w") as f:
        json.dump(meta, f)


class TestWriterShapeIsStable:
    """Pins today's exact on-disk shape so a future rewrite can diff against it."""

    def test_metadata_json_keys(self, tmp_path):
        run_dir = _write_run_dir(tmp_path, with_scoring_keys=True)
        with open(run_dir / "metadata.json") as f:
            meta = json.load(f)

        assert set(meta.keys()) == {
            "timestamp",
            "format_version",
            "use_case",
            "original_prompt",
            "field_metrics_config",
            "response_format_schema",
            "task_config",
            "documents",
            "total_cost",
            "cost",
        }
        assert meta["format_version"] == FORMAT_VERSION

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


class TestReadersAgreeOnMissingScoreDefaults:
    """The reconciled rule: every reader defaults a missing score to None.

    Before the reconciliation commit, three of these four asserted
    ``False``/``0.0`` here instead (see this module's own docstring). Kept as
    one test per reader, same shape as ``TestReadersAgreeWhenKeysArePresent``
    above, so a future regression in any single reader still fails precisely.
    """

    def test_model_eval_defaults_to_none(self, tmp_path):
        run_dir = _write_run_dir(tmp_path, with_scoring_keys=False)
        model_file = run_dir / "models" / "gpt-4o-mini.json"
        md = ModelEval._model_data_from_file(model_file)
        result = ModelEval._result_from_model_data(md, label_map={"d1": "positive"})

        assert result.predictions[0].is_correct is None
        assert result.predictions[0].example_score is None

    def test_referenced_eval_defaults_to_none(self, tmp_path):
        run_dir = _write_run_dir(tmp_path, with_scoring_keys=False)
        loaded = ReferencedEval.load_experiment_results(run_dir)

        assert loaded.results[0].predictions[0].is_correct is None
        assert loaded.results[0].predictions[0].example_score is None

    def test_evaluation_runner_defaults_to_none(self, mock_llm_client, tmp_path):
        from valtron_core.runner import EvaluationRunner

        run_dir = _write_run_dir(tmp_path, with_scoring_keys=False)
        runner = EvaluationRunner(client=mock_llm_client)
        results, _metadata = runner._load_results_from_run_dir(run_dir)

        assert results[0].predictions[0].is_correct is None
        assert results[0].predictions[0].example_score is None

    def test_aggregate_reports_defaults_to_none(self, tmp_path):
        from valtron_core.utilities.aggregate_reports import load_results_from_run_dir

        run_dir = _write_run_dir(tmp_path, with_scoring_keys=False)
        results, _metadata = load_results_from_run_dir(run_dir)

        assert results[0].predictions[0].is_correct is None
        assert results[0].predictions[0].example_score is None


class TestFormatVersion:
    """``format_version`` is written going forward and defaults to 1 on read."""

    def test_read_format_version_missing_key_is_version_1(self, tmp_path):
        run_dir = _write_run_dir(tmp_path, with_scoring_keys=True)
        _strip_format_version(run_dir)

        meta = RunDirectoryCodec.read_json(run_dir / "metadata.json")

        assert RunDirectoryCodec.read_format_version(meta) == 1

    def test_read_format_version_present_key(self, tmp_path):
        run_dir = _write_run_dir(tmp_path, with_scoring_keys=True)
        meta = RunDirectoryCodec.read_json(run_dir / "metadata.json")

        assert RunDirectoryCodec.read_format_version(meta) == FORMAT_VERSION

    def test_legacy_run_dir_with_no_format_version_still_gets_reconciled_defaults(self, tmp_path):
        """A version-1 file (predating format_version entirely) is not a special
        case: it reloads under the same None-default rule as a brand-new run,
        not whatever behavior its original writer happened to have."""
        run_dir = _write_run_dir(tmp_path, with_scoring_keys=False)
        _strip_format_version(run_dir)

        loaded = ReferencedEval.load_experiment_results(run_dir)

        assert loaded.results[0].predictions[0].is_correct is None
        assert loaded.results[0].predictions[0].example_score is None
