"""Tests for the EvaluationRunner module."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest

from valtron_core.models import (
    Document,
    Label,
    EvaluationResult,
    EvaluationMetrics,
    PredictionResult,
)
from valtron_core.runner import EvaluationRunner


class TestEvaluationRunnerInit:
    """Tests for EvaluationRunner initialization."""

    def test_init_with_default_client(self):
        """Test initialization with default client."""
        with patch("valtron_core.runner.LLMClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            runner = EvaluationRunner()

            assert runner.client is not None
            assert runner.evaluator is not None
            assert runner.loader is not None

    def test_init_with_custom_client(self, mock_llm_client):
        """Test initialization with custom client."""
        runner = EvaluationRunner(client=mock_llm_client)

        assert runner.client is mock_llm_client


class TestSaveResultToRunDir:
    """Tests for _save_result_to_run_dir method."""

    def test_creates_metadata_and_model_file(self, mock_llm_client, tmp_path):
        """Test that metadata.json and models/*.json are written."""
        runner = EvaluationRunner(client=mock_llm_client)

        result = EvaluationResult(
            run_id="test-run",
            predictions=[
                PredictionResult(
                    document_id="doc-1",
                    predicted_value="positive",
                    expected_value="positive",
                    is_correct=True,
                    response_time=0.5,
                    llm_cost=0.001,
                    model="gpt-3.5-turbo",
                )
            ],
            model="gpt-3.5-turbo",
            prompt_template="Classify: {content}",
            status="completed",
            metrics=EvaluationMetrics(
                total_documents=1,
                correct_predictions=1,
                accuracy=1.0,
                average_example_score=1.0,
                total_cost=0.001,
                total_time=1.0,
                average_cost_per_document=0.001,
                average_time_per_document=1.0,
                model="gpt-3.5-turbo",
            ),
        )

        doc_content_map = {"doc-1": "Good product"}
        doc_label_map = {"doc-1": "positive"}

        runner._save_result_to_run_dir(result, tmp_path, doc_content_map, doc_label_map)

        assert (tmp_path / "metadata.json").exists()
        assert (tmp_path / "models" / "gpt-3.5-turbo.json").exists()

        with open(tmp_path / "metadata.json") as f:
            meta = json.load(f)
        assert len(meta["documents"]) == 1
        assert meta["documents"][0]["id"] == "doc-1"
        assert meta["documents"][0]["label"] == "positive"

        with open(tmp_path / "models" / "gpt-3.5-turbo.json") as f:
            model_data = json.load(f)
        assert model_data["model"] == "gpt-3.5-turbo"
        assert len(model_data["predictions"]) == 1
        assert model_data["predictions"][0]["expected_value"] == "positive"

    def test_does_not_overwrite_metadata(self, mock_llm_client, tmp_path):
        """Test that metadata.json is not overwritten when a second model is added."""
        runner = EvaluationRunner(client=mock_llm_client)

        existing_meta = {"timestamp": "fixed", "documents": [], "use_case": "test"}
        (tmp_path / "metadata.json").write_text(json.dumps(existing_meta))
        (tmp_path / "models").mkdir()

        result = EvaluationResult(
            run_id="run-2",
            predictions=[],
            model="gpt-4o",
            prompt_template="test",
            status="completed",
        )
        runner._save_result_to_run_dir(result, tmp_path, {}, {})

        with open(tmp_path / "metadata.json") as f:
            meta = json.load(f)
        assert meta["timestamp"] == "fixed"


class TestLoadResultsFromRunDir:
    """Tests for _load_results_from_run_dir method."""

    def test_loads_results_with_label_join(self, mock_llm_client, tmp_path):
        """Test that expected_value is joined from metadata.json."""
        runner = EvaluationRunner(client=mock_llm_client)

        meta = {
            "timestamp": "20260213_023910",
            "use_case": "test",
            "original_prompt": "Classify: {content}",
            "field_config": None,
            "documents": [
                {"id": "doc-1", "content": "Good product", "label": "positive"}
            ],
        }
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (tmp_path / "metadata.json").write_text(json.dumps(meta))

        model_file_data = {
            "run_id": "run-1",
            "model": "gpt-3.5-turbo",
            "status": "completed",
            "prompt_template": "Classify: {content}",
            "prompt_manipulations": [],
            "llm_config": {},
            "metrics": {
                "total_documents": 1,
                "correct_predictions": 1,
                "accuracy": 1.0,
                "average_example_score": 1.0,
                "total_cost": 0.001,
                "total_time": 1.0,
                "average_cost_per_document": 0.001,
                "average_time_per_document": 1.0,
                "model": "gpt-3.5-turbo",
                "aggregated_field_metrics": {},
            },
            "predictions": [
                {
                    "document_id": "doc-1",
                    "predicted_value": "positive",
                    "original_cost": 0.001,
                    "llm_cost": 0.001,
                    "evaluation_cost": 0.0,
                    "response_time": 0.5,
                    "is_correct": True,
                    "example_score": 1.0,
                }
            ],
        }
        (models_dir / "gpt-3.5-turbo.json").write_text(json.dumps(model_file_data))

        results, metadata = runner._load_results_from_run_dir(tmp_path)

        assert len(results) == 1
        assert results[0].model == "gpt-3.5-turbo"
        assert results[0].predictions[0].expected_value == "positive"
        assert metadata["use_case"] == "test"
        assert metadata["original_prompt"] == "Classify: {content}"


class TestEvaluate:
    """Tests for the evaluate method."""

    @pytest.mark.asyncio
    async def test_evaluate_success(
        self, mock_llm_client, sample_documents, sample_labels
    ):
        """Test successful single-model evaluation."""
        runner = EvaluationRunner(client=mock_llm_client)

        mock_result = EvaluationResult(
            run_id="test-run",
            predictions=[],
            model="gpt-3.5-turbo",
            prompt_template="test",
            status="completed",
            metrics=EvaluationMetrics(
                total_documents=5,
                correct_predictions=4,
                accuracy=0.8,
                average_example_score=0.8,
                total_cost=0.001,
                total_time=2.0,
                average_cost_per_document=0.0002,
                average_time_per_document=0.4,
                model="gpt-3.5-turbo",
            ),
        )

        with patch.object(runner.evaluator, "evaluate", AsyncMock(return_value=mock_result)):
            result = await runner.evaluate(
                documents=sample_documents,
                labels=sample_labels,
                prompt_template="Classify: {content}",
                model="gpt-3.5-turbo",
            )

            assert result.status == "completed"
            assert result.model == "gpt-3.5-turbo"

    @pytest.mark.asyncio
    async def test_evaluate_with_dict_model(
        self, mock_llm_client, sample_documents, sample_labels
    ):
        """Test evaluate extracts temperature/max_tokens from model dict."""
        runner = EvaluationRunner(client=mock_llm_client)

        mock_result = EvaluationResult(
            run_id="test-run",
            predictions=[],
            model="gpt-4",
            prompt_template="test",
            status="completed",
        )

        model_dict = {"model": "gpt-4", "temperature": 0.3, "max_tokens": 512}

        with patch.object(runner.evaluator, "evaluate", AsyncMock(return_value=mock_result)) as mock_eval:
            await runner.evaluate(
                documents=sample_documents,
                labels=sample_labels,
                prompt_template="Classify: {content}",
                model=model_dict,
            )

            call_kwargs = mock_eval.call_args.kwargs
            assert call_kwargs["eval_input"].temperature == 0.3
            assert call_kwargs["eval_input"].max_tokens == 512


class TestSaveResults:
    """Tests for the save_results method."""

    def test_save_results_to_file(self, mock_llm_client, tmp_path):
        """Test saving results to file."""
        runner = EvaluationRunner(client=mock_llm_client)

        result = EvaluationResult(
            run_id="test-run",
            predictions=[],
            model="gpt-3.5-turbo",
            prompt_template="test",
            status="completed",
        )

        output_file = tmp_path / "output.json"

        with patch.object(runner.loader, "save_results_to_json") as mock_save:
            runner.save_results(result, output_file)

            mock_save.assert_called_once_with(result, output_file)


class TestGenerateReport:
    """Tests for the generate_report method."""

    def test_generate_report_with_results(
        self, mock_llm_client, sample_evaluation_results, tmp_path
    ):
        """Test generating report with results."""
        runner = EvaluationRunner(client=mock_llm_client)

        with patch("valtron_core.reports.ReportGenerator") as mock_report_class:
            mock_generator = Mock()
            mock_report_class.return_value = mock_generator
            mock_generator.generate_html_report = Mock(
                return_value=(tmp_path / "report.html", "Recommendation text")
            )
            mock_generator.generate_pdf_report = Mock(
                return_value=tmp_path / "report.pdf"
            )

            # Create the HTML file since the method expects it to exist
            (tmp_path / "report.html").write_text("<html></html>")

            report_path = runner.generate_report(
                results=sample_evaluation_results,
                output_dir=tmp_path,
                include_recommendation=False,
            )

            assert report_path is not None
            mock_generator.generate_html_report.assert_called_once()

    def test_generate_report_from_run_dir(
        self, mock_llm_client, tmp_path
    ):
        """Test generating report from run directory with metadata.json + models/."""
        runner = EvaluationRunner(client=mock_llm_client)

        # Create a run directory
        meta = {
            "timestamp": "20260213_023910",
            "use_case": "test",
            "original_prompt": "Test prompt",
            "field_config": None,
            "documents": [{"id": "doc-1", "content": "Test", "label": "positive"}],
        }
        (tmp_path / "metadata.json").write_text(json.dumps(meta))
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        model_data = {
            "run_id": "test-run",
            "model": "gpt-3.5-turbo",
            "status": "completed",
            "prompt_template": "test",
            "prompt_manipulations": [],
            "llm_config": {},
            "metrics": None,
            "predictions": [],
        }
        (models_dir / "gpt-3.5-turbo.json").write_text(json.dumps(model_data))

        with patch("valtron_core.reports.ReportGenerator") as mock_report_class:
            mock_generator = Mock()
            mock_report_class.return_value = mock_generator
            mock_generator.generate_html_report = Mock(
                return_value=(tmp_path / "report.html", None)
            )
            mock_generator.generate_pdf_report = Mock(
                return_value=tmp_path / "report.pdf"
            )

            (tmp_path / "report.html").write_text("<html></html>")

            report_path = runner.generate_report(
                output_dir=tmp_path,
                include_recommendation=False,
            )

            assert report_path is not None

    def test_generate_report_no_results_error(self, mock_llm_client, tmp_path):
        """Test error when no results provided and no JSON found."""
        runner = EvaluationRunner(client=mock_llm_client)

        # Empty directory with no JSON files
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No metadata.json found|Must provide"):
            runner.generate_report(output_dir=empty_dir)

    def test_generate_report_extracts_prompt_from_results(
        self, mock_llm_client, sample_evaluation_results, tmp_path
    ):
        """Test that original_prompt is extracted from results if not provided."""
        runner = EvaluationRunner(client=mock_llm_client)

        with patch("valtron_core.reports.ReportGenerator") as mock_report_class:
            mock_generator = Mock()
            mock_report_class.return_value = mock_generator
            mock_generator.generate_html_report = Mock(
                return_value=(tmp_path / "report.html", None)
            )
            mock_generator.generate_pdf_report = Mock(
                return_value=tmp_path / "report.pdf"
            )

            (tmp_path / "report.html").write_text("<html></html>")

            runner.generate_report(
                results=sample_evaluation_results,
                output_dir=tmp_path,
                include_recommendation=False,
            )

            # Check that original_prompt was passed
            call_kwargs = mock_generator.generate_html_report.call_args.kwargs
            assert "original_prompt" in call_kwargs



class TestPrintMethods:
    """Tests for the print helper methods."""

    def test_print_summary_table_no_metrics(self, mock_llm_client):
        """Test printing summary table when result has no metrics."""
        runner = EvaluationRunner(client=mock_llm_client)

        result = EvaluationResult(
            run_id="test",
            predictions=[],
            model="gpt-3.5-turbo",
            prompt_template="test",
            status="completed",
            metrics=None,
        )

        # Should not raise
        runner._print_summary_table([result])

    def test_print_summary_table_with_metrics(self, mock_llm_client):
        """Test printing summary table with a single model result."""
        runner = EvaluationRunner(client=mock_llm_client)

        result = EvaluationResult(
            run_id="test",
            predictions=[],
            model="gpt-3.5-turbo",
            prompt_template="test",
            status="completed",
            metrics=EvaluationMetrics(
                total_documents=10,
                correct_predictions=8,
                accuracy=0.8,
                average_example_score=0.8,
                total_cost=0.01,
                total_time=5.0,
                average_cost_per_document=0.001,
                average_time_per_document=0.5,
                model="gpt-3.5-turbo",
            ),
        )

        # Should not raise
        runner._print_summary_table([result])

    def test_print_summary_table_multi_model(self, mock_llm_client, sample_evaluation_results):
        """Test printing summary table with multiple models."""
        runner = EvaluationRunner(client=mock_llm_client)

        # Should not raise
        runner._print_summary_table(sample_evaluation_results)

    def test_print_summary_table_with_field_metrics(self, mock_llm_client, sample_evaluation_results):
        """Test printing summary table with field metrics enabled."""
        runner = EvaluationRunner(client=mock_llm_client)

        # Should not raise
        runner._print_summary_table(sample_evaluation_results, show_field_metrics=True)


class TestCheckApiKeys:
    """Tests for EvaluationRunner._check_api_keys and its integration into _preflight_check."""

    @pytest.mark.unit
    def test_missing_key_raises(self, mock_llm_client):
        runner = EvaluationRunner(client=mock_llm_client)
        with patch(
            "valtron_core.runner.litellm.validate_environment",
            return_value={"keys_in_environment": False, "missing_keys": ["OPENAI_API_KEY"]},
        ):
            with pytest.raises(ValueError, match="missing API key"):
                runner._check_api_keys(["gpt-4o-mini"])

    @pytest.mark.unit
    def test_present_key_passes(self, mock_llm_client):
        runner = EvaluationRunner(client=mock_llm_client)
        with patch(
            "valtron_core.runner.litellm.validate_environment",
            return_value={"keys_in_environment": True, "missing_keys": []},
        ):
            runner._check_api_keys(["gpt-4o-mini"])  # should not raise

    @pytest.mark.unit
    def test_model_dict_with_explicit_api_key_skipped(self, mock_llm_client):
        runner = EvaluationRunner(client=mock_llm_client)
        with patch(
            "valtron_core.runner.litellm.validate_environment",
            return_value={"keys_in_environment": False, "missing_keys": ["OPENAI_API_KEY"]},
        ) as mock_env:
            runner._check_api_keys([{"model": "gpt-4o-mini", "api_key": "sk-test"}])
            mock_env.assert_not_called()

    @pytest.mark.unit
    def test_model_dict_with_api_base_skipped(self, mock_llm_client):
        runner = EvaluationRunner(client=mock_llm_client)
        with patch(
            "valtron_core.runner.litellm.validate_environment",
            return_value={"keys_in_environment": False, "missing_keys": ["OPENAI_API_KEY"]},
        ) as mock_env:
            runner._check_api_keys([{"model": "gpt-4o-mini", "api_base": "http://localhost:11434"}])
            mock_env.assert_not_called()

    @pytest.mark.unit
    def test_transformer_model_skipped(self, mock_llm_client):
        runner = EvaluationRunner(client=mock_llm_client)

        class FakeTransformer:
            type = "transformer"

        with patch(
            "valtron_core.runner.litellm.validate_environment",
            return_value={"keys_in_environment": False, "missing_keys": ["OPENAI_API_KEY"]},
        ) as mock_env:
            runner._check_api_keys([FakeTransformer()])
            mock_env.assert_not_called()

    @pytest.mark.unit
    def test_unknown_model_exception_is_ignored(self, mock_llm_client):
        runner = EvaluationRunner(client=mock_llm_client)
        with patch(
            "valtron_core.runner.litellm.validate_environment",
            side_effect=Exception("unknown model"),
        ):
            runner._check_api_keys(["totally-unknown-model"])  # should not raise

    @pytest.mark.unit
    def test_multiple_models_all_missing_reported(self, mock_llm_client):
        runner = EvaluationRunner(client=mock_llm_client)

        def fake_validate(model):
            return {
                "gpt-4o": {"keys_in_environment": False, "missing_keys": ["OPENAI_API_KEY"]},
                "claude-sonnet-4-6": {"keys_in_environment": False, "missing_keys": ["ANTHROPIC_API_KEY"]},
            }.get(model, {"keys_in_environment": True, "missing_keys": []})

        with patch("valtron_core.runner.litellm.validate_environment", side_effect=fake_validate):
            with pytest.raises(ValueError) as exc_info:
                runner._check_api_keys(["gpt-4o", "claude-sonnet-4-6"])
        msg = str(exc_info.value)
        assert "gpt-4o" in msg
        assert "claude-sonnet-4-6" in msg

    @pytest.mark.unit
    def test_preflight_check_calls_key_check_when_models_provided(self, mock_llm_client):
        runner = EvaluationRunner(client=mock_llm_client)
        with patch.object(runner, "_check_api_keys") as mock_check:
            runner._preflight_check(None, 10, 1, ["gpt-4o-mini"])
            mock_check.assert_called_once_with(["gpt-4o-mini"])

    @pytest.mark.unit
    def test_preflight_check_skips_key_check_when_no_models(self, mock_llm_client):
        runner = EvaluationRunner(client=mock_llm_client)
        with patch.object(runner, "_check_api_keys") as mock_check:
            runner._preflight_check(None, 10, 1)
            mock_check.assert_not_called()
