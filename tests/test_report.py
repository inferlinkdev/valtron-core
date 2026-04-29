"""Tests for the ReportGenerator module."""

import json
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch, MagicMock

import pytest

from valtron_core.models import (
    Document,
    EvaluationResult,
    EvaluationMetrics,
    PredictionResult,
)
from valtron_core.reports import ReportGenerator
from valtron_core.reports.generate_pdf_report import _check_weasyprint_available


class TestReportGeneratorInit:
    """Tests for ReportGenerator initialization."""

    def test_init_with_default_client(self):
        """Test initialization with default client."""
        with patch("valtron_core.reports._base.LLMClient") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            generator = ReportGenerator()

            assert generator.client is not None

    def test_init_with_custom_client(self, mock_llm_client):
        """Test initialization with custom client."""
        generator = ReportGenerator(client=mock_llm_client)

        assert generator.client is mock_llm_client


class TestGenerateRecommendation:
    """Tests for the generate_recommendation method."""

    def test_generate_recommendation_success(
        self, mock_llm_client, sample_evaluation_results
    ):
        """Test successful recommendation generation."""
        # Create mock response with proper structure
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Based on the results, gpt-3.5-turbo is recommended for its excellent accuracy-to-cost ratio."

        mock_llm_client.complete_sync = Mock(return_value=mock_response)
        generator = ReportGenerator(client=mock_llm_client)

        recommendation = generator.generate_recommendation(
            results=sample_evaluation_results,
            use_case="sentiment classification",
        )

        assert "gpt-3.5-turbo" in recommendation or "recommended" in recommendation.lower()
        mock_llm_client.complete_sync.assert_called_once()

    def test_generate_recommendation_with_custom_use_case(
        self, mock_llm_client, sample_evaluation_results
    ):
        """Test recommendation with custom use case."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "Custom recommendation for legal use case"

        mock_llm_client.complete_sync = Mock(return_value=mock_response)
        generator = ReportGenerator(client=mock_llm_client)

        generator.generate_recommendation(
            results=sample_evaluation_results,
            use_case="legal document classification requiring high accuracy",
        )

        call_args = mock_llm_client.complete_sync.call_args
        # The use case should be included in the prompt
        messages = call_args.kwargs.get("messages") or call_args[1].get("messages")
        assert any("legal" in str(msg).lower() for msg in messages)

    def test_generate_recommendation_empty_results(self, mock_llm_client):
        """Test recommendation with empty results."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "No data available for recommendation"

        mock_llm_client.complete_sync = Mock(return_value=mock_response)
        generator = ReportGenerator(client=mock_llm_client)

        recommendation = generator.generate_recommendation(
            results=[],
            use_case="test",
        )

        assert recommendation is not None

    def test_generate_recommendation_llm_error(
        self, mock_llm_client, sample_evaluation_results
    ):
        """Test recommendation handles LLM errors gracefully."""
        mock_llm_client.complete_sync = Mock(side_effect=Exception("API Error"))
        generator = ReportGenerator(client=mock_llm_client)

        recommendation = generator.generate_recommendation(
            results=sample_evaluation_results,
            use_case="test",
        )

        # Should return error message, not raise exception
        assert "error" in recommendation.lower() or "could not" in recommendation.lower()


class TestEncodeImage:
    """Tests for the _encode_image method."""

    def test_encode_image(self, mock_llm_client, tmp_path):
        """Test encoding image to base64."""
        generator = ReportGenerator(client=mock_llm_client)

        # Create a simple test file
        test_file = tmp_path / "test.png"
        test_file.write_bytes(b"PNG test data")

        result = generator._encode_image(test_file)

        # Should be a base64 encoded string
        assert isinstance(result, str)
        assert len(result) > 0


class TestPrepareChartData:
    """Tests for the _prepare_chart_data method."""

    def test_prepare_chart_data_basic(self, mock_llm_client, sample_evaluation_results):
        """Test preparing chart data from results."""
        generator = ReportGenerator(client=mock_llm_client)

        data = generator._prepare_chart_data(sample_evaluation_results)

        assert "models" in data
        assert "accuracy" in data
        assert "avg_time" in data
        assert "total_time" in data
        assert "avg_cost" in data
        assert "total_cost" in data
        assert len(data["models"]) == 3  # gpt-3.5-turbo, gpt-4, gemini

    def test_prepare_chart_data_with_predictions(self, mock_llm_client):
        """Test chart data includes prediction-level histogram data."""
        results = [
            EvaluationResult(
                run_id="test",
                predictions=[
                    PredictionResult(
                        document_id="doc-1",
                        predicted_value="positive",
                        expected_value="positive",
                        is_correct=True,
                        response_time=0.5,
                        cost=0.001,
                        model="gpt-3.5-turbo",
                    ),
                    PredictionResult(
                        document_id="doc-2",
                        predicted_value="negative",
                        expected_value="negative",
                        is_correct=True,
                        response_time=0.3,
                        cost=0.0008,
                        model="gpt-3.5-turbo",
                    ),
                ],
                model="gpt-3.5-turbo",
                prompt_template="test",
                status="completed",
                metrics=EvaluationMetrics(
                    total_documents=2,
                    correct_predictions=2,
                    accuracy=1.0,
                    average_example_score=1.0,
                    total_cost=0.0018,
                    total_time=0.8,
                    average_cost_per_document=0.0009,
                    average_time_per_document=0.4,
                    model="gpt-3.5-turbo",
                ),
            )
        ]

        generator = ReportGenerator(client=mock_llm_client)
        data = generator._prepare_chart_data(results)

        assert "histogram_cost" in data
        assert "histogram_time" in data
        assert "histogram_score" in data

    def test_prepare_chart_data_empty_results(self, mock_llm_client):
        """Test chart data with empty results."""
        generator = ReportGenerator(client=mock_llm_client)

        data = generator._prepare_chart_data([])

        assert data["models"] == []
        assert data["accuracy"] == []


class TestComputePerformanceBestValues:
    """Tests for the _compute_performance_best_values method."""

    def test_compute_performance_best_values(self, mock_llm_client, sample_evaluation_results):
        """Test computing best values across results."""
        generator = ReportGenerator(client=mock_llm_client)

        best = generator._compute_performance_best_values(sample_evaluation_results)

        assert "best_accuracy" in best
        assert "best_total_cost" in best
        assert "best_avg_cost" in best
        assert "best_total_time" in best
        assert "best_avg_time" in best

        # Best accuracy should be 1.0 (100%)
        assert best["best_accuracy"] == 1.0

    def test_compute_performance_best_values_empty(self, mock_llm_client):
        """Test computing best values with empty results."""
        generator = ReportGenerator(client=mock_llm_client)

        best = generator._compute_performance_best_values([])

        # Should return defaults
        assert best["best_accuracy"] == -1.0
        assert best["best_total_cost"] == float('inf')


class TestFindBinIndex:
    """Tests for the _find_bin_index method."""

    def test_find_bin_index_in_range(self, mock_llm_client):
        """Test finding bin index for value in range."""
        generator = ReportGenerator(client=mock_llm_client)

        bins = [0.0, 0.25, 0.5, 0.75, 1.0]

        assert generator._find_bin_index(0.1, bins) == 0
        assert generator._find_bin_index(0.3, bins) == 1
        assert generator._find_bin_index(0.6, bins) == 2
        assert generator._find_bin_index(0.9, bins) == 3

    def test_find_bin_index_at_boundaries(self, mock_llm_client):
        """Test finding bin index at boundary values."""
        generator = ReportGenerator(client=mock_llm_client)

        bins = [0.0, 0.5, 1.0]

        # Lower boundary should be in first bin
        assert generator._find_bin_index(0.0, bins) == 0
        # Upper boundary should be in last bin
        assert generator._find_bin_index(1.0, bins) == 1

    def test_find_bin_index_out_of_range(self, mock_llm_client):
        """Test finding bin index for out-of-range value."""
        generator = ReportGenerator(client=mock_llm_client)

        bins = [0.0, 0.5, 1.0]

        # Values outside range should return None
        assert generator._find_bin_index(-0.1, bins) is None
        assert generator._find_bin_index(1.5, bins) is None


class TestPrepareHistogramData:
    """Tests for the _prepare_histogram_data method."""

    def test_prepare_histogram_data_basic(self, mock_llm_client):
        """Test preparing histogram data."""
        generator = ReportGenerator(client=mock_llm_client)

        all_doc_data = {
            "gpt-3.5-turbo": [
                {"id": "doc-1", "cost": 0.001, "time": 0.5, "score": 100},
                {"id": "doc-2", "cost": 0.002, "time": 0.6, "score": 100},
            ],
            "gpt-4": [
                {"id": "doc-1", "cost": 0.01, "time": 1.0, "score": 100},
                {"id": "doc-2", "cost": 0.015, "time": 1.2, "score": 100},
            ],
        }
        models = ["gpt-3.5-turbo", "gpt-4"]

        result = generator._prepare_histogram_data(all_doc_data, models)

        assert "cost" in result
        assert "time" in result
        assert "score" in result
        assert "bin_labels" in result["cost"]
        assert "bins" in result["cost"]

    def test_prepare_histogram_data_empty(self, mock_llm_client):
        """Test histogram data with empty input."""
        generator = ReportGenerator(client=mock_llm_client)

        result = generator._prepare_histogram_data({}, [])

        assert result == {"cost": {}, "time": {}, "score": {}}


class TestGenerateHtmlReport:
    """Tests for the generate_html_report method."""

    def test_generate_html_report_basic(
        self, mock_llm_client, sample_evaluation_results, tmp_path
    ):
        """Test basic HTML report generation."""
        generator = ReportGenerator(client=mock_llm_client)

        report_path, recommendation = generator.generate_html_report(
            results=sample_evaluation_results,
            output_path=tmp_path / "report.html",
            include_recommendation=False,
        )

        assert report_path.exists()
        assert report_path.suffix == ".html"

        # Verify HTML content
        content = report_path.read_text()
        assert "<html" in content.lower() or "<!doctype" in content.lower()

    def test_generate_html_report_with_recommendation(
        self, mock_llm_client, sample_evaluation_results, tmp_path
    ):
        """Test HTML report with AI recommendation."""
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message = Mock()
        mock_response.choices[0].message.content = "I recommend using gpt-3.5-turbo for cost efficiency."

        mock_llm_client.complete_sync = Mock(return_value=mock_response)
        generator = ReportGenerator(client=mock_llm_client)

        report_path, recommendation = generator.generate_html_report(
            results=sample_evaluation_results,
            output_path=tmp_path / "report.html",
            include_recommendation=True,
            use_case="sentiment analysis",
        )

        assert report_path.exists()
        assert recommendation is not None

    def test_generate_html_report_without_recommendation(
        self, mock_llm_client, sample_evaluation_results, tmp_path
    ):
        """Test HTML report without recommendation."""
        generator = ReportGenerator(client=mock_llm_client)

        report_path, recommendation = generator.generate_html_report(
            results=sample_evaluation_results,
            output_path=tmp_path / "report.html",
            include_recommendation=False,
        )

        assert report_path.exists()
        assert recommendation is None

    def test_generate_html_report_with_prompt_optimizations(
        self, mock_llm_client, sample_evaluation_results, tmp_path
    ):
        """Test HTML report with prompt optimizations."""
        generator = ReportGenerator(client=mock_llm_client)

        report_path, _ = generator.generate_html_report(
            results=sample_evaluation_results,
            output_path=tmp_path / "report.html",
            include_recommendation=False,
            prompt_optimizations={"gpt-3.5-turbo": ["few_shot", "explanation"]},
            original_prompt="Classify: {document}",
        )

        assert report_path.exists()
        content = report_path.read_text()
        # Should contain the original prompt or optimization info
        assert len(content) > 100

    def test_generate_html_report_empty_results(self, mock_llm_client, tmp_path):
        """Test HTML report with empty results."""
        generator = ReportGenerator(client=mock_llm_client)

        report_path, _ = generator.generate_html_report(
            results=[],
            output_path=tmp_path / "report.html",
            include_recommendation=False,
        )

        assert report_path.exists()


class TestGeneratePdfReport:
    """Tests for the generate_pdf_report method (using weasyprint)."""

    def test_generate_pdf_report_basic(
        self, mock_llm_client, sample_evaluation_results, tmp_path
    ):
        """Test basic PDF report generation."""
        generator = ReportGenerator(client=mock_llm_client)

        with patch("valtron_core.reports.generate_pdf_report._check_weasyprint_available"), \
             patch("weasyprint.HTML") as mock_html_class:
            mock_html = MagicMock()
            mock_html_class.return_value = mock_html
            mock_html.write_pdf = Mock()

            result = generator.generate_pdf_report(
                results=sample_evaluation_results,
                output_path=tmp_path / "report",
            )

            mock_html.write_pdf.assert_called_once()

    def test_generate_pdf_report_with_recommendation(
        self, mock_llm_client, sample_evaluation_results, tmp_path
    ):
        """Test PDF report with recommendation."""
        generator = ReportGenerator(client=mock_llm_client)

        with patch("valtron_core.reports.generate_pdf_report._check_weasyprint_available"), \
             patch("weasyprint.HTML") as mock_html_class:
            mock_html = MagicMock()
            mock_html_class.return_value = mock_html
            mock_html.write_pdf = Mock()

            generator.generate_pdf_report(
                results=sample_evaluation_results,
                output_path=tmp_path / "report",
                recommendation="Use gpt-3.5-turbo for best value",
            )

            mock_html.write_pdf.assert_called_once()

    def test_generate_pdf_report_model_name_shortening(
        self, mock_llm_client, tmp_path
    ):
        """Test that long model names with prefixes are handled."""
        # Create result with long model name
        result = EvaluationResult(
            run_id="test",
            predictions=[],
            model="gemini/gemini-2.5-flash-preview",
            prompt_template="test",
            status="completed",
            metrics=EvaluationMetrics(
                total_documents=10,
                correct_predictions=10,
                accuracy=1.0,
                average_example_score=1.0,
                total_cost=0.001,
                total_time=5.0,
                average_cost_per_document=0.0001,
                average_time_per_document=0.5,
                model="gemini/gemini-2.5-flash-preview",
            ),
        )
        generator = ReportGenerator(client=mock_llm_client)

        with patch("valtron_core.reports.generate_pdf_report._check_weasyprint_available"), \
             patch("weasyprint.HTML") as mock_html_class:
            mock_html = MagicMock()
            mock_html_class.return_value = mock_html
            mock_html.write_pdf = Mock()

            generator.generate_pdf_report(
                results=[result],
                output_path=tmp_path / "report",
            )

            # The method should complete without error
            mock_html.write_pdf.assert_called_once()

    def test_generate_pdf_report_special_characters(
        self, mock_llm_client, tmp_path
    ):
        """Test that special characters are handled in HTML."""
        # Create result with special characters
        result = EvaluationResult(
            run_id="test_&_special",
            predictions=[],
            model="test_model_with_underscore",
            prompt_template="Test with $pecial & characters",
            status="completed",
            metrics=EvaluationMetrics(
                total_documents=10,
                correct_predictions=10,
                accuracy=1.0,
                average_example_score=1.0,
                total_cost=0.001,
                total_time=5.0,
                average_cost_per_document=0.0001,
                average_time_per_document=0.5,
                model="test_model_with_underscore",
            ),
        )
        generator = ReportGenerator(client=mock_llm_client)

        with patch("valtron_core.reports.generate_pdf_report._check_weasyprint_available"), \
             patch("weasyprint.HTML") as mock_html_class:
            mock_html = MagicMock()
            mock_html_class.return_value = mock_html
            mock_html.write_pdf = Mock()

            generator.generate_pdf_report(
                results=[result],
                output_path=tmp_path / "report",
            )

            mock_html.write_pdf.assert_called_once()

    def test_generate_pdf_report_empty_results(self, mock_llm_client, tmp_path):
        """Test PDF report with empty results."""
        generator = ReportGenerator(client=mock_llm_client)

        with patch("valtron_core.reports.generate_pdf_report._check_weasyprint_available"), \
             patch("weasyprint.HTML") as mock_html_class:
            mock_html = MagicMock()
            mock_html_class.return_value = mock_html
            mock_html.write_pdf = Mock()

            generator.generate_pdf_report(
                results=[],
                output_path=tmp_path / "report",
            )

            mock_html.write_pdf.assert_called_once()

    def test_check_weasyprint_available_raises_on_missing_python_package(self):
        import sys
        with patch.dict(sys.modules, {"weasyprint": None}):
            with pytest.raises(ImportError, match="doc.courtbouillon.org/weasyprint"):
                _check_weasyprint_available()

    def test_check_weasyprint_available_raises_on_missing_system_deps(self):
        original_import = __import__

        def mock_import(name, *args, **kwargs):
            if name == "weasyprint":
                raise OSError("cannot load library 'gobject-2.0-0'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            with pytest.raises(ImportError, match="doc.courtbouillon.org/weasyprint"):
                _check_weasyprint_available()
