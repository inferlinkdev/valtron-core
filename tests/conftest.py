"""Pytest configuration and fixtures."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, AsyncIterator, Iterator
from unittest.mock import AsyncMock, Mock, MagicMock

import pytest
from litellm.utils import ModelResponse

from valtron_core.models import (
    Document,
    Label,
    EvaluationInput,
    EvaluationResult,
    EvaluationMetrics,
    PredictionResult,
)
from valtron_core.evaluation.json_eval import EvalResult

BENCHMARK_DATA_DIR = Path(__file__).parent / "benchmark_data"


def load_benchmark(filename: str) -> list[dict[str, Any]]:
    """Load benchmark data from a JSON file in tests/benchmark_data/."""
    with open(BENCHMARK_DATA_DIR / filename) as f:
        return json.load(f)


@pytest.fixture
def mock_env_vars() -> Iterator[dict[str, str]]:
    """Mock environment variables for testing."""
    original_env = os.environ.copy()

    test_env = {
        "OPENAI_API_KEY": "test-openai-key",
        "ANTHROPIC_API_KEY": "test-anthropic-key",
        "GOOGLE_API_KEY": "test-google-key",
        "LOG_LEVEL": "DEBUG",
        "MAX_RETRIES": "3",
    }

    os.environ.update(test_env)

    yield test_env

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


@pytest.fixture
def sample_messages() -> list[dict[str, str]]:
    """Sample messages for testing."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
    ]


@pytest.fixture
def mock_model_response() -> ModelResponse:
    """Create a mock ModelResponse object."""
    response = Mock(spec=ModelResponse)
    response.choices = [Mock()]
    response.choices[0].message = Mock()
    response.choices[0].message.content = "The capital of France is Paris."
    response.choices[0].delta = Mock()
    response.choices[0].delta.content = "Paris"
    response._hidden_params = {"response_cost": 0.0001}
    return response


@pytest.fixture
def mock_streaming_response() -> AsyncIterator[ModelResponse]:
    """Create a mock streaming response."""

    async def stream_generator() -> AsyncIterator[ModelResponse]:
        chunks = ["The ", "capital ", "of ", "France ", "is ", "Paris."]
        for chunk in chunks:
            response = Mock(spec=ModelResponse)
            response.choices = [Mock()]
            response.choices[0].delta = Mock()
            response.choices[0].delta.content = chunk
            yield response

    return stream_generator()


@pytest.fixture
def mock_litellm_completion(mock_model_response: ModelResponse) -> Mock:
    """Mock litellm.completion function."""
    mock = Mock(return_value=mock_model_response)
    return mock


@pytest.fixture
def mock_litellm_acompletion(mock_model_response: ModelResponse) -> AsyncMock:
    """Mock litellm.acompletion function."""
    mock = AsyncMock(return_value=mock_model_response)
    return mock


# ============================================================================
# Document and Label Fixtures
# ============================================================================


@pytest.fixture
def sample_documents() -> list[Document]:
    """Create sample Document objects for testing."""
    return [
        Document(id="doc-1", content="This product is amazing! I love it.", metadata={"source": "review"}),
        Document(id="doc-2", content="Terrible experience, would not recommend.", metadata={"source": "review"}),
        Document(id="doc-3", content="It's okay, nothing special.", metadata={"source": "review"}),
        Document(id="doc-4", content="Best purchase I've ever made!", metadata={"source": "review"}),
        Document(id="doc-5", content="Complete waste of money.", metadata={"source": "review"}),
    ]


@pytest.fixture
def sample_labels() -> list[Label]:
    """Create sample Label objects for testing."""
    return [
        Label(document_id="doc-1", value="positive"),
        Label(document_id="doc-2", value="negative"),
        Label(document_id="doc-3", value="neutral"),
        Label(document_id="doc-4", value="positive"),
        Label(document_id="doc-5", value="negative"),
    ]


@pytest.fixture
def sample_json_labels() -> list[Label]:
    """Create sample Label objects with JSON values for testing."""
    return [
        Label(document_id="doc-1", value='{"sentiment": "positive", "confidence": 0.9}'),
        Label(document_id="doc-2", value='{"sentiment": "negative", "confidence": 0.85}'),
        Label(document_id="doc-3", value='{"sentiment": "neutral", "confidence": 0.7}'),
    ]


# ============================================================================
# Evaluation Result Fixtures
# ============================================================================


@pytest.fixture
def sample_prediction_results() -> list[PredictionResult]:
    """Create sample PredictionResult objects for testing."""
    return [
        PredictionResult(
            document_id="doc-1",
            predicted_value="positive",
            expected_value="positive",
            is_correct=True,
            response_time=0.5,
            cost=0.0001,
            model="gpt-3.5-turbo",
            metadata={"content": "This product is amazing!"},
        ),
        PredictionResult(
            document_id="doc-2",
            predicted_value="negative",
            expected_value="negative",
            is_correct=True,
            response_time=0.4,
            cost=0.0001,
            model="gpt-3.5-turbo",
            metadata={"content": "Terrible experience"},
        ),
        PredictionResult(
            document_id="doc-3",
            predicted_value="positive",
            expected_value="neutral",
            is_correct=False,
            response_time=0.6,
            cost=0.0001,
            model="gpt-3.5-turbo",
            metadata={"content": "It's okay"},
        ),
    ]


@pytest.fixture
def sample_evaluation_metrics() -> EvaluationMetrics:
    """Create sample EvaluationMetrics for testing."""
    return EvaluationMetrics(
        total_documents=10,
        correct_predictions=8,
        accuracy=0.8,
        average_example_score=0.8,
        total_cost=0.001,
        total_time=5.0,
        average_cost_per_document=0.0001,
        average_time_per_document=0.5,
        model="gpt-3.5-turbo",
    )


@pytest.fixture
def sample_evaluation_result(
    sample_prediction_results: list[PredictionResult],
) -> EvaluationResult:
    """Create a sample EvaluationResult for testing."""
    result = EvaluationResult(
        run_id="test-run-123",
        started_at=datetime(2026, 1, 15, 10, 0, 0),
        completed_at=datetime(2026, 1, 15, 10, 0, 5),
        predictions=sample_prediction_results,
        model="gpt-3.5-turbo",
        prompt_template="Classify the sentiment: {document}",
        status="completed",
    )
    # Compute metrics
    result.compute_metrics()
    return result


@pytest.fixture
def sample_evaluation_results() -> list[EvaluationResult]:
    """Create multiple sample EvaluationResult objects for comparison testing."""
    results = []

    # GPT-3.5-turbo result
    gpt35_predictions = [
        PredictionResult(
            document_id=f"doc-{i}",
            predicted_value="positive" if i % 2 == 0 else "negative",
            expected_value="positive" if i % 2 == 0 else "negative",
            is_correct=True,
            response_time=0.3 + i * 0.1,
            cost=0.00002,
            model="gpt-3.5-turbo",
        )
        for i in range(10)
    ]
    gpt35_result = EvaluationResult(
        run_id="gpt35-run",
        predictions=gpt35_predictions,
        model="gpt-3.5-turbo",
        prompt_template="Classify: {document}",
        status="completed",
        metrics=EvaluationMetrics(
            total_documents=10,
            correct_predictions=10,
            accuracy=1.0,
            average_example_score=1.0,
            total_cost=0.0002,
            total_time=4.0,
            average_cost_per_document=0.00002,
            average_time_per_document=0.4,
            model="gpt-3.5-turbo",
        ),
    )
    results.append(gpt35_result)

    # GPT-4 result
    gpt4_predictions = [
        PredictionResult(
            document_id=f"doc-{i}",
            predicted_value="positive" if i % 2 == 0 else "negative",
            expected_value="positive" if i % 2 == 0 else "negative",
            is_correct=i != 5,  # One incorrect
            response_time=0.5 + i * 0.1,
            cost=0.0003,
            model="gpt-4",
        )
        for i in range(10)
    ]
    gpt4_result = EvaluationResult(
        run_id="gpt4-run",
        predictions=gpt4_predictions,
        model="gpt-4",
        prompt_template="Classify: {document}",
        status="completed",
        metrics=EvaluationMetrics(
            total_documents=10,
            correct_predictions=9,
            accuracy=0.9,
            average_example_score=0.9,
            total_cost=0.003,
            total_time=10.0,
            average_cost_per_document=0.0003,
            average_time_per_document=1.0,
            model="gpt-4",
        ),
    )
    results.append(gpt4_result)

    # Gemini result
    gemini_predictions = [
        PredictionResult(
            document_id=f"doc-{i}",
            predicted_value="positive" if i % 2 == 0 else "negative",
            expected_value="positive" if i % 2 == 0 else "negative",
            is_correct=True,
            response_time=0.8 + i * 0.1,
            cost=0.0001,
            model="gemini/gemini-2.0-flash",
        )
        for i in range(10)
    ]
    gemini_result = EvaluationResult(
        run_id="gemini-run",
        predictions=gemini_predictions,
        model="gemini/gemini-2.0-flash",
        prompt_template="Classify: {document}",
        status="completed",
        metrics=EvaluationMetrics(
            total_documents=10,
            correct_predictions=10,
            accuracy=1.0,
            average_example_score=1.0,
            total_cost=0.001,
            total_time=12.0,
            average_cost_per_document=0.0001,
            average_time_per_document=1.2,
            model="gemini/gemini-2.0-flash",
        ),
    )
    results.append(gemini_result)

    return results


@pytest.fixture
def sample_evaluation_result_with_field_metrics() -> EvaluationResult:
    """Create an EvaluationResult with field-level metrics."""
    predictions = [
        PredictionResult(
            document_id="doc-1",
            predicted_value='{"name": "John", "age": 30}',
            expected_value='{"name": "John", "age": 30}',
            is_correct=True,
            response_time=0.5,
            cost=0.0001,
            model="gpt-4",
        ),
    ]

    return EvaluationResult(
        run_id="field-metrics-run",
        predictions=predictions,
        model="gpt-4",
        prompt_template="Extract: {document}",
        status="completed",
        metrics=EvaluationMetrics(
            total_documents=1,
            correct_predictions=1,
            accuracy=1.0,
            average_example_score=1.0,
            total_cost=0.0001,
            total_time=0.5,
            average_cost_per_document=0.0001,
            average_time_per_document=0.5,
            model="gpt-4",
            aggregated_field_metrics={
                "name": EvalResult(
                    path="root.name",
                    score=1.0,
                    weight=1.0,
                    metric="exact",
                    is_correct=True,
                ),
                "age": EvalResult(
                    path="root.age",
                    score=1.0,
                    weight=1.0,
                    metric="exact",
                    is_correct=True,
                ),
            },
        ),
    )


# ============================================================================
# File-based Fixtures
# ============================================================================


@pytest.fixture
def sample_json_results_file(tmp_path: Path, sample_evaluation_results: list[EvaluationResult]) -> Path:
    """Create a sample evaluation results JSON file."""
    results_data = {
        "timestamp": "20260115_100000",
        "use_case": "sentiment classification",
        "original_prompt": "Classify the sentiment of the text: {document}",
        "results": [
            {
                "run_id": r.run_id,
                "started_at": r.started_at.isoformat() if r.started_at else None,
                "completed_at": r.completed_at.isoformat() if r.completed_at else None,
                "predictions": [p.model_dump() for p in r.predictions],
                "metrics": r.metrics.model_dump() if r.metrics else None,
                "model": r.model,
                "prompt_template": r.prompt_template,
                "status": r.status,
            }
            for r in sample_evaluation_results
        ],
    }

    json_file = tmp_path / "evaluation_results_20260115_100000.json"
    with open(json_file, "w") as f:
        json.dump(results_data, f)

    return json_file


@pytest.fixture
def sample_data_file(tmp_path: Path, sample_documents: list[Document], sample_labels: list[Label]) -> Path:
    """Create a sample data file with documents and labels."""
    data = [
        {
            "id": doc.id,
            "content": doc.content,
            "label": next((l.value for l in sample_labels if l.document_id == doc.id), None),
            "metadata": doc.metadata,
        }
        for doc in sample_documents
    ]

    data_file = tmp_path / "test_data.json"
    with open(data_file, "w") as f:
        json.dump(data, f)

    return data_file


@pytest.fixture
def temp_output_dir(tmp_path: Path) -> Path:
    """Create a temporary output directory for tests."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


# ============================================================================
# Mock Client Fixtures
# ============================================================================


@pytest.fixture
def mock_llm_client() -> Mock:
    """Create a mock LLMClient for testing."""
    client = Mock()
    client.complete = AsyncMock(return_value="The capital of France is Paris.")
    client.get_call_count = Mock(return_value=1)
    client.get_total_cost = Mock(return_value=0.0001)
    client.reset_stats = Mock()
    return client


@pytest.fixture
def mock_bert_trainer() -> Mock:
    """Create a mock BERTTrainer for testing."""
    trainer = Mock()
    trainer.predict_single = Mock(return_value="positive")
    trainer.label_map = {"positive": 0, "negative": 1, "neutral": 2}
    trainer.reverse_label_map = {0: "positive", 1: "negative", 2: "neutral"}
    return trainer


@pytest.fixture
def mock_transformer_classifier() -> Mock:
    """Create a mock TransformerClassifier for testing."""
    classifier = Mock()
    classifier.predict_single = Mock(return_value="yes")
    classifier.predict = Mock(return_value=["yes", "no", "yes"])
    classifier.load_model = Mock()
    classifier.model = Mock()
    classifier.tokenizer = Mock()
    return classifier
