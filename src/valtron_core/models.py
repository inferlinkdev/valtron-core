"""Data models for evaluation."""

from collections import defaultdict
from datetime import datetime
import logging
import re
from typing import Any, Callable, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from valtron_core.scoring.json_eval import EvalResult


# Type alias for supported scoring strategies
ScoringStrategy = Literal[
    "all_or_nothing",
    "average_item_score",
    "average_field_score",
    "macro_f1",
    "custom",
]


class OverallScoringConfig(BaseModel):
    """Configuration for how the overall example score should be computed."""

    strategy: ScoringStrategy = Field(
        default="all_or_nothing",
        description="Scoring strategy: 'all_or_nothing', 'average_item_score', 'average_field_score', 'macro_f1', 'custom'",
    )
    threshold: float = Field(
        default=0.5,
        description="Score threshold (0-1) for is_correct=True. Only used when strategy != 'all_or_nothing'",
    )
    custom_scorer: Callable[[dict[str, float]], float] | None = Field(
        default=None,
        description="Custom scoring function that receives dict of field_name -> avg_score and returns overall score (0-1)",
    )

    class Config:
        arbitrary_types_allowed = True


class Document(BaseModel):
    """Represents a document to be evaluated."""

    id: str = Field(..., description="Unique identifier for the document")
    content: str | dict[str, str | None] = Field(
        ...,
        description=(
            "The document content. Either a plain string inserted at {content}, "
            "or a dict mapping placeholder names to their values. None values are treated as empty strings."
        ),
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")
    attachments: list[str] = Field(
        default_factory=list,
        description=(
            "Optional list of file attachments to send alongside the text. "
            "Each entry is either an HTTP/HTTPS URL or a local file path. "
            "The file type and MIME type are auto-detected from the extension or content."
        ),
    )


class Label(BaseModel):
    """Represents the expected label/output for a document."""

    document_id: str = Field(..., description="ID of the associated document")
    value: str = Field(..., description="Expected label value")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")


class EvaluationInput(BaseModel):
    """Input configuration for an evaluation run."""

    documents: list[Document] = Field(..., description="List of documents to evaluate")
    labels: list[Label] = Field(..., description="Expected labels for documents")
    prompt_template: str = Field(..., description="Prompt template with {content} placeholder")
    model: str | dict[str, Any] = Field(
        ...,
        description=(
            "Model to use for evaluation. Can be a string model name or a dict with LiteLLM parameters. "
            "The dict may also include evaltron_core-specific keys:"
            "'cost_rate' (float) — cost per cost_rate_time_unit, replaces token-based cost tracking when set; "
            "'cost_rate_time_unit' (str, default '1hr') — duration that cost_rate applies to, "
            "e.g. '1s', '30s', '1hr', '2h', '5min'."
        ),
    )
    temperature: float = Field(default=0.0, description="Temperature for sampling")
    max_tokens: int | None = Field(default=None, description="Max tokens to generate")


class PredictionResult(BaseModel):
    """Result of a single prediction."""

    document_id: str = Field(..., description="ID of the document")
    predicted_value: str | dict[str, Any] | list[Any] = Field(
        ..., description="Predicted value from the model"
    )
    expected_value: str | None = Field(
        default=None,
        description="Expected label value; None for a task with no ground truth for this document",
    )
    is_correct: bool | None = Field(
        default=None,
        description=(
            "Whether the prediction was correct; None when there is no correctness "
            "notion for this document (e.g. no ground truth, or an open-ended task)"
        ),
    )
    example_score: float | None = Field(
        default=None,
        description=(
            "Continuous score (0-1) indicating prediction quality; None means "
            "not scored, distinct from a score of 0"
        ),
    )
    error: str | None = Field(
        default=None,
        description="Error message if generating or scoring this prediction failed",
    )
    task_scores: dict[str, float] | None = Field(
        default=None,
        description=(
            "Open-ended per-document scalar scores for tasks whose signal doesn't fit "
            "is_correct/example_score (e.g. a summarization quality axis). Aggregated "
            "into EvaluationMetrics.aggregated_task_scores."
        ),
    )
    response_time: float = Field(..., description="Time taken for prediction in seconds")
    original_cost: float = Field(
        default=0.0, description="Cost as reported by litellm; never overwritten by cost_rate"
    )
    llm_cost: float = Field(
        default=0.0,
        description="Effective inference cost; initially equals original_cost, overwritten when cost_rate is applied",
    )
    evaluation_cost: float = Field(
        default=0.0,
        description="LLM-as-judge / embedding comparison cost incurred while scoring this prediction",
    )
    model: str = Field(..., description="Model used for prediction")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    field_metrics: EvalResult | None = Field(default=None, description="Per-field accuracy metrics")
    confidence_score: float | None = Field(
        default=None,
        description="Classifier confidence for the predicted label (transformer models only)",
    )


class EvaluationMetrics(BaseModel):
    """Aggregated metrics for an evaluation run."""

    total_documents: int = Field(..., description="Total number of documents evaluated")
    correct_predictions: int | None = Field(
        default=None,
        description="Number of correct predictions; None when no prediction has a defined is_correct",
    )
    accuracy: float | None = Field(
        default=None,
        description="Accuracy score (0-1) over predictions with a defined is_correct; None if none do",
    )
    average_example_score: float | None = Field(
        default=None,
        description=(
            "Mean per-document score (0-1) over predictions with a defined "
            "example_score; None if none do"
        ),
    )
    aggregated_task_scores: dict[str, float] | None = Field(
        default=None,
        description=(
            "Mean of each key present across predictions' task_scores; None if no "
            "prediction set any"
        ),
    )
    total_cost: float = Field(
        ..., description="Total cost of all API calls (LLM inference + evaluation)"
    )
    cost: dict[str, float] = Field(
        default_factory=dict,
        description="Cost breakdown: total_llm_cost (inference) and total_evaluation_cost (judge/embedding)",
    )
    total_time: float = Field(..., description="Total time in seconds")
    average_time_per_document: float = Field(
        ..., description="Average time per document in seconds"
    )
    average_cost_per_document: float = Field(..., description="Average cost per document")
    model: str = Field(..., description="Model used for evaluation")
    aggregated_field_metrics: dict[str, EvalResult] = Field(
        default_factory=dict, description="Aggregated per-field metrics across all predictions"
    )


class EvaluationResult(BaseModel):
    """Complete evaluation result."""

    run_id: str = Field(..., description="Unique identifier for this evaluation run")
    started_at: datetime = Field(default_factory=datetime.now, description="Start timestamp")
    completed_at: datetime | None = Field(default=None, description="Completion timestamp")
    predictions: list[PredictionResult] = Field(
        default_factory=list, description="Individual prediction results"
    )
    metrics: EvaluationMetrics | None = Field(default=None, description="Aggregated metrics")
    prompt_template: str = Field(..., description="Prompt template used")
    model: str = Field(..., description="Model used")
    llm_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Full model configuration dict as supplied by the caller (LiteLLM params, cost_rate, cost_rate_time_unit, etc.). Empty if model was specified as a plain string.",
    )
    field_config: dict[str, Any] | None = Field(
        default=None,
        description="Raw field metrics configuration as originally provided by the caller.",
    )
    status: str = Field(
        default="pending", description="Status: pending, running, completed, failed"
    )
    error: str | None = Field(default=None, description="Error message if failed")

    def add_prediction(self, prediction: PredictionResult) -> None:
        """Add a prediction result."""
        self.predictions.append(prediction)

    def compute_metrics(self) -> EvaluationMetrics:
        """Compute and update metrics from predictions."""
        if not self.predictions:
            raise ValueError("No predictions to compute metrics from")

        total_docs = len(self.predictions)
        total_llm_cost = sum(p.llm_cost for p in self.predictions)
        total_evaluation_cost = sum(p.evaluation_cost for p in self.predictions)
        total_cost = total_llm_cost + total_evaluation_cost
        total_time = sum(p.response_time for p in self.predictions)

        # Aggregate field-level metrics across all predictions
        aggregated_field_metrics: dict[str, EvalResult] = {}
        try:
            aggregated_field_metrics = self._aggregate_field_metrics()
        except Exception:
            logger.exception("Failed to aggregate field metrics")

        # is_correct / example_score / task_scores are all optional per prediction.
        # Aggregate over whatever's defined; None when nothing is defined. One rule,
        # no separate "does this run have a correctness notion" flag -- a run with
        # zero ground truth falls out of this the same way partial ground truth does.
        scored_correctness = [p.is_correct for p in self.predictions if p.is_correct is not None]
        correct_predictions = sum(scored_correctness) if scored_correctness else None
        accuracy = sum(scored_correctness) / len(scored_correctness) if scored_correctness else None

        scored_examples = [p.example_score for p in self.predictions if p.example_score is not None]
        average_example_score = (
            sum(scored_examples) / len(scored_examples) if scored_examples else None
        )

        task_score_lists: dict[str, list[float]] = defaultdict(list)
        for p in self.predictions:
            if p.task_scores:
                for key, value in p.task_scores.items():
                    task_score_lists[key].append(value)
        aggregated_task_scores = (
            {key: sum(values) / len(values) for key, values in task_score_lists.items()}
            if task_score_lists
            else None
        )

        self.metrics = EvaluationMetrics(
            total_documents=total_docs,
            correct_predictions=correct_predictions,
            accuracy=accuracy,
            average_example_score=average_example_score,
            aggregated_task_scores=aggregated_task_scores,
            total_cost=total_cost,
            cost={
                "total_llm_cost": total_llm_cost,
                "total_evaluation_cost": total_evaluation_cost,
            },
            total_time=total_time,
            average_time_per_document=total_time / total_docs if total_docs > 0 else 0.0,
            average_cost_per_document=total_cost / total_docs if total_docs > 0 else 0.0,
            model=self.model,
            aggregated_field_metrics=aggregated_field_metrics,
        )

        return self.metrics

    def _aggregate_field_metrics(self) -> dict[str, EvalResult]:
        """
        Aggregate field metrics across all predictions.

        Returns:
            Dictionary mapping field names to aggregated EvalResult
        """
        acc = self._walk_and_accumulate()

        aggregated: dict[str, EvalResult] = {}
        for path, data in acc.items():
            tp = data["tp"]
            fp = data["fp"]
            fn = data["fn"]
            tn = data["tn"]
            score_count = data["score_count"]
            is_object_field = str(data["metric"] or "").startswith("agg:")
            if is_object_field:
                # Object fields don't set tp/fp/fn; precision/recall are the
                # weighted average of subfields computed per-document by
                # _eval_object, so average those values across documents.
                precision = data["precision_sum"] / score_count if score_count > 0 else 0.0
                recall = data["recall_sum"] / score_count if score_count > 0 else 0.0
            else:
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            avg_score = data["score_sum"] / score_count if score_count > 0 else 0.0

            aggregated[path] = EvalResult(
                path=path,
                score=avg_score,
                weight=data["weight"],
                metric=data["metric"] or "aggregated",
                tp=tp,
                fp=fp,
                fn=fn,
                tn=tn,
                precision=precision,
                recall=recall,
                params=data["params"] or {},
                details={
                    "aggregation": "avg_score",
                    "num_examples": len(self.predictions),
                },
            )

        self._rederive_object_metrics(aggregated)
        return aggregated

    def _walk_and_accumulate(self) -> dict[str, dict[str, Any]]:
        """Walk all prediction field_metrics trees and accumulate stats per path."""
        acc: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "tp": 0.0,
                "fp": 0.0,
                "fn": 0.0,
                "tn": 0.0,
                "score_sum": 0.0,
                "score_count": 0,
                "precision_sum": 0.0,
                "recall_sum": 0.0,
                "weight": 0.0,
                "metric": None,
                "params": None,
            }
        )

        synthetic_metrics = {"missing_field", "unexpected_field", "optional_missing_both"}
        list_metrics = ("list_greedy_f1", "list_ordered_f1")

        def walk(res: EvalResult) -> None:
            children: list[EvalResult] = []
            if res.children:
                children.extend(res.children.values())
            if res.alignment:
                for align in res.alignment:
                    if align.result and align.result.children:
                        children.extend(align.result.children.values())

            path_base = re.sub(r"\[\d+\]", "", res.path) if "[" in res.path else res.path
            path_base = re.sub(r"^root\.", "", path_base)

            if path_base == "root" and res.metric in list_metrics:
                key = "[*]"
            elif path_base == "root":
                for child in children:
                    walk(child)
                return
            else:
                key = path_base

            a = acc[key]
            a["tp"] += res.tp
            a["fp"] += res.fp
            a["fn"] += res.fn
            a["tn"] += res.tn
            a["score_sum"] += res.score
            a["score_count"] += 1
            a["precision_sum"] += res.precision
            a["recall_sum"] += res.recall
            if not a["weight"]:
                a["weight"] = res.weight
            # Prefer real metric names over synthetic ones
            if a["metric"] is None or (
                a["metric"] in synthetic_metrics and res.metric not in synthetic_metrics
            ):
                a["metric"] = res.metric
            if a["params"] is None and res.params is not None:
                a["params"] = res.params

            for child in children:
                walk(child)

        for prediction in self.predictions or []:
            if prediction.field_metrics:
                walk(prediction.field_metrics)

        return acc

    def _rederive_object_metrics(self, aggregated: dict[str, EvalResult]) -> None:
        """Re-derive precision/recall for object fields from their direct children.

        This ensures the displayed parent figure equals the weighted average of
        the displayed child figures. Process deepest paths first so intermediate
        object fields are re-derived before their parents consume them.
        """
        for path, result in sorted(aggregated.items(), key=lambda x: -x[0].count(".")):
            if result.metric and result.metric.startswith("agg:"):
                prefix = path + "."
                direct_children = [
                    v
                    for p, v in aggregated.items()
                    if p.startswith(prefix) and "." not in p[len(prefix) :]
                ]
                if direct_children:
                    total_weight = sum(c.weight for c in direct_children) or 1.0
                    result.precision = (
                        sum(c.precision * c.weight for c in direct_children) / total_weight
                    )
                    result.recall = sum(c.recall * c.weight for c in direct_children) / total_weight


class FieldMetricsConfig(BaseModel):
    """Parsed field-level metrics configuration, including custom scoring and aggregation callables."""

    config: dict[str, Any] = Field(
        description="Per-field FieldConfig tree, as accepted by valtron_core.scoring.json_eval."
    )
    custom_metrics: dict[str, Callable[[Any, Any, dict[str, Any]], tuple[float, bool]]] = Field(
        default_factory=dict,
        description="Custom leaf metrics registered by name, callable as metric_config.metric.",
    )
    custom_aggs: dict[str, Callable[[list[EvalResult]], float]] = Field(
        default_factory=dict,
        description="Custom aggregation functions registered by name for combining child EvalResults.",
    )
