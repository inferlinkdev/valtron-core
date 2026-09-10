"""Scorer protocol: decide correctness/score for one prediction.

Extracted from evaluator._score_prediction, kept behavior-identical (see
that function's own docstring, which now delegates here). Ground truth vs.
no ground truth is a matter of which Scorer a recipe constructs, never a
branch inside shared code: ExactMatchScorer/FieldMetricsScorer require a
real expected_value; a future reference-free Scorer (see JudgeScorer)
ignores it entirely. Both kinds return the same ScoreOutcome shape, so
nothing calling a Scorer needs to know which kind it holds.

The current method signature mirrors what _score_prediction already takes.
It is expected to change shape once the Generator/RawPrediction/Ingestor
split lands and every call site is rewired onto the fuller pipeline at
once, rather than being finalized before those pieces exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import structlog

from valtron_core.models import FieldMetricsConfig
from valtron_core.scoring.json_eval import EvalResult, JsonEvaluator

logger = structlog.get_logger()


@dataclass
class ScoreOutcome:
    """Uniform scoring result, mirroring PredictionResult's own optional scoring fields.

    All fields are optional/defaulted so a Scorer only needs to set what it
    actually knows: a ground-truth Scorer sets is_correct/example_score
    (and field_metrics, if field-level); a reference-free Scorer sets
    task_scores and leaves the rest at their None/0.0 defaults.
    """

    is_correct: bool | None = None
    example_score: float | None = None
    field_metrics: EvalResult | None = None
    task_scores: "dict[str, float] | None" = None
    evaluation_cost: float = 0.0


class Scorer(Protocol):
    """Turns one (predicted_value, expected_value) pair into a ScoreOutcome."""

    def score(
        self,
        predicted_value: str,
        expected_value: str,
        *,
        extra_template_vars: "dict[str, Any] | None" = None,
        document_id: str = "",
    ) -> ScoreOutcome: ...


class ExactMatchScorer:
    """Case-insensitive exact string match; the default when no field_metrics_config is set."""

    def score(
        self,
        predicted_value: str,
        expected_value: str,
        *,
        extra_template_vars: "dict[str, Any] | None" = None,
        document_id: str = "",
    ) -> ScoreOutcome:
        is_correct = predicted_value.strip().lower() == expected_value.strip().lower()
        return ScoreOutcome(is_correct=is_correct, example_score=1.0 if is_correct else 0.0)


class FieldMetricsScorer:
    """Field-level JSON comparison via JsonEvaluator; falls back to exact-match on error.

    Holds a JsonEvaluator instance (shared across documents in a run, if one
    is passed in) so its match-key/embedding caches survive across calls.
    """

    def __init__(
        self,
        field_metrics_config: FieldMetricsConfig,
        *,
        json_evaluator: "JsonEvaluator | None" = None,
    ) -> None:
        self._field_metrics_config = field_metrics_config
        self._json_evaluator = json_evaluator

    def score(
        self,
        predicted_value: str,
        expected_value: str,
        *,
        extra_template_vars: "dict[str, Any] | None" = None,
        document_id: str = "",
    ) -> ScoreOutcome:
        fallback = ExactMatchScorer().score(predicted_value, expected_value)
        try:
            evaluator = self._json_evaluator or JsonEvaluator(
                custom_metrics=self._field_metrics_config.custom_metrics,
                custom_aggs=self._field_metrics_config.custom_aggs,
            )
            result, evaluation_cost = evaluator.evaluate(
                self._field_metrics_config.config,
                expected_value,
                predicted_value,
                extra_template_vars=extra_template_vars or {},
            )
            return ScoreOutcome(
                is_correct=result.is_correct,
                example_score=result.score,
                field_metrics=result,
                evaluation_cost=evaluation_cost,
            )
        except Exception as e:
            logger.warning("field_metrics_error", document_id=document_id, error=str(e))
            return fallback
