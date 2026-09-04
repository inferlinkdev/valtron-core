"""Evaluation engine for LLM prompt testing."""

import asyncio
import traceback
import uuid
from datetime import datetime
from typing import Any, Callable, cast

import structlog
from litellm import BaseModel

from valtron_core.attachments import check_attachment_support
from valtron_core.client import LLMClient
from valtron_core.evaluation.stages import (
    ExactMatchScorer,
    FieldMetricsScorer,
    LLMGenerator,
    Scorer,
    format_prompt,
)
from valtron_core.scoring.json_eval import JsonEvaluator
from valtron_core.models import (
    Document,
    EvaluationInput,
    EvaluationResult,
    FieldMetricsConfig,
    Label,
    PredictionResult,
)
from valtron_core.cost_utils import _get_fallback_rate_info

logger = structlog.get_logger()


def _score_prediction(
    predicted_value: str,
    expected_value: str,
    field_metrics_config: FieldMetricsConfig | None,
    extra_template_vars: dict[str, Any] | None = None,
    document_id: str = "",
    json_evaluator: JsonEvaluator | None = None,
) -> tuple[Any, float, bool, float]:
    """Compute (field_metrics, example_score, is_correct, evaluation_cost).

    Thin wrapper kept for existing callers/importers: builds the Scorer this
    call needs (FieldMetricsScorer when field_metrics_config is given, else
    ExactMatchScorer) and delegates to it. See
    valtron_core.evaluation.stages.score for the Scorer protocol and its
    implementations, which is where this logic now actually lives.

    Pass a pre-built ``json_evaluator`` to share its cache across documents in a run.
    If omitted, a fresh JsonEvaluator is constructed from ``field_metrics_config``.
    """
    scorer: Scorer = (
        FieldMetricsScorer(field_metrics_config, json_evaluator=json_evaluator)
        if field_metrics_config
        else ExactMatchScorer()
    )
    outcome = scorer.score(
        predicted_value,
        expected_value,
        extra_template_vars=extra_template_vars,
        document_id=document_id,
    )
    # ExactMatchScorer/FieldMetricsScorer (unlike a reference-free Scorer) always
    # set example_score/is_correct; ScoreOutcome types them Optional for the
    # Protocol as a whole, so the cast documents that narrower guarantee here.
    return (
        outcome.field_metrics,
        cast(float, outcome.example_score),
        cast(bool, outcome.is_correct),
        outcome.evaluation_cost,
    )


class PromptEvaluator:
    """Evaluates prompts against labeled documents."""

    def __init__(self, client: LLMClient | None = None) -> None:
        """
        Initialize the evaluator.

        Args:
            client: Optional LLMClient instance. Creates new one if not provided.
        """
        self.client = client or LLMClient()
        self._generator = LLMGenerator(client=self.client)

    def _format_prompt(self, template: str, document: Document) -> str:
        """
        Format a prompt template with document content.

        Args:
            template: Prompt template with {content} placeholder (string content)
                      or arbitrary {key} placeholders (dict content)
            document: Document to insert

        Returns:
            Formatted prompt string

        Thin wrapper over evaluation.stages.generate.format_prompt, kept on this
        class (rather than called directly) so its exact tested name/behavior,
        including logging via this module's own ``logger``, is unchanged.
        """
        return format_prompt(
            template,
            document,
            on_missing_key=lambda key: logger.warning(
                "prompt_variable_missing", document_id=document.id, key=key
            ),
        )

    def _normalize_value(self, value: str) -> str:
        """
        Normalize a value for comparison.

        Args:
            value: Value to normalize

        Returns:
            Normalized value (lowercase, stripped)
        """
        return value.strip().lower()

    def _compare_values(
        self,
        predicted: str,
        expected: str,
        comparison_fn: Callable[..., bool] | None = None,
        context: str | None = None,
    ) -> bool:
        """
        Compare predicted and expected values.

        Args:
            predicted: Predicted value
            expected: Expected value
            comparison_fn: Optional custom comparison function
            context: Optional source document text for comparison context

        Returns:
            True if values match
        """
        if comparison_fn:
            return comparison_fn(predicted, expected, context)

        # Default: case-insensitive string comparison
        return self._normalize_value(predicted) == self._normalize_value(expected)

    async def evaluate_single(
        self,
        document: Document,
        label: Label,
        prompt_template: str,
        model: str | dict[str, Any],
        temperature: float = 0.0,
        max_tokens: int | None = None,
        response_format: type[BaseModel] | None = None,
        field_metrics_config: FieldMetricsConfig | None = None,
        post_extraction_filter: Callable[[Any, Document], Any] | None = None,
        multi_pass: int = 1,
        json_evaluator: JsonEvaluator | None = None,
    ) -> PredictionResult:
        """
        Evaluate a single document.

        Args:
            document: Document to evaluate
            label: Expected label
            prompt_template: Prompt template
            model: Model to use (string name or dict with model parameters)
            temperature: Sampling temperature
            max_tokens: Max tokens to generate
            comparison_fn: Optional custom comparison function
            response_format: Optional pydantic model for response parsing
            field_metrics_config: Configuration for field-level metrics. If provided, field-level
                metrics will be computed automatically.

        Returns:
            PredictionResult

        Thin wrapper: generation (this method's former body) now lives in
        LLMGenerator.generate(); this composes it with a Scorer, exactly the
        Generator-then-Scorer shape the rest of the shared engine is moving to.
        """
        model_name = model if isinstance(model, str) else model.get("model", "unknown")

        raw = await self._generator.generate(
            document,
            prompt_template,
            model,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
            post_extraction_filter=post_extraction_filter,
            multi_pass=multi_pass,
        )

        if raw.error is not None:
            # Generation itself failed: an automatic wrong answer, same as before,
            # without asking a Scorer to judge a non-existent prediction.
            return PredictionResult(
                document_id=document.id,
                predicted_value=raw.predicted_value,
                expected_value=label.value,
                is_correct=False,
                response_time=raw.response_time,
                original_cost=raw.original_cost,
                llm_cost=raw.llm_cost,
                model=model_name,
                metadata=raw.metadata,
            )

        # Build template vars for field metrics (prompt_used + doc content fields)
        if isinstance(document.content, dict):
            doc_vars: dict[str, Any] = {f"example_{k}": v for k, v in document.content.items()}
        else:
            doc_vars = {"example_content": document.content}
        extra_template_vars = {"prompt_used": raw.prompt, **doc_vars}

        # Score prediction (string comparison + optional JsonEvaluator)
        field_metrics, example_score, is_correct, evaluation_cost = _score_prediction(
            predicted_value=raw.predicted_value,
            expected_value=label.value,
            field_metrics_config=field_metrics_config,
            extra_template_vars=extra_template_vars,
            document_id=document.id,
            json_evaluator=json_evaluator,
        )

        return PredictionResult(
            document_id=document.id,
            predicted_value=raw.predicted_value,
            expected_value=label.value,
            is_correct=is_correct,
            example_score=example_score,
            response_time=raw.response_time,
            original_cost=raw.original_cost,
            llm_cost=raw.llm_cost,
            evaluation_cost=evaluation_cost,
            model=model_name,
            field_metrics=field_metrics,
            metadata=raw.metadata,
        )

    async def evaluate(
        self,
        eval_input: EvaluationInput,
        max_concurrent: int = 5,
        response_format: type[BaseModel] | None = None,
        field_metrics_config: FieldMetricsConfig | None = None,
        post_extraction_filter: Callable[[Any, Document], Any] | None = None,
        multi_pass: int = 1,
        on_document_complete: Callable[["PredictionResult"], None] | None = None,
    ) -> EvaluationResult:
        """
        Evaluate all documents against their labels.

        Args:
            eval_input: Evaluation input configuration
            max_concurrent: Maximum concurrent API calls
            response_format: Optional pydantic model for response parsing
            field_metrics_config: Configuration for field-level metrics. If provided, field-level
                metrics will be computed automatically.

        Returns:
            EvaluationResult with all predictions and metrics
        """
        run_id = str(uuid.uuid4())

        # Extract model name for result storage
        model_name = (
            eval_input.model
            if isinstance(eval_input.model, str)
            else eval_input.model.get("model", "unknown")
        )

        result = EvaluationResult(
            run_id=run_id,
            started_at=datetime.now(),
            prompt_template=eval_input.prompt_template,
            model=model_name,
            llm_config=eval_input.model if isinstance(eval_input.model, dict) else {},
            status="running",
        )

        # Create label lookup
        label_map = {label.document_id: label for label in eval_input.labels}

        # Validate all documents have labels
        for doc in eval_input.documents:
            if doc.id not in label_map:
                logger.warning("missing_label", document_id=doc.id)

        # Preflight: verify model supports all attachment types before running anything
        check_attachment_support(eval_input.documents, model_name)

        try:
            # Use semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)
            _fallback_warning_logged = False
            _has_user_cost_rate = (
                isinstance(eval_input.model, dict) and eval_input.model.get("cost_rate") is not None
            )

            json_evaluator = (
                JsonEvaluator(
                    custom_metrics=field_metrics_config.custom_metrics,
                    custom_aggs=field_metrics_config.custom_aggs,
                )
                if field_metrics_config is not None
                else None
            )

            async def evaluate_with_semaphore(doc: Document) -> PredictionResult | None:
                nonlocal _fallback_warning_logged
                if doc.id not in label_map:
                    return None

                async with semaphore:
                    pred = await self.evaluate_single(
                        document=doc,
                        label=label_map[doc.id],
                        prompt_template=eval_input.prompt_template,
                        model=eval_input.model,
                        temperature=eval_input.temperature,
                        max_tokens=eval_input.max_tokens,
                        response_format=response_format,
                        field_metrics_config=field_metrics_config,
                        post_extraction_filter=post_extraction_filter,
                        multi_pass=multi_pass,
                        json_evaluator=json_evaluator,
                    )
                    if pred is not None:
                        if (
                            not _fallback_warning_logged
                            and not _has_user_cost_rate
                            and pred.original_cost == 0.0
                            and pred.llm_cost > 0.0
                        ):
                            logger.warning(
                                "using_estimated_cost",
                                model=model_name,
                                note="no litellm pricing found; costs are approximate",
                            )
                            _fallback_warning_logged = True
                        if on_document_complete is not None:
                            on_document_complete(pred)
                    return pred

            # Evaluate all documents concurrently
            predictions = await asyncio.gather(
                *[evaluate_with_semaphore(doc) for doc in eval_input.documents]
            )

            # Filter out None predictions (documents without labels)
            result.predictions = [p for p in predictions if p is not None]

            # Propagate fallback rate info to result metadata if it was used
            fallback_rate_info = _get_fallback_rate_info(eval_input.model)
            if fallback_rate_info and all(p.original_cost == 0.0 for p in result.predictions):
                result.llm_config.update(fallback_rate_info)

            # Compute metrics
            result.compute_metrics()
            result.completed_at = datetime.now()
            result.status = "completed"

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.completed_at = datetime.now()

            tb_str = "".join(traceback.format_tb(e.__traceback__))

            logger.error(
                "evaluation_failed",
                run_id=run_id,
                error=str(e),
                error_type=type(e).__name__,
                error_repr=repr(e),
                traceback=tb_str[:500],  # Limit traceback length
            )

        return result
