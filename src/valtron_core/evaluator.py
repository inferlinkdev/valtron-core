"""Evaluation engine for LLM prompt testing."""

import asyncio
import json
import re
import time
import traceback
import uuid
from datetime import datetime
from typing import Any, AsyncIterator, Callable

import structlog
from litellm import BaseModel, completion_cost
from litellm.utils import ModelResponse  # type: ignore[attr-defined]

from valtron_core.attachments import build_message_content, check_attachment_support
from valtron_core.client import LLMClient
from valtron_core.scoring.json_eval import JsonEvaluator
from valtron_core.models import (
    Document,
    EvaluationInput,
    EvaluationResult,
    FieldMetricsConfig,
    Label,
    PredictionResult,
)
from valtron_core.cost_utils import (
    _TIME_UNIT_RE,
    _fallback_cost,
    _get_fallback_rate_info,
    _parse_time_unit_to_seconds,
)

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

    Uses JsonEvaluator when field_metrics_config is provided; falls back to
    case-insensitive string comparison otherwise. evaluation_cost is non-zero
    only when JsonEvaluator makes LLM-as-judge calls.

    Pass a pre-built ``json_evaluator`` to share its cache across documents in a run.
    If omitted, a fresh JsonEvaluator is constructed from ``field_metrics_config``.
    """
    is_correct = predicted_value.strip().lower() == expected_value.strip().lower()
    example_score = 1.0 if is_correct else 0.0
    field_metrics = None
    evaluation_cost = 0.0

    if field_metrics_config:
        try:
            evaluator = json_evaluator or JsonEvaluator(
                custom_metrics=field_metrics_config.custom_metrics,
                custom_aggs=field_metrics_config.custom_aggs,
            )
            result, evaluation_cost = evaluator.evaluate(
                field_metrics_config.config,
                expected_value,
                predicted_value,
                extra_template_vars=extra_template_vars or {},
            )
            field_metrics = result
            example_score = result.score
            is_correct = result.is_correct
        except Exception as e:
            logger.warning(
                "field_metrics_error",
                document_id=document_id,
                error=str(e),
            )

    return field_metrics, example_score, is_correct, evaluation_cost


class PromptEvaluator:
    """Evaluates prompts against labeled documents."""

    def __init__(self, client: LLMClient | None = None) -> None:
        """
        Initialize the evaluator.

        Args:
            client: Optional LLMClient instance. Creates new one if not provided.
        """
        self.client = client or LLMClient()

    def _format_prompt(self, template: str, document: Document) -> str:
        """
        Format a prompt template with document content.

        Args:
            template: Prompt template with {content} placeholder (string content)
                      or arbitrary {key} placeholders (dict content)
            document: Document to insert

        Returns:
            Formatted prompt string
        """
        if isinstance(document.content, str):
            # Use replace() instead of format() to avoid issues with curly braces in document content
            # This prevents JSON examples in prompts from being interpreted as format placeholders
            return template.replace("{content}", document.content)

        result = template
        for key in set(re.findall(r"\{(\w+)\}", template)):
            if key in document.content:
                result = result.replace(f"{{{key}}}", document.content[key] or "")
            else:
                logger.warning(
                    "prompt_variable_missing",
                    document_id=document.id,
                    key=key,
                )
                result = result.replace(f"{{{key}}}", "")
        return result

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
        """
        # Extract model name for logging
        model_name = model if isinstance(model, str) else model.get("model", "unknown")

        # Format prompt and build message content (may include attachment parts)
        prompt = self._format_prompt(prompt_template, document)
        content = build_message_content(prompt, document.attachments, model_name)
        messages = [{"role": "user", "content": content}]

        # Track time
        start_time = time.time()

        try:
            # Multi-pass: run N completions with varying temperatures, then merge
            if multi_pass > 1:
                temperatures = [0.0, 0.3]

                async def _single_pass(temp: float) -> ModelResponse | AsyncIterator[ModelResponse]:
                    return await self.client.complete(
                        model=model,
                        messages=messages,
                        temperature=temp,
                        max_tokens=max_tokens,
                        response_format=response_format,
                    )

                responses = await asyncio.gather(*[_single_pass(t) for t in temperatures])

                raw_values = [r.choices[0].message.content.strip() for r in responses]

                from valtron_core.decompose import _multi_pass_merge

                predicted_value = _multi_pass_merge(raw_values)

                end_time = time.time()
                response_time = end_time - start_time

                cost = 0.0
                for resp in responses:
                    try:
                        cost += completion_cost(completion_response=resp)
                    except Exception:
                        pass
            else:
                # Get prediction
                response = await self.client.complete(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )

                end_time = time.time()
                response_time = end_time - start_time

                # Extract predicted value
                predicted_value = response.choices[0].message.content.strip()

                cost = 0.0
                try:
                    cost = completion_cost(completion_response=response)
                except Exception:
                    pass

            # Resolve effective cost: user cost_rate > litellm pricing > fallback estimate
            original_cost = cost
            if isinstance(model, dict) and model.get("cost_rate") is not None:
                unit_seconds = _parse_time_unit_to_seconds(model.get("cost_rate_time_unit", "1hr"))
                cost = float(model["cost_rate"]) * (response_time / unit_seconds)
            elif cost == 0.0:
                cost = _fallback_cost(model, response_time)

            # Apply post-extraction filter (e.g. hallucination filter)
            if post_extraction_filter is not None:
                predicted_value = await post_extraction_filter(predicted_value, document)

            # Build template vars for field metrics (prompt_used + doc content fields)
            if isinstance(document.content, dict):
                doc_vars: dict[str, Any] = {f"example_{k}": v for k, v in document.content.items()}
            else:
                doc_vars = {"example_content": document.content}
            extra_template_vars = {"prompt_used": prompt, **doc_vars}

            # Score prediction (string comparison + optional JsonEvaluator)
            field_metrics, example_score, is_correct, evaluation_cost = _score_prediction(
                predicted_value=predicted_value,
                expected_value=label.value,
                field_metrics_config=field_metrics_config,
                extra_template_vars=extra_template_vars,
                document_id=document.id,
                json_evaluator=json_evaluator,
            )

            return PredictionResult(
                document_id=document.id,
                predicted_value=predicted_value,
                expected_value=label.value,
                is_correct=is_correct,
                example_score=example_score,
                response_time=response_time,
                original_cost=original_cost,
                llm_cost=cost,
                evaluation_cost=evaluation_cost,
                model=model_name,
                field_metrics=field_metrics,
                metadata={"content": document.content, "attachments": document.attachments},
            )

        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time

            logger.error(
                "evaluation_error",
                document_id=document.id,
                error=str(e),
                time=response_time,
            )

            # Return a failed prediction
            return PredictionResult(
                document_id=document.id,
                predicted_value=f"ERROR: {str(e)}",
                expected_value=label.value,
                is_correct=False,
                response_time=response_time,
                original_cost=0.0,
                llm_cost=0.0,
                model=model_name,
                metadata={"error": str(e), "content": document.content},
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
