"""Evaluation engine for LLM prompt testing."""

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, Callable

import structlog
from litellm import BaseModel, completion_cost

from valtron_core.client import LLMClient
from valtron_core.evaluation.json_eval import JsonEvaluator
from valtron_core.models import (
    Document,
    EvaluationInput,
    EvaluationResult,
    FieldMetricsConfig,
    Label,
    PredictionResult,
)

import base64
import urllib.request
from pathlib import Path
import traceback
import litellm

from valtron_core.attachments import _EXT_MIME, _MAGIC, detect_mime_hint
from valtron_core.cost_utils import (
    _TIME_UNIT_RE,
    _fallback_cost,
    _get_fallback_rate_info,
    _parse_time_unit_to_seconds,
)

logger = structlog.get_logger()


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
            template: Prompt template with {content} placeholder
            document: Document to insert

        Returns:
            Formatted prompt string
        """
        # Use replace() instead of format() to avoid issues with curly braces in document content
        # This prevents JSON examples in prompts from being interpreted as format placeholders
        return template.replace("{content}", document.content)

    def _preflight_attachment_check(self, documents: list[Document], model_name: str) -> None:
        """
        Verify the model supports every attachment type across all documents before
        any evaluation runs. Uses extension/data-URI detection only — no I/O.

        Raises:
            ValueError: If any document has an attachment type the model cannot handle,
                        or if an attachment's type cannot be determined from its extension.
        """
        supported_exts = ", ".join(_EXT_MIME.keys())

        for doc in documents:
            if not doc.attachments:
                continue
            for attachment in doc.attachments:
                mime_type = detect_mime_hint(attachment)

                if not mime_type:
                    raise ValueError(
                        f"Cannot determine attachment type for document '{doc.id}' "
                        f"(attachment: '{attachment}'). "
                        f"Supported extensions: {supported_exts}."
                    )

                if mime_type.startswith("image/") and not litellm.supports_vision(model_name):
                    raise ValueError(
                        f"Model '{model_name}' does not support image inputs, "
                        f"but document '{doc.id}' has an image attachment."
                    )
                if mime_type == "application/pdf" and not litellm.supports_pdf_input(model_name):
                    raise ValueError(
                        f"Model '{model_name}' does not support PDF inputs, "
                        f"but document '{doc.id}' has a PDF attachment."
                    )

    def _load_attachment(self, s: str) -> tuple[bytes, str, bool]:
        """
        Load attachment data and detect its MIME type.

        Args:
            s: HTTP/HTTPS URL or local file path.

        Returns:
            (data, mime_type, is_url) where is_url indicates the source was a URL.
            For URL sources where MIME was determined from the extension alone,
            data is empty bytes — callers that support URL passthrough can skip fetching.
        """
        is_url = s.startswith(("http://", "https://"))

        # Data URI: data:<mime>;base64,<data>
        if s.startswith("data:"):
            header, _, b64 = s.partition(",")
            mime_type = header.split(":")[1].split(";")[0]
            return base64.b64decode(b64), mime_type, False

        # Detect MIME from extension first (strips query strings for URLs)
        mime_type = detect_mime_hint(s)

        if is_url:
            if mime_type:
                # Extension was sufficient — skip the network fetch.
                # Callers that support URL passthrough won't need the bytes.
                return b"", mime_type, True
            with urllib.request.urlopen(s) as resp:
                raw = resp.read()
                ct = resp.headers.get("Content-Type", "").split(";")[0].strip()
                mime_type = ct if ct else ""
        else:
            raw = Path(s).read_bytes()

        # Magic-byte fallback if MIME still unknown
        if not mime_type:
            for magic, mime in _MAGIC:
                if raw[: len(magic)] == magic:
                    mime_type = mime
                    break

        if not mime_type:
            mime_type = "application/octet-stream"

        return raw, mime_type, is_url

    def _build_message_content(
        self, prompt: str, document: Document, model: str
    ) -> str | list[dict]:
        """
        Build the user message content, adding attachment parts when present.

        Returns a plain string when there are no attachments, or a list of
        provider-appropriate content parts when there are.

        Each entry in document.attachments is an HTTP/HTTPS URL or a local file
        path. The file type is auto-detected from the extension or magic bytes.
        LiteLLM translates the content blocks to the correct format per provider.

        Raises:
            ValueError: If the model does not support the required input type.
        """
        if not document.attachments:
            return prompt

        import base64
        import litellm

        parts: list[dict] = [{"type": "text", "text": prompt}]

        for s in document.attachments:
            try:
                raw, mime_type, is_url = self._load_attachment(s)
            except Exception as e:
                logger.warning("attachment_load_failed", attachment=s, error=str(e))
                continue

            is_image = mime_type.startswith("image/")
            is_pdf = mime_type == "application/pdf"

            if not is_image and not is_pdf:
                logger.warning("attachment_unsupported_mime", attachment=s, mime_type=mime_type)
                continue

            is_data_uri = s.startswith("data:")

            if is_image:
                if not litellm.supports_vision(model):
                    raise ValueError(f"Model '{model}' does not support image inputs.")
                # image_url is LiteLLM's universal format for images across all providers.
                if is_url or is_data_uri:
                    # URL and data URIs can be passed directly — no encode/decode needed.
                    parts.append({"type": "image_url", "image_url": {"url": s}})
                else:
                    b64 = base64.b64encode(raw).decode()
                    parts.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64}"},
                        }
                    )

            elif is_pdf:
                if not litellm.utils.supports_pdf_input(model):
                    raise ValueError(f"Model '{model}' does not support PDF inputs.")
                if is_data_uri:
                    # Already a data URI — pass directly as file_data.
                    parts.append({"type": "file", "file": {"file_data": s}})
                elif is_url and not raw:
                    # URL passthrough — LiteLLM fetches and routes per-provider.
                    parts.append(
                        {"type": "file", "file": {"file_id": s, "format": "application/pdf"}}
                    )
                else:
                    # Local file or URL whose bytes were already fetched during MIME detection.
                    b64_data = f"data:application/pdf;base64,{base64.b64encode(raw).decode()}"
                    parts.append({"type": "file", "file": {"file_data": b64_data}})

        return parts

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
        post_extraction_filter: Callable | None = None,
        multi_pass: int = 1,
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
        content = self._build_message_content(prompt, document, model_name)
        messages = [{"role": "user", "content": content}]

        # Track time
        start_time = time.time()

        try:
            # Multi-pass: run N completions with varying temperatures, then merge
            if multi_pass > 1:
                temperatures = [0.0, 0.3]

                async def _single_pass(temp: float):
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

            # Apply post-extraction filter (e.g. hallucination filter)
            if post_extraction_filter is not None:
                predicted_value = await post_extraction_filter(predicted_value, document)

            # Compare with expected
            is_correct = self._compare_values(predicted_value, label.value, None, document.content)
            example_score = 1.0 if is_correct else 0.0

            # Compute field-level metrics if config is provided
            field_metrics = None

            if field_metrics_config:
                try:
                    evaluator = JsonEvaluator(
                        custom_metrics=field_metrics_config.custom_metrics,
                        custom_aggs=field_metrics_config.custom_aggs,
                    )
                    result = evaluator.evaluate(
                        field_metrics_config.config,
                        label.value,
                        predicted_value,
                    )

                    field_metrics = result
                    example_score = result.score
                    is_correct = result.is_correct

                except Exception as e:
                    logger.warning(
                        "field_metrics_error",
                        document_id=document.id,
                        error=str(e),
                    )

            logger.info(
                "evaluation_single",
                document_id=document.id,
                predicted=predicted_value,
                expected=label.value,
                correct=is_correct,
                time=response_time,
                cost=cost,
            )

            return PredictionResult(
                document_id=document.id,
                predicted_value=predicted_value,
                expected_value=label.value,
                is_correct=is_correct,
                example_score=example_score,
                response_time=response_time,
                original_cost=cost,
                cost=cost,
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
                cost=0.0,
                model=model_name,
                metadata={"error": str(e), "content": document.content},
            )

    async def evaluate(
        self,
        eval_input: EvaluationInput,
        max_concurrent: int = 5,
        response_format: type[BaseModel] | None = None,
        field_metrics_config: FieldMetricsConfig | None = None,
        post_extraction_filter: Callable | None = None,
        multi_pass: int = 1,
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
        self._preflight_attachment_check(eval_input.documents, model_name)

        logger.info(
            "evaluation_started",
            run_id=run_id,
            total_documents=len(eval_input.documents),
            model=model_name,
        )

        try:
            # Use semaphore to limit concurrent requests
            semaphore = asyncio.Semaphore(max_concurrent)

            async def evaluate_with_semaphore(doc: Document) -> PredictionResult | None:
                if doc.id not in label_map:
                    return None

                async with semaphore:
                    return await self.evaluate_single(
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
                    )

            # Evaluate all documents concurrently
            predictions = await asyncio.gather(
                *[evaluate_with_semaphore(doc) for doc in eval_input.documents]
            )

            # Filter out None predictions (documents without labels)
            result.predictions = [p for p in predictions if p is not None]

            # Apply cost overrides now that all predictions are collected.
            # This ensures consistent cost treatment across all predictions.
            cost_rate = eval_input.model.get("cost_rate") if isinstance(eval_input.model, dict) else None
            if cost_rate is not None:
                # Explicit cost_rate: override every prediction's cost with time-based pricing
                time_unit_str = eval_input.model.get("cost_rate_time_unit", "1hr") if isinstance(eval_input.model, dict) else "1hr"
                unit_seconds = _parse_time_unit_to_seconds(time_unit_str)
                for p in result.predictions:
                    p.cost = float(cost_rate) * (p.response_time / unit_seconds)
            elif all(p.cost == 0.0 for p in result.predictions):
                # LiteLLM had no pricing data — apply fallback rate to all predictions
                fallback_rate_info = _get_fallback_rate_info(eval_input.model)
                if fallback_rate_info:
                    for p in result.predictions:
                        p.cost = _fallback_cost(eval_input.model, p.response_time)
                    result.llm_config.update(fallback_rate_info)

            # Compute metrics
            result.compute_metrics()
            result.completed_at = datetime.now()
            result.status = "completed"

            logger.info(
                "evaluation_completed",
                run_id=run_id,
                accuracy=result.metrics.accuracy if result.metrics else 0,
                total_cost=result.metrics.total_cost if result.metrics else 0,
                total_time=result.metrics.total_time if result.metrics else 0,
            )

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

