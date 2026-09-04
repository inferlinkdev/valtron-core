"""Generator protocol: turn one document into raw model output, no scoring attached.

Extracted from evaluator.PromptEvaluator.evaluate_single, kept behavior-identical
(see that method's own docstring, which now delegates here). Generator.generate()
always returns a RawPrediction; scoring is a separate step (see
evaluation.stages.score), so a Generator never needs to know about labels, ground
truth, or how its output gets judged: the same seam that lets a
reference-free recipe plug in without a shared engine ever branching on
"does this have ground truth."
"""

from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Protocol

import structlog
from litellm import BaseModel, completion_cost
from litellm.utils import ModelResponse  # type: ignore[attr-defined]

from valtron_core.attachments import build_message_content
from valtron_core.client import LLMClient
from valtron_core.cost_utils import _fallback_cost, _parse_time_unit_to_seconds
from valtron_core.models import Document

logger = structlog.get_logger()


def format_prompt(
    template: str,
    document: Document,
    *,
    on_missing_key: "Callable[[str], None] | None" = None,
) -> str:
    """Fill a prompt template from a document's content (string or dict).

    Shared by PromptEvaluator._format_prompt (which passes its own
    logger-backed callback, to keep that method's exact tested behavior) and
    LLMGenerator.generate() below. String content fills the single
    ``{content}`` placeholder; dict content fills one placeholder per key,
    calling ``on_missing_key`` (if given) for any placeholder the dict lacks
    and substituting an empty string for it either way.
    """
    if isinstance(document.content, str):
        # Use replace() instead of format() to avoid issues with curly braces in
        # document content (e.g. JSON examples in prompts).
        return template.replace("{content}", document.content)

    result = template
    for key in set(re.findall(r"\{(\w+)\}", template)):
        if key in document.content:
            result = result.replace(f"{{{key}}}", document.content[key] or "")
        else:
            if on_missing_key is not None:
                on_missing_key(key)
            result = result.replace(f"{{{key}}}", "")
    return result


@dataclass
class RawPrediction:
    """One document's raw model output, before any scoring is attached.

    ``prompt`` is the fully-formatted prompt actually sent, kept so a caller
    can build a Scorer's ``extra_template_vars`` (e.g. field-metrics templating)
    without reformatting it. ``error`` is set (and ``predicted_value`` holds an
    ``"ERROR: ..."`` string) when generation itself failed; a caller composing
    a Scorer on top should treat that as an automatic wrong answer rather than
    calling the Scorer at all, matching today's behavior.
    """

    document_id: str
    predicted_value: Any
    prompt: str = ""
    original_cost: float = 0.0
    llm_cost: float = 0.0
    response_time: float = 0.0
    error: "str | None" = None
    confidence_score: "float | None" = None
    metadata: "dict[str, Any]" = field(default_factory=dict)


class Generator(Protocol):
    """Formats a prompt, calls a model, and resolves cost; no scoring."""

    async def generate(
        self,
        document: Document,
        prompt_template: str,
        model: "str | dict[str, Any]",
        *,
        temperature: float = 0.0,
        max_tokens: "int | None" = None,
        response_format: "type[BaseModel] | None" = None,
        post_extraction_filter: "Callable[[Any, Document], Any] | None" = None,
        multi_pass: int = 1,
    ) -> RawPrediction: ...


class LLMGenerator:
    """Formats a prompt, calls one LLM via LLMClient, and resolves cost. No scoring."""

    def __init__(self, client: "LLMClient | None" = None) -> None:
        self.client = client or LLMClient()

    async def generate(  # noqa: C901, PLR0912
        self,
        document: Document,
        prompt_template: str,
        model: "str | dict[str, Any]",
        *,
        temperature: float = 0.0,
        max_tokens: "int | None" = None,
        response_format: "type[BaseModel] | None" = None,
        post_extraction_filter: "Callable[[Any, Document], Any] | None" = None,
        multi_pass: int = 1,
    ) -> RawPrediction:
        model_name = model if isinstance(model, str) else model.get("model", "unknown")

        prompt = format_prompt(
            prompt_template,
            document,
            on_missing_key=lambda key: logger.warning(
                "prompt_variable_missing", document_id=document.id, key=key
            ),
        )
        content = build_message_content(prompt, document.attachments, model_name)
        messages = [{"role": "user", "content": content}]

        start_time = time.time()
        try:
            if multi_pass > 1:
                temperatures = [0.0, 0.3]

                async def _single_pass(
                    temp: float,
                ) -> "ModelResponse | AsyncIterator[ModelResponse]":
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

                response_time = time.time() - start_time
                cost = 0.0
                for resp in responses:
                    try:
                        cost += completion_cost(completion_response=resp)
                    except Exception:
                        pass
            else:
                response = await self.client.complete(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    response_format=response_format,
                )
                response_time = time.time() - start_time
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

            if post_extraction_filter is not None:
                predicted_value = await post_extraction_filter(predicted_value, document)

            return RawPrediction(
                document_id=document.id,
                predicted_value=predicted_value,
                prompt=prompt,
                original_cost=original_cost,
                llm_cost=cost,
                response_time=response_time,
                metadata={"content": document.content, "attachments": document.attachments},
            )

        except Exception as e:
            response_time = time.time() - start_time
            logger.error(
                "evaluation_error",
                document_id=document.id,
                error=str(e),
                time=response_time,
            )
            return RawPrediction(
                document_id=document.id,
                predicted_value=f"ERROR: {str(e)}",
                prompt=prompt,
                original_cost=0.0,
                llm_cost=0.0,
                response_time=response_time,
                error=str(e),
                metadata={"error": str(e), "content": document.content},
            )


class TransformerGenerator:
    """Wraps a local TransformerModelWrapper: synchronous inference, no scoring.

    Not (yet) typed as Generator: it takes no prompt_template/model/temperature/
    etc. at all, since a local transformer classifier has no prompt or sampling
    params to speak of. Same deferral as JudgeScorer vs. Scorer: expected to
    unify once the fuller Generator/RawPrediction/Ingestor shape exists and
    every implementation is rewired at once.

    Constructed fresh per call (matching today's ``_evaluate_transformer``,
    which builds a new ``TransformerModelWrapper`` every time it runs, not a
    cached one) so that tests patching
    ``valtron_core.transformer_wrapper.TransformerModelWrapper`` keep working:
    the import inside ``__init__`` is deliberately inline, resolved fresh on
    each construction, not hoisted to module level.
    """

    def __init__(self, model_path: str, model_name: str) -> None:
        from valtron_core.transformer_wrapper import TransformerModelWrapper

        self._model = TransformerModelWrapper(model_path, model_name)

    async def generate(self, document: Document) -> RawPrediction:
        start_time = time.time()
        prediction, confidence = self._model.predict_with_confidence(document.content)
        return RawPrediction(
            document_id=document.id,
            predicted_value=prediction,
            response_time=time.time() - start_time,
            confidence_score=confidence,
            metadata={"content": document.content},
        )
