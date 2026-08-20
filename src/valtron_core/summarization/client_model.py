"""A :class:`Model` implemented over this codebase's ``LLMClient``.

This is the whole of what the ported summarization logic needs in order to run
here rather than standalone. Everything above it -- the judge, the pipeline, the
scoring -- depends only on the abstract "run this prompt, get text or a
validated object back" call that :mod:`valtron_core.summarization.model`
defines, so swapping the implementation is the entire adaptation.

Three things this adds on top of ``LLMClient.complete``, none of which it
provides and all of which the method needs:

* **Structured output.** The judge's verdicts are Pydantic models. ``complete``
  will pass a ``response_format`` through to litellm, but hands back a raw
  completion; validating it is done here, with the model named in the error,
  because a judge returning prose instead of JSON is the likeliest operational
  failure and the least obvious one from a bare ``ValidationError``.
* **Per-call accounting.** ``LLMClient`` tracks a process-wide call count and
  cost total, which cannot attribute spend to one candidate on one document.
  Each call is recorded into the caller's :class:`Usage` instead, which is what
  lets the recipe report a per-prediction cost and split judge spend from
  generation spend.
* **A fixed temperature.** ``complete`` defaults to 0.7; an evaluation wants 0.0
  so that a re-run is comparable.

What it deliberately does *not* add is retries, rate limiting or provider
setup. ``LLMClient`` owns all three, and a second layer of them here is exactly
what this port set out to avoid.
"""

from __future__ import annotations

from typing import Any

from litellm import completion_cost
from litellm.types.utils import Choices
from litellm.utils import ModelResponse  # type: ignore[attr-defined]
from pydantic import BaseModel, ValidationError

from valtron_core.client import LLMClient

from .model import Model, Usage
from .prompts import Prompt


class ClientModel(Model):
    """A :class:`Model` that reaches an LLM through :class:`LLMClient`.

    ``model`` is whatever ``LLMClient.complete`` accepts: a plain model name, or
    a dict of litellm parameters as built by ``ModelEval._build_model_arg``. The
    dict form is how per-model ``params`` and ``cost_rate`` reach the call, so
    prefer it for candidate models and the plain string for the judge.
    """

    def __init__(
        self,
        model: str | dict[str, Any],
        *,
        client: LLMClient,
        name: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        usage: Usage | None = None,
    ) -> None:
        """Initialize the model.

        Args:
            model: Model name, or a litellm parameter dict containing ``model``.
            client: The shared client every call goes through.
            name: Display/scores key; defaults to the model name.
            temperature: Sampling temperature; 0.0 for repeatable evaluation. A
                ``temperature`` inside ``model`` takes precedence, since
                ``LLMClient`` applies the dict last.
            max_tokens: Optional cap on response length.
            usage: Optional accumulator recording every call this model makes,
                in addition to any passed per call.
        """
        resolved = model if isinstance(model, str) else str(model.get("model", "unknown"))
        super().__init__(name or resolved)
        self._model = model
        self._client = client
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._usage = usage

    async def run(self, prompt: Prompt, *, usage: Usage | None = None) -> str:
        return await self._call(prompt, response_format=None, usage=usage)

    async def run_structured[
        T: BaseModel
    ](self, prompt: Prompt, schema: type[T], *, usage: Usage | None = None) -> T:
        content = await self._call(prompt, response_format=schema, usage=usage)
        try:
            return schema.model_validate_json(content)
        except ValidationError as error:
            # Almost always one of two things: the model does not support
            # structured output (litellm's drop_params silently removes the
            # response_format), or it wrapped the JSON in prose. Neither is
            # apparent from the raw validation error, so say which model.
            raise ValueError(
                f"model {self.name!r} did not return valid {schema.__name__} JSON; "
                f"got: {content[:200]!r}"
            ) from error

    async def _call(
        self,
        prompt: Prompt,
        *,
        response_format: type[BaseModel] | None,
        usage: Usage | None,
    ) -> str:
        response = await self._client.complete(
            model=self._model,
            messages=[{"role": "user", "content": str(prompt)}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
            response_format=response_format,
        )
        if not isinstance(response, ModelResponse):
            raise TypeError(f"expected a ModelResponse, got {type(response).__name__}")
        choice = response.choices[0]
        if not isinstance(choice, Choices):
            raise TypeError("expected a non-streaming completion choice")
        content = choice.message.content
        if content is None:
            raise ValueError(f"model {self.name!r} returned no content")

        self._record(response, usage)
        return content

    def _record(self, response: ModelResponse, usage: Usage | None) -> None:
        """Charge one call to this model's accumulator and the caller's, if any."""
        if self._usage is None and usage is None:
            return
        stats = getattr(response, "usage", None)
        hidden: dict[str, Any] = getattr(response, "_hidden_params", None) or {}
        # Only ever true if a host has enabled litellm's cache globally; this
        # codebase does not, but a cached response must still be free if so.
        cache_hit = bool(hidden.get("cache_hit"))
        cost = 0.0
        if not cache_hit:
            # litellm attaches the price it worked out to every completion, so
            # prefer that over recomputing it; completion_cost is the fallback
            # for responses that arrived by some other route.
            recorded = hidden.get("response_cost")
            if recorded is not None:
                cost = float(recorded)
            else:
                try:
                    cost = float(completion_cost(completion_response=response))
                except Exception:
                    cost = 0.0
        for accumulator in (self._usage, usage):
            if accumulator is not None:
                accumulator.record(
                    self.name,
                    prompt_tokens=int(getattr(stats, "prompt_tokens", 0) or 0),
                    completion_tokens=int(getattr(stats, "completion_tokens", 0) or 0),
                    cost_usd=cost,
                    cache_hit=cache_hit,
                )
