"""The LLM seam: an abstract :class:`Model` and the usage it accounts for.

Everything above this module -- the judge, the prompts, the scoring, the
pipeline -- depends only on "run this prompt, get text or a validated object
back". Nothing above it knows which provider answers, how the call is retried,
or whether a cache served it. That is the whole point: the same domain logic
runs against litellm here and against a host application's own client
elsewhere, by supplying a different :class:`Model`.

:class:`Usage` is the other half of the seam. A model records each call into
the accumulator it was built with, and optionally into a second one supplied
per call -- which is how a caller attributes spend to a *scope* (one candidate
on one document, say) without the judge or the pipeline having to know what a
scope is.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from pydantic import BaseModel

from .prompts import Prompt


@dataclass
class Usage:
    """What a set of LLM calls spent, accumulated as they complete."""

    calls: int = 0
    cache_hits: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    by_model: dict[str, int] = field(default_factory=dict)

    def record(
        self,
        model: str,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        cache_hit: bool,
    ) -> None:
        """Add one call's usage to the running totals."""
        self.calls += 1
        self.cache_hits += int(cache_hit)
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        self.cost_usd += cost_usd
        self.by_model[model] = self.by_model.get(model, 0) + 1

    def merge(self, other: Usage) -> None:
        """Fold another accumulator's totals into this one.

        How a run-wide figure is built from the per-scope accumulators the
        pipeline hands back, without any of them having to know it exists.
        """
        self.calls += other.calls
        self.cache_hits += other.cache_hits
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cost_usd += other.cost_usd
        for model, count in other.by_model.items():
            self.by_model[model] = self.by_model.get(model, 0) + count

    @property
    def total_tokens(self) -> int:
        """Prompt and completion tokens together."""
        return self.prompt_tokens + self.completion_tokens


class Model(ABC):
    """An LLM that can run a prompt and return text.

    Both methods take an optional ``usage``: an accumulator to record this call
    into *in addition to* whatever the model already accounts to. It is how the
    pipeline separates a candidate's generation spend from the judge spend
    incurred grading it, and both from the per-document work they share.
    """

    def __init__(self, name: str) -> None:
        """Initialize the model.

        Args:
            name: Identifier used for display and as the key this model's
                scores are recorded under.
        """
        self.name = name

    @abstractmethod
    async def run(self, prompt: Prompt, *, usage: Usage | None = None) -> str:
        """Run the prompt and return the model's text response."""

    async def run_structured[
        T: BaseModel
    ](self, prompt: Prompt, schema: type[T], *, usage: Usage | None = None) -> T:
        """Run the prompt and return a validated instance of ``schema``.

        Raises:
            NotImplementedError: If this model does not support structured
                output. Models that do override this method.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support structured output")

    def __str__(self) -> str:
        return self.name
