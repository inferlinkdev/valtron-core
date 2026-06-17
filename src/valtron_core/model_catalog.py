# Copyright 2026 InferLink
# SPDX-License-Identifier: Apache-2.0
"""Token-based cost catalog over the models LiteLLM knows about.

LiteLLM ships a pricing table (``litellm.model_cost``) keyed by model name. For
each model it records, among other things, ``input_cost_per_token`` and
``output_cost_per_token`` (USD per token). This module distils that table into a
simple per-model cost model and exposes two things:

- :meth:`ModelCatalog.get_expected_cost` — the dollar cost of a hypothetical call
  with a given input/output token count.
- :meth:`ModelCatalog.get_spread` — a uniform linear spread of models, by cost,
  from the cheapest known model up to a given "current" model.

Scope note — other cost variables. Real LiteLLM pricing has many dimensions
beyond plain input/output tokens: prompt-cache read/creation rates, batch and
priority/flex service tiers, tiered rates above a token threshold
(``*_above_128k_tokens`` etc.), reasoning-token surcharges, per-second pricing
(audio/video), per-character pricing (Gemini), per-image and per-request costs,
and regional uplift multipliers. The cost model here intentionally uses only the
two flat per-token rates, which is the right approximation for ordinary text
chat/completion calls and the only dimension that depends solely on token
counts. The catalog is restricted to chat-completion models (LiteLLM
``mode == "chat"``): embedding, image, audio, rerank, moderation, legacy text
``completion`` and Responses-API models are all excluded, as is anything lacking
both per-token rates.

This complements the runtime cost paths elsewhere in the codebase: LiteLLM's
``completion_cost`` measures the *actual* spend of a completed call (client.py,
evaluator.py), and ``cost_utils.py`` imputes a *time-based* estimate when LiteLLM
has no pricing at all. This module is the *a-priori, token-based* estimator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import litellm

litellm.suppress_debug_info = True

# LiteLLM's `mode` value for chat-completion models; the catalog is limited to these.
_CHAT_MODE = "chat"


@dataclass(frozen=True)
class ModelCost:
    """Flat per-token cost model for a single LiteLLM model."""

    name: str
    input_cost_per_token: float
    output_cost_per_token: float
    provider: str | None = None
    mode: str | None = None

    def expected_cost(self, num_input_tokens: int, num_output_tokens: int) -> float:
        """Dollar cost of a call with the given input/output token counts."""
        return (
            num_input_tokens * self.input_cost_per_token
            + num_output_tokens * self.output_cost_per_token
        )

    @property
    def filter_target(self) -> str:
        """The ``"{provider}/{name}"`` string that include/exclude filters match.

        Matching against this combined string (rather than the bare name) lets a
        filter address the provider too — e.g. ``"^openai/"`` selects first-party
        OpenAI models while skipping ``azure/``- or ``bedrock/``-hosted ones.
        """
        return f"{self.provider or ''}/{self.name}"


def _validate_tokens(num_input_tokens: int, num_output_tokens: int) -> None:
    if num_input_tokens < 0 or num_output_tokens < 0:
        raise ValueError(
            f"Token counts must be non-negative, got input={num_input_tokens}, "
            f"output={num_output_tokens}."
        )


def _linspace(lo: float, hi: float, n: int) -> list[float]:
    """Return ``n`` evenly spaced values from ``lo`` to ``hi`` (both inclusive)."""
    if n == 1:
        return [hi]
    step = (hi - lo) / (n - 1)
    return [lo + step * i for i in range(n)]


def _compile_filter(pattern: str | None, label: str) -> re.Pattern[str] | None:
    """Compile a filter regex, or return None when no filter is set."""
    if pattern is None:
        return None
    try:
        return re.compile(pattern)
    except re.error as exc:
        raise ValueError(f"Invalid {label} regex {pattern!r}: {exc}") from exc


def _passes_filters(
    target: str, include: re.Pattern[str] | None, exclude: re.Pattern[str] | None
) -> bool:
    """Whether ``target`` survives the include/exclude filters.

    ``target`` is a model's :attr:`ModelCost.filter_target` — its
    ``"{provider}/{name}"`` string — so filters can address provider and model
    name together. Kept iff it matches ``include`` (when set) and does not match
    ``exclude`` (when set). The two are independent predicates, so include-first
    vs exclude-first is irrelevant; on a target matching both, exclude wins.
    """
    if include is not None and not include.search(target):
        return False
    if exclude is not None and exclude.search(target):
        return False
    return True


class ModelCatalog:
    """A token-cost catalog built from LiteLLM's pricing table.

    The catalog is built lazily from ``litellm.model_cost`` and cached on the
    class. Call :meth:`refresh` to rebuild it (e.g. after LiteLLM's pricing data
    is updated, or to pick up patched data in tests).
    """

    _catalog: dict[str, ModelCost] | None = None

    @staticmethod
    def _build_catalog() -> dict[str, ModelCost]:
        """Distil ``litellm.model_cost`` into per-model :class:`ModelCost` entries.

        Only chat-completion models (LiteLLM ``mode == "chat"``) that expose
        *both* an input and an output per-token rate are kept; embedding, image,
        audio, rerank, legacy ``completion`` / Responses-API models and the
        ``sample_spec`` schema entry are all dropped.
        """
        catalog: dict[str, ModelCost] = {}
        for name, info in litellm.model_cost.items():
            if name == "sample_spec" or not isinstance(info, dict):
                continue
            if info.get("mode") != _CHAT_MODE:
                continue
            input_cost = info.get("input_cost_per_token")
            output_cost = info.get("output_cost_per_token")
            if input_cost is None or output_cost is None:
                continue
            catalog[name] = ModelCost(
                name=name,
                input_cost_per_token=float(input_cost),
                output_cost_per_token=float(output_cost),
                provider=info.get("litellm_provider"),
                mode=info.get("mode"),
            )
        return catalog

    @classmethod
    def _ensure_loaded(cls) -> dict[str, ModelCost]:
        if cls._catalog is None:
            cls._catalog = cls._build_catalog()
        return cls._catalog

    @classmethod
    def refresh(cls) -> None:
        """Rebuild the cached catalog from the current ``litellm.model_cost``."""
        cls._catalog = cls._build_catalog()

    @classmethod
    def available_models(cls) -> list[str]:
        """Names of every model with a token-based cost model, sorted."""
        return sorted(cls._ensure_loaded())

    @classmethod
    def get_model_cost(cls, model_name: str) -> ModelCost:
        """Return the :class:`ModelCost` for ``model_name``.

        Raises:
            ValueError: if the model has no token-based cost model in the catalog.
        """
        catalog = cls._ensure_loaded()
        try:
            return catalog[model_name]
        except KeyError:
            raise ValueError(
                f"Model {model_name!r} is not in the LiteLLM token-cost catalog. "
                "It may be unknown to LiteLLM, not a chat-completion model, or "
                "priced on a non-token basis (embedding/image/audio/rerank)."
            ) from None

    @classmethod
    def get_expected_cost(
        cls, model_name: str, num_input_tokens: int, num_output_tokens: int
    ) -> float:
        """Expected dollar cost of one call to ``model_name``.

        Args:
            model_name: A model name known to LiteLLM (e.g. ``"gpt-4o-mini"``).
            num_input_tokens: Number of prompt/input tokens.
            num_output_tokens: Number of completion/output tokens.

        Returns:
            Cost in US dollars: ``input * input_rate + output * output_rate``.

        Raises:
            ValueError: if the model is not in the catalog, or token counts are
                negative.
        """
        _validate_tokens(num_input_tokens, num_output_tokens)
        return cls.get_model_cost(model_name).expected_cost(num_input_tokens, num_output_tokens)

    @classmethod
    def get_spread(
        cls,
        current_model_name: str,
        num_models: int,
        num_input_tokens: int,
        num_output_tokens: int,
        include_filter: str | None = None,
        exclude_filter: str | None = None,
    ) -> list[str]:
        """Models forming a uniform linear cost spread up to ``current_model_name``.

        Costs are evaluated for the given token mix (relative model ordering
        depends on the input/output ratio, so the spread is workload-specific).
        ``num_models`` target costs are spaced evenly between the cheapest known
        model and ``current_model_name``; each target is mapped to the model
        whose expected cost is nearest. The endpoints are pinned to the cheapest
        model and ``current_model_name`` respectively.

        Models more expensive than ``current_model_name`` are excluded — the
        spread spans ``[cheapest, current]`` only.

        Args:
            current_model_name: The upper anchor of the spread.
            num_models: Maximum number of models to return (must be >= 1).
            num_input_tokens: Input tokens used to price every model.
            num_output_tokens: Output tokens used to price every model.
            include_filter: Optional regex; only models whose
                ``"{provider}/{name}"`` string matches (``re.search``) are
                eligible for the spread.
            exclude_filter: Optional regex; models whose ``"{provider}/{name}"``
                string matches are dropped. Both filters are matched against that
                combined provider-and-name string, so they can address either
                (e.g. ``"^openai/"`` for a provider, ``"haiku"`` for a name).
                They are independent predicates — order is irrelevant, and a
                model matching both is excluded. ``current_model_name`` must
                itself survive both filters (it is the anchor of the spread).

        Returns:
            Up to ``num_models`` distinct model names, ordered from cheapest to
            current. Fewer are returned when the catalog is too sparse to fill
            every target — duplicates are collapsed rather than padded — so
            requesting more models than exist in range yields all of them.

        Raises:
            ValueError: if ``current_model_name`` is not in the catalog or does
                not pass the filters, a filter regex is invalid,
                ``num_models < 1``, or token counts are negative.
        """
        _validate_tokens(num_input_tokens, num_output_tokens)
        if num_models < 1:
            raise ValueError(f"num_models must be >= 1, got {num_models}.")

        catalog = cls._ensure_loaded()
        if current_model_name not in catalog:
            raise ValueError(
                f"Model {current_model_name!r} is not in the LiteLLM token-cost catalog."
            )

        include = _compile_filter(include_filter, "include_filter")
        exclude = _compile_filter(exclude_filter, "exclude_filter")
        if not _passes_filters(catalog[current_model_name].filter_target, include, exclude):
            raise ValueError(
                f"current_model {current_model_name!r} does not pass the include/exclude "
                "filters; it must, since it anchors the spread."
            )

        current_cost = catalog[current_model_name].expected_cost(
            num_input_tokens, num_output_tokens
        )
        # Filtered candidates within [cheapest, current], sorted by (cost, name)
        # for determinism. current always survives (it passed the filters above).
        candidates = sorted(
            (
                (cost, name)
                for name, mc in catalog.items()
                if _passes_filters(mc.filter_target, include, exclude)
                and (cost := mc.expected_cost(num_input_tokens, num_output_tokens)) <= current_cost
            ),
            key=lambda pair: (pair[0], pair[1]),
        )
        cheapest_cost, cheapest_name = candidates[0]

        if num_models == 1:
            return [current_model_name]

        targets = _linspace(cheapest_cost, current_cost, num_models)
        spread: list[str] = []
        for i, target in enumerate(targets):
            if i == 0:
                spread.append(cheapest_name)
            elif i == num_models - 1:
                spread.append(current_model_name)
            else:
                # Nearest by |cost - target|; tie-break by lower cost then name.
                _, name = min(
                    candidates,
                    key=lambda pair: (abs(pair[0] - target), pair[0], pair[1]),
                )
                spread.append(name)
        # Collapse duplicates (preserving cheapest -> current order): when the
        # catalog can't fill every target, return the smaller distinct set.
        return list(dict.fromkeys(spread))
