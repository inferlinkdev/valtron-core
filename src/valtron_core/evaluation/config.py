"""Typed configuration models for recipe classes."""

import re
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Manipulation(str, Enum):
    """Prompt manipulation strategies that can be layered onto an LLM model config.

    ``decompose``, ``hallucination_filter``, and ``multi_pass`` require a Pydantic
    ``response_format`` to be provided (see ``requires_response_format``).
    """

    few_shot = "few_shot"
    explanation = "explanation"
    prompt_repetition = "prompt_repetition"
    #: Repeats the prompt three times per call instead of once.
    prompt_repetition_x3 = "prompt_repetition_x3"
    decompose = "decompose"
    hallucination_filter = "hallucination_filter"
    multi_pass = "multi_pass"

    @property
    def requires_response_format(self) -> bool:
        """True when this manipulation requires a Pydantic response_format to be provided."""
        return self in STRUCTURED_MANIPULATIONS


# Manipulations that only work in structured-output mode (response_format required).
# Used by ModelEval.__init__ to validate configuration at construction time.
STRUCTURED_MANIPULATIONS: frozenset[Manipulation] = frozenset(
    {
        Manipulation.decompose,
        Manipulation.hallucination_filter,
        Manipulation.multi_pass,
    }
)


class DecomposeConfig(BaseModel):
    """Configuration for the ``decompose`` prompt manipulation."""

    rewrite_model: str = Field(
        default="gpt-4o-mini",
        description="Model used to auto-generate per-field rewrite prompts.",
    )
    sub_prompts: dict[str, str] | None = Field(
        default=None,
        description=(
            "Manual override of the auto-generated per-field rewrite prompts, keyed by field name."
        ),
    )


class LLMModelConfig(BaseModel):
    """Config for a single LLM model entry in a recipe."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["llm"] = "llm"
    name: str
    label: str | None = None
    params: dict[str, Any] = Field(
        default_factory=dict, description="Extra keyword arguments passed through to litellm."
    )
    prompt_manipulation: list[Manipulation] = Field(
        default_factory=list,
        description="Prompt manipulations to apply, in order, when calling this model.",
    )
    decompose_config: DecomposeConfig | None = Field(
        default=None,
        description="Settings for the decompose manipulation; ignored unless decompose is in prompt_manipulation.",
    )
    cost_rate: float | None = Field(
        default=None,
        description="Cost per cost_rate_time_unit used to compute billed cost; if unset, cost is taken as-reported from litellm.",
    )
    cost_rate_time_unit: str = Field(
        default="1hr",
        description="Time unit for cost_rate (e.g. '1hr'); only meaningful when cost_rate is set.",
    )
    prompt: str | None = None

    @model_validator(mode="after")
    def model_prompt_has_placeholder(self) -> "LLMModelConfig":
        if self.prompt is not None and not re.search(r"\{\w+\}", self.prompt):
            raise ValueError("model prompt must contain at least one {placeholder}")
        return self


class TransformerModelConfig(BaseModel):
    """Config for a local transformer model entry in a recipe.

    ``model_path`` must point to the ``final_model/`` directory produced by
    ``train_transformer()`` (or the ``TransformerClassifier.train()`` method
    directly). That directory must contain a ``label_mapping.json``.
    """

    model_config = ConfigDict(extra="forbid")

    type: Literal["transformer"]
    label: str
    model_path: str
    cost_rate: float | None = Field(
        default=None,
        description="Cost per cost_rate_time_unit used to compute billed cost; if unset, no cost is tracked.",
    )
    cost_rate_time_unit: str = Field(
        default="1hr",
        description="Time unit for cost_rate (e.g. '1hr'); only meaningful when cost_rate is set.",
    )


# Discriminated union — Pydantic routes to the correct variant based on `type`.
# Use as the type annotation wherever a single model config is accepted.
ModelConfig = Annotated[
    LLMModelConfig | TransformerModelConfig,
    Field(discriminator="type"),
]


def _inject_default_llm_type(data: Any) -> Any:
    """Pre-validator helper: insert ``type: "llm"`` when the field is absent.

    This preserves backwards compatibility with configs that omit ``type``
    entirely (the historical default was ``"llm"``).
    """
    if isinstance(data, dict):
        for m in data.get("models", []):
            if isinstance(m, dict) and "type" not in m:
                m["type"] = "llm"
    return data


class FewShotConfig(BaseModel):
    """Auto-generates and injects few-shot examples into the prompt.

    ``max_seed_examples`` real labeled documents seed generation of ``num_examples``
    synthetic candidates, of which the best ``max_few_shots`` are kept and injected
    into the prompt.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    generator_model: str = Field(
        default="gpt-4o-mini", description="Model used to generate synthetic few-shot examples."
    )
    num_examples: int = Field(
        default=50, description="Number of synthetic examples to generate before selection."
    )
    max_seed_examples: int = Field(
        default=10, description="Number of real labeled documents used to seed generation."
    )
    max_few_shots: int = Field(
        default=10,
        description="Maximum number of generated examples actually injected into the prompt.",
    )


class BaseRecipeConfig(BaseModel):
    """Shared configuration fields for all recipe classes.

    Any field added here is automatically available to every recipe.
    """

    model_config = ConfigDict(extra="forbid")

    # Required by subclasses
    models: list[ModelConfig]
    prompt: str

    # Output location, optional here; must be set either in config or per save_*() call
    output_dir: str | None = Field(
        default=None,
        description="Output directory for run(); if unset, must be provided per save_*() call instead.",
    )
    use_case: str = Field(
        default="evaluation",
        description="Free-text label describing the task, shown in generated reports.",
    )

    # Evaluation defaults
    temperature: float = 0.0
    few_shot: FewShotConfig | None = None
    field_metrics_config: dict[str, Any] | None = Field(
        default=None,
        description="Raw field-metrics config dict; see FieldMetricsConfig for the parsed structure.",
    )

    # Optional structured output schema in litellm format:
    # {"type": "json_schema", "json_schema": {"name": str, "strict": bool, "schema": {...}}}
    # Takes lower priority than a Pydantic response_format passed to the recipe constructor.
    response_format_schema: dict[str, Any] | None = Field(
        default=None,
        description=(
            "Structured output schema in litellm format; lower priority than a response_format "
            "passed to the recipe constructor."
        ),
    )

    # Saving behaviour when using run() — individual save_*() methods always work
    # regardless of this setting.
    output_formats: list[str] = Field(
        default=["html"], description="Report formats to write on run(), e.g. 'html', 'pdf'."
    )

    @model_validator(mode="before")
    @classmethod
    def _default_model_type(cls, data: Any) -> Any:
        return _inject_default_llm_type(data)

    @model_validator(mode="after")
    def prompt_has_placeholder(self) -> "BaseRecipeConfig":
        if not re.search(r"\{\w+\}", self.prompt):
            raise ValueError("prompt must contain at least one {placeholder}")
        return self


class ModelEvalConfig(BaseRecipeConfig):
    use_case: str = "model evaluation"
