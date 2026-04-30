"""Typed configuration models for recipe classes."""

import re
from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Manipulation(str, Enum):
    few_shot = "few_shot"
    explanation = "explanation"
    prompt_repetition = "prompt_repetition"
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
    rewrite_model: str = "gpt-4o-mini"
    sub_prompts: dict[str, str] | None = None


class LLMModelConfig(BaseModel):
    """Config for a single LLM model entry in a recipe."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["llm"] = "llm"
    name: str
    label: str | None = None
    params: dict[str, Any] = Field(default_factory=dict)
    prompt_manipulation: list[Manipulation] = Field(default_factory=list)
    decompose_config: DecomposeConfig | None = None
    cost_rate: float | None = None
    cost_rate_time_unit: str = "1hr"
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
    cost_rate: float | None = None
    cost_rate_time_unit: str = "1hr"


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
    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    generator_model: str = "gpt-4o-mini"
    num_examples: int = 50
    max_seed_examples: int = 10
    max_few_shots: int = 10


class BaseRecipeConfig(BaseModel):
    """Shared configuration fields for all recipe classes.

    Any field added here is automatically available to every recipe.
    """

    model_config = ConfigDict(extra="forbid")

    # Required by subclasses
    models: list[ModelConfig]
    prompt: str

    # Output location — optional here; must be set either in config or per save_*() call
    output_dir: str | None = None
    use_case: str = "evaluation"

    # Evaluation defaults
    temperature: float = 0.0
    few_shot: FewShotConfig | None = None
    field_metrics_config: dict[str, Any] | None = None

    # Saving behaviour when using run() — individual save_*() methods always work
    # regardless of this setting.
    output_formats: list[str] = ["html"]

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
