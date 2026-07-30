from __future__ import annotations
from typing import Literal, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator


MAX_LIST_LENGTH_FOR_EXPENSIVE_COMPARE = 4

# --- Embedding alignment defaults (unordered lists with LLM-judge leaves) ---
# Unordered-list items are aligned by optimal one-to-one assignment over embedding cosine
# similarity (Hungarian). These settings are overridable per list via ListMetricConfig.
DEFAULT_ALIGN_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MATCH_KEY_MODEL = "gpt-5.4-mini"
DEFAULT_ALIGN_LO = 0.35  # cosine floor: pairs below this are left unmatched
# Safety cap on the text embedded per item. Top-level-only rendering keeps items small; this
# bounds the request size even when a top-level string is unusually long, so a batched
# embedding over a long list can't grow into an oversized request.
MATCH_KEY_MAX_CHARS = 512


class LeafMetricConfig(BaseModel):
    """Metric config for leaf (scalar) fields."""

    model_config = ConfigDict(extra="forbid")
    metric: str = Field(
        default="exact",
        description="Comparison strategy name; see the built-in metrics table for accepted values and their params.",
    )
    params: dict[str, Any] = Field(
        default_factory=dict,
        description="Metric-specific parameters; shape depends on the chosen metric.",
    )


class ObjectMetricConfig(BaseModel):
    """Metric config for object fields."""

    model_config = ConfigDict(extra="forbid")
    propagation: str = Field(
        default="weighted_avg",
        description="How child field scores combine into this object's score.",
    )


class AlignmentConfig(BaseModel):
    """Embedding alignment settings for unordered lists with LLM-judge leaves."""

    model_config = ConfigDict(extra="forbid")
    match_key_fields: list[str] | None = Field(
        default=None,
        description="Top-level item fields to embed for alignment; if unset, all fields are used.",
    )
    match_key_model: str = Field(
        default=DEFAULT_MATCH_KEY_MODEL,
        description="Model used to select match_key_fields when they are not explicitly set.",
    )
    embed_model: str = Field(
        default=DEFAULT_ALIGN_EMBEDDING_MODEL,
        description="Embedding model used to compute item similarity for alignment.",
    )
    lo: float = Field(
        default=DEFAULT_ALIGN_LO,
        description="Cosine similarity floor; item pairs scoring below this are left unmatched.",
    )


class ListMetricConfig(BaseModel):
    """Metric config for list fields."""

    model_config = ConfigDict(extra="forbid")
    ordered: bool = Field(
        default=False,
        description="Compare items positionally when True; otherwise align items before comparing.",
    )
    match_threshold: float = Field(
        default=0.5, description="Simple greedy path only; ignored when alignment is set."
    )
    item_logic: FieldConfig | None = Field(
        default=None, description="FieldConfig describing how to score each list item."
    )
    required_fields_to_match: list[str] | None = Field(
        default=None,
        description="Item fields that must match for two items to be considered aligned.",
    )
    allow_expensive_comparisons_for: list[str] | None = Field(
        default=None,
        description=(
            "Opt-in list of 3rd-party metrics allowed despite list length exceeding "
            "MAX_LIST_LENGTH_FOR_EXPENSIVE_COMPARE; otherwise raises ExpensiveListComparisonError."
        ),
    )
    alignment: AlignmentConfig | None = Field(
        default=None, description="Embedding-based alignment settings for unordered lists."
    )


class FieldConfig(BaseModel):
    """Recursive scoring configuration for a single field in the expected schema."""

    type: Literal["object", "list", "leaf"] = "leaf"
    weight: float = Field(
        default=1.0, description="Relative weight of this field in weighted score aggregation."
    )
    optional: bool = Field(
        default=False,
        description="If True, both sides missing this field counts as correct instead of penalized.",
    )
    metric_config: LeafMetricConfig | ObjectMetricConfig | ListMetricConfig | None = None
    fields: dict[str, FieldConfig] | None = Field(
        default=None,
        description="Nested per-field configs, keyed by field name, for object/list item schemas.",
    )

    @model_validator(mode="before")
    @classmethod
    def _route_metric_config(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        field_type = data.get("type", "leaf")
        mc = data.get("metric_config") or {}
        if not isinstance(mc, dict):
            return data  # already a model instance, leave as-is

        if field_type == "object":
            data["metric_config"] = {
                "propagation": mc.get("propagation", "weighted_avg"),
            }
        elif field_type == "list":
            flat_keys = (
                "match_threshold",
                "item_logic",
                "required_fields_to_match",
                "allow_expensive_comparisons_for",
            )
            filtered: dict[str, Any] = {k: mc[k] for k in flat_keys if k in mc}
            filtered["ordered"] = mc.get("ordered", False)
            alignment_keys = ("match_key_fields", "match_key_model", "embed_model", "lo")
            alignment = {k: mc[k] for k in alignment_keys if k in mc}
            if alignment:
                filtered["alignment"] = alignment
            data["metric_config"] = filtered
        else:  # leaf
            data["metric_config"] = {
                "metric": mc.get("metric", "exact"),
                "params": mc.get("params", {}),
            }
        return data


# Resolve circular dependencies
ListMetricConfig.model_rebuild()
FieldConfig.model_rebuild()


def _leaf_mc(config: FieldConfig) -> LeafMetricConfig:
    """Narrow a leaf field's metric config to :class:`LeafMetricConfig`.

    The ``_route_metric_config`` validator guarantees a leaf field always carries a
    ``LeafMetricConfig``; this helper encodes that invariant for the type checker.

    :param config: A leaf :class:`FieldConfig`.
    :return: The field's leaf metric config.
    """
    mc = config.metric_config
    assert isinstance(mc, LeafMetricConfig)
    return mc


def _object_mc(config: FieldConfig) -> ObjectMetricConfig:
    """Narrow an object field's metric config to :class:`ObjectMetricConfig`.

    :param config: An object :class:`FieldConfig`.
    :return: The field's object metric config.
    """
    mc = config.metric_config
    assert isinstance(mc, ObjectMetricConfig)
    return mc


def _list_mc(config: FieldConfig) -> ListMetricConfig:
    """Narrow a list field's metric config to :class:`ListMetricConfig`.

    :param config: A list :class:`FieldConfig`.
    :return: The field's list metric config.
    """
    mc = config.metric_config
    assert isinstance(mc, ListMetricConfig)
    return mc


class ExpensiveListComparisonError(Exception):
    """Raised when an unordered list field uses a 3rd-party metric without
    explicit opt-in via ``allow_expensive_list_comparison: true``."""


class _MatchKeyFields(BaseModel):
    """LLM response selecting the identity-bearing fields of a list item.

    Used once per list (cached) to decide which fields to embed when aligning items for an
    unordered list.  Embedding only the identity fields keeps the cosine signal sharp;
    boilerplate/enum fields would otherwise dilute it.

    :param fields: Field names (top level of the item) that together identify an item.
    """

    fields: list[str]


class AlignmentItem(BaseModel):
    e_idx: int
    a_idx: int
    score: float
    result: EvalResult


class EvalResult(BaseModel):
    """Scoring result for one field path, aggregated bottom-up from leaf comparisons."""

    path: str
    score: float
    weight: float

    is_correct: bool = Field(
        default=False, description="Pass/fail outcome for threshold-based metrics."
    )

    tp: float = Field(
        default=0.0, description="True positives: sum of match scores for correctly matched items."
    )
    tn: float = 0.0
    fp: float = Field(
        default=0.0,
        description="False positives: extra items or hallucinations not present in the expected data.",
    )
    fn: float = Field(
        default=0.0, description="False negatives: expected items missing from the prediction."
    )

    precision: float = 0.0
    recall: float = 0.0

    metric: str
    params: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    children: dict[str, EvalResult] = {}
    alignment: list[AlignmentItem] | None = None
