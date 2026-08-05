from __future__ import annotations
from typing import Literal, Any
from pydantic import BaseModel, Field, ConfigDict, model_validator


MAX_LIST_LENGTH_FOR_EXPENSIVE_COMPARE = 4

# --- Embedding alignment defaults (unordered lists with LLM-judge leaves) ---
# Unordered-list items are aligned by optimal one-to-one assignment over embedding cosine
# similarity (Hungarian). These settings are overridable per list via ListMetricConfig.
DEFAULT_ALIGN_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_MATCH_KEY_MODEL = "gpt-5.4-mini"
DEFAULT_ALIGN_LO = 0.35      # cosine floor: pairs below this are left unmatched
# Safety cap on the text embedded per item. Top-level-only rendering keeps items small; this
# bounds the request size even when a top-level string is unusually long, so a batched
# embedding over a long list can't grow into an oversized request.
MATCH_KEY_MAX_CHARS = 512


class LeafMetricConfig(BaseModel):
    """Metric config for leaf (scalar) fields."""
    model_config = ConfigDict(extra='forbid')
    metric: str = "exact"
    params: dict[str, Any] = Field(default_factory=dict)


class ObjectMetricConfig(BaseModel):
    """Metric config for object fields."""
    model_config = ConfigDict(extra='forbid')
    propagation: str = "weighted_avg"


class AlignmentConfig(BaseModel):
    """Embedding alignment settings for unordered lists with LLM-judge leaves."""
    model_config = ConfigDict(extra='forbid')
    match_key_fields: list[str] | None = None
    match_key_model: str = DEFAULT_MATCH_KEY_MODEL
    embed_model: str = DEFAULT_ALIGN_EMBEDDING_MODEL
    lo: float = DEFAULT_ALIGN_LO


class ListMetricConfig(BaseModel):
    """Metric config for list fields."""
    model_config = ConfigDict(extra='forbid')
    ordered: bool = False
    match_threshold: float = Field(default=0.5, description="Simple greedy path only; ignored when alignment is set.")
    item_logic: FieldConfig | None = None
    required_fields_to_match: list[str] | None = None
    allow_expensive_comparisons_for: list[str] | None = None
    alignment: AlignmentConfig | None = None



class FieldConfig(BaseModel):
    type: Literal["object", "list", "leaf"] = "leaf"
    weight: float = 1.0
    optional: bool = False
    metric_config: LeafMetricConfig | ObjectMetricConfig | ListMetricConfig | None = None
    fields: dict[str, FieldConfig] | None = None

    @model_validator(mode='before')
    @classmethod
    def _route_metric_config(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        field_type = data.get('type', 'leaf')
        mc = data.get('metric_config') or {}
        if not isinstance(mc, dict):
            return data  # already a model instance, leave as-is

        if field_type == 'object':
            data['metric_config'] = {
                'propagation': mc.get('propagation', 'weighted_avg'),
            }
        elif field_type == 'list':
            flat_keys = (
                'match_threshold', 'item_logic', 'required_fields_to_match',
                'allow_expensive_comparisons_for',
            )
            filtered: dict[str, Any] = {k: mc[k] for k in flat_keys if k in mc}
            filtered['ordered'] = mc.get('ordered', False)
            alignment_keys = ('match_key_fields', 'match_key_model', 'embed_model', 'lo')
            alignment = {k: mc[k] for k in alignment_keys if k in mc}
            if alignment:
                filtered['alignment'] = alignment
            data['metric_config'] = filtered
        else:  # leaf
            data['metric_config'] = {
                'metric': mc.get('metric', 'exact'),
                'params': mc.get('params', {}),
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
    path: str
    score: float
    weight: float

    is_correct: bool = False  # For threshold-based metrics

    tp: float = 0.0  # True Positives (Sum of match scores)
    tn: float = 0.0
    fp: float = 0.0  # False Positives (Extra items / hallucinations)
    fn: float = 0.0  # False Negatives (Missing items)

    precision: float = 0.0
    recall: float = 0.0

    metric: str
    params: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    children: dict[str, EvalResult] = {}
    alignment: list[AlignmentItem] | None = None
