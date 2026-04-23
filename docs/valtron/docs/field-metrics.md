---
sidebar_position: 5
---

# Field Metrics

Field metrics enable per-field precision/recall/F1 scoring for structured extraction. Instead of a single pass/fail on the whole JSON output, each field in your schema gets its own score, so you can see that a model gets `city` right 95% of the time but `country` only 70%.

Field metrics are configured via `field_metrics_config` in the recipe config and require `response_format` to be set.

---

## FieldConfig schema

The `field_metrics_config` mirrors the shape of your Pydantic schema. Each node describes how that field should be scored.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `"leaf"` \| `"object"` \| `"list"` | `"leaf"` | Shape of this field |
| `weight` | `float` | `1.0` | Weight when this field is rolled up into a parent score |
| `optional` | `bool` | `false` | If true, missing values don't penalize the score |
| `metric_config` | object | `null` | How to score this field (see below) |
| `fields` | object | `null` | Child field configs for `object` and `list` types |

---

## Leaf fields

A `"leaf"` is a scalar value: a string, number, or boolean at the bottom of your schema.

### `metric_config` for leaves

```json
{
  "metric": "<metric-name>",
  "params": {}
}
```

There are three built-in leaf metrics:

---

### `"exact"`

Exact string equality after whitespace normalization. No params.

```json
{"metric": "exact"}
```

Returns `1.0` if equal, `0.0` otherwise.

---

### `"threshold"`

Numeric comparison: passes if the actual value is at or above a minimum.

```json
{"metric": "threshold", "params": {"min": 0.75}}
```

Returns `1.0` if `actual >= min`, `0.0` otherwise. Useful for confidence scores or numeric fields.

---

### `"comparator"`

The flexible comparison metric. Wraps the `Comparator` class and supports four comparison strategies controlled by `element_compare`.

```json
{
  "metric": "comparator",
  "params": {
    "element_compare": "text_similarity",
    "text_similarity_metric": "fuzz_ratio",
    "text_similarity_threshold": 0.8,
    "case_sensitive": false,
    "ignore_spaces": false
  }
}
```

#### `element_compare` strategies

**`"exact"`**: Same as the standalone `exact` metric but with normalization options.

```json
{
  "metric": "comparator",
  "params": {
    "element_compare": "exact",
    "case_sensitive": false,
    "ignore_spaces": true
  }
}
```

---

**`"text_similarity"`**: Fuzzy string similarity.

```json
{
  "metric": "comparator",
  "params": {
    "element_compare": "text_similarity",
    "text_similarity_metric": "fuzz_ratio",
    "text_similarity_threshold": 0.8
  }
}
```

| Param | Type | Default | Description |
|---|---|---|---|
| `text_similarity_metric` | string | `"fuzz_ratio"` | `"fuzz_ratio"` (rapidfuzz), `"bleu"`, `"gleu"`, or `"cosine"` |
| `text_similarity_threshold` | float | `null` | If set, score becomes a bool (pass/fail at this threshold); if null, score is the raw similarity float |

`"cosine"` calls an embedding API and counts as an expensive comparison. See [List fields](#list-fields) for implications.

---

**`"llm"`**: Uses an LLM to judge whether the two values refer to the same entity or concept. Returns a boolean.

```json
{
  "metric": "comparator",
  "params": {
    "element_compare": "llm",
    "llm_model": "gpt-4o-mini"
  }
}
```

| Param | Type | Default | Description |
|---|---|---|---|
| `llm_model` | string | `"gpt-4o-mini"` | Model used for the comparison call |

Incurs one LLM call per field per document. Costs are tracked.

---

**`"embedding"`**: Computes cosine similarity between embedding vectors.

```json
{
  "metric": "comparator",
  "params": {
    "element_compare": "embedding",
    "embedding_model": "text-embedding-3-small",
    "embedding_threshold": 0.85
  }
}
```

| Param | Type | Default | Description |
|---|---|---|---|
| `embedding_model` | string | `"text-embedding-3-small"` | Embedding model to use |
| `embedding_threshold` | float | `null` | If set, score becomes a bool; if null, score is raw cosine similarity float |

Incurs one embedding API call per field per document. Costs are tracked.

---

#### Shared `comparator` params

These apply to any `element_compare` strategy:

| Param | Type | Default | Description |
|---|---|---|---|
| `case_sensitive` | bool | `false` | Whether to preserve case when comparing |
| `ignore_spaces` | bool | `false` | Whether to strip all spaces before comparing |

---

## Object fields

An `"object"` node groups child fields. Its score is aggregated from its children.

```json
{
  "type": "object",
  "metric_config": {"propagation": "weighted_avg"},
  "fields": {
    "name": {"type": "leaf", "metric_config": {"metric": "exact"}},
    "city": {"type": "leaf", "metric_config": {"metric": "exact"}, "weight": 2.0}
  }
}
```

### Built-in aggregation strategies

| Strategy | Description |
|---|---|
| `"weighted_avg"` | Weighted mean of child scores (default) |
| `"min"` | Minimum child score |
| `"max"` | Maximum child score |

---

## List fields

A `"list"` node scores a predicted array against an expected array. Two matching modes:

### Unordered (default)

Builds an N×M score matrix across all predicted/expected pairs, then greedily matches the highest-scoring pairs first (1-to-1). Reports F1 from the resulting precision/recall.

```json
{
  "type": "list",
  "metric_config": {
    "ordered": false,
    "match_threshold": 0.5
  },
  "fields": {
    "name": {"type": "leaf", "metric_config": {"metric": "exact"}},
    "city": {"type": "leaf", "metric_config": {"metric": "exact"}}
  }
}
```

| Option | Type | Default | Description |
|---|---|---|---|
| `ordered` | bool | `false` | If true, compare positionally (exp[i] vs act[i]) |
| `match_threshold` | float | `0.5` | Minimum score for a pair to be considered a match |
| `required_fields_to_match` | array | `null` | Fields that must match before a pair is even considered (pre-filter before expensive comparison) |
| `allow_expensive_comparisons_for` | array | `null` | Fields allowed to use LLM/embedding comparison in unordered mode (see below) |

### Ordered mode

Compares element by element positionally: `expected[i]` vs `actual[i]`. Score is F1 based on matched positions.

```json
{"type": "list", "metric_config": {"ordered": true}}
```

### Expensive comparisons in lists

`"llm"` and `"embedding"` comparators in unordered list fields trigger a warning at config validation time because they run N×M calls (one per pair). You must explicitly opt in per field:

```json
{
  "type": "list",
  "metric_config": {
    "allow_expensive_comparisons_for": ["description"]
  },
  "fields": {
    "name": {"type": "leaf", "metric_config": {"metric": "exact"}},
    "description": {
      "type": "leaf",
      "metric_config": {
        "metric": "comparator",
        "params": {"element_compare": "llm"}
      }
    }
  }
}
```

---

## Custom metrics

You can register your own leaf metric functions. The signature is:

```python
def my_metric(expected: str, actual: str, params: dict) -> tuple[float, bool]:
    score = ...    # float between 0.0 and 1.0
    is_correct = score >= params.get("threshold", 0.5)
    return score, is_correct
```

Pass it via `FieldMetricsConfig`:

```python
from valtron_core.models import FieldMetricsConfig
from valtron_core.recipes import ModelEval

def starts_with_metric(expected: str, actual: str, params: dict) -> tuple[float, bool]:
    prefix = params.get("prefix_length", 3)
    match = expected[:prefix].lower() == actual[:prefix].lower()
    return (1.0 if match else 0.0), match

field_metrics = FieldMetricsConfig(
    config={
        "type": "object",
        "fields": {
            "code": {
                "type": "leaf",
                "metric_config": {"metric": "starts_with", "params": {"prefix_length": 4}}
            }
        }
    },
    custom_metrics={
        "starts_with": starts_with_metric
    }
)

experiment = ModelEval(
    config={
        "models": [...],
        "prompt": "...",
        "field_metrics_config": field_metrics,
    },
    data=data,
    response_format=MySchema,
)
```

Reference the metric by name in your `field_metrics_config` and register the callable under the same name key.

---

## Custom aggregators

You can also register your own object aggregation strategies. The signature is:

```python
def my_agg(results: list[EvalResult]) -> float:
    ...
    return score  # float between 0.0 and 1.0
```

```python
from valtron_core.evaluation.json_eval import EvalResult
from valtron_core.models import FieldMetricsConfig

def harmonic_mean_agg(results: list[EvalResult]) -> float:
    scores = [r.score for r in results if r.score > 0]
    if not scores:
        return 0.0
    return len(scores) / sum(1.0 / s for s in scores)

field_metrics = FieldMetricsConfig(
    config={
        "type": "object",
        "metric_config": {"propagation": "harmonic_mean"},
        "fields": { ... }
    },
    custom_aggs={
        "harmonic_mean": harmonic_mean_agg
    }
)
```

---

## Full example

```python
from valtron_core.models import FieldMetricsConfig

field_metrics = FieldMetricsConfig(
    config={
        "type": "object",
        "fields": {
            "title": {
                "type": "leaf",
                "weight": 2.0,
                "metric_config": {
                    "metric": "comparator",
                    "params": {
                        "element_compare": "text_similarity",
                        "text_similarity_metric": "fuzz_ratio",
                        "text_similarity_threshold": 0.85
                    }
                }
            },
            "authors": {
                "type": "list",
                "metric_config": {
                    "ordered": false,
                    "match_threshold": 0.6,
                    "required_fields_to_match": ["last_name"]
                },
                "fields": {
                    "first_name": {
                        "type": "leaf",
                        "metric_config": {"metric": "comparator", "params": {"element_compare": "exact"}}
                    },
                    "last_name": {
                        "type": "leaf",
                        "metric_config": {"metric": "exact"}
                    }
                }
            },
            "year": {
                "type": "leaf",
                "metric_config": {"metric": "exact"}
            }
        }
    }
)
```
