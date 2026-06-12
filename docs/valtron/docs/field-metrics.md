---
sidebar_position: 5
---

# Field Metrics

Field metrics enable per-field precision/recall/F1 scoring for structured extraction. Instead of a single pass/fail on the whole JSON output, each field in your schema gets its own score, so you can see that a model gets `city` right 95% of the time but `country` only 70%.

Field metrics are configured via `field_metrics_config` in the recipe config and require `response_format` to be set.

---

## FieldConfig schema

The `config` value inside `field_metrics_config` mirrors the shape of your Pydantic schema. Each node describes how that field should be scored.

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `"leaf"` \| `"object"` \| `"list"` | `"leaf"` | Shape of this field |
| `weight` | `float` | `1.0` | Weight when this field is rolled up into a parent score |
| `optional` | `bool` | `false` | If true, missing values don't penalize the score |
| `metric_config` | object | `null` | Scoring/matching/aggregation config. Shape depends on `type` -- see each section below. |
| `fields` | object | `null` | Child field configs for `object` and `list` types |

---

The three node types -- `"leaf"`, `"object"`, and `"list"` -- each use `metric_config` differently. The sections below cover each in turn.

---

## Leaf nodes

A `"leaf"` is a scalar value: a string, number, or boolean at the bottom of your schema.

Leaf nodes use `metric_config` to specify a comparison metric and its parameters:

```json
{
  "metric": "<metric-name>",
  "params": {}
}
```

There are six built-in leaf metrics:

---

### `"exact"`

Strict equality check (`predicted == expected`). No params. No normalization is applied.

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

### `"exact_compare"`

Normalized string comparison with optional case and whitespace handling.

```json
{
  "metric": "exact_compare",
  "params": {
    "case_sensitive": false,
    "ignore_spaces": false
  }
}
```

| Param | Type | Default | Description |
|---|---|---|---|
| `case_sensitive` | bool | `false` | Whether to preserve case when comparing |
| `ignore_spaces` | bool | `false` | Whether to strip all spaces before comparing |

Returns `1.0` if the normalized strings match, `0.0` otherwise.

---

### `"text_similarity"`

Fuzzy string similarity using your choice of algorithm.

```json
{
  "metric": "text_similarity",
  "params": {
    "metric": "fuzz_ratio",
    "threshold": 0.8
  }
}
```

| Param | Type | Default | Description |
|---|---|---|---|
| `metric` | string | `"fuzz_ratio"` | `"fuzz_ratio"` (rapidfuzz), `"bleu"`, `"gleu"`, or `"cosine"` |
| `threshold` | float | `null` | If set, score becomes a bool (pass/fail at this threshold); if null, score is the raw similarity float |
| `case_sensitive` | bool | `false` | Whether to preserve case when comparing |
| `ignore_spaces` | bool | `false` | Whether to strip all spaces before comparing |
| `embedding_model` | string | `"text-embedding-3-small"` | Embedding model used when `metric` is `"cosine"` |

`"cosine"` calls an embedding API and counts as an expensive comparison. See [List fields](#list-fields) for implications.

---

### `"llm"`

Uses an LLM to judge whether the two values refer to the same entity or concept. Returns a boolean.

```json
{
  "metric": "llm",
  "params": {
    "model": "gpt-4o-mini"
  }
}
```

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | string | `"gpt-4o-mini"` | Any [LiteLLM-supported model string](https://docs.litellm.ai/docs/providers) |
| `prompt_template` | string | `null` | Custom prompt template. Must contain `{predicted}` and `{expected}` placeholders and end with an instruction to respond with only `YES` or `NO`. When omitted, the built-in entity-matching prompt is used. |

**Custom prompt templates**

By default the following prompt is used:

```
Do these two values refer to the same entity or concept? Consider them a match even if one is more specific, has extra qualifiers, or is an abbreviation of the other.

Value 1: <predicted>
Value 2: <expected>

Respond with only "YES" or "NO".
```

Set `prompt_template` to replace it with a prompt suited to your domain. The template supports the following placeholders, which are filled in automatically at evaluation time:

| Placeholder | Value |
|---|---|
| `{predicted}` | The model's predicted value for the field |
| `{expected}` | The ground-truth value for the field |
| `{prompt_used}` | The full prompt that was sent to the model being evaluated |
| `{example_content}` | The document content (when the document has a plain-string `content` field) |
| `{example_<key>}` | One placeholder per key when the document `content` is a dict (e.g. `{example_title}`, `{example_body}`) |

Your template must end with an instruction to respond with only `YES` or `NO`.

```json
{
  "metric": "llm",
  "params": {
    "model": "claude-sonnet-4-6",
    "prompt_template": "Source document:\n{example_content}\n\nDoes '{predicted}' refer to the same entity as '{expected}'?\nRespond with only YES or NO."
  }
}
```

When the model supports structured outputs (e.g. GPT-4o, Claude 3.5+), Valtron uses a JSON schema response format to enforce a boolean `match` field, making the result reliable regardless of how the model phrases its answer. For models that do not support structured outputs, the response must start with `YES` or `NO` (case insensitive).

Incurs one LLM call per field per document.

---

### `"embedding"`

Computes cosine similarity between embedding vectors.

```json
{
  "metric": "embedding",
  "params": {
    "model": "text-embedding-3-small",
    "threshold": 0.85
  }
}
```

| Param | Type | Default | Description |
|---|---|---|---|
| `model` | string | `"text-embedding-3-small"` | Any [LiteLLM-supported embedding model](https://docs.litellm.ai/docs/embedding/supported_providers) |
| `threshold` | float | `null` | If set, score becomes a bool; if null, score is raw cosine similarity float |

Incurs one embedding API call per field per document.

---

## Object nodes

An `"object"` node groups child fields. Its score is aggregated from its children.

Object nodes use `metric_config` to specify how child scores are aggregated:

```json
{"metric_config": {"propagation": "weighted_avg"}}
```

The default propagation is `"weighted_avg"`. Full example:

```json
{
  "type": "object",
  "metric_config": {"propagation": "weighted_avg"},
  "fields": {
    "name": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
    "city": {"type": "leaf", "metric_config": {"metric": "exact_compare"}, "weight": 2.0}
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

## List nodes

A `"list"` node scores a predicted array against an expected array.

List nodes use `metric_config` to control matching behavior:

| Option | Type | Default | Description |
|---|---|---|---|
| `ordered` | bool | `false` | If true, compare positionally (`exp[i]` vs `act[i]`); if false, greedy best-match |
| `match_threshold` | float | `0.5` | Minimum score for a pair to be considered a match |
| `required_fields_to_match` | array | `null` | Fields that must match before a pair is even considered (pre-filter before expensive comparison) |
| `allow_expensive_comparisons_for` | array | `null` | Fields allowed to use `"llm"` or `"embedding"` metrics in unordered mode (see below) |

Two matching modes:

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
    "name": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
    "city": {"type": "leaf", "metric_config": {"metric": "exact_compare"}}
  }
}
```

### Ordered mode

Compares element by element positionally: `expected[i]` vs `actual[i]`. Score is F1 based on matched positions.

```json
{"type": "list", "metric_config": {"ordered": true}}
```

### Expensive comparisons in lists

The `"llm"` and `"embedding"` metrics in unordered list fields trigger an error at evaluation time because they run N×M API calls (one per pair). You must explicitly opt in per field path:

```json
{
  "type": "list",
  "metric_config": {
    "allow_expensive_comparisons_for": ["description"]
  },
  "fields": {
    "name": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
    "description": {
      "type": "leaf",
      "metric_config": {
        "metric": "llm",
        "params": {"model": "gpt-4o-mini"}
      }
    }
  }
}
```

For a flat list of primitives, use `"$item"` as the path:

```json
{
  "type": "list",
  "metric_config": {
    "ordered": false,
    "allow_expensive_comparisons_for": ["$item"],
    "item_logic": {
      "type": "leaf",
      "metric_config": {"metric": "llm", "params": {}}
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
                    "metric": "text_similarity",
                    "params": {
                        "metric": "fuzz_ratio",
                        "threshold": 0.85
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
                        "metric_config": {
                            "metric": "exact_compare",
                            "params": {"case_sensitive": false}
                        }
                    },
                    "last_name": {
                        "type": "leaf",
                        "metric_config": {"metric": "exact_compare"}
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
