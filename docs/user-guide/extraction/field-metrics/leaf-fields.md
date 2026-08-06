# Leaf Fields

A `"leaf"` is a scalar at the bottom of your schema: a string, number, or boolean. It's the simplest [`FieldConfig`](../../../api/field_config) node, and the one every object or list eventually bottoms out at.

Say the model predicted `"Stanford Univ."` for a `name` field whose expected value is `"Stanford University"`. Scored with the default strict comparison, that's a miss:

```python
from valtron_core.models import FieldMetricsConfig

field_metrics = FieldMetricsConfig(
    config={"type": "leaf", "metric_config": {"metric": "exact"}}
)
# predicted="Stanford Univ." vs expected="Stanford University" -> score 0.0, is_correct=False
```

Swap `"exact"` for `"text_similarity"` and set a threshold, and the same prediction passes:

```python
field_metrics = FieldMetricsConfig(
    config={
        "type": "leaf",
        "metric_config": {"metric": "text_similarity", "params": {"metric": "fuzz_ratio", "threshold": 0.8}},
    }
)
# same prediction/expected pair -> fuzz_ratio similarity ~0.87 >= 0.8 -> score 1.0, is_correct=True
```

Which metric is right depends entirely on how strict your task needs to be. The sections below cover each one.

## [`"exact"`](../../../api/leaf_metric_config)

Passes only if the predicted value equals the expected value, with no normalization:

```python
field_metrics = FieldMetricsConfig(
    config={"type": "leaf", "metric_config": {"metric": "exact"}}
)
# predicted="positive" vs expected="positive" -> score 1.0, is_correct=True
# predicted="Positive" vs expected="positive" -> score 0.0, is_correct=False
```

Use it for fields drawn from a fixed set of values, where any deviation is a real error.

## [`"threshold"`](../../../api/leaf_metric_config)

For numeric or confidence fields. Passes if the predicted value is at or above `min`:

```python
field_metrics = FieldMetricsConfig(
    config={"type": "leaf", "metric_config": {"metric": "threshold", "params": {"min": 0.9}}}
)
# predicted=0.95 -> 0.95 >= 0.9 -> score 1.0, is_correct=True
# predicted=0.82 -> 0.82 >= 0.9 -> score 0.0, is_correct=False
```

## [`"exact_compare"`](../../../api/leaf_metric_config)

`"exact"` with case and whitespace normalization applied first:

```python
field_metrics = FieldMetricsConfig(
    config={
        "type": "leaf",
        "metric_config": {
            "metric": "exact_compare",
            "params": {"case_sensitive": False, "ignore_spaces": True},
        },
    }
)
# predicted="STANFORD UNIVERSITY" vs expected="Stanford University" -> score 1.0, is_correct=True
```

`case_sensitive` and `ignore_spaces` both default to `False`.

## [`"text_similarity"`](../../../api/leaf_metric_config)

Scores how close two strings are instead of requiring an exact match, using `metric` to pick the underlying comparison (`"fuzz_ratio"` by default, or `"bleu"`, `"gleu"`, `"cosine"`):

```python
field_metrics = FieldMetricsConfig(
    config={
        "type": "leaf",
        "metric_config": {"metric": "text_similarity", "params": {"metric": "fuzz_ratio", "threshold": 0.8}},
    }
)
# predicted="Stanford Univ." vs expected="Stanford University" -> fuzz_ratio ~0.87 >= 0.8 -> score 1.0, is_correct=True
```

Leave `threshold` as `null` to get the raw similarity score back instead of a pass/fail. `"cosine"` calls an embedding API to compare meaning rather than spelling, and accepts its own `embedding_model` param.

## [`"llm"`](../../../api/leaf_metric_config)

Sends one LLM call per field per document, asking whether the two values "refer to the same entity or concept," and expects a boolean back:

```python
field_metrics = FieldMetricsConfig(
    config={
        "type": "leaf",
        "metric_config": {"metric": "llm", "params": {"model": "gpt-4o-mini"}},
    }
)
# predicted="NYC" vs expected="New York City" -> judge answers YES -> score 1.0, is_correct=True
```

Customize the judge with your own `prompt_template`, as long as it contains `{predicted}` and `{expected}` and ends with an instruction to answer only `YES` or `NO`:

```python
field_metrics = FieldMetricsConfig(
    config={
        "type": "leaf",
        "metric_config": {
            "metric": "llm",
            "params": {
                "model": "claude-sonnet-4-6",
                "prompt_template": (
                    "Source document:\n{example_content}\n\n"
                    "Does '{predicted}' refer to the same entity as '{expected}'?\n"
                    "Respond with only YES or NO."
                ),
            },
        },
    }
)
```

`{example_content}` (or `{example_<key>}` for dict-shaped document content) and `{prompt_used}` are also available as placeholders, filled in automatically at evaluation time.

## [`"embedding"`](../../../api/leaf_metric_config)

Compares two values by cosine similarity between their embedding vectors:

```python
field_metrics = FieldMetricsConfig(
    config={
        "type": "leaf",
        "metric_config": {"metric": "embedding", "params": {"model": "text-embedding-3-small", "threshold": 0.85}},
    }
)
# predicted="NYC" vs expected="New York City" -> cosine similarity ~0.9 >= 0.85 -> score 1.0, is_correct=True
```

Leave `threshold` as `null` to get the raw similarity score back instead of a pass/fail.

`"llm"` and `"embedding"` (and `"text_similarity"` with `metric: "cosine"`) all make a network call per comparison. See [List Fields: Expensive comparisons](./list-fields.md#expensive-comparisons) for the guardrail that applies when a leaf like this sits inside a list.

```{rubric} What's next?
```

- Group leaf fields together: [Object Fields](./object-fields)
- Score a predicted array of them: [List Fields](./list-fields)
