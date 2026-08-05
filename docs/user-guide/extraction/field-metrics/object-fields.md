# Object Fields

An `"object"` node groups child fields into a single score instead of leaving them as several disconnected numbers. Suppose you have a schema like this:

```python
from pydantic import BaseModel

class Institution(BaseModel):
    name: str
    city: str
    country: str
```

The `Institution` object aggregates the scores of `name`, `city`, and `country` into one overall score. Two settings control that aggregation. `weight` on each child [`FieldConfig`](../../../api/field_config) sets how much that child counts toward the parent. `propagation` on the object's own [`ObjectMetricConfig`](../../../api/object_metric_config) sets how the weighted children combine into one score.

## Weighted Average

The default propagation method. Every child field is weighted equally by default. Score an institution where `name` matters most by doing something like:

```python
from valtron_core.models import FieldMetricsConfig

field_metrics = FieldMetricsConfig(
    config={
        "type": "object",
        "metric_config": {"propagation": "weighted_avg"},
        "fields": {
            "name": {"type": "leaf", "weight": 3.0, "metric_config": {"metric": "text_similarity", "params": {"threshold": 0.85}}},
            "city": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
            "country": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
        },
    }
)
```

With `weight: 3.0` on `name` and the default weight of `1.0` on the other two, a wrong `name` drags the object's score down far more than a wrong `city` would.

## Worst-Scoring Child

Setting `propagation` to `"min"` scores the object however badly its worst child did, regardless of weight. Use it when every field genuinely has to be right for the record to count as correct at all:

```python
field_metrics = FieldMetricsConfig(
    config={
        "type": "object",
        "metric_config": {"propagation": "min"},
        "fields": {
            "name": {"type": "leaf", "metric_config": {"metric": "text_similarity", "params": {"threshold": 0.85}}},
            "city": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
            "country": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
        },
    }
)
```

## Best-Scoring Child

Setting `propagation` to `"max"` scores the object however well its best child did, regardless of weight. Use it when a record counts as correct as long as at least one field was extracted right, e.g. any one of several alternate identifiers for the same institution:

```python
field_metrics = FieldMetricsConfig(
    config={
        "type": "object",
        "metric_config": {"propagation": "max"},
        "fields": {
            "name": {"type": "leaf", "metric_config": {"metric": "text_similarity", "params": {"threshold": 0.85}}},
            "ror_id": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
        },
    }
)
```

Need a strategy that isn't one of these three? See [Custom Evaluation](./custom-evaluation.md#custom-aggregators).

```{rubric} What's next?
```

- The leaf metrics available for each child field: [Leaf Fields](./leaf-fields)
- Scoring a predicted array of objects like this one: [List Fields](./list-fields)
