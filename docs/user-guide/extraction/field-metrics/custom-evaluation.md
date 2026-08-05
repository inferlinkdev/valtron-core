# Custom Evaluation

When none of the built-in leaf metrics or aggregation strategies fit, register your own. Both are plain callables passed to [`FieldMetricsConfig`](../../../api/field_metrics_config).

## Custom Metrics

Say correctness for a `code` field means the first four characters match, regardless of what follows:

```python
from valtron_core.models import FieldMetricsConfig
from valtron_core.evaluation import ModelEval

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
                "metric_config": {"metric": "starts_with", "params": {"prefix_length": 4}},
            }
        },
    },
    custom_metrics={"starts_with": starts_with_metric},
)

experiment = ModelEval(
    config={"models": [{"name": "gpt-4o-mini"}], "prompt": "...", "field_metrics_config": field_metrics},
    data=data,
    response_format=MySchema,
)
```

Reference the metric by name in `metric_config`, the same as any built-in metric, and register the callable under that same name in `custom_metrics`. The signature is fixed: `(expected, actual, params) -> (score, is_correct)`, where `score` is a float between `0.0` and `1.0`.

## Custom Aggregators

Object nodes support `"weighted_avg"`, `"min"`, and `"max"` out of the box (see [Object Fields](./object-fields)). Register your own the same way for anything else, e.g. a harmonic mean that penalizes any single weak field harder than a plain average would:

```python
from valtron_core.scoring.json_eval import EvalResult

def harmonic_mean_agg(results: list[EvalResult]) -> float:
    scores = [r.score for r in results if r.score > 0]
    if not scores:
        return 0.0
    return len(scores) / sum(1.0 / s for s in scores)

field_metrics = FieldMetricsConfig(
    config={
        "type": "object",
        "metric_config": {"propagation": "harmonic_mean"},
        "fields": {
            "name": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
            "city": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
        },
    },
    custom_aggs={"harmonic_mean": harmonic_mean_agg},
)
```

Reference the strategy by name in `propagation`, the same as any built-in strategy, and register the callable under that same name in `custom_aggs`. The signature: `(results: list[EvalResult]) -> float`, one entry in `results` per child field, returning the rolled-up object score.

```{rubric} What's next?
```

- See the field-level breakdown these produce in the generated reports: [Report Formats](../report-formats)
