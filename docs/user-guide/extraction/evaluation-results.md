# Evaluation Results

Classification results center on a single `accuracy` number: each document's predicted label either matches the expected one or it doesn't. Extraction results are scored per field instead. `result.metrics.aggregated_field_metrics` breaks scoring down into precision, recall, and F1 for every field path in your schema, so a model that nails `city` but misses `country` shows up as two separate numbers instead of one blended score.

```python
experiment.evaluate()

for result in experiment.results:
    for path, field_result in result.metrics.aggregated_field_metrics.items():
        print(f"{path:<20} precision={field_result.precision:.0%}  recall={field_result.recall:.0%}")
```

```text
institutions.name    precision=91%  recall=88%
institutions.city    precision=97%  recall=95%
institutions.country precision=99%  recall=99%
```

`result.metrics.aggregated_field_metrics` holds one [`EvalResult`](../../api/eval_result) per field path, keyed the same way as `field_metrics_config`. Each [`PredictionResult`](../../api/prediction_result)'s `predicted_value` is a JSON string matching your schema instead of plain text, for example `'{"name": "Apple Inc.", "city": "Cupertino"}'`, and its `field_metrics` holds that one document's own `EvalResult` tree.

`prediction.is_correct` reflects the root `EvalResult.is_correct` once `field_metrics_config` is set. Without one, it falls back to a case-insensitive exact match between the two JSON strings, which rarely matches what you want beyond a single-field schema.

```{rubric} What's next?
```

- View the HTML and PDF reports: [Report Formats](./report-formats)
- Configure per-field comparators and list matching: [Field Metrics](./field-metrics/index)
