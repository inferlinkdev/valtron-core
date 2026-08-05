# Evaluation Results

## Reading results after a run

After `evaluate()` (or `run()`), `experiment.results` is a list of one [`EvaluationResult`](../../api/evaluation_result) per model:

```python
experiment.evaluate()

for result in experiment.results:
    print(result.model, result.metrics.accuracy, result.metrics.total_cost)

    for prediction in result.predictions:
        print(" ", prediction.document_id, prediction.predicted_value, prediction.is_correct)
```

```text
gpt-4o-mini 0.92 0.0043
  1 positive True
  2 negative True
  3 neutral False
claude-haiku-4-5-20251001 0.88 0.0021
  1 positive True
  2 negative True
  3 positive False
```

`result.metrics` is an [`EvaluationMetrics`](../../api/evaluation_metrics) which provides accuracy, cost breakdown, timing, and additional information on the model result. Each entry in `result.predictions` is a [`PredictionResult`](../../api/prediction_result): one document's predicted value, cost, latency, and correctness. Transformer models also add a `confidence_score`, which [Combining Multiple Models](../combining-models) uses to find deferral thresholds.

## Output directory layout

`run(output_dir="./results")` writes:

```text
results/
├── metadata.json
├── models/
│   ├── gpt-4o.json
│   └── claude-sonnet-4-6.json
├── evaluation_report.html
├── detailed_analysis.html
└── evaluation_report.pdf        ← only if "pdf" in output_formats
```

`metadata.json` holds experiment-level state (timestamp, `use_case`, the original prompt, `field_metrics_config`, and the input documents). `models/<label>.json` holds one full serialized [`EvaluationResult`](../../api/evaluation_result) per model. This is exactly what `ModelEval.load_experiment_results()` reads back for [incremental evaluation](./evaluation-api.md#incremental-evaluation).

```{rubric} What's next?
```

- View the HTML and PDF reports: [Report Formats](./report-formats)
- Add new models to an existing run: [Evaluation API: Incremental Evaluation](./evaluation-api.md#incremental-evaluation)
