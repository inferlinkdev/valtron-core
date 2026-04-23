---
sidebar_position: 5
---

# Response Format

This page covers what Valtron expects from models and how results are stored on disk.

## Evaluation modes

### Label / classification mode

Used when no `response_format` is passed to `ModelEval`. The model is expected to return a plain string that matches the `label` field in your data.

```python
experiment = ModelEval(config=config, data=data)
```

- The model's full response text is compared against the `label` using string equality (or a custom comparator if configured in `field_metrics_config`)
- Score is binary: `1.0` (correct) or `0.0` (incorrect)
- Works with any model type including transformers

### Structured extraction mode

Used when a Pydantic `BaseModel` is passed as `response_format`. The model returns structured JSON matching the schema, which is then compared field-by-field against the `label` (also a JSON string).

```python
from pydantic import BaseModel

class Institution(BaseModel):
    name: str
    city: str
    country: str

class ExtractionResult(BaseModel):
    institutions: list[Institution]

experiment = ModelEval(config=config, data=data, response_format=ExtractionResult)
```

Structured mode enables the `decompose`, `hallucination_filter`, and `multi_pass` optimizers, and unlocks field-level metrics (precision/recall/F1 per field).

---

## Metrics

### EvaluationMetrics (per model)

After evaluation, each model has an `EvaluationMetrics` object:

| Field | Type | Description |
|---|---|---|
| `accuracy` | `float` | Fraction of documents with a correct prediction (0–1) |
| `average_example_score` | `float` | Mean continuous score across documents (0–1); uses field-level scores if available |
| `total_documents` | `int` | Total number of documents evaluated |
| `correct_predictions` | `int` | Number of exact matches |
| `total_cost` | `float` | Total API cost in USD |
| `average_cost_per_document` | `float` | Mean cost per document |
| `total_time` | `float` | Total response time in seconds |
| `average_time_per_document` | `float` | Mean latency per document |
| `aggregated_field_metrics` | `dict` | Per-field `EvalResult` (structured mode only) |

### EvalResult (field-level scoring)

Each field in a structured schema gets an `EvalResult`:

| Field | Type | Description |
|---|---|---|
| `path` | `string` | Dot-separated field path (e.g. `root.institutions.city`) |
| `score` | `float` | Field score (0–1) |
| `is_correct` | `bool` | Whether score meets the correctness threshold |
| `precision` | `float` | TP / (TP + FP) |
| `recall` | `float` | TP / (TP + FN) |
| `tp` / `fp` / `fn` | `float` | True/false positives and false negatives |
| `metric` | `string` | Comparison method used |
| `params` | `dict` | Parameters passed to the metric |
| `children` | `dict[str, EvalResult]` | Nested results for object/list fields |
| `alignment` | `list` | Item-level matching results for list fields |

---

## Output directory layout

When you call `run(output_dir="./results")`, Valtron writes:

```
results/
├── metadata.json
├── models/
│   ├── gpt-4o.json
│   └── claude-sonnet-4-6.json
├── evaluation_report.html
├── detailed_analysis.html
├── chart_accuracy.png
├── chart_cost.png
├── chart_time.png
└── evaluation_report.pdf        ← only if "pdf" in output_formats
```

### `metadata.json`

Contains experiment-level information:

```json
{
  "timestamp": "2024-01-15T10:30:00",
  "use_case": "sentiment classification",
  "original_prompt": "Classify the sentiment: {document}\n\nSentiment:",
  "field_config": { ... },
  "documents": [
    {
      "id": "1",
      "content": "...",
      "label": "positive",
      "attachments": []
    }
  ]
}
```

### `models/<model-label>.json`

One file per model. Contains the full evaluation result:

```json
{
  "run_id": "abc123",
  "model": "gpt-4o",
  "started_at": "2024-01-15T10:30:01",
  "completed_at": "2024-01-15T10:30:45",
  "status": "completed",
  "prompt_template": "Classify the sentiment: {document}\n\nSentiment:",
  "prompt_manipulations": ["few_shot"],
  "override_prompt": null,
  "llm_config": {
    "model": "gpt-4o",
    "temperature": 0.0
  },
  "metrics": {
    "accuracy": 0.92,
    "average_example_score": 0.92,
    "total_cost": 0.0043,
    "total_time": 12.4,
    "average_time_per_document": 1.24,
    "average_cost_per_document": 0.00043,
    "total_documents": 100,
    "correct_predictions": 92,
    "aggregated_field_metrics": {}
  },
  "predictions": [
    {
      "document_id": "1",
      "predicted_value": "positive",
      "expected_value": "positive",
      "is_correct": true,
      "example_score": 1.0,
      "response_time": 1.1,
      "cost": 0.000041,
      "model": "gpt-4o",
      "field_metrics": null
    }
  ]
}
```

The `override_prompt` field is only present when the model has a per-model prompt override configured.

---

## Loading results programmatically

You can reload a saved run without re-evaluating:

```python
experiment = ModelEval.load_experiment_results("./results")

# Access results
for result in experiment.results:
    print(result.model, result.metrics.accuracy, result.metrics.total_cost)

# Regenerate the report
experiment.save_html_report("./results")
```

See [Recipes → Incremental Evaluation](./recipes#incremental-evaluation) for adding new models to an existing run.
