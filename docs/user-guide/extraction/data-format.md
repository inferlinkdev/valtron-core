# Data Format

Every model you evaluate is scored against the same `data`. Each document in it needs three things: a unique `id`, the `content` the model sees, and the `label` it should have produced, here shaped as a dict or JSON string matching your schema instead of a plain string. [`ExtractionExperiment`](../../api/extraction_experiment) builds each model's input from that `content`, and checks its output field by field against `label`.

## Label Format

`label` is a dict, or a JSON string, matching your response schema, instead of a plain string:

```python
{
    "id": "1",
    "content": "Apple Inc. was founded in Cupertino, California.",
    "label": {"name": "Apple Inc.", "city": "Cupertino", "state": "California"},
}
```

A JSON string (`'{"name": "Apple Inc.", ...}'`) works the same as a dict; Valtron normalizes either to the same internal form before scoring.

A `response_format` schema is required for extraction tasks. Passing plain string labels with no schema raises a `ValueError` pointing at `ClassificationExperiment` instead.

## Validation

Every label is validated against the schema before evaluation starts. A mismatch raises `ValueError` naming the failing record `id`s.

```{rubric} What's next?
```

- Configuring the schema itself: [Config Format](./config-format)
- Field-level scoring: [Field Metrics](./field-metrics/index)
