# Common Pitfalls

Warnings and gotchas that are easy to miss because they surface as silent behavior changes or log warnings rather than errors.

## `temperature` silently dropped for unsupported models

```python
{"name": "o1-mini", "params": {"temperature": 0.0}}  # o1-mini doesn't accept temperature
```

Not every model accepts a `temperature` parameter. Valtron logs a warning (`temperature_not_supported`) and drops it rather than failing the run. See [`ModelEvalConfig`](../api/model_eval_config).

## `response_format` silently skipped for unsupported models

```python
experiment = ModelEval(config=config, data=data, response_format=MySchema)
# a model in `config["models"]` that doesn't support structured output logs
# `response_format_not_supported` and falls back to unstructured text for that call
```

The run continues, but that model's predictions won't match the schema you expect. Check your provider's structured-output support before relying on [extraction mode](./extraction/index) for a given model.

## Plain string labels + a multi-field `response_format`

```python
data = [{"id": "1", "content": "...", "label": "positive"}]  # plain string

class MultiField(BaseModel):
    label: str
    confidence: float

experiment = ModelEval(config=config, data=data, response_format=MultiField)
# logs a warning and scores unreliably, since MultiField has two fields, not one,
# so "positive" can't be auto-wrapped into it
```

The one case handled automatically is a schema with exactly one `label` field. See [Extraction Config Format](./extraction/config-format.md).

## Duplicate model `name` without a unique `label`

```python
"models": [
    {"name": "gpt-4o-mini"},
    {"name": "gpt-4o-mini", "prompt_manipulation": ["few_shot"]},  # overwrites the first!
]
```

Give the second entry an explicit `label` (`"gpt-4o-mini + few_shot"`) or its results silently overwrite the first's in `experiment.results` and in `models/*.json`. See [`LLMModelConfig`](../api/llm_model_config).

## Structured-only manipulations fail fast in classification mode

```python
experiment = ModelEval(
    config={"models": [{"name": "gpt-4o", "prompt_manipulation": ["decompose"]}], "prompt": "..."},
    data=data,
    # no response_format ->
)
# ValueError: 'decompose' requires response_format
```

This one's a hard failure at construction time, not a silent gotcha. It's intentional: none of `decompose`, `hallucination_filter`, and `multi_pass` mean anything without a schema. See [Extraction-Based Manipulations](./manipulations/extraction-manipulations).

## Expensive comparisons in field metrics

```python
{"type": "list", "metric_config": {"ordered": False}, "fields": {
    "description": {"type": "leaf", "metric_config": {"metric": "llm", "params": {}}}
}}
# raises: "llm"/"embedding" in an unordered list means N x M API calls,
# so add "allow_expensive_comparisons_for": ["description"] to opt in explicitly
```

See [List Fields: Expensive comparisons](./extraction/field-metrics/list-fields.md#expensive-comparisons).

## `TradeoffAnalyzer` is binary-only, today

```python
# data with three labels: "positive" / "negative" / "neutral"
analyzer = TradeoffAnalyzer.from_model_eval(experiment)
# ValueError: ground truth has more than two unique labels
```

Multi-class cascades aren't supported yet. See [Chapter 6: Constraints](./combining-models.md#constraints).

## Requirements checklist scored but never shown to the candidate

```python
config = {
    "prompt": "Summarize the document.\n\n{content}",  # no {requirements} placeholder
    "requirements": ["Name the parties.", "State the outcome."],
    "judge_model": "gpt-4o",
    "models": [{"name": "gpt-4o-mini"}],
}
# logs `requirements_not_in_prompt` and still grades against the checklist,
# but the candidate is never told what's on it
```

Valtron doesn't fail the run, but scoring a candidate against criteria it never saw rarely matches what you want. Add a `{requirements}` placeholder to the prompt (`SALIENCE_SUMMARY_PROMPT` has one built in) so the checklist is rendered into what the candidate actually reads. See [Summarization Config Format: The Requirements Checklist](./summarization/config-format.md#the-requirements-checklist).

## The `transformers` extra isn't installed by default

```python
config = {"models": [{"type": "transformer", "label": "clf", "model_path": "./model"}], "prompt": "..."}
experiment = ModelEval(config=config, data=data)
experiment.run()
# fails when the transformer model is actually evaluated, not at construction time
```

```bash
pip install "valtron-core[transformers]"  # torch, transformers, scikit-learn, datasets, accelerate
```

Install the extra up front if you know you'll need it. A config referencing `"type": "transformer"` parses fine either way.

```{rubric} What's next?
```

- Still not sure which chapter covers your case? [Choosing the Right Approach](./choosing-the-right-approach)
