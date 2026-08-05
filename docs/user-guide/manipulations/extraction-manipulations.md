# Extraction-Based Manipulations

`decompose`, `hallucination_filter`, and `multi_pass` all operate on a structured response, so they only make sense in extraction mode, where a schema is already in play. Each one requires `response_format` and raises a validation error without it:

```python
experiment = ModelEval(config=config, data=data, response_format=MySchema)
```

## Field Decomposition

Take a schema with five fields where a single call keeps missing one or two of them:

```python
from pydantic import BaseModel
from valtron_core.evaluation import ModelEval

class Institution(BaseModel):
    name: str
    city: str
    state: str
    country: str

class AffiliationResult(BaseModel):
    institutions: list[Institution]

config = {
    "prompt": "List all institutions in the following affiliation string.\n\n{content}",
    "models": [
        {"name": "gpt-4o", "label": "baseline"},
        {
            "name": "gpt-4o",
            "label": "+ decompose",
            "prompt_manipulation": ["decompose"],
            "decompose_config": {"rewrite_model": "gpt-4o-mini"},
        },
    ],
}

experiment = ModelEval(config=config, data=data, response_format=AffiliationResult)
experiment.run("./results/decompose_ablation")
```

`decompose` finds the list field (`institutions`) as the "split point," generates one focused sub-prompt per entity field via `rewrite_model` (or your own overrides in `decompose_config.sub_prompts`), runs one call per field per document, and merges the results back into the full schema before scoring. `+ decompose` costs five calls per document here instead of one; whether that trade is worth it is exactly what the side-by-side report answers. `decompose_config` validates as [`DecomposeConfig`](../../api/decompose_config).

## Hallucination Filtering

`hallucination_filter` needs no config. After the model responds, every predicted string value is checked against the source document text; anything not found there is dropped:

```python
{"name": "gpt-4o", "prompt_manipulation": ["hallucination_filter"]}
```

If the source document is `"Founded by Marie Curie in Paris."` and the model predicts `{"name": "Marie Curie", "city": "Paris", "country": "France"}`, `"France"` never appears in the source text, so it gets set to `null` even though it's factually correct: the filter checks *presence in the document*, not real-world truth. That trade (higher precision, lower recall on inferred-but-correct values) is the reason to reach for it specifically when a model tends to invent or infer values, not as a default. It adds no extra model calls.

## Multi-Pass Reconciliation

`multi_pass` needs no config. It calls the model twice with the same prompt and reconciles the two outputs, deduplicating overlapping items and merging anything one pass caught that the other missed:

```python
{"name": "gpt-4o", "prompt_manipulation": ["multi_pass"]}
```

Useful for variable-length list extraction where the model is inconsistent about how many items it returns run to run, at 2x the inference cost and latency, since both calls always happen.

```{rubric} What's next?
```

- The manipulations that don't need a schema: [Universal Manipulations](./universal-manipulations)
- Field-level scoring for the schema these manipulations operate on: [Field Metrics](../extraction/field-metrics/index)
