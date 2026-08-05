# Config Format

The evaluation config is where you set up the experiment: which models to compare, what prompt they receive, and any manipulations to layer on top of each one.

You can pass a config as a Python dict or a path to a JSON file. It validates as a plain [`ModelEvalConfig`](../../api/model_eval_config).

You can also generate configs like these with the [Configuration Wizard](../../getting-started/configuration-wizard) instead of hand-writing them.

## Basic Structure

Similar to classification, a config is a plain dict, or a path to a JSON file with the same shape, and the same fields are available: `prompt`, `models`, `output_dir`, `few_shot`, `temperature`, and so on. The main difference is that `response_format` is required.

```python
from valtron_core.evaluation import ExtractionExperiment

config = {
    "prompt": "List all institutions in the following affiliation string.\n\n{content}",
    "models": [{"name": "gpt-4o"}],
    "output_dir": "./results",
}

experiment = ExtractionExperiment(config=config, data=data, response_format=AffiliationResult)
experiment.run()
```

Model-level LLM parameters work the same way here, under `params` on the model entry:

```python
{"name": "gpt-4o", "params": {"temperature": 0.0, "max_tokens": 256}}
```

Full field lists are on [`ModelEvalConfig`](../../api/model_eval_config) and [`LLMModelConfig`](../../api/llm_model_config).

## Structured Outputs

This follows the same pattern as [Classification Config Format: Structured Outputs](../classification/config-format.md#structured-outputs): pass a Pydantic class as `response_format` to `ExtractionExperiment(...)`, as above, or set `response_format_schema` in the config itself, in litellm's JSON schema format, for example a config saved to disk or generated automatically by the [configuration wizard](../../getting-started/configuration-wizard):

```python
config["response_format_schema"] = {
    "type": "json_schema",
    "json_schema": {
        "name": "AffiliationResult",
        "strict": True,
        "schema": {
            "type": "object",
            "title": "AffiliationResult",
            "properties": {
                "institutions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "city": {"type": "string"},
                            "country": {"type": "string"},
                        },
                        "required": ["name", "city", "country"],
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["institutions"],
            "additionalProperties": False,
        },
    },
}
```

The schema is stored under `response_format_schema` in `metadata.json` for every run, so results can be replayed with the exact schema that produced them.

A Pydantic `response_format` passed to the constructor takes priority over `config.response_format_schema`, which takes priority over the schema loaded from a previous run's metadata. See [Data Format: Label Format](./data-format.md#label-format) for what happens to your labels once a schema is in play.

```{rubric} What's next?
```

- To run your evaluation, see [Evaluation API](./evaluation-api).
- For field-level scoring configuration, see [Field Metrics](./field-metrics/index).
