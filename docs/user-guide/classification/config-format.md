# Config Format

The evaluation config is where you set up the experiment: which models to compare, what prompt they receive, and any manipulations to layer on top of each one.

You can pass a config as a Python dict or a path to a JSON file. It validates as [`ClassificationConfig`](../../api/classification_config), which extends `ModelEvalConfig`.

You can also generate configs like these with the [Configuration Wizard](../../getting-started/configuration-wizard) instead of hand-writing them.

## Basic Structure

A config is a plain dict, or a path to a JSON file with the same shape, with two required fields: `prompt` and `models`. Additional optional parameters can be provided.

A minimal config only needs those two fields and output location:

```python
from valtron_core.evaluation import ClassificationExperiment

config = {
    "prompt": "Classify the sentiment: {content}\n\nSentiment:",
    "models": [{"name": "gpt-4o-mini"}],
    "output_dir": "./results",
}

experiment = ClassificationExperiment(config=config, data=data)
experiment.run()
```

A more advanced config layers on more:

```python
config = {
    "prompt": "Classify the sentiment: {content}\n\nSentiment:",
    "use_case": "sentiment classification",
    "output_dir": "./results",
    "output_formats": ["html", "pdf"],
    "temperature": 0.0,
    "few_shot": {"enabled": True, "generator_model": "gpt-4o-mini", "num_examples": 30},
    "models": [
        {"name": "gpt-4o-mini", "label": "GPT-4o mini (baseline)"},
        {"name": "gpt-4o-mini", "label": "GPT-4o mini + few-shot", "prompt_manipulation": ["few_shot"]},
        {
            "name": "ollama/llama3.1",
            "label": "Llama 3.1 (local)",
            "cost_rate": 0.05,
            "cost_rate_time_unit": "1hr",
        },
    ],
}

experiment = ClassificationExperiment(config=config, data=data)
report_path = experiment.run()
```

Model-level LLM parameters (temperature, max_tokens, and other Chat Completions/Responses API fields) go under `params` on that model's entry:

```python
{"name": "gpt-4o-mini", "label": "GPT-4o mini (low temp)", "params": {"temperature": 0.0, "max_tokens": 256}}
```

Full field lists are on [`ModelEvalConfig`](../../api/model_eval_config) and [`LLMModelConfig`](../../api/llm_model_config).

[Transformer Models](../self-hosting/transformer-models) slots into the same `models` list via a [`TransformerModelConfig`](../../api/transformer_model_config):

```python
config["models"].append({
    "type": "transformer",
    "label": "distilbert-sentiment",
    "model_path": "./transformer_models/final_model",
    "cost_rate": 0.50,
})
```

---

## Structured Outputs

`response_format_schema` configures structured output. This forces models to return JSON matching the provided schema instead of free text, and Valtron scores it field by field instead of by exact string match. See [Extraction](../extraction/index).


Structured outputs can also be set by passing a Pydantic class as `response_format` to `ClassificationExperiment(...)`.
Passing a Pydantic class as `response_format` to `ClassificationExperiment(...)` is the usual way to set a schema. The same schema can also travel inside the config itself as `response_format_schema`, in litellm's JSON schema format, for example a config saved to disk or generated automatically by the [configuration wizard](../../getting-started/configuration-wizard):

```python
config["response_format_schema"] = {
    "type": "json_schema",
    "json_schema": {
        "name": "ResponseModel",
        "strict": True,
        "schema": {
            "type": "object",
            "title": "ResponseModel",
            "properties": {
                "label": {"type": "string", "description": "Predicted class label"}
            },
            "required": ["label"],
            "additionalProperties": False,
        },
    },
}
```

The schema is stored under `response_format_schema` in `metadata.json` for every run, so results can be replayed with the exact schema that produced them.

A Pydantic `response_format` passed to the constructor takes priority over `config.response_format_schema`, which takes priority over the schema loaded from a previous run's metadata. See [Data Format: Label Format](./data-format.md#label-format) for what happens to your labels once a schema is in play.

### Structured Output Schema Inference

By default `ClassificationConfig` infers a response format schema based on the unique label values in your data. `infer_schema: False` opts out of this behavior and gets the model's raw, unconstrained text output instead.

```python
config = {
    "prompt": "Classify the sentiment: {content}\n\nSentiment:",
    "models": [{"name": "gpt-4o-mini"}],
    "infer_schema": False,
}

experiment = ClassificationExperiment(config=config, data=data)
```
---

## Passing Config to ClassificationExperiment

`config` can be passed as a Python dict, as shown throughout this page, or as a path to a JSON file with the same shape, useful for configs generated by the [configuration wizard](../../getting-started/configuration-wizard) or checked into version control:

```python
experiment = ClassificationExperiment(config={"models": [...], "prompt": "..."}, data=data)
experiment = ClassificationExperiment(config="./config.json", data="./data.json")
```

---

```{rubric} What's next?
```

- To run your evaluation, see [Evaluation API](./evaluation-api).
- To apply prompt strategies per model, see [Prompt Manipulations](../manipulations/index).
- For field-level scoring on structured extraction, see [Field Metrics](../extraction/field-metrics/index).
