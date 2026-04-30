---
sidebar_position: 4
---

# Config Format

The config controls which models to run, the prompt template, evaluation options, and output settings. You can pass it as a Python dict or a path to a JSON file. See [Data Format](./data-format) for the input schema and [Field Metrics](../field-metrics) for per-field scoring on structured extraction.

## Top-level fields

| Field | Type | Default | Description |
|---|---|---|---|
| `models` | `array` | required | List of model configs (see below) |
| `prompt` | `string` | required | Prompt template with placeholders. Use `{content}` for string document content, or any named key (e.g. `{text}`, `{topic}`) when `content` is a dict. See [Data Format: content placeholders](./data-format#how-content-fills-your-prompt). |
| `output_dir` | `string` | `null` | Directory to write results to |
| `use_case` | `string` | `"evaluation"` | Describes your task. Used in the report header and passed directly to the LLM that generates the AI recommendation (e.g. `"sentiment classification"`, `"medical entity extraction"`) |
| `temperature` | `float` | `0.0` | Default temperature for all models (can be overridden per model) |
| `few_shot` | `object` | `null` | Few-shot generation config (see below) |
| `field_metrics_config` | `object` | `null` | Field-level scoring for structured extraction (see below) |
| `disable_auto_response_format` | `boolean` | `false` | Set `true` to disable auto-enum and use free-text mode. See [Classification mode](#classification-mode). |
| `output_formats` | `array[string]` | `["html"]` | Report formats to generate. Acceptable formats include: `"html"`, `"pdf"` |

```json
{
  "models": [...],
  "prompt": "Classify the sentiment: {content}\n\nSentiment:",
  "output_dir": "./results",
  "use_case": "sentiment classification",
  "temperature": 0.0,
  "output_formats": ["html", "pdf"]
}
```

---

## Classification mode

When `label` fields in your dataset are plain strings (not JSON), Valtron automatically builds a `Literal` enum from all unique label values and uses it as the LLM's required output schema. This constrains the model to return one of the known classes exactly, reducing hallucinations and making correctness checking unambiguous.

For a dataset with labels `"positive"`, `"negative"`, and `"neutral"`, the generated schema looks like:

```python
class ResponseModel(BaseModel):
    label: Literal["negative", "neutral", "positive"]
```

The schema is serialized and stored in `metadata.json` under `response_format_schema` for every run.

**Cardinality guard**: if your dataset has more than 50 unique label values, evaluation raises a `ValueError` before any LLM call is made.

**Opt out**: set `disable_auto_response_format: true` to disable enum generation entirely. The LLM will return free text and correctness is determined by string equality. The >50-value cardinality guard is also suppressed.

```json
{
  "disable_auto_response_format": true
}
```

An explicit `response_format` passed to `ModelEval(...)` always takes priority; auto-enum logic is skipped entirely when one is provided.

See also: [`label` field in Data Format](./data-format#label-format).

---

## Model config

Each entry in `models` is either an LLM model or a transformer model.

### LLM model

| Field | Type | Default | Description |
|---|---|---|---|
| `name` | `string` | required | LiteLLM model identifier (e.g. `gpt-4o`, `claude-sonnet-4-6`, `ollama/llama3`) |
| `label` | `string` | `null` | **Unique identifier** used in results, output files, and the report. Defaults to `name`. Must be set when the same `name` appears more than once (e.g. same model with different manipulations). |
| `params` | `object` | `{}` | Extra LiteLLM parameters (e.g. `{"temperature": 0.2, "max_tokens": 512}`) |
| `prompt_manipulation` | `array[string]` | `[]` | Optimizer strategies to apply (see [Optimizers](./optimizers)) |
| `decompose_config` | `object` | `null` | Required when using `"decompose"` (see [Optimizers](./optimizers#decompose)) |
| `cost_rate` | `float` | `null` | Override token-based cost with a fixed rate (e.g. hourly server cost) |
| `cost_rate_time_unit` | `string` | `"1hr"` | Unit for `cost_rate`: `"1hr"`, `"1min"`, etc. |
| `prompt` | `string` | `null` | Per-model prompt override; replaces the top-level `prompt` for this model only |

```json
{
  "name": "gpt-4o",
  "label": "GPT-4o (with few-shot)",
  "params": {"temperature": 0.0, "max_tokens": 256},
  "prompt_manipulation": ["few_shot", "explanation"]
}
```

**Per-model prompt override**: useful for testing different phrasings per model:
```json
{
  "name": "claude-sonnet-4-6",
  "prompt": "You are a sentiment classifier. Output only: positive, negative, or neutral.\n\n{content}"
}
```
The report will show a "Base" / "Overridden" toggle for models with a custom prompt.

### Transformer model

| Field | Type | Default | Description |
|---|---|---|---|
| `type` | `string` | required | Must be `"transformer"` |
| `label` | `string` | required | **Unique identifier** used in results, output files, and the report |
| `model_path` | `string` | required | Path to the `final_model/` directory from training |
| `cost_rate` | `float` | `null` | Fixed cost rate (usually your server's hourly cost) |
| `cost_rate_time_unit` | `string` | `"1hr"` | Unit for `cost_rate` |

```json
{
  "type": "transformer",
  "label": "distilbert-sentiment",
  "model_path": "./transformer_models/final_model",
  "cost_rate": 0.50,
  "cost_rate_time_unit": "1hr"
}
```

See [Transformer Models](./transformer-models) for how to train a model.

---

## Few-shot config

Controls automatic few-shot example generation. When enabled, Valtron generates synthetic document+label pairs from your seed data and prepends the best examples into prompts for any model that has `"few_shot"` in its `prompt_manipulation` list.

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `false` | Enable few-shot generation |
| `generator_model` | `string` | `"gpt-4o-mini"` | LLM used to generate synthetic examples |
| `num_examples` | `int` | `50` | Number of synthetic examples to attempt generating |
| `max_seed_examples` | `int` | `10` | Max real data rows to use as seeds |
| `max_few_shots` | `int` | `10` | Max examples injected into each prompt |

```json
{
  "few_shot": {
    "enabled": true,
    "generator_model": "gpt-4o-mini",
    "num_examples": 30,
    "max_few_shots": 5
  }
}
```

---

## Field metrics config

Enables field-level precision/recall/F1 scoring for structured extraction. See **[Field Metrics](./field-metrics)** for the full reference: metric types, `comparator` params, list matching options, and how to define custom metrics and aggregators.

Quick example for a schema like `{"name": str, "institutions": [{"city": str, "country": str}]}`:

```json
{
  "field_metrics_config": {
    "type": "object",
    "fields": {
      "name": {
        "type": "leaf",
        "metric_config": {
          "metric": "comparator",
          "params": {"element_compare": "text_similarity", "text_similarity_threshold": 0.8}
        }
      },
      "institutions": {
        "type": "list",
        "fields": {
          "city": {"type": "leaf", "metric_config": {"metric": "exact"}},
          "country": {"type": "leaf", "metric_config": {"metric": "exact"}}
        }
      }
    }
  }
}
```

---

## Decompose config

Required when using the `"decompose"` prompt manipulation. Controls how the schema is split into sub-calls.

| Field | Type | Default | Description |
|---|---|---|---|
| `rewrite_model` | `string` | `"gpt-4o-mini"` | LLM used to generate per-field sub-prompts |
| `sub_prompts` | `object` | `null` | Manual prompt overrides per field (keys are field names, values are prompt templates containing `{content}`) |

```json
{
  "decompose_config": {
    "rewrite_model": "gpt-4o-mini",
    "sub_prompts": {
      "institutions": "Extract only the list of institutions from this text: {content}"
    }
  }
}
```

---

## Environment variables

All LLM credentials and runtime settings are read from environment variables (or a `.env` file).

### API keys

| Variable | Provider |
|---|---|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GOOGLE_API_KEY` | Google Gemini |
| `COHERE_API_KEY` | Cohere |
| `AZURE_API_KEY` | Azure OpenAI |
| `HUGGINGFACE_API_KEY` | HuggingFace |
| `REPLICATE_API_KEY` | Replicate |
| `TOGETHER_API_KEY` | Together AI |
| `AWS_ACCESS_KEY_ID` | AWS Bedrock |
| `AWS_SECRET_ACCESS_KEY` | AWS Bedrock |
| `OLLAMA_API_BASE` | Ollama (default: `http://localhost:11434`) |

### Runtime settings

| Variable | Default | Description |
|---|---|---|
| `MAX_RETRIES` | `3` | Retry count on transient failures |
| `RETRY_DELAY` | `1.0` | Seconds between retries |
| `REQUESTS_PER_MINUTE` | - | Rate limit cap; omit for no limit |
| `LOG_LEVEL` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR` |

---

## Passing config to ModelEval

**As a Python dict:**
```python
experiment = ModelEval(config={"models": [...], "prompt": "..."}, data=data)
```

**As a JSON file path:**
```python
experiment = ModelEval(config="./config.json", data="./data.json")
```

---

## What's next?

- Run your evaluation: [Evaluation API](./recipes)
- Apply prompt strategies per model: [Optimizers](../optimizers)
- For field-level scoring on structured extraction: [Field Metrics](../field-metrics)
