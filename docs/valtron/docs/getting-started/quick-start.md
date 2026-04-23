---
sidebar_position: 2
---

# Quick Start

This walks through a minimal end-to-end evaluation: two models, five documents, one report.

## 1. Prepare your data

Data is a list of dicts, each with an `id`, `content`, and `label`.

```python
data = [
    {"id": "1", "content": "The product arrived damaged and the support was unhelpful.", "label": "negative"},
    {"id": "2", "content": "Fast shipping, exactly what I ordered. Very happy!", "label": "positive"},
    {"id": "3", "content": "Average experience, nothing special.", "label": "neutral"},
    {"id": "4", "content": "Outstanding quality. Will definitely buy again.", "label": "positive"},
    {"id": "5", "content": "Wrong item sent. Refund process was painful.", "label": "negative"},
]
```

To load from a JSON file, see [Data Format](../data-format) for the full schema.

## 2. Write a config

The config specifies which models to run and the prompt template. The prompt must contain `{document}`, which is where each document's content gets inserted.

```python
config = {
    "models": [
        {"name": "gpt-4o-mini"},
        {"name": "claude-haiku-4-5-20251001"},
    ],
    "prompt": "Classify the sentiment of the following review as positive, negative, or neutral.\n\nReview: {document}\n\nSentiment:",
    "output_dir": "./results",
}
```

:::tip[Prefer a guided setup?]
Use the [Configuration Wizard](./configuration-wizard) to build your config interactively in a browser (no JSON editing required).
:::

See [Config Format](../config-format) for all options including prompt optimizers, cost overrides, and field metrics.

## 3. Run the evaluation

```python
from valtron_core.recipes import ModelEval

experiment = ModelEval(config=config, data=data)
report_path = experiment.run()

print(f"Report: {report_path}")
```

`run()` is synchronous. It evaluates all models, computes metrics, saves results to `output_dir`, and returns the path to the HTML report.

## 4. View the report

Open `./results/evaluation_report.html` in a browser. You'll see:

- An AI-generated recommendation for the best model
- Accuracy, cost, and speed comparison across models
- Per-document predictions for each model (in `detailed_analysis.html`)

## What's next?

- **Add prompt optimizers** — apply few-shot, chain-of-thought, or other strategies per model: [Prompt Optimizers](../optimizers)
- **Structured extraction** — extract nested JSON with field-level scoring: [Response Format](../response-format)
- **PDF output** — add `"output_formats": ["html", "pdf"]` to your config: [Report Formats](../report-formats)
- **Add a model later** — load a prior run and append a new model without re-running: [Recipes → Incremental Evaluation](../recipes#incremental-evaluation)
- **Run locally for free** — use Ollama: [Self-Hosted Models](../self-hosted-models)
