---
sidebar_position: 9
---

# Evaluation API

`ModelEval` is Valtron's primary API. It ties together your data, config, and models to run the full evaluation pipeline. See [Data Format](./data-format) and [Config Format](./config-format) for the inputs it expects.

## ModelEval

`ModelEval` takes a config and dataset, runs all models, computes metrics, and generates reports.

### Constructor

```python
from valtron_core.recipes import ModelEval

experiment = ModelEval(
    config,                  # dict, JSON file path, or ModelEvalConfig
    data,                    # list of dicts or JSON file path
    response_format=None,    # optional Pydantic BaseModel for structured extraction
)
```

| Parameter | Type | Description |
|---|---|---|
| `config` | `dict \| str \| Path` | Evaluation config (see [Config Format](./config-format)) |
| `data` | `list \| str \| Path` | Input documents and labels (see [Data Format](./data-format)) |
| `response_format` | `type[BaseModel] \| None` | Pydantic schema for structured extraction; enables decompose/hallucination_filter/multi_pass |

---

## Sync vs async

All primary methods have both synchronous and asynchronous variants.

**Use the sync variants in scripts:**

```python
experiment = ModelEval(config=config, data=data)

# Runs the full pipeline: evaluate → save results → generate report
report_path = experiment.run(output_dir="./results")

# Or step by step:
experiment.evaluate()
experiment.save_html_report("./results")
experiment.save_pdf_report("./results")
```

`run()` and `evaluate()` call `asyncio.run()` internally. Calling them from within an existing event loop raises `RuntimeError`.

**Use the async variants when inside an existing async context** (FastAPI, Jupyter with `await`, etc.):

```python
import asyncio

async def run_experiment():
    experiment = ModelEval(config=config, data=data)

    # Full pipeline
    report_path = await experiment.arun(output_dir="./results")

    # Or step by step
    await experiment.aevaluate()
    experiment.save_html_report("./results")

asyncio.run(run_experiment())
```

| Sync method | Async equivalent | Description |
|---|---|---|
| `run(output_dir)` | `arun(output_dir)` | Full pipeline: evaluate + save + report |
| `evaluate()` | `aevaluate()` | Evaluate all models, populate `self.results` |

---

## Pipeline stages

When you call `run()`, Valtron executes these stages in order:

1. **Load and validate data**: parse documents and labels, validate config
2. **Validate manipulations**: check that structured-only manipulations have `response_format`
3. **Generate few-shot examples**: if `few_shot.enabled` is true, generate and validate examples before evaluation starts
4. **Prepare per-model prompts**: apply manipulations to build the final prompt for each model
5. **Evaluate all models concurrently**: LLM calls run in parallel across models (up to `max_concurrent` per model)
6. **Compute metrics**: accuracy, cost, latency, and field-level scores per model
7. **Save results**: write `metadata.json` and `models/*.json` to `output_dir`
8. **Generate reports**: write HTML and/or PDF reports to `output_dir`

---

## Incremental evaluation

You can load a previously completed run, add new models, and re-evaluate; only the new models are called. Existing results are preserved and the report is regenerated with all models combined.

**Step 1: Load an existing run**

```python
experiment = ModelEval.load_experiment_results("./results/my_run")
```

This loads the saved `models/*.json` files into `experiment.results`.

**Step 2: Add new models**

```python
experiment.add_models([
    {"name": "claude-sonnet-4-6", "label": "Claude Sonnet"},
    {"name": "gemini-1.5-flash"},
])
```

**Step 3: Run**

```python
experiment.run("./results/my_run")
```

Valtron checks which models already have results in `self.results` and skips them. Only the newly added models are evaluated. Afterward, the report is regenerated with all models.

**Calling `evaluate()` a second time on an unmodified experiment is a no-op** because all models are already in `self.results`.

---

## Saving results and reports manually

After `evaluate()` or `aevaluate()`, you can save results and reports independently:

```python
experiment.evaluate()

# Save raw JSON results
experiment.save_experiment_results(output_dir="./results")

# Generate reports
html_path = experiment.save_html_report(output_dir="./results")
pdf_path = experiment.save_pdf_report(output_dir="./results")
```

---

## Accessing results programmatically

After evaluation, `experiment.results` is a list of `EvaluationResult` objects:

```python
experiment.evaluate()

for result in experiment.results:
    print(result.model)
    print(result.metrics.accuracy)
    print(result.metrics.total_cost)
    print(result.metrics.average_time_per_document)

    for prediction in result.predictions:
        print(prediction.document_id, prediction.predicted_value, prediction.is_correct)
```

See [Evaluation Results](./evaluation-results) for the full schema of `EvaluationResult` and `EvaluationMetrics`.

---

## What's next?

- Understand the output schema: [Evaluation Results](./evaluation-results)
- View the HTML and PDF reports: [Report Formats](./report-formats)
- Apply prompt strategies per model: [Optimizers](./optimizers)
