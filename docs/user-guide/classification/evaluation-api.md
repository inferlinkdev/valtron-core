# Evaluation API

Classification and extraction tasks share the same evaluation API. `ClassificationExperiment` and `ExtractionExperiment` both subclass [`ModelEval`](../../api/model_eval) and inherit the same methods and pipeline; they differ only in how they validate the config and data (see [Config Format](./config-format)). Examples on this page use `ClassificationExperiment`.

## Running an Evaluation

```python
from valtron_core.evaluation import ClassificationExperiment

experiment = ClassificationExperiment(config, data)
report_path = experiment.run(output_dir="./results")
```

`config` and `data` are covered in [Config Format](./config-format) and [Data Format](./data-format). `response_format`, an optional third constructor argument, is covered in [Structured Output Schema Inference](./config-format.md#structured-output-schema-inference).

`run()` executes these stages in order:

1. **Load and validate data**: parse documents and labels, validate config
2. **Preflight checks**:
   - Validate that structured-only manipulations have `response_format`
   - Warn when a model does not support `temperature` or `response_format` (those params are dropped at call time rather than erroring)
   - Detect auto-wrap: if a schema is configured with exactly one `label` field and all data labels are plain strings, labels are automatically wrapped as `{"label": value}` before scoring
   - Validate all JSON-structured labels (and auto-wrapped labels) against the schema; raises `ValueError` with failing record ids if any mismatch
3. **Generate few-shot examples**: if `few_shot.enabled` is true, generate and validate examples before evaluation starts
4. **Prepare per-model prompts**: apply manipulations to build the final prompt for each model
5. **Evaluate all models concurrently**: LLM calls run in parallel across models (up to `max_concurrent` per model)
6. **Compute metrics**: accuracy, cost, latency, and field-level scores per model
7. **Save results**: write `metadata.json` and `models/*.json` to `output_dir`
8. **Generate reports**: write HTML and/or PDF reports to `output_dir`

Alternatively, you can call evaluation and save method separately:

```python
experiment.evaluate()                                             # stages 1-6, populates experiment.results
experiment.save_experiment_results(output_dir="./results")        # stage 7
html_path = experiment.save_html_report(output_dir="./results")   # stage 8
pdf_path = experiment.save_pdf_report(output_dir="./results")
```

---

## Accessing Results

After `evaluate()` (or `run()`), `experiment.results` is a list of one [`EvaluationResult`](../../api/evaluation_result) per model:

```python
for result in experiment.results:
    print(result.model, result.metrics.accuracy, result.metrics.total_cost)

    for prediction in result.predictions:
        print(" ", prediction.document_id, prediction.predicted_value, prediction.is_correct)
```

See [Evaluation Results](./evaluation-results) for the full `EvaluationResult`/`EvaluationMetrics` schema.

---

## Incremental Evaluation

You can load a previously completed run to regenerate its report or re-evaluate it with new models. Re-evaluating only queries the newly added models.

See the full example in [Incremental Evaluation](../../examples/incremental-evaluation).

**Step 1: Load an existing run**

```python
experiment = ClassificationExperiment.load_experiment_results("./results/my_run")
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

---

## Sync vs Async

Every method that runs models (`run`, `evaluate`) has an async twin (`arun`, `aevaluate`) for use inside an existing event loop, e.g. FastAPI or a Jupyter cell with `await`:

```python
report_path = await experiment.arun(output_dir="./results")
```

---

```{rubric} What's next?
```

- To understand the output schema, see [Evaluation Results](./evaluation-results).
- To view the HTML and PDF reports, see [Report Formats](./report-formats).
- To apply prompt strategies per model, see [Prompt Manipulations](../manipulations/index).
