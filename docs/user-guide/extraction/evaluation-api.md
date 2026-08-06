# Evaluation API

`ExtractionExperiment` shares its pipeline with `ClassificationExperiment`: both subclass [`ModelEval`](../../api/model_eval) and inherit the same methods, differing only in config and data validation. Examples below use `ExtractionExperiment`.

## Running an Evaluation

```python
from valtron_core.evaluation import ExtractionExperiment

experiment = ExtractionExperiment(config, data, response_format=AffiliationResult)
report_path = experiment.run(output_dir="./results")
```

`run()` executes the same eight stages as [Classification Evaluation API: Running an Evaluation](../classification/evaluation-api.md#running-an-evaluation), including the `evaluate()` / `save_*` split.

## Accessing Results

`experiment.results` populates the same way as classification, one [`EvaluationResult`](../../api/evaluation_result) per model. See [Evaluation Results](./evaluation-results) for what `predicted_value` and `aggregated_field_metrics` look like once a schema is involved.

## Incremental Evaluation

Loading, adding models, and rerunning follows the same three steps as [Classification Evaluation API: Incremental Evaluation](../classification/evaluation-api.md#incremental-evaluation):

```python
experiment = ExtractionExperiment.load_experiment_results("./results/my_run")
experiment.add_models([{"name": "claude-sonnet-4-6"}])
experiment.run("./results/my_run")
```

The schema saved in `metadata.json` is restored automatically, so the reloaded experiment scores new models against the same schema the original run used.

## Sync vs Async

Same async twins (`arun`, `aevaluate`) as [Classification Evaluation API: Sync vs Async](../classification/evaluation-api.md#sync-vs-async).

```{rubric} What's next?
```

- To understand the output schema, see [Evaluation Results](./evaluation-results).
- To view the field-level breakdown in the reports, see [Report Formats](./report-formats).
