# Evaluation API

`SummarizationExperiment` shares the same constructor, `run`/`evaluate` pipeline shape, and incremental evaluation as the other recipes, but replaces the reports and the notion of a "result" with its own.

## Running an Evaluation

```python
from valtron_core.evaluation import SummarizationExperiment

experiment = SummarizationExperiment(config, data)
report_path = experiment.run(output_dir="./results")
```

`config` and `data` are covered in [Config Format](./config-format) and [Data Format](./data-format). `run()` executes these stages in order:

1. **Load and validate data**: parse documents (`id` and `content`), validate config
2. **Preflight checks**: reject non-string or blank `content`, a prompt missing `{content}`, and any transformer models in `models`; warn if a requirements checklist has no `{requirements}` placeholder to render into
3. **Extract shared document facts**: the judge decomposes every document into atomic facts and marks which are must-convey, once per document, shared by every candidate model
4. **Prepare per-model prompts**: fill the `{requirements}` placeholder into each model's prompt
5. **Evaluate all models concurrently**: each candidate writes a summary and the judge grades it on the four axes
6. **Compute the ranking**: axes are averaged across the whole document set and scored once under `salience-f+reqs`, then tiered
7. **Save results**: write `metadata.json` and `models/*.json` to `output_dir`
8. **Generate reports**: write HTML and/or PDF reports to `output_dir`

The same `evaluate()` / `save_*()` split classification and extraction offer works here too:

```python
experiment.evaluate()                                             # stages 1-6, populates experiment.results and experiment.ranking
experiment.save_experiment_results(output_dir="./results")        # stage 7
html_path = experiment.save_html_report(output_dir="./results")   # stage 8
pdf_path = experiment.save_pdf_report(output_dir="./results")
```

---

## Accessing Results

After `evaluate()` (or `run()`), `experiment.results` is a list of one [`EvaluationResult`](../../api/evaluation_result) per model, the same shape classification and extraction produce:

```python
for result in experiment.results:
    print(result.model, result.metrics.aggregated_task_scores)

    for prediction in result.predictions:
        print(" ", prediction.document_id, prediction.task_scores)
```

`prediction.task_scores` holds whichever of the four axes were defined for that document; `prediction.is_correct` and `prediction.expected_value` are left `None`, since neither concept applies. The cross-model comparison lives on `experiment.ranking` instead, a [`SummarizationRanking`](../../api/summarization_ranking):

```python
print(experiment.ranking.best)              # top tier's model labels
for entry in experiment.ranking.scores:     # every model, best first
    print(entry.model, entry.score, entry.axes())
```

See [Evaluation Results](./evaluation-results) for the full breakdown of both.

---

## Incremental Evaluation

You can load a previously completed run to regenerate its report or evaluate it against new models. This follows the same three steps as [Classification Evaluation API: Incremental Evaluation](../classification/evaluation-api.md#incremental-evaluation):

```python
experiment = SummarizationExperiment.load_experiment_results("./results/my_run")
experiment.add_models([{"name": "claude-sonnet-4-6"}])
experiment.run("./results/my_run")
```

Models already present in `self.results` are skipped; only the newly added ones are evaluated, and the report is regenerated with all of them. `judge_model` and the rest of the scoring config are restored from `metadata.json`, so the reloaded experiment grades new candidates under the exact configuration the original run used.

Shared document facts aren't persisted to disk, only kept on the live instance that extracted them. Calling `add_models()` and `evaluate()` again on the *same* instance reuses them for free, but a fresh instance from `load_experiment_results()` re-extracts them, and their cost is split only across the models being added in that pass, not the full original field.

---

## Reweighting and Regrading a Run

A finished summarization run can be re-scored without regenerating a single candidate summary, via `reevaluate()` / `areevaluate()`. Every stored `predicted_value` is replayed rather than requeried, so a regrade only ever pays for judge calls, never a second generation call.

```python
experiment = SummarizationExperiment.load_experiment_results("./results/my_run")

# Pure arithmetic, no LLM calls: try a stricter faithfulness gate
experiment.reevaluate(gate=0.7, output_dir="./results/my_run_strict")

# Add a checklist after the fact: only the requirements_met axis is regraded
experiment.reevaluate(requirements=["Name the parties.", "State the outcome."], output_dir="./results/my_run_checklist")

# Swap the judge: correctness, coverage, and precision are all recomputed from scratch
experiment.reevaluate(judge_model="gpt-4o", output_dir="./results/my_run_rejudged")
```

`reevaluate()` picks the cheapest tier that covers what changed:

- **Reweight** (`gate`, `beta`, `requirement_weight`, `tier_gap`): pure arithmetic over axes already stored in `task_scores`. No LLM calls at all.
- **Requirements-only regrade** (`requirements` changes, `judge_model` doesn't): only `requirements_met` can possibly change, so only that judge call reruns per prediction; the other three axes are untouched.
- **Full regrade** (`judge_model` changes): a different judge has its own opinions about which facts are salient, so document facts and all four axes are recomputed from scratch, exactly as a fresh `evaluate()` would.

`output_dir`, if given, writes the re-scored results the same way `save_experiment_results()` does. As with the base `reevaluate()` classification and extraction use, `metadata.json` is not overwritten if it already exists there, so pass a fresh directory to persist an updated `judge_model` or `requirements`.

---

## Sync vs Async

`evaluate`/`run`/`reevaluate` each have an async twin (`aevaluate`/`arun`/`areevaluate`) for use inside an existing event loop, e.g. FastAPI or a Jupyter cell with `await`:

```python
report_path = await experiment.arun(output_dir="./results")
```

```{rubric} What's next?
```

- To understand the output schema and the ranking, see [Evaluation Results](./evaluation-results).
- To view the HTML and PDF reports, see [Report Formats](./report-formats).
