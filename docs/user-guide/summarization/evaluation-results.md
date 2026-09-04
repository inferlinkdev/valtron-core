# Evaluation Results

## Reading a model's axes

Summarization results center on the four axes described in [Summarization](./index): `correctness`, `salient_coverage`, `salient_precision`, and `requirements_met`.

Each [`PredictionResult`](../../api/prediction_result)'s `task_scores` holds whichever of the four were defined for that document, and each [`EvaluationMetrics`](../../api/evaluation_metrics)'s `aggregated_task_scores` is their mean across every prediction for that model, skipping documents where an axis was undefined rather than treating it as zero:

```python
experiment.evaluate()

for result in experiment.results:
    print(result.model, result.metrics.aggregated_task_scores)

    for prediction in result.predictions:
        print(" ", prediction.document_id, prediction.task_scores)
```

```text
gpt-4o {'correctness': 1.0, 'salient_coverage': 0.88, 'salient_precision': 0.92}
  1 {'correctness': 1.0, 'salient_coverage': 0.83, 'salient_precision': 1.0}
  2 {'correctness': 1.0, 'salient_coverage': 0.92, 'salient_precision': 0.85}
gpt-4o-mini {'correctness': 1.0, 'salient_coverage': 0.71, 'salient_precision': 0.85}
  1 {'correctness': 1.0, 'salient_coverage': 0.67, 'salient_precision': 0.80}
  2 {'correctness': 1.0, 'salient_coverage': 0.75, 'salient_precision': 0.90}
```

`requirements_met` is absent from both dicts above because no `requirements` checklist was configured; see [Config Format: The Requirements Checklist](./config-format.md#the-requirements-checklist). `prediction.is_correct`, `prediction.expected_value`, and `prediction.example_score` are all left `None` on every prediction. There is no ground truth to be correct against, and leaving them unset is deliberate rather than an oversight, so nothing downstream mistakes a summarization run for a classification one.

`prediction.metadata` carries the evidence behind those numbers: the document's own facts, which of them the judge marked salient, the candidate summary's own extracted facts, and the per-fact verdicts behind each axis. That's what [Report Formats](./report-formats) renders per document.

## Reading the ranking

`aggregated_task_scores` is a per-model average; it doesn't say which model won or by how much. That's what `experiment.ranking` is for, a [`SummarizationRanking`](../../api/summarization_ranking) built by scoring each model's averaged axes once under `salience-f+reqs`:

```python
ranking = experiment.ranking

print(ranking.best)           # ['gpt-4o'] -- the top tier's model labels
print(ranking.ranked_models)  # ['gpt-4o', 'gpt-4o-mini'] -- every model, best first

for entry in ranking.scores:  # SummarizationScore, best first
    print(entry.model, entry.score, entry.axes(), entry.documents_scored)
```

```text
['gpt-4o']
['gpt-4o', 'gpt-4o-mini']
gpt-4o 0.9091 {'correctness': 1.0, 'salient_coverage': 0.88, 'salient_precision': 0.92, 'requirements_met': None} 2
gpt-4o-mini 0.7734 {'correctness': 1.0, 'salient_coverage': 0.71, 'salient_precision': 0.85, 'requirements_met': None} 2
```

`ranking.tiers` groups model labels into ordered lists; two models only share a tier when their scores differ by no more than `tier_gap` (`0.0` by default, so in practice only on an exact tie). `ranking.parameters` records the four scoring parameters the ranking was computed with, and `ranking.usage` splits token/cost accounting three ways: what candidates spent generating summaries, what the judge spent grading each of them, and what the judge spent on the one-per-document work (fact extraction and salience marking) that every candidate shares. `ranking.to_dict()` gives the same information as a plain JSON-serializable dict, which is what the HTML and PDF reports render from.

Reading `experiment.ranking` before `evaluate()` has run raises `RuntimeError`. After `load_experiment_results()`, reading it computes the ranking on demand from each prediction's stored `task_scores`, since the ranking itself isn't one of the files a run writes to disk; see [Output directory layout](#output-directory-layout) below.

## Output directory layout

`run(output_dir="./results")` writes:

```text
results/
├── metadata.json
├── models/
│   ├── gpt-4o.json
│   └── gpt-4o-mini.json
├── html_report/
│   └── summarization_report.html
└── summarization_report.pdf        ← only if "pdf" in output_formats
```

`metadata.json` holds experiment-level state (timestamp, `use_case`, the original prompt, and the input documents), plus the scoring configuration (`judge_model`, `requirements`, `gate`, `beta`, `requirement_weight`, `tier_gap`) needed to reproduce this run's score on reload. `models/<label>.json` holds one full serialized [`EvaluationResult`](../../api/evaluation_result) per model, including every prediction's `task_scores`. This is what `SummarizationExperiment.load_experiment_results()` reads back for [incremental evaluation](./evaluation-api.md#incremental-evaluation) and [regrading](./evaluation-api.md#reweighting-and-regrading-a-run).

Note the HTML report's path nests one level deeper than classification and extraction's `evaluation_report.html`, and there is no separate `detailed_analysis.html`: the per-document breakdown lives inside `summarization_report.html` itself. The ranking is not written as its own file; it is rebuilt from the predictions' `task_scores` whenever `experiment.ranking` or a report is requested.

```{rubric} What's next?
```

- View the HTML and PDF reports: [Report Formats](./report-formats)
- Reweight or regrade a run after the fact: [Evaluation API: Reweighting and Regrading](./evaluation-api.md#reweighting-and-regrading-a-run)
