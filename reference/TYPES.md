# Types behind the output files

Maps each field group in the example JSON files to the Pydantic model / code that
produces it. See [`README.md`](README.md) for the directory layout.

## `metadata.json` -> [`examples/metadata.example.json`](examples/metadata.example.json)

Built by `save_run_dir()` ([`runner.py:125`](../src/valtron_core/runner.py#L125)),
not a Pydantic model itself, assembled from:

| Field | Comes from |
|---|---|
| `use_case`, `original_prompt` | `BaseRecipeConfig` ([`evaluation/config.py:164`](../src/valtron_core/evaluation/config.py#L164)) |
| `field_metrics_config` | `FieldMetricsConfig` ([`models.py:431`](../src/valtron_core/models.py#L431)) |
| `response_format_schema` | JSON Schema derived from the recipe's `response_format` |
| `documents` | one entry per input dict, built by `ModelEval._build_save_documents()` ([`evaluation/model_eval.py:538`](../src/valtron_core/evaluation/model_eval.py#L538)) from `Document` ([`models.py:46`](../src/valtron_core/models.py#L46)) |
| `cost` / `total_cost` | summed from each model's `PredictionResult.llm_cost` / `evaluation_cost` |

`metadata.json` only holds run-wide totals per model. Per-document cost lives
in `models/<label>.json` instead: `metrics.average_cost_per_document`
([models.py:191](../src/valtron_core/models.py#L191)) and each
`predictions[i].llm_cost` / `.evaluation_cost` / `.original_cost`
([models.py:134-144](../src/valtron_core/models.py#L134)), since that file
already scopes to one model.

`PredictionResult.is_correct` / `.example_score` / `expected_value` are all
`Optional` and `None` by default ([models.py:96](../src/valtron_core/models.py#L96)) --
a task with no single ground-truth value per document (e.g. a future
summarization task) leaves these unset rather than faking a value. The
`task_scores: dict[str, float] | None` field ([models.py:125](../src/valtron_core/models.py#L125))
and `EvaluationMetrics.aggregated_task_scores` ([models.py:173](../src/valtron_core/models.py#L173))
exist for exactly that case: open-ended per-document scalar scores (e.g. a
requirement-coverage score, a hallucination rate) that don't fit
`is_correct`/`example_score`, aggregated by key across all predictions.

## `progress.json` -> [`examples/progress.example.json`](examples/progress.example.json)

Not a Pydantic model; a plain dict written atomically by `ProgressTracker`
([`progress.py:89`](../src/valtron_core/progress.py#L89)). `on_doc_complete()`
increments `docs_done` per model as each document finishes; `completed` flips
to `true` once `docs_done == docs_total`.

## `models/<label>.json` -> [`examples/model_result.example.json`](examples/model_result.example.json)

Built by `save_single_model_result()` ([`runner.py:41`](../src/valtron_core/runner.py#L41))
from an `EvaluationResult` ([`models.py:198`](../src/valtron_core/models.py#L198)):

| Field | Type |
|---|---|
| `run_id`, `started_at`, `completed_at`, `status`, `llm_config` | `EvaluationResult` attributes |
| `metrics` | `EvaluationMetrics` ([`models.py:154`](../src/valtron_core/models.py#L154)) |
| `predictions[]` | list of `PredictionResult` ([`models.py:96`](../src/valtron_core/models.py#L96)) |
| `predictions[].field_metrics`, `metrics.aggregated_field_metrics[*]` | `EvalResult` ([`scoring/json_eval/schema.py:215`](../src/valtron_core/scoring/json_eval/schema.py#L215)), the per-field scoring tree |

## Producing code

`ModelEval` ([`evaluation/model_eval.py`](../src/valtron_core/evaluation/model_eval.py))
is the generic, correctness-agnostic abstract pipeline these files come out of:
`_run_evaluations()` persists each model's `EvaluationResult` as it finishes,
`save_experiment_results()` writes `metadata.json`, and
`load_experiment_results()` reads both back into a live instance. Its one
required override is `_evaluate_model_documents()` -- call a model, score its
output; everything else (construction, model management, data loading,
preflight, prompt prep, persistence, traces, a default `reevaluate()`) is
shared. `save_html_report()` / `save_pdf_report()` are *not* implemented at
this level -- they assume a correctness/accuracy notion `ModelEval` deliberately
makes no assumption about.

`ReferencedEval` ([`evaluation/referenced_eval.py`](../src/valtron_core/evaluation/referenced_eval.py))
is the concrete subclass used today: it implements `_evaluate_model_documents()`
for the classification/extraction shape (schema inference, few-shot, prompt
manipulations, decompose/hallucination-filter, transformer models) and adds
the HTML/PDF reports. `ClassificationExperiment` / `ExtractionExperiment`
([`evaluation/classification.py`](../src/valtron_core/evaluation/classification.py) /
[`evaluation/extraction.py`](../src/valtron_core/evaluation/extraction.py)) are
thin `ReferencedEval` subclasses that add label-shape validation and schema
auto-inference; neither overrides `_evaluate_model_documents()` itself.

This used to be two separate, unwired pipelines (a draft generic base kept in
this `reference/` folder, and a since-renamed `ModelEval` that did everything
classification/extraction needs inline); they were merged so the draft became
the real base class every recipe -- present and future -- shares. A task with
no single ground-truth value per document (e.g. summarization, scored against
extracted requirements/facts rather than an exact label) would extend
`ModelEval` directly as a sibling of `ReferencedEval`, not a child of it --
see the note on `task_scores` above.
