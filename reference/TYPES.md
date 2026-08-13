# Types behind the output files

Maps each field group in the example JSON files to the Pydantic model / code that
produces it. See [`README.md`](README.md) for the directory layout.

## `metadata.json` -> [`examples/metadata.example.json`](examples/metadata.example.json)

Built by `save_run_dir()` ([`runner.py:124`](../src/valtron_core/runner.py#L124)),
not a Pydantic model itself, assembled from:

| Field | Comes from |
|---|---|
| `use_case`, `original_prompt` | `BaseRecipeConfig` ([`evaluation/config.py:164`](../src/valtron_core/evaluation/config.py#L164)) |
| `field_metrics_config` | `FieldMetricsConfig` ([`models.py:368`](../src/valtron_core/models.py#L368)) |
| `response_format_schema` | JSON Schema derived from the recipe's `response_format` |
| `documents` | one entry per input dict, built by `BaseRecipe._build_save_documents()` ([`evaluation/base.py`](../src/valtron_core/evaluation/base.py)) from `Document` ([`models.py:46`](../src/valtron_core/models.py#L46)) |
| `cost` / `total_cost` | summed from each model's `PredictionResult.llm_cost` / `evaluation_cost` |

`metadata.json` only holds run-wide totals per model. Per-document cost lives
in `models/<label>.json` instead: `metrics.average_cost_per_document`
([models.py:148](../src/valtron_core/models.py#L148)) and each
`predictions[i].llm_cost` / `.evaluation_cost` / `.original_cost`
([models.py:107-117](../src/valtron_core/models.py#L107)), since that file
already scopes to one model.

## `progress.json` -> [`examples/progress.example.json`](examples/progress.example.json)

Not a Pydantic model; a plain dict written atomically by `ProgressTracker`
([`progress.py:89`](../src/valtron_core/progress.py#L89)). `on_doc_complete()`
increments `docs_done` per model as each document finishes; `completed` flips
to `true` once `docs_done == docs_total`.

## `models/<label>.json` -> [`examples/model_result.example.json`](examples/model_result.example.json)

Built by `save_single_model_result()` ([`runner.py:40`](../src/valtron_core/runner.py#L40))
from an `EvaluationResult` ([`models.py:155`](../src/valtron_core/models.py#L155)):

| Field | Type |
|---|---|
| `run_id`, `started_at`, `completed_at`, `status`, `llm_config` | `EvaluationResult` attributes |
| `metrics` | `EvaluationMetrics` ([`models.py:127`](../src/valtron_core/models.py#L127)) |
| `predictions[]` | list of `PredictionResult` ([`models.py:96`](../src/valtron_core/models.py#L96)) |
| `predictions[].field_metrics`, `metrics.aggregated_field_metrics[*]` | `EvalResult` ([`scoring/json_eval/schema.py:215`](../src/valtron_core/scoring/json_eval/schema.py#L215)), the per-field scoring tree |

## Producing code

[`base_model_eval.py`](base_model_eval.py) is the abstract pipeline
(`BaseModelEval`) these files come out of: `_run_evaluations()` persists each
model's `EvaluationResult` as it finishes, `save_experiment_results()` (from
`BaseRecipe`, [`evaluation/base.py`](../src/valtron_core/evaluation/base.py))
writes `metadata.json`, and `load_experiment_results()` reads both back into a
live instance. It is not yet wired up to the in-use `ModelEval`
([`evaluation/model_eval.py`](../src/valtron_core/evaluation/model_eval.py)),
which produces the same file shapes via its own, separate pipeline today.
