# Evaluation

Three ready-to-use recipes for evaluating LLMs (and, for classification, transformer
models) against a dataset, plus the shared engine and extension seams behind them.

## Recipes

```python
from valtron_core.evaluation import (
    ClassificationExperiment,
    ExtractionExperiment,
    SummarizationExperiment,
)
```

| Recipe | Data shape | Ground truth | Report |
|---|---|---|---|
| `ClassificationExperiment` | `label` is a plain string | Required | Accuracy, per-model comparison |
| `ExtractionExperiment` | `label` is a dict/list (or a JSON string of one) | Required | Accuracy plus per-field precision/recall |
| `SummarizationExperiment` | no `label` at all | None; judged corpus-wide by an LLM judge | A ranking, not a per-document score |

All three share the same construction and run shape:

```python
experiment = ClassificationExperiment(
    config={
        "models": [{"name": "gpt-4o-mini"}, {"name": "gpt-4o"}],
        "prompt": "Classify: {content}",
        "output_dir": "./results",
    },
    data=[{"id": "1", "content": "...", "label": "positive"}],
)
report_path = experiment.run()          # sync; also writes output_dir per config.output_formats
report_path = await experiment.arun()   # same, async (e.g. inside a notebook)

# Or drive it by hand:
await experiment.aevaluate()
experiment.save_experiment_results()    # metadata.json + models/*.json
experiment.save_html_report()           # html_report/evaluation_report.html
experiment.save_pdf_report()            # a PDF alongside it
```

`ExtractionExperiment` requires a schema (`response_format` or
`config.response_format_schema`); `ClassificationExperiment` can auto-infer one
from the unique label values instead (`config.infer_schema`, default `True`).
`SummarizationExperiment` takes `judge_model` and `requirements` instead of a
schema; see its own module docstring (`evaluation/summarization.py`) for the
scoring scheme.

Reloading a previous run and re-scoring or extending it:

```python
experiment = ClassificationExperiment.load_experiment_results("./results/run_.../")
experiment.add_models(["claude-sonnet-4-6"])   # only the new model gets evaluated
await experiment.aevaluate()
experiment.reevaluate(field_metrics_config=new_config)  # re-score, no new LLM calls
```

## Architecture

Every recipe is a `ModelEval` subclass. `ModelEval` owns everything that doesn't
vary by recipe: the fan-out over models and documents, caching and resume,
run-directory persistence, and the report-writing seam. What *does* vary is
expressed as four narrow structural-typing `Protocol`s (`evaluation/stages/`,
`reports/_base.py`), so a recipe's constructor wires up concrete
implementations instead of a shared method branching on task type:

```
                    ModelEval (shared engine)
                   /      |       |        \
          Ingestor   Generator   Scorer   ReportWriter
        (records ->  (call a    (decide  (render one
         documents)   model)   correctness) format)
```

- **`Ingestor`** (`evaluation/stages/ingest.py`): turns `self.data` into
  `(documents, labels)`. `DefaultIngestor` (label optional, `ModelEval`'s own
  default) and `StructuredLabelIngestor` (label required, JSON-serialized;
  `ReferencedEval`, the shared base of `ClassificationExperiment`/
  `ExtractionExperiment`, sets one of these up in `_post_init`).
- **`Generator`** (`evaluation/stages/generate.py`): `generate(document, ...) ->
  RawPrediction`, no scoring involved. `LLMGenerator` (any litellm model,
  including multi-pass), `TransformerGenerator` (a locally trained classifier),
  `HallucinationFilterGenerator` (decorates any `Generator`, filters
  unsupported values out of the result). `decompose.py`'s `DecomposedGenerator`
  (splits a multi-entity extraction prompt into parallel per-entity calls)
  lives in that module rather than here, since putting it here would create an
  import cycle through `evaluator.py`.
- **`Scorer`** (`evaluation/stages/score.py`): `score(predicted, expected, ...)
  -> ScoreOutcome`. `ExactMatchScorer` (case-insensitive string compare) and
  `FieldMetricsScorer` (per-field precision/recall via `JsonEvaluator`, falling
  back to exact match on error).
- **`ReportWriter`** (`reports/_base.py`): `write(results, output_path,
  **context) -> (Path, recommendation | None)`. Concrete writers in
  `reports/writers.py`: `HtmlReportWriter`/`PdfReportWriter` (wrap
  `HtmlReportGenerator`/`PdfReportGenerator`), `SummarizationHtmlReportWriter`/
  `SummarizationPdfReportWriter` (wrap `SummarizationReportGenerator`, and take
  a `ranking` in `**context` that the other two don't need). `**context` rather
  than a fixed parameter list exists because the underlying generators
  genuinely need different things (a `ranking` only summarization has; a
  pre-computed `recommendation` only the PDF writers take).

Summarization has no ground truth, so it has no `Ingestor`/`Scorer` role to
fill the usual way; `JudgeScorer`/`JudgeCandidateGenerator`
(`evaluation/stages/summarization_*.py`) wrap `evaluate_candidate`'s two
halves (`summarization/pipeline.py`'s `generate_summary`/`grade_summary`) but
are deliberately not typed as `Generator`/`Scorer` and not re-exported from
`stages/__init__.py`: their shapes (`Doc`/`Model`/checklist in,
`Summary`/`GradeResult` out) don't match the ground-truth recipes', and
importing the generic `Scorer` protocol should not have to pull in the whole
summarization subsystem for a recipe that never uses it.

### Per-document fan-out

`PromptEvaluator.evaluate()` (`evaluator.py`) and
`SummarizationExperiment._evaluate_model_documents()` (`summarization.py`)
both bound per-document concurrency with a semaphore, run one callable per
document, and report progress/cost through a shared bar. That bookkeeping
(not the callable itself, which differs by recipe) is
`evaluation/document_fan_out.py::fan_out_over_documents(documents,
max_concurrent, process_one, *, on_document_complete=None, progress_bar=None)`.
It has no dependency on `evaluator.py`/`model_eval.py`/`runner.py`, so those
can depend on it without a cycle.

`ModelEval._run_evaluations()` (the *per-model* concurrency layer above this,
handling progress tracking, partial-result staging, and per-model disk
persistence) is a different concern and does not use this helper; see
**Debt ledger disposition** below for why.

### Persistence

Every recipe writes the same run-directory shape (`metadata.json` +
`models/<name>.json`) through `runner.save_run_dir`/`save_single_model_result`,
and reads it back through `evaluation/persistence.py`'s `RunDirectoryCodec`,
which every reader (`ModelEval.load_experiment_results`,
`ReferencedEval.load_experiment_results`,
`EvaluationRunner._load_results_from_run_dir`,
`utilities/aggregate_reports.py`'s loader) now shares. A run directory
written before `RunDirectoryCodec.build_prediction`'s reconciliation reads
back with `is_correct`/`example_score` defaulting to `None` ("not scored")
regardless of format version, including a version-1 file (no
`format_version` key at all, i.e. a run written before that field existed).
`FORMAT_VERSION` exists so a future format change has something real to
branch on, not because this one needs a branch.

### Registering an experiment type

`evaluation/registry.py`'s `ExperimentRegistry` maps a `task_type` name (e.g.
`"classification"`) to the `ModelEval` subclass that implements it, plus an
optional sniff predicate deciding whether a raw dataset "looks like" that
type. `utilities/config_wizard.py`'s `/api/analyze-data` endpoint uses
`ExperimentRegistry.sniff_best_match(data)` to guess a task type from an
uploaded dataset's shape, instead of a hardcoded if/elif.

## Extending

**A new ground-truth experiment type**: subclass `ReferencedEval` (or
`ModelEval` directly, for a shape `ReferencedEval` doesn't fit), reuse
`ExactMatchScorer`/`FieldMetricsScorer` or write a new `Scorer`, and register:

```python
from valtron_core.evaluation.registry import register_experiment

@register_experiment("my_task", sniff=lambda data: ...)
class MyExperiment(ReferencedEval):
    ...
```

Zero edits to `ModelEval`, any existing recipe, or any existing `Generator`/
`Scorer`.

**A new no-ground-truth experiment type**: subclass `ModelEval` directly (see
`SummarizationExperiment`), write a `Scorer`-shaped or custom scoring
collaborator that never reads `label`, and register the same way.

**A new report format for an existing recipe**: implement `ReportWriter`
(one `write()` method), register an instance under a new key in that
recipe's `self._report_writers` (set up in its own `_post_init`), and call
`experiment.save_report("your_format")`. Zero edits to any existing
`ReportWriter`, `HtmlReportGenerator`/`PdfReportGenerator`/
`SummarizationReportGenerator`, or to `ModelEval.save_report()` itself; see
`tests/test_report_writer_extension_point.py` for a worked (throwaway)
example.

## Debt ledger disposition

This refactor's own planning document (kept locally, untracked, not part of
this repository) measured a debt ledger up front and predicted a fix or a
deferral for each item. Some predictions held; a few did not once the actual
implementation forced a closer look. Recorded here so the correction is not
lost once that planning document is gone.

### Complexity (mccabe/branch-count, measured with `ruff`'s own gates)

| Hot spot | Final status |
|---|---|
| `PromptEvaluator.evaluate_single`/`.evaluate` (`evaluator.py`) | **Fixed.** Split across `LLMGenerator` and `Scorer`; no suppression needed. |
| `ModelEval._result_from_model_data`/`.load_experiment_results` | **Fixed.** Unified into `persistence.py`. |
| `EvaluationRunner._load_results_from_run_dir` | **Fixed.** Unified into `persistence.py`. |
| `ReferencedEval.add_models` | **Fixed.** Delegates to `super().add_models()`. |
| `ReferencedEval.load_experiment_results` | **Fixed.** Unified into `persistence.py`; the complexity suppression this method carried is gone, not just narrowed. |
| `ReferencedEval._evaluate_transformer` | **Fixed, better than planned.** Originally expected to stay deferred (wrapped as-is by `TransformerGenerator`); the label-serialization dedup that landed for an unrelated reason (see Redundant code, below) happened to drop it under threshold too. |
| `config_wizard.py::api_analyze_data` | **Mostly fixed.** Pointing its dispatch at `ExperimentRegistry` dropped two of its three suppressed checks; one (`PLR0911`, return-statement count) remains, from logic this refactor never touched. |
| `ModelEval._run_evaluations` | **Deferred, and the original plan's framing of it was wrong.** It was originally counted as the third of "3 fan-out-with-semaphore loops" duplicating `PromptEvaluator.evaluate`/`SummarizationExperiment._evaluate_model_documents`. It is not: those two bound *per-document* concurrency with a semaphore; this one gathers *per-model* tasks (no semaphore) and owns progress tracking, partial-result staging, and per-model disk persistence, a different concern `fan_out_over_documents` was never meant to cover. Still complex, still suppressed, correctly untouched. |
| `ReferencedEval._evaluate_model_documents` | **Partially fixed.** Generation now delegates to a `Generator` (`LLMGenerator`/`TransformerGenerator`/`DecomposedGenerator`) instead of calling a model inline, which dropped its branch count under threshold; the mccabe complexity of choosing *which* `Generator`/manipulations apply remains above it. |
| `ReferencedEval.reevaluate` | **Deferred**, as planned; untouched by this migration. |
| `EvaluationRunner.generate_report` | **Deferred**, as planned; report internals stay byte-identical. |
| `JsonEvaluator._eval_list_unordered_with_alignment` and siblings | **Deferred**, as planned; `FieldMetricsScorer` wraps `JsonEvaluator.evaluate` unchanged. |
| `PdfReportGenerator._build_perf_table`, `HtmlReportGenerator._prepare_detailed_analysis_data` | **Deferred**, as planned; `ReportWriter` wraps both unchanged. |
| `decompose.py` helpers (`_deep_merge_dicts`, `filter_hallucinated_values`) | **Deferred**, as planned; `DecomposedGenerator` wraps `DecomposedEvaluator` as-is. Untouched beyond one unrelated dead-variable removal (see below). |

### Redundant code

| Item | Final status |
|---|---|
| 4 near-duplicate disk-reconstruction implementations (`model_eval.py`, `referenced_eval.py`, `runner.py`, `utilities/aggregate_reports.py`) | **Fixed.** All four now share `persistence.py`'s `RunDirectoryCodec`. |
| `decompose.py` inline `JsonEvaluator` construction bypassing the shared scorer | **Fixed**, and it was also a real bug: decomposed extraction with no `field_metrics_config`/`comparison_fn` scored `is_correct=False` unconditionally regardless of whether the merged result actually matched. Its own dedicated, reviewable commit. |
| "3 fan-out-with-semaphore loops" | **2 of the 3 fixed**, not 3; see `ModelEval._run_evaluations` above for why the third was never the same shape. `PromptEvaluator.evaluate`/`SummarizationExperiment._evaluate_model_documents` now share `fan_out_over_documents`. |
| Three copies of the same label-serialization rule inside `referenced_eval.py` (`_load_documents_and_labels`, `reevaluate`, `_evaluate_transformer`) | **Fixed**, found and taken while introducing `Ingestor`, not originally on the ledger. All three now call `evaluation/stages/ingest.py::serialize_structured_label`. |
| `ReferencedEval.save_html_report`/`save_pdf_report` near-identical bodies | **Still open, not fixed as originally expected.** The plan assumed this would resolve automatically once both became one-line `save_report(...)` calls; `ReferencedEval` was deliberately not migrated onto `ReportWriter`/`self._report_writers` (its report generation goes through `EvaluationRunner.generate_report()`'s own cross-format recommendation coordination, a separate, larger piece of work), so the two near-identical bodies remain. A real candidate for a future commit, once someone re-derives that coordination inside `save_report()`. |
| "Dark GitHub-style" header/CSS duplicated per template | **Fixed for exactly the two templates that actually duplicated it.** Measured (not assumed) with a script that diffed every `<style>` rule pairwise: `evaluation_report.jinja2.html` and `summarization_report.jinja2.html` shared one `body` rule, `.header`, and eight `.recommendation` child selectors, verbatim, now in `common.css`. `detailed_analysis.jinja2.html` (despite being named in the original plan as a third beneficiary) shares **zero** byte-identical rules with either; its `.header` is a genuinely different sticky-nav layout, not a copy, and it was already relying on `common.css`'s shared `body`/`h1`/`.footer`. `tradeoff_report.jinja2.html` stays out of scope, untouched. |
| Duplicate Jinja `Environment`/`TEMPLATES_DIR` objects (`reports/_base.py` vs `analysis/_report.py`) | **Deferred**, as planned; `analysis/` is out of scope. |
| Duplicated ReportLab helpers (`generate_pdf_report.py` vs `generate_summarization_report.py`) | **Deferred**, as planned. |
| Six small pre-existing dead-code items found incidentally while working in this area (unused imports/variable in `decompose.py`, `runner.py`, `reports/generate_html_report.py`, `reports/generate_pdf_report.py`) | **Fixed**, in a dedicated commit with its own removal manifest, cross-checked against the pre-refactor baseline to confirm each was genuinely pre-existing. |

### Size

Predicted directional changes mostly held, with one correction worth
recording: `model_eval.py` was predicted to shrink ("loses the reconstruction
methods to `persistence.py`"); it grew slightly instead. It did lose that
code, but gained more from becoming the one shared home for the `Ingestor`
delegation and the `save_report`/`ReportWriter` dispatch machinery that
previously would have been duplicated (or simply absent) per recipe, an
intentional trade: a little more code in the one place every recipe shares,
in exchange for real duplication eliminated at every recipe using it.
`decompose.py` was predicted "roughly unchanged" and grew instead, since
`DecomposedGenerator` had to live inside it (see the `Generator` section
above for the import-cycle reason), not in `stages/`.

New files landed close to the "roughly 300 lines" target from the original
plan, with one file (`stages/generate.py`, at just over 300) left slightly
over rather than split further: it holds `LLMGenerator`,
`TransformerGenerator`, `HallucinationFilterGenerator`, `RawPrediction`, and
`format_prompt`, which are all "generate stage" implementations and would
gain nothing from separate files beyond satisfying the line count itself.
One planned file, `reports/registry.py`, was never created: with only one
recipe (`SummarizationExperiment`) actually populating a `self._report_writers`
dict so far, a separate class-level registry module for report formats would
have been speculative machinery ahead of a second consumer, not something an
actual caller needed yet. `ModelEval.save_report()`'s per-instance dict
already does the job that would justify one; a real second use case is the
right time to decide whether it should grow into more than that.
