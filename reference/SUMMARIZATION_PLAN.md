# Plan: a summarization recipe

Outline for adding a summarization evaluation recipe to this codebase, based on
reviewing an external summarization-quality evaluation codebase built around
LLM-judged fact decomposition. See [`README.md`](README.md) and
[`TYPES.md`](TYPES.md) for the class hierarchy this plan builds on.

## Why this doesn't fit the existing hierarchy

`ReferencedEval` (and `ClassificationExperiment`/`ExtractionExperiment` on top
of it) assume every document has a known label/`expected_value` to score a
prediction against, exactly or per-field. Summarization quality has no single
ground-truth string to diff against: a good summary can vary in wording,
compression, and structure while still covering everything it needs to. The
approach reviewed here scores summaries a different way: a judge model
decomposes the document, the summary, and any reference summaries into atomic
facts, and one shared entailment kernel ("is each claim fact supported by
these reference facts?") gets reused with different claim/reference role
assignments to compute several axes: faithfulness to the source, coverage of
what matters, and satisfaction of an optional requirements checklist. An
offline second pass turns those recorded per-fact verdicts into named axes and
reduces them to a ranking via a pluggable scoring scheme, replayable without
new LLM calls.

`ModelEval` already anticipates a task shaped like this: `PredictionResult`
has `is_correct` / `example_score` / `expected_value` as `Optional`, plus a
`task_scores: dict[str, float] | None` field described as being "for tasks
whose signal doesn't fit is_correct/example_score (e.g. a summarization
quality axis)" (see [`TYPES.md`](TYPES.md)). So the shape to build toward is a
new subclass of `ModelEval` directly, a sibling of `ReferencedEval`, not a
child of it.

## What's reusable versus what should be dropped

The reviewed codebase cleanly separates task logic from infrastructure: its
prompt construction, fact extraction/matching, requirements alignment,
salience marking, and the whole offline analyzer only depend on an abstract
"run this prompt, get structured output back" call, not on a particular LLM
client. Its own LLM wrapper, disk cache, and per-vendor concurrency limiter
are a separate, swappable layer underneath that.

This codebase already has its own version of that infrastructure layer:
`client.py` (`LLMClient` over litellm, per project convention: "All LLM calls
go through litellm via client.py. Do not call provider SDKs directly."),
`PartialResultStore` for caching/resume, and `ProgressTracker` for progress
reporting. So the port should bring over the *domain logic* (fact
decomposition, the entailment kernel, requirements alignment, salience,
the offline axis/scoring-scheme analyzer) rewritten against `client.py`, and
should not bring over a second cache/concurrency/client layer alongside the
one this codebase already has.

## Class hierarchy

```
abc.ABC
 └── ModelEval(ABC)                    [existing: evaluation/model_eval.py]
      ├── ReferencedEval(ModelEval)    [existing: evaluation/referenced_eval.py]
      │    ├── ClassificationExperiment
      │    └── ExtractionExperiment
      └── SummarizationExperiment(ModelEval)   <- new, evaluation/summarization.py
```

```
pydantic.BaseModel
 └── BaseRecipeConfig                  [existing: evaluation/config.py]
      └── ModelEvalConfig
           ├── ClassificationConfig
           └── SummarizationConfig     <- new: judge_model, ground_truth_models,
                                          requirements, important_set_source,
                                          scoring_scheme
```

## Methods `SummarizationExperiment` would need to implement

Everything else (`__init__`, `add_models`, the default `_load_documents_and_labels`
with optional labels, `_run_evaluations` concurrency/resume/persistence,
`get_traces`, `save_experiment_results`) is inherited unchanged from
`ModelEval`. The seams that need filling in:

```python
class SummarizationExperiment(ModelEval):
    @classmethod
    def _config_model(cls) -> type[BaseRecipeConfig]:
        return SummarizationConfig                     # required override

    async def _before_evaluation(self, field_metrics_config) -> None:
        # runs once per document, shared across every model being compared:
        # extract document facts, mark salient ("must-convey") facts,
        # build the reference universe from any gold/panel summaries.
        # This is exactly the extension point ModelEval already provides
        # for "setup after preflight but before prompt resolution".
        ...

    async def _evaluate_model_documents(                # the ONE required override
        self, model_config, documents, labels, prompt,
        field_metrics_config, on_document_complete=None, progress_bar=None,
    ) -> tuple[EvaluationResult, str | None]:
        # generate (or accept a precomputed) summary -> extract its facts ->
        # run the entailment kernel against document/gold/panel/salient
        # facts -> requirements alignment -> build a PredictionResult per doc with
        #   predicted_value = summary text
        #   expected_value  = None            (no single ground truth)
        #   is_correct / example_score = None (no binary correctness notion)
        #   task_scores = {"correctness": ..., "salient_coverage": ...,
        #                  "requirements_met": ..., ...}
        ...

    def compute_task_statistics(self, results) -> dict:
        # the offline analyzer stage: aggregate axes across documents/models
        # and reduce via the chosen scoring scheme into a ranking. This is
        # exactly the documented "general-purpose alternative" to field
        # metrics for a task whose signal doesn't fit EvaluationMetrics.
        ...

    def save_html_report(self, output_dir=None) -> Path:  # base raises NotImplementedError
        ...                                                 # new template: per-axis
    def save_pdf_report(self, output_dir=None) -> Path:      # breakdown, salience hits,
        ...                                                   # requirement coverage, ranking
```

## Open design questions

Not settled by this outline; need a decision before implementation starts:

- **`reevaluate()` semantics.** Genuinely useful here: re-run just the offline
  analyzer/scoring-scheme stage against previously recorded fact verdicts,
  with no new LLM calls. The generic `_score_predictions()` hook assumes a
  simpler "predictions in, rescored predictions out" shape than this needs.
  `ReferencedEval` already sets the precedent of overriding `reevaluate()`
  directly instead of using that hook; `SummarizationExperiment` likely
  should too.
- **`PartialResultStore` / cache-key granularity.** The existing resume model
  assumes one `predicted_value` per document per model. Here a single
  document involves several expensive sub-calls per model (summary
  generation, fact extraction per text, several judge calls for matching,
  requirements, and salience). Whether each of those sub-calls gets its own
  resumability, or only the final `PredictionResult` does, needs to be
  decided up front, since it affects how much LLM spend a crash mid-run
  costs to redo.
