# Summarization: what landed, and what it touched

Notes from mpg for Dennis, on the branch `mpg/260820-izzy`. This is the
implementation of [`SUMMARIZATION_PLAN.md`](SUMMARIZATION_PLAN.md), which you
wrote from a review of our external codebase. The plan held up well; this
records where the implementation followed it, where it diverged and why, and
the five files of yours it touches.

Short version: `SummarizationExperiment` extends `ModelEval` directly, exactly
as you sketched. 898 tests pass, mypy reports the same 188 pre-existing errors
it did before (none in the new code), and the additions to your files come to
93 lines across five files, all additive.

## What it does

Summarization has no ground-truth string to diff against, so a judge model
decomposes each document into atomic facts and marks which of them a good
summary *must* convey. That is the reference-free part: coverage is recall, and
recall needs a target set, which here comes from the source document itself
rather than from a human summary or a panel of frontier models. Each candidate
then summarizes the document and the judge grades it on four axes -- faithfulness
to the source, coverage of the must-convey facts, precision against them, and an
optional per-class requirements checklist.

The per-document work is done once and shared by every candidate, which is why
cost does not grow with the size of the model field.

## New files

```text
src/valtron_core/summarization/          # the method, ported near-verbatim
    judge.py            the five judge operations
    prompts.py          the five prompts, plus SALIENCE_SUMMARY_PROMPT
    scoring.py          the salience-f+reqs metric
    text.py             Doc / Summary / Fact / Requirement
    model.py            the Model ABC + Usage accounting
    pipeline.py         per-document, per-candidate flow
    client_model.py     the one adapter: Model implemented over LLMClient
src/valtron_core/evaluation/summarization.py          SummarizationExperiment
src/valtron_core/reports/generate_summarization_report.py
src/valtron_core/templates/summarization_report.jinja2.html
tests/summarization/                     40 tests, incl. 3 real billsum documents
```

Everything under `summarization/` except `client_model.py` is shared verbatim
with the standalone package it came from, so that we can keep tweaking the
method in isolation and copy changes across. Please treat those files as a unit:
if something in them needs changing, it is worth a word so the two copies do not
drift.

## Changes to your files

All five are additive; nothing was removed or rewritten.

| File | Change | Lines |
|---|---|---|
| `evaluation/config.py` | `SummarizationConfig(ModelEvalConfig)` | +77 |
| `evaluation/__init__.py` | export the recipe and its config | +4 |
| `reports/__init__.py` | export `SummarizationReportGenerator` | +2 |
| `runner.py` | **bug fix**, see below | +5 |
| `partial_results.py` | **bug fix**, see below | +5 |

### The two bug fixes

These are not summarization-specific, and they were not optional for us.

**`save_single_model_result` dropped fields that `load_experiment_results`
reads.** The writer emitted `is_correct` and `example_score` but never
`task_scores` or `error`, while the reader looks for both
(`task_scores=p.get("task_scores")`, `error=p.get("error")`). For
classification that is invisible, since the signal lives in `is_correct`. For a
task whose *only* signal is `task_scores`, a saved run reloaded from disk had no
scores at all and could not be ranked. Two keys added to the dict.

**`PartialResultStore.record` had the same gap.** A prediction staged before a
crash and reused on resume came back without its `task_scores`, so the resumed
documents silently contributed nothing to the aggregate while the run still
looked complete. Same two keys.

Both are covered by regression tests in
`tests/summarization/test_experiment.py::TestPersistence`.

## Where the implementation diverged from the plan

**Your `SummarizationConfig` sketch is out of date, and smaller now.** It listed
`ground_truth_models` and `important_set_source`, which belong to the version of
our design you reviewed. We have since retired the silver panel of frontier
models and the reference-regime switch entirely: the method is reference free,
with one scoring scheme. What is left is `judge_model`, `requirements`, the four
scheme scalars (`gate`, `beta`, `requirement_weight`, `tier_gap`), and
`max_concurrent_documents`.

**We kept the `Model` abstraction rather than rewriting the judge against
`client.py`.** Reading your note again, the thing you objected to was "a second
cache/concurrency/client layer" -- and you singled out the abstract "run this
prompt, get structured output back" call as the clean part. So the abstraction
stayed and everything underneath it went: our litellm wrapper, our disk cache,
our per-vendor concurrency limiter, our retry handling. `client_model.py` is the
whole of the adaptation, and it adds only what `LLMClient` does not provide:
structured-output validation, per-call usage attribution, and a temperature
pinned to 0.

**We did not bring litellm's response cache.** You named it, and
`PartialResultStore` is this codebase's answer. Worth knowing what it costs:
re-running a finished evaluation here pays full price, where in our own repo it
is nearly free. If cheap re-runs ever matter, that is the lever.

**We generate summaries ourselves rather than reusing `EvaluationRunner`.** Not
by preference -- `evaluator.py` cannot run a reference-free task as it stands.
`PromptEvaluator.evaluate` returns `None` for any document without a label, and
`evaluate_single` unconditionally sets `expected_value` and computes
`is_correct`. See "worth considering" below.

**`reevaluate()` is deliberately unimplemented.** Your plan flagged it as an
open question and worried it would need the recorded per-fact verdicts. It
would not: the scheme's only inputs are the four axes, and those are in
`task_scores`, so re-scoring under different scheme parameters needs no LLM
calls and no verdicts. We left it out because nobody has asked to move those
knobs and the defaults are the tuned ones. It is perhaps twenty lines whenever
someone wants it.

## Things worth knowing

**Cost attribution is a judgement call, and here is the one we made.** The
per-document judge work belongs to no single candidate. Leaving it out would
make `metadata.json`'s `total_cost` understate the run, so it is divided evenly
across the candidates into each prediction's `evaluation_cost`. The consequence
is that a model's reported cost depends on how many models it was compared
against. The undivided split -- generation, per-candidate judging, shared work,
with token counts -- is on `experiment.ranking.usage` and in both reports.

**The prompt has no default.** `SummarizationConfig.prompt` is required like any
other recipe's, and `valtron_core.summarization.SALIENCE_SUMMARY_PROMPT` is the
exact text every number we have published was derived under. A `{requirements}`
placeholder, if present, gets the checklist substituted into it in
`_prepare_model_prompts`, so the prompt that is saved and displayed is the one
the candidate actually saw. Without that placeholder the checklist is still
scored but never shown to the model, which is not the configuration the method
was validated under -- we log a warning rather than refuse.

**The ranking is corpus-level, and the order of operations matters.** Axes are
averaged across the corpus first and scored once. Scoring each document and
averaging the scores is a different and much noisier number, because the
salience axes carry roughly a third of the per-document signal of
reference-based ones. `aggregated_task_scores` happens to compute exactly the
average we want, which is why the axes go in `task_scores`.

**`compute_task_statistics` output is still not persisted.** Your docstring
already notes this as a follow-up. We left it alone: the ranking is available in
memory as `experiment.ranking` (typed) and rendered into both reports, and the
per-model axes do reach disk via `aggregated_task_scores`. If you want the
ranking itself in the run directory, the natural place is a `task_statistics.json`
written by `save_run_dir` -- about six lines, and every future recipe using the
hook would get it.

**`save_run_dir` writes `metadata.json` only when it is absent.** Re-saving into
an existing run directory keeps the old run-wide metadata while refreshing the
per-model files. Not something we hit, but likely to surprise someone.

**`HtmlReportGenerator.generate_recommendation` is not None-safe.** It formats
`result.metrics.accuracy:.2%`, which raises `TypeError` when accuracy is `None`
-- so any recipe without ground truth cannot reuse it, even though
`_compute_performance_best_values` right next to it was made None-safe. The
summarization report has its own recommendation, so we did not touch it, but the
next reference-free recipe will hit the same wall.

**The report is separate, not a widening of yours.** `evaluation_report.jinja2.html`
has around forty-five references to accuracy; rather than thread None through
all of them, summarization gets its own template and generator. It reuses
`_ReportBase` for the Jinja environment and the cost/latency chart data, both of
which were already correctness-agnostic. Output is
`html_report/summarization_report.html` and `summarization_report.pdf`.

## Worth considering, not done

**Finish the label-optional conversion in `evaluator.py`.** You converted
`PredictionResult`, `EvaluationMetrics` and `ModelEval` deliberately -- "leaves
these unset rather than faking a value" -- but `evaluator.py` and `runner.py`
still assume a label per document. If they tolerated a missing one
(`expected_value=None`, `is_correct=None`, skip the string match), a
reference-free recipe could route generation through `EvaluationRunner` and
inherit `cost_rate`, attachments, retries and the rest for free. We avoided it
because it is shared code that classification and extraction both run through,
and that is your call rather than ours. It is the single change that would most
reduce what summarization has to reimplement.

## Next steps

1. **Read the numbers before trusting them.** Every test here fakes the judge.
   The method itself is validated in our own repo against a live judge and a
   research harness, but nothing in *this* repo has yet run against a real model.
   A small live run on a handful of documents is the obvious next step, and the
   thing most likely to surface a rough edge in the adapter.
2. **Decide on `evaluator.py`** (above). Everything else is downstream of it.
3. **Decide whether the ranking should be persisted**, and if so whether
   `task_statistics.json` is the right shape for every recipe rather than just
   this one.
4. **Docs.** Nothing has been added to the Docusaurus site under `docs/valtron/`
   yet; the user-guide section for summarization would mirror
   `docs/user-guide/classification/`.
5. **A judge-model preflight.** `litellm.drop_params` silently removes
   `response_format` for models that do not support structured output, and the
   judge then returns prose. `client_model.py` raises with the model named, but
   catching it in the preflight would be friendlier than failing mid-run.
