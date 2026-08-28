# How the summarization recipe works

A full walkthrough of `SummarizationExperiment`, end to end, with code
references. Companion to [`SUMMARIZATION_PLAN.md`](SUMMARIZATION_PLAN.md) (the
original design) and [`summarization-mpg.md`](summarization-mpg.md) (what
landed vs. the plan, and open follow-ups) -- this document is the "how it
actually works" reference, for someone reading the code for the first time.

## The problem this solves

Every other recipe (`ClassificationExperiment`, `ExtractionExperiment`) scores
a prediction against a known label: exact match, or per-field diff. That does
not work for summarization -- there is no single correct summary. Two summaries
can differ in wording, length and structure and both be excellent, so there is
nothing to diff a candidate's output against.

The method used here (`salience-f+reqs`) sidesteps the need for a reference
summary entirely. Instead of comparing a candidate's summary to a *gold*
summary, it asks a judge model to read the **source document** and decide
which facts a good summary would have to convey. That judgment becomes the
target: a candidate's coverage is how many of those must-convey facts it
captured, and its precision is how much of what it wrote actually lands on
them. No human-written reference, no panel of frontier models -- the source
document is the only input the importance judgment needs.

See the module docstring at
[`summarization/__init__.py:1-30`](../src/valtron_core/summarization/__init__.py#L1-L30)
for the same summary in the code itself.

## The four axes

Every candidate summary is graded on four independent axes, computed by
[`evaluate_candidate()`](../src/valtron_core/summarization/pipeline.py#L119-L182):

| Axis | Question it answers | Judge call behind it |
|---|---|---|
| `correctness` | Is the summary faithful to the source? What fraction of the summary's own facts are supported by the document? | `fraction_supported(summary_facts, document_facts)` |
| `salient_coverage` | Of the facts a good summary *must* convey, how many did this one convey? | `fraction_supported(document_facts, summary_facts)`, then masked down to the salient subset |
| `salient_precision` | Of what the summary said, how much of it landed on must-convey material (rather than padding)? | `fraction_supported(summary_facts, salient_document_facts)` |
| `requirements_met` | Optional: how much of a per-document-class checklist did the summary satisfy? | `requirements_met(summary, checklist)` |

`correctness`, `salient_coverage` and `salient_precision` are all the *same*
judge primitive, `Judge.fraction_supported` -- "what fraction of these claims
are supported by this reference set?" -- called three times with the claim and
reference roles swapped. That's the whole trick that keeps the judge's surface
area small: see the comment at
[`judge.py:8-11`](../src/valtron_core/summarization/judge.py#L8-L11).

Each axis is `float | None`, never faked to `0.0` when undefined -- see
[`scoring.py:64-68`](../src/valtron_core/summarization/scoring.py#L64-L68). A
summary with zero extracted facts has no precision; that's different from
precision `0`, and averaging must skip it rather than drag the axis down (see
`mean_axes`, [`scoring.py:76-92`](../src/valtron_core/summarization/scoring.py#L76-L92)).

## The scoring formula (`salience-f+reqs`)

Defined in [`scoring.py:95-120`](../src/valtron_core/summarization/scoring.py#L95-L120):

```
score = 0                                              if correctness < gate
      = (1 - w) * F(salient_coverage, salient_precision) + w * requirements_met   with a checklist
      = F(salient_coverage, salient_precision)                                     without one
```

Where `F` is the harmonic mean (F-measure, parameterized by `beta`) of coverage
and precision, computed by `_f_measure()` at
[`scoring.py:142-149`](../src/valtron_core/summarization/scoring.py#L142-L149).

Three design choices worth understanding, straight from the module docstring
at [`scoring.py:1-36`](../src/valtron_core/summarization/scoring.py#L1-L36):

1. **Faithfulness is a gate, not a term.** A summary below `gate` (default
   `0.5`, see `DEFAULT_GATE`) scores exactly `0`, full stop. It is not averaged
   in with the other axes -- a fluent fabrication must never outrank a dull,
   correct summary by trading accuracy off against coverage.
2. **Coverage and precision are combined by harmonic mean, not summed.** This
   is what stops the score being a length proxy: padding a summary with extra
   claims can raise coverage, but it costs precision, and the harmonic mean
   punishes that trade rather than rewarding it.
3. **The checklist is a separate additive term, not folded into recall.** The
   docstring explains why folding it into coverage was tried and rejected: it
   inverts quality on document classes where a strong model answers a
   requirement abstractively rather than by matching a literal slot, and that
   inversion would contaminate the whole score. As its own term, a missing
   checklist just means the score falls back to the plain salience F-measure
   (`requirement_weight` has no effect when `requirements_met is None`).

`rank()` ([`scoring.py:123-139`](../src/valtron_core/summarization/scoring.py#L123-L139))
turns a `{model: score}` dict into ordered tiers, splitting a new tier only when
a score drop exceeds `tier_gap` (default `0.0` -- only an exact tie shares a
tier).

**Aggregation order matters.** Scores are computed once, from axes *averaged
across the whole corpus first* -- never as a mean of per-document scores. This
is deliberate (see `compute_task_statistics`,
[`evaluation/summarization.py:539-543`](../src/valtron_core/evaluation/summarization.py#L539-L543)):
the salience axes carry roughly a third of the per-document signal-to-noise of
a reference-based metric, so a single document is a noisy way to compare two
competent models. Averaging axes over many documents before scoring is what
recovers a reliable signal.

## The judge

[`Judge`](../src/valtron_core/summarization/judge.py#L94-L272) wraps whatever
model plays the judge and exposes exactly four operations:

- **`facts(text, source)`** -- decompose a piece of text (a document, or a
  candidate's summary) into a flat list of atomic, decontextualized facts.
  Memoized per `(text, source)` via an `asyncio.Task` cache
  ([`judge.py:108-131`](../src/valtron_core/summarization/judge.py#L108-L131)),
  so concurrent callers race onto the *same* extraction rather than each
  triggering their own -- important, since every candidate summarizing the same
  document needs that document's facts, and only the first caller should pay
  for the call.
- **`mark_salient(document_facts)`** -- for each of a document's own facts,
  would a reader be materially misinformed if a summary omitted it? This is
  the reference-free importance oracle
  ([`judge.py:183-226`](../src/valtron_core/summarization/judge.py#L183-L226)):
  it runs once per document, shared by every candidate, and its output (the
  "salient" subset) is the target set that `salient_coverage` and
  `salient_precision` are measured against.
- **`fraction_supported(claims, references)`** -- what fraction of `claims`
  does the judge find entailed by `references`. The one primitive behind three
  of the four axes (see the table above).
- **`requirements_met(summary, requirements)`** -- fraction of an optional
  checklist the summary satisfies, plus the per-item verdict.

Two robustness patterns run through all four ([`judge.py:17-22`](../src/valtron_core/summarization/judge.py#L17-L22)):

- **Chunking.** Long fact lists are split into chunks of at most
  `MAX_CLAIMS_PER_CALL` (20) and judged concurrently
  (`_chunked()`, [`judge.py:274-279`](../src/valtron_core/summarization/judge.py#L274-L279)).
  The comment there says why: omissions were "routine above ~45 claims," so
  chunk size is set well under where that started, not just under it.
- **Re-ask, never guess.** If the judge's response omits a verdict for some
  facts, the missing ones are re-asked -- up to `_MAX_VERDICT_ATTEMPTS` (3)
  times (see `fraction_supported`,
  [`judge.py:158-181`](../src/valtron_core/summarization/judge.py#L158-L181), and
  `mark_salient`,
  [`judge.py:204-226`](../src/valtron_core/summarization/judge.py#L204-L226)).
  If a fact is still unjudged after all retries, the call raises rather than
  inventing a verdict -- a guessed verdict would quietly distort a score
  instead of failing visibly. `_warn_retry()`
  ([`judge.py:262-271`](../src/valtron_core/summarization/judge.py#L262-L271))
  is what logs the `"judge omitted N ...(s); re-asking"` lines you see during a
  run.

Each judge operation is backed by its own prompt in
[`prompts.py`](../src/valtron_core/summarization/prompts.py), and those five
prompt strings are treated as a fixed interface -- see the module docstring at
[`prompts.py:1-8`](../src/valtron_core/summarization/prompts.py#L1-L8): they are
the litellm cache key, and they are the exact text every published number for
this method was derived under. Changing so much as a comma is a deliberate
deviation, not a tidy-up.

## The per-document flow

[`pipeline.py`](../src/valtron_core/summarization/pipeline.py) holds the method
itself, decoupled from any policy about concurrency, persistence, or what a
score means -- that separation is what lets the same flow serve both this
codebase's recipe and a bare research script. Two functions, composed by a
third:

1. **`extract_document_facts(doc, judge)`**
   ([`pipeline.py:104-116`](../src/valtron_core/summarization/pipeline.py#L104-L116)) --
   `judge.facts()` then `judge.mark_salient()`, in sequence (salience needs the
   fact list first). Runs once per document, its result (`DocumentFacts`)
   shared by every candidate.
2. **`evaluate_candidate(doc, model, judge, shared_facts, checklist)`**
   ([`pipeline.py:119-194`](../src/valtron_core/summarization/pipeline.py#L119-L194)) --
   one candidate's showing on one document:
   - the candidate model writes a summary (`model.run(...)`),
   - the judge extracts *that summary's* own facts (`judge.facts(summary, GENERATED)`),
   - then the four grading calls run **concurrently** via `asyncio.gather`
     ([`pipeline.py:158-168`](../src/valtron_core/summarization/pipeline.py#L158-L168)):
     faithfulness, coverage (unmasked, then filtered to the salient subset),
     precision, and requirements.
3. **`evaluate_document(doc, judge, candidates, checklist)`**
   ([`pipeline.py:197-225`](../src/valtron_core/summarization/pipeline.py#L197-L225)) --
   composes the two for the common case: extract once, then evaluate every
   candidate concurrently, catching a candidate's failure so one bad response
   doesn't void the whole document. **Not used by the recipe** --
   `SummarizationExperiment` re-implements this composition itself (see below)
   because it needs to interleave with this codebase's own caching, resume and
   progress machinery. `evaluate_document` exists for a standalone caller.

## How the recipe wires into `ModelEval`

[`SummarizationExperiment`](../src/valtron_core/evaluation/summarization.py#L178-L916)
extends `ModelEval` directly -- a **sibling** of `ReferencedEval`
(`ClassificationExperiment`/`ExtractionExperiment`'s base), not a child of it.
`ModelEval` is deliberately correctness-agnostic; `ReferencedEval` is the
subclass that layers on "score against a known label." Summarization has no
such label, so it skips `ReferencedEval` and implements `ModelEval`'s
extension points directly. See
[`reference/README.md`](README.md#class-hierarchy-behind-this) for the class
hierarchy this fits into.

### Construction

```python
from valtron_core.evaluation import SummarizationExperiment

experiment = SummarizationExperiment(
    config={
        "models": [{"name": "gpt-4.1"}, {"name": "gpt-4.1-mini"}],
        "judge_model": "gpt-4.1",
        "requirements": ["Name the parties.", "State the outcome."],
        "output_dir": "./results",
    },
    data=[{"id": "0001", "content": "..."}],
)
experiment.evaluate()
print(experiment.ranking.best)
```

(the same shape used by [`examples/summarization_example.py`](../examples/summarization_example.py)).
`data` needs only `id` and `content`; a `label` is ignored if present -- see
the class docstring at
[`evaluation/summarization.py:178-200`](../src/valtron_core/evaluation/summarization.py#L178-L200).

`SummarizationConfig`
([`evaluation/config.py:251-325`](../src/valtron_core/evaluation/config.py#L251-L325))
is the config model, extending `ModelEvalConfig`:

| Field | Default | What it controls |
|---|---|---|
| `prompt` | `SALIENCE_SUMMARY_PROMPT` | The prompt template, requiring a `{content}` placeholder. Overrides `BaseRecipeConfig.prompt`, which has no default, with the prompt this scoring method was validated under. |
| `judge_model` | `gemini/gemini-2.5-pro` | The model that decomposes facts, marks salience, and grades every candidate. A plain `str` only -- unlike a candidate `LLMModelConfig`, there is currently no way to pass litellm params (e.g. `reasoning_effort`) to it. |
| `requirements` | `[]` | The optional per-class checklist, authored once rather than per document. |
| `gate` | `0.5` (`DEFAULT_GATE`) | Minimum `correctness` to score above zero. |
| `beta` | `1.0` (`DEFAULT_BETA`) | F-measure beta; above 1 favors coverage over precision. |
| `requirement_weight` | `0.6` (`DEFAULT_REQUIREMENT_WEIGHT`) | Weight on the requirements term when a checklist is supplied. |
| `tier_gap` | `0.0` (`DEFAULT_TIER_GAP`) | Score drop that starts a new ranking tier. |
| `max_concurrent_documents` | `5` | Documents in flight at once, per model, during both fact extraction and candidate evaluation. The main lever on how hard a run leans on provider rate limits. |

`prompt` overrides `BaseRecipeConfig.prompt`, which has no default, with
`SALIENCE_SUMMARY_PROMPT` -- the configuration the published numbers came from.
Pass your own to deviate deliberately (see the class docstring,
[`evaluation/config.py:260-265`](../src/valtron_core/evaluation/config.py#L260-L265)).
If your prompt contains a `{requirements}` placeholder, the checklist gets
substituted in; without one, the checklist is still scored but never shown to
the candidate, and `_validate_task_data` logs a warning
([`evaluation/summarization.py:268-276`](../src/valtron_core/evaluation/summarization.py#L268-L276)).

`_post_init()`
([`evaluation/summarization.py:217-242`](../src/valtron_core/evaluation/summarization.py#L217-L242))
builds the `Judge` once (wrapping a `ClientModel` around `judge_model`) and
three separate `Usage` accumulators -- generation, per-candidate judging, and
shared judging -- because "what did this cost?" has three different honest
answers, not one.

### Preflight: `_validate_task_data()`

([`evaluation/summarization.py:248-281`](../src/valtron_core/evaluation/summarization.py#L248-L281))
Runs before anything else and rejects, up front:

- structured (non-string) document content -- summarization needs one string
  per document, and there's no convention for which field of a dict is "the
  document",
- blank documents,
- a prompt missing the `{content}` placeholder,
- transformer models in `models` -- summarization needs free text generation,
  which a local classifier can't do.

### Phase 1: shared fact extraction -- `_before_evaluation()`

([`evaluation/summarization.py:287-325`](../src/valtron_core/evaluation/summarization.py#L287-L325))
Overrides the `ModelEval` extension point that runs "after preflight, before
prompt resolution." For every document, concurrently (capped by
`asyncio.Semaphore(max_concurrent_documents)`), calls
`extract_document_facts()` and stashes the result in
`self._document_facts[document.id]`. This is the phase with **no per-document
progress bar of its own** beyond the `tqdm` "Reading documents" bar -- a judge
retry inside `mark_salient` can make it look stalled for a while even though
it's working.

### Phase 2: prompt preparation -- `_prepare_model_prompts()`

([`evaluation/summarization.py:337-346`](../src/valtron_core/evaluation/summarization.py#L337-L346))
Fills the `{requirements}` placeholder into every model's prompt *before*
evaluation, not at call time -- so the prompt persisted and shown in the report
is the exact prompt the candidate actually saw.

### Phase 3: per-model, per-document evaluation

- **`_run_evaluations()`**
  ([`evaluation/summarization.py:366-380`](../src/valtron_core/evaluation/summarization.py#L366-L380))
  records how many models are in the current pass (`self._models_in_pass`)
  before delegating to the base class, which runs every model concurrently via
  `asyncio.gather` (see `ModelEval._run_evaluations`,
  [`evaluation/model_eval.py:845-847`](../src/valtron_core/evaluation/model_eval.py#L845-L847)).
  This count is the divisor for dividing up each document's shared judge cost
  -- it's the number of models *actually running in this pass*, not the number
  configured, because `add_models()` + another `evaluate()` re-does the shared
  work only for the new models, and only they should pay for it.
- **`_evaluate_model_documents()`**
  ([`evaluation/summarization.py:382-426`](../src/valtron_core/evaluation/summarization.py#L382-L426))
  runs one model's documents concurrently (its own `max_concurrent_documents`
  semaphore) via `_evaluate_one()` per document.
- **`_evaluate_one()`**
  ([`evaluation/summarization.py:428-510`](../src/valtron_core/evaluation/summarization.py#L428-L510))
  is where `evaluate_candidate()` from `pipeline.py` actually gets called,
  using the shared `DocumentFacts` from phase 1. Builds the `PredictionResult`:
  - `task_scores` carries the four axes (whichever are defined) -- this is the
    seam that makes `aggregated_task_scores` give the corpus-level axes "for
    free."
  - `expected_value`, `is_correct`, `example_score` are left `None` --
    deliberately not faked.
  - `evaluation_cost` = this candidate's own judge spend (faithfulness/
    coverage/precision/requirements calls) **plus an even split of the shared
    per-document extraction cost** across `self._models_in_pass`. The
    docstring at [`evaluation/summarization.py:472-477`](../src/valtron_core/evaluation/summarization.py#L472-L477)
    is explicit that this is a judgment call: the shared work belongs to no
    single candidate, but omitting it from `total_cost` would understate the
    run.
  - `metadata` carries everything needed to argue with the verdict later: the
    document's own facts, the salient subset, the summary's facts, and all
    four kinds of per-fact verdicts.
  - A document that raises is caught and recorded as an errored prediction
    ([`evaluation/summarization.py:439-465`](../src/valtron_core/evaluation/summarization.py#L439-L465))
    rather than voiding the whole model's run.

### Phase 4: ranking -- `compute_task_statistics()`

([`evaluation/summarization.py:532-588`](../src/valtron_core/evaluation/summarization.py#L532-L588))
For each model's results: pulls every prediction's `task_scores`, rebuilds an
`Axes` per document (`_axes_from()`,
[`evaluation/summarization.py:919-927`](../src/valtron_core/evaluation/summarization.py#L919-L927)),
averages them corpus-wide with `mean_axes()`, and scores the average once with
`score()`. Builds the `self._ranking` (`SummarizationRanking`,
[`evaluation/summarization.py:97-117`](../src/valtron_core/evaluation/summarization.py#L97-L117)) --
tiers, per-model `SummarizationScore` entries, the scheme's four scalars (so
the number is reproducible), and the three-way usage split. Note this method
works from each prediction's stored `task_scores` rather than from
`self`-held state, so it works identically on a run reloaded from disk via
`load_experiment_results()`.

`experiment.ranking` (a property,
[`evaluation/summarization.py:908-916`](../src/valtron_core/evaluation/summarization.py#L908-L916))
exposes it as typed objects rather than the raw dict `compute_task_statistics`
returns.

### Reevaluation -- `reevaluate()` / `areevaluate()`

([`evaluation/summarization.py:594-708`](../src/valtron_core/evaluation/summarization.py#L594-L708),
plus the two regrade helpers below.) Three tiers, cheapest first, chosen by
which of `judge_model`/`requirements`/the four scheme scalars are passed:

- **Reweight** (`gate`/`beta`/`requirement_weight`/`tier_gap`) -- pure
  arithmetic over axes already sitting in `task_scores`; `compute_task_statistics()`
  is simply re-run. No LLM calls at all.
- **Requirements-only regrade** (`requirements` changes, `judge_model` doesn't,
  `_regrade_requirements()`,
  [`evaluation/summarization.py:798-844`](../src/valtron_core/evaluation/summarization.py#L798-L844)) --
  correctness/coverage/precision cannot depend on the checklist, so only
  `Judge.requirements_met()` reruns per prediction; the other three axes are
  untouched, and cost is additive on top of the original run's `evaluation_cost`.
- **Full regrade** (`judge_model` changes, `_regrade_fully()`,
  [`evaluation/summarization.py:715-796`](../src/valtron_core/evaluation/summarization.py#L715-L796)) --
  a different judge has its own opinions about salience from the ground up, so
  document facts, salience, and all four axes are recomputed via
  `extract_document_facts()` and `evaluate_candidate()`, exactly as a fresh
  `evaluate()` would.

No tier ever regenerates a candidate's summary: every stored `predicted_value`
is wrapped in a small private `Model` (`_StoredSummary`,
[`evaluation/summarization.py:161-175`](../src/valtron_core/evaluation/summarization.py#L161-L175))
that replays it instead of calling an LLM, so `evaluate_candidate()` is reused
unchanged and a regrade only ever pays for judge calls. `reevaluate()` wraps
`areevaluate()` in `asyncio.run()`, the same split `evaluate()`/`aevaluate()`
already use -- call `areevaluate()` directly from inside a running event loop.

### Reports

`save_html_report()` / `save_pdf_report()`
([`evaluation/summarization.py:850-886`](../src/valtron_core/evaluation/summarization.py#L850-L886))
override the base class's default `NotImplementedError` (the base report
assumes an accuracy notion this task doesn't have) and delegate to
[`SummarizationReportGenerator`](../src/valtron_core/reports/generate_summarization_report.py#L66),
which reuses `_ReportBase` for the Jinja environment and cost/latency chart
data (both already correctness-agnostic) but builds its own recommendation
prompt and its own template
([`templates/summarization_report.jinja2.html`](../src/valtron_core/templates/summarization_report.jinja2.html)),
answering: which model won and by how much (tiers), *why* (the four axes shown
alongside every score, since a zero from a failed faithfulness gate means
something very different from a zero from thin coverage), what the run spent
(generation vs. per-candidate judging vs. shared judging), and what the judge
actually decided per document.

## Cost accounting, end to end

Three separate `Usage` accumulators
([`model.py:27-72`](../src/valtron_core/summarization/model.py#L27-L72) for the
type) live on the experiment:

- `self._generation_usage` -- what every candidate spent writing summaries.
- `self._candidate_judge_usage` -- what the judge spent grading each candidate
  (the four per-candidate calls).
- `self._shared_judge_usage` -- what the judge spent on the one-per-document
  work (fact extraction + salience marking), merged in during
  `_before_evaluation`.

Per prediction, `evaluation_cost` = that candidate's own judge spend + an even
share of the document's extraction cost across however many models are in the
current evaluation pass. This is why the recipe's own docstring
([`evaluation/summarization.py:1-30`](../src/valtron_core/evaluation/summarization.py#L1-L30))
notes that "the per-document work is shared across candidates, so cost does
not grow with the size of the model field" -- but also why a model's *reported*
cost depends on how many other models it was compared against in the same run.

## Where the real LLM calls happen, and how many

Every `ClientModel._call()`
([`client_model.py:104-121`](../src/valtron_core/summarization/client_model.py#L104-L121))
goes through the **same shared `LLMClient` instance** (`self.client` on the
experiment), whether it's a candidate generating a summary or the judge
grading one. That client owns retries (`max_retries`/`retry_delay`, default 3
retries / 1s base delay, see
[`client.py:159-160`](../src/valtron_core/client.py#L159-L160)) and an optional
`requests_per_minute` throttle
([`client.py:48-64`](../src/valtron_core/client.py#L48-L64)) -- both shared
across every call the experiment makes, judge and candidates alike. There is
currently no per-call request timeout configured anywhere in that path; a slow
or hung call is bounded only by litellm's own default.

For `N` documents and `M` candidate models with no checklist, expect roughly:

- `N` calls to `judge.facts()` (document decomposition) -- deduplicated by the
  fact cache if any document text repeats.
- up to `N * _MAX_VERDICT_ATTEMPTS` calls to the salience judge, per document
  (usually just 1 if nothing is omitted).
- `N * M` candidate generation calls.
- `N * M` calls to `judge.facts()` on each summary.
- `N * M` batches of 4 concurrent grading calls (faithfulness, coverage,
  precision, requirements -- fewer if a checklist is absent, since
  `requirements_met` short-circuits with no checklist).

Concurrency is capped at `max_concurrent_documents` *per model*, and all models
run concurrently with each other -- see the earlier parallelism breakdown in
this project's session notes, or re-derive it from
[`evaluation/model_eval.py:845-847`](../src/valtron_core/evaluation/model_eval.py#L845-L847)
(all models via `asyncio.gather`) and
[`evaluation/summarization.py:382-426`](../src/valtron_core/evaluation/summarization.py#L382-L426)
(`_evaluate_model_documents`, documents within a model via a semaphore).

## Persistence and resume

Two pre-existing bugs (not summarization-specific, but invisible until this
recipe existed) were fixed as part of this work, both in the same commit:

- [`runner.py`'s `save_single_model_result`](../src/valtron_core/runner.py#L78-L88)
  was writing `is_correct`/`example_score` to disk but never `task_scores` or
  `error`, even though `load_experiment_results` reads both back. Invisible
  for classification, whose only signal is `is_correct`; fatal for a task whose
  *only* signal is `task_scores` -- a reloaded run had nothing to rank.
- [`partial_results.py`'s `PartialResultStore.record`](../src/valtron_core/partial_results.py#L167-L175)
  had the same gap for a run resumed mid-flight: a prediction staged before a
  crash came back on resume without its `task_scores`, so the resumed
  documents silently contributed nothing to the aggregate score while the run
  still looked complete.

Both are now fixed by writing the two extra keys, and both are covered by
regression tests in `tests/summarization/test_experiment.py::TestPersistence`.

## What's deliberately not implemented yet

From [`summarization-mpg.md`](summarization-mpg.md), for anyone extending
this:

- **`compute_task_statistics`'s output isn't persisted to disk.** The ranking
  lives in memory (`experiment.ranking`) and reaches both reports, and the
  per-model axes reach disk via `aggregated_task_scores`, but the ranking
  object itself is not written as its own file (e.g. a `task_statistics.json`).
- **Generation bypasses `EvaluationRunner`.** `PromptEvaluator.evaluate()`
  assumes a label exists; a reference-free recipe can't route through it
  without a label-optional conversion of `evaluator.py`/`runner.py` that
  hasn't been made yet.
- **No response cache.** Unlike the standalone research package this method
  was ported from, this codebase's answer to "make a re-run cheap" is
  `PartialResultStore` (crash resume), not a cache -- re-running a *finished*
  evaluation here pays full price again.
- **No timeout on LLM calls** (see the cost-accounting section above) -- a
  real gap surfaced by testing this recipe against a reasoning-heavy judge
  model, not specific to summarization itself.

## Try it yourself

[`examples/summarization_example.py`](../examples/summarization_example.py)
runs the recipe end to end against the three public-domain billsum documents
in `tests/summarization/data/billsum/` (see that directory's `README.md` for
their provenance). Run with:

```bash
poetry run python examples/summarization_example.py
```

It needs `OPENAI_API_KEY` set and makes real, billed calls to OpenAI for both
candidates and the judge.
