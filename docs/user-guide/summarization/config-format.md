# Config Format

The evaluation config is where you set up the experiment: which candidate models to compare, what prompt they receive, and the judge model that scores them.

To run this, set your provider API keys in a `.env` file. See LiteLLM's [provider documentation](https://docs.litellm.ai/docs/providers) for the available providers and their key names, for example `OPENAI_API_KEY` or `GEMINI_API_KEY`.

You can pass a config as a Python dict or a path to a JSON file. It validates as [`SummarizationConfig`](../../api/summarization_config), which extends `ModelEvalConfig`.

## Basic Structure

A config is a plain dict, or a path to a JSON file with the same shape, with one field required by every recipe, `models`, plus one that's specific to this one: `judge_model`.

```python
from valtron_core.evaluation import SummarizationExperiment

config = {
    "models": [{"name": "gpt-4o-mini"}, {"name": "gpt-4o"}],
    "judge_model": "gpt-4o",
    "output_dir": "./results",
}

experiment = SummarizationExperiment(config=config, data=data)
experiment.run()
```

`prompt` is a required field on every other recipe's config, but `SummarizationConfig` gives it a default: `SALIENCE_SUMMARY_PROMPT`, imported from `valtron_core.summarization`, the prompt this scoring method was validated under. Leave `prompt` unset, as above, to use it, or set your own to deviate deliberately. Whichever prompt is used, it must contain a `{content}` placeholder for the document, the same as in classification and extraction experiment prompts.

`judge_model` is the model that decides which of a document's facts are must-convey and then grades every candidate against that decision, so its own quality bounds how trustworthy every score in the run is. It defaults to `gemini/gemini-2.5-pro`. It can be changed to any other LLM, but keep it fixed across runs you intend to compare. Changing it changes what "must-convey" means for every document, not just how strictly the grading is applied.

[Transformer models](../self-hosting/transformer-models) are not supported in `models`.

---

## The Requirements Checklist

`requirements` is an optional list of criteria a good summary of this document *class* should satisfy, such as the tone, style, or required information to include:

```python
config["requirements"] = [
    "Name the parties involved.",
    "State the outcome or decision.",
    "Mention any dollar amount involved.",
]
```

Each item is scored independently by the judge as a `requirements_met` fraction. That fraction is added into the final score as its own weighted quantity, separate from how coverage and precision are computed (see [Summarization: the scoring formula](./index.md)). With no `requirements` configured, the score is just the harmonic mean of `salient_coverage` and `salient_precision`, and `requirements_met` reports as `n/a` on every candidate.

If your `prompt` contains a `{requirements}` placeholder, the checklist is rendered into it before evaluation starts, so the candidate actually sees what it's being asked to cover. `SALIENCE_SUMMARY_PROMPT` has one built in. Without that placeholder, the checklist is still scored but never shown to the candidate, which is a valid but different configuration than the one the method was validated under. `SummarizationExperiment` logs a warning in that case rather than failing the run.

---

## Scoring Parameters

Four fields tune how the four axes combine into a score, all with defaults chosen for the published numbers behind this method:

| Field | Default | What it controls |
|---|---|---|
| `gate` | `0.5` | Minimum `correctness` for a summary to score above zero at all. |
| `beta` | `1.0` | F-measure beta over `salient_coverage` and `salient_precision`; above `1` weights coverage more heavily than precision. |
| `requirement_weight` | `0.6` | Weight on the `requirements_met` term when a checklist is supplied; ignored otherwise. |
| `tier_gap` | `0.0` | Score drop that starts a new tier in the ranking. Zero by default, so only an exact tie shares a tier. |

```python
config.update({"gate": 0.6, "beta": 1.5, "requirement_weight": 0.4})
```

These are also the arguments to `reevaluate()`, which lets you try different values against an already-run evaluation without paying for new judge calls; see [Evaluation API: Reweighting and Regrading](./evaluation-api.md#reweighting-and-regrading-a-run).

`max_concurrent_documents` (default `5`) is a performance parameter rather than a scoring one: how many documents are in flight at once per model, during both fact extraction and candidate grading. Each document fans out into several judge calls, so this is the main lever on how hard a run leans on your provider's rate limits; see [LLM Call Volume](#llm-call-volume) below for what that fan-out actually looks like.

---

## LLM Call Volume

For `N` documents and `M` candidate models, a run makes roughly:

- `N` calls to decompose each document into facts, plus about `N` more to mark which of them are salient (up to 3x that if the judge omits a verdict and has to be re-asked, but usually just one pass). Shared once per document, so this part doesn't scale with `M`.
- `N * M` candidate generation calls, one summary per model per document.
- `N * M` calls to decompose each of those summaries into its own facts.
- `N * M` grading calls per axis: `correctness`, `salient_coverage`, and `salient_precision` always run, and `requirements_met` runs as a fourth call only when a `requirements` checklist is configured; it makes no call at all otherwise.

The per-candidate work is what scales, and with a checklist configured it's 4 grading calls plus the generation and fact-extraction calls above, 6 calls per document per candidate model. So a run's total is ultimately **`6 * N * M`** LLM calls, on top of the roughly `2 * N` shared calls for fact extraction and salience marking that don't grow with `M`. Drop the checklist and the per-candidate figure falls to `5 * N * M`, since `requirements_met` no longer makes a call.

This is why `max_concurrent_documents` matters more here than in classification or extraction: doubling `M` roughly doubles the total call volume, not just the number of candidates being compared.

---

## Passing Config to SummarizationExperiment

`config` can be passed as a Python dict, as shown throughout this page, or as a path to a JSON file with the same shape, the same as the other recipes:

```python
experiment = SummarizationExperiment(config={"models": [...], "judge_model": "gpt-4o"}, data=data)
experiment = SummarizationExperiment(config="./config.json", data="./data.json")
```

```{rubric} What's next?
```

- To run your evaluation, see [Evaluation API](./evaluation-api).
- To read the axes and the ranking off a run, see [Evaluation Results](./evaluation-results).
