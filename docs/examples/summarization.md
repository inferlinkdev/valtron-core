# Summarization

**File:** [`examples/summarization_example.py`](https://github.com/inferlinkdev/valtron-core/blob/main/examples/summarization_example.py)

Ranks `gpt-4.1` against `gpt-4.1-mini` on three public-domain Congressional bills. An LLM judge decomposes each bill into the facts a good summary must convey, then grades both candidates' summaries against that, with no reference summary required.

## What it demonstrates

- Minimal [`SummarizationExperiment`](../api/summarization_experiment) setup: no `label`, no `response_format`
- The default `prompt` (`SALIENCE_SUMMARY_PROMPT`), left unset in the config
- Reading the cross-model [`SummarizationRanking`](../api/summarization_ranking) off `experiment.ranking`

## Run it

```bash
python examples/summarization_example.py
```

Requires `OPENAI_API_KEY` (see `.env`); makes real, billed calls to OpenAI for both candidates and the judge.

## Data files

Data is loaded from three public-domain Congressional bill texts in [`examples/summarization/`](https://github.com/inferlinkdev/valtron-core/tree/main/examples/summarization). Each document needs only an `id` and its `content`, no `label`:

```python
DATA_DIR = Path(__file__).resolve().parent / "summarization"
DOCUMENT_IDS = ["0001", "0003", "0006"]

DATA = [
    {"id": doc_id, "content": (DATA_DIR / f"{doc_id}.txt").read_text()} for doc_id in DOCUMENT_IDS
]
```

See [Data Format](../user-guide/summarization/data-format) for more details.

## Full code

```python
from pathlib import Path

from valtron_core.evaluation import SummarizationExperiment

DATA_DIR = Path(__file__).resolve().parent / "summarization"
DOCUMENT_IDS = ["0001", "0003", "0006"]

DATA = [
    {"id": doc_id, "content": (DATA_DIR / f"{doc_id}.txt").read_text()} for doc_id in DOCUMENT_IDS
]

CONFIG = {
    "models": [{"name": "gpt-4.1"}, {"name": "gpt-4.1-mini"}],
    "judge_model": "gpt-5.4-mini",
}

if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "results" / "summarization"

    experiment = SummarizationExperiment(config=CONFIG, data=DATA)
    report_path = experiment.run(output_dir=output_dir)

    print(f"\nReport: {report_path}\n")
    print("Best model(s):", experiment.ranking.best)
    for candidate in experiment.ranking.scores:
        axes = ", ".join(
            f"{name}={value:.0%}" if value is not None else f"{name}=n/a"
            for name, value in candidate.axes().items()
        )
        print(f"  {candidate.model:<20}  score={candidate.score:.0%}  {axes}")
```

## Key points

- `CONFIG` sets no `prompt`, so `SummarizationConfig` fills in `SALIENCE_SUMMARY_PROMPT`, the prompt this scoring method was validated under. See [Config Format](../user-guide/summarization/config-format).
- `judge_model` here (`gpt-5.4-mini`) is a separate model from either candidate; its own quality bounds how trustworthy every score in the run is, so it's worth choosing deliberately.
- `experiment.ranking` is only available after `run()` (or `evaluate()`); unlike classification and extraction, there is no `label`, `accuracy`, or `is_correct` on the results.
- Each candidate's `.axes()` gives the four scoring axes: `correctness`, `salient_coverage`, `salient_precision`, and `requirements_met`. The last reports `n/a` here, since no `requirements` checklist was configured.

## What's next

- Add a `requirements` checklist to score against a per-document-class rubric. See [Config Format: The Requirements Checklist](../user-guide/summarization/config-format.md#the-requirements-checklist).
- Reweight or regrade this run after the fact without regenerating a single summary. See [Evaluation API: Reweighting and Regrading](../user-guide/summarization/evaluation-api.md#reweighting-and-regrading-a-run).
- Read the full chapter: [Summarization](../user-guide/summarization/index).
