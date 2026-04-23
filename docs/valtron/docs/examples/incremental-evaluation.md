---
sidebar_position: 6
---

# Incremental Evaluation

**File:** [`examples/incremental_evaluation.py`](https://github.com/your-org/valtron-core/blob/main/examples/incremental_evaluation.py)

Shows how to extend a completed evaluation with a new model, without re-running models that were already evaluated. Prior results are loaded from disk, the new model is appended, and the report is regenerated with all models combined.

## What it demonstrates

- `ModelEval.load_experiment_results()` to restore a prior run
- `experiment.add_models()` to register new models
- How Valtron skips already-evaluated models on the second `run()` call

## Run it

```bash
python examples/incremental_evaluation.py
```

The script is self-contained: it runs the initial evaluation inline, then demonstrates loading and extending it.

## How it works

### Step 1: Initial evaluation

```python
experiment = ModelEval(config=INITIAL_CONFIG, data=DATA)
experiment.run(output_dir="examples/results/incremental")
```

`run()` saves `metadata.json` and per-model result files under `output_dir`. These are what `load_experiment_results` reads back.

### Step 2: Load and extend

```python
experiment = ModelEval.load_experiment_results("examples/results/incremental")

experiment.add_models([
    {"name": "claude-haiku-4-5-20251001", "label": "Claude Haiku"},
])

experiment.run(output_dir="examples/results/incremental")
```

Valtron checks `experiment.results` before evaluating. Any model whose label already appears in the loaded results is skipped. Only the newly added model is called.

### Step 3: Report regenerated

After the second `run()`, the HTML report at `output_dir` is rewritten to include all models: the original GPT-4o mini result and the new Claude Haiku result.

## Key points

- `load_experiment_results` reads `models/*.json` from the output directory. Results are only loadable if `save_experiment_results` (or `run`) was called previously.
- Calling `evaluate()` on an already-complete experiment with no new models added is a no-op; nothing runs.
- You can add as many new models as needed at once; they all run in parallel.
- The same `output_dir` can be passed to the second `run()`; existing result files for old models are preserved and new ones are added.

## What's next

- Run the [Sentiment Classification](./sentiment-classification) example first, then load those results here instead of running the initial evaluation inline.
- Use `save_pdf_report()` after loading to produce a PDF without re-evaluating. See [Recipes](../recipes#saving-results-and-reports-manually).
