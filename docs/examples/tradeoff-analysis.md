# Tradeoff Analysis

**File:** [`examples/tradeoff_analysis_workflow.py`](https://github.com/your-org/valtron-core/blob/main/examples/tradeoff_analysis_workflow.py)

Runs the full three-stage pipeline on a customer review sentiment dataset: train a DistilBERT classifier, evaluate it against two LLM tiers with [`ClassificationExperiment`](../api/classification_experiment), then use [`TradeoffAnalyzer`](../api/tradeoff_analyzer) to find the confidence threshold at which predictions should be escalated from the transformer to an LLM.

## What it demonstrates

- Chaining `TransformerClassifier`, `ClassificationExperiment`, and `TradeoffAnalyzer` in one script
- Building `TradeoffAnalyzer.from_model_eval()` from a completed evaluation run, with no re-evaluation
- Saving both an HTML and a JSON tradeoff report from a single `analyze()` call
- A two-tier LLM setup (`gpt-4o-mini`, `gpt-4o`) so the report also sweeps a multi-tier cascade

## Requirements

```bash
pip install "valtron-core[transformers]"
```

Set `OPENAI_API_KEY` (or the equivalent variable for whichever LLM you point the config at).

## Run it

```bash
python examples/tradeoff_analysis_workflow.py
```

## How it works

### Stage 1: Train

A `TransformerClassifier` fine-tunes `distilbert-base-uncased` on 20 labeled reviews (binary positive / negative) and saves it to `examples/results/tradeoff_workflow/transformer/final_model/`:

```python
classifier = TransformerClassifier(
    model_name="distilbert-base-uncased",
    output_dir=str(TRANSFORMER_DIR),
)
train_dataset, test_dataset = classifier.prepare_data(documents=documents, labels=labels, test_size=0.2)
classifier.train(train_dataset=train_dataset, test_dataset=test_dataset)
```

### Stage 2: Evaluate

`ClassificationExperiment` runs the transformer next to two LLM tiers. `cost_rate` on the transformer config puts its cost on the same footing as the LLMs' token cost:

```python
CONFIG = {
    "models": [
        {
            "type": "transformer",
            "label": "DistilBERT",
            "model_path": str(TRANSFORMER_PATH),
            "cost_rate": 0.085,
            "cost_rate_time_unit": "1hr",
        },
        {"name": "gpt-4o-mini", "label": "GPT-4o Mini"},
        {"name": "gpt-4o",      "label": "GPT-4o"},
    ],
}

experiment = ClassificationExperiment(config=CONFIG, data=DATA)
experiment.run(output_dir=RESULTS_DIR / "eval")
```

### Stage 3: Analyze

`TradeoffAnalyzer.from_model_eval()` reuses the predictions already recorded on `experiment.results`. Calling `analyze()` once lets you save both report formats without recomputing the sweep:

```python
analyzer = TradeoffAnalyzer.from_model_eval(experiment)
analyzer.analyze()
analyzer.save_html_report(RESULTS_DIR / "tradeoff_report.html")
analyzer.save_json_report(RESULTS_DIR / "tradeoff_report.json")
```

Open the HTML report to see, at each confidence threshold, what fraction of predictions the transformer can handle on its own versus escalating to `gpt-4o-mini` or `gpt-4o`, and what that mix costs compared to sending everything to a single model.

## Key points

- Because there are two LLM tiers in the config, the report includes a multi-tier cascade sweep, not just a single transformer-to-LLM threshold. See [Combining Multiple Models: Multi-tier cascades](../user-guide/combining-models.md#multi-tier-cascades).
- `TradeoffAnalyzer` supports binary classification only; this dataset is positive/negative sentiment for that reason.
- The 20-item dataset is for demonstration. A real cost/accuracy tradeoff sweep is more reliable with a larger, held-out evaluation set.

## What's next

- Read the full API reference: [Combining Multiple Models](../user-guide/combining-models)
- Train on your own dataset: [Transformer Models](../user-guide/self-hosting/transformer-models)
