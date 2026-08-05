# Combining Multiple Models

A transformer classifier is fast and free to run, and it handles most cases well. It still struggles on the harder, more ambiguous cases: the ones an LLM is more likely to get right. An LLM is more capable across the board, but it costs money on every call, including the majority the transformer already handles correctly for free.

Valtron lets you route between the two instead of picking one for every document: send each document to the transformer first, keep its answer when it's confident, defer to an LLM otherwise. A support-ticket classifier that's correct on 90% of tickets for free, and silently wrong on the rest, only needs to pay for LLM calls on that harder 10%:

```python
if transformer_confidence >= threshold:
    label = transformer_prediction
else:
    label = call_llm(document)
```

[`TradeoffAnalyzer`](../api/tradeoff_analyzer) is what finds that `threshold`. It's a meta-analysis tool: it doesn't classify any documents itself. It takes predictions a transformer and one or more LLMs already produced on the same documents and works out the breaking point, the confidence level below which the transformer's predictions are unreliable enough that deferring to an LLM is worth the cost. It does this by sweeping the confidence values the transformer actually produced, computing the blended cost and accuracy of deferring below each one, and reporting the frontier of thresholds that no cheaper threshold beats on both cost and accuracy. This is useful for:

- Finding the cheapest confidence threshold that still hits a target accuracy
- Comparing a multi-tier cascade (transformer, then a cheap LLM, then an expensive LLM) against sending everything to a single model
- Estimating cost at scale under different accuracy targets, before committing to an all-LLM or all-transformer approach

The rest of this chapter walks through producing that analysis end to end: train and evaluate a transformer alongside your LLMs with [Transformer Models](./self-hosting/transformer-models) and the [Evaluation API](./classification/evaluation-api), then feed the completed run into `TradeoffAnalyzer`.

See the full example in [Tradeoff Analysis](../examples/tradeoff-analysis).

## Constraints

- Binary classification only. `TradeoffAnalyzer.from_model_eval()` raises `ValueError` if the ground truth has more than two unique labels.
- Requires exactly one transformer model config and at least one LLM model config in the `ModelEval` experiment, and the transformer predictions must have `confidence_score` populated (see [`PredictionResult`](../api/prediction_result)).
- Requires the `transformers` extra:

```bash
pip install "valtron-core[transformers]"
```

---

## Step 1: Train and evaluate

Reuse [`TransformerClassifier`](../api/transformer_classifier) and [`ModelEval`](../api/model_eval) exactly as described in [Transformer Models](./self-hosting/transformer-models). Set `cost_rate` and `cost_rate_time_unit` on the transformer model config so its cost is measured the same way as the LLMs (see [Transformer Models: Cost tracking for transformers](./self-hosting/transformer-models.md#cost-tracking-for-transformers)):

```python
from valtron_core.evaluation import ModelEval
from valtron_core.training import TransformerClassifier

classifier = TransformerClassifier(model_name="distilbert-base-uncased", output_dir="./model")
train_dataset, test_dataset = classifier.prepare_data(documents=documents, labels=labels, test_size=0.2)
classifier.train(train_dataset=train_dataset, test_dataset=test_dataset)

config = {
    "prompt": "Classify the sentiment: {content}",
    "models": [
        {
            "type": "transformer",
            "label": "DistilBERT",
            "model_path": "./model/final_model",
            "cost_rate": 0.085,
            "cost_rate_time_unit": "1hr",
        },
        {"name": "gpt-4o-mini"},
        {"name": "gpt-4o"},
    ],
}

experiment = ModelEval(config=config, data=data)
experiment.run(output_dir="./results/eval")
```

## Step 2: Analyze the tradeoff

Build a `TradeoffAnalyzer` from the completed `ModelEval` run with `from_model_eval()`. It expects `experiment` to already hold predictions from the transformer and every LLM you want it to consider, so call it only after `experiment.run()` or `experiment.evaluate()` has finished. It reuses those predictions rather than calling any model itself, and raises `ValueError` if `experiment.results` doesn't already contain exactly one transformer result and at least one LLM result:

```python
from valtron_core.analysis import TradeoffAnalyzer

analyzer = TradeoffAnalyzer.from_model_eval(
    experiment,
    transformer_instance_hourly=0.085,      # only used if cost_rate is not set on the transformer config
    transformer_samples_per_second=8.0,     # only used if cost_rate is not set on the transformer config
)
analyzer.analyze()
```

`analyze()` (and its async form `aanalyze()`) computes the confidence-threshold sweep and stores it in memory, mirroring `ModelEval.evaluate()`. `transformer_instance_hourly` and `transformer_samples_per_second` only come into play when the transformer config has no `cost_rate`; when it does, cost per call is derived from the recorded response times instead.

## Step 3: Save the report

```python
analyzer.save_html_report("./results/tradeoff_report.html")
analyzer.save_json_report("./results/tradeoff_report.json")
```

The HTML report renders the interactive threshold sweep. The JSON report contains the full sweep, cells, Pareto indices, and baselines for a custom UI. Both require `analyze()` to have run first.

For the common case of one HTML report, `run()` (or its async form `arun()`) combines `analyze()` and `save_html_report()` into a single call:

```python
TradeoffAnalyzer.from_model_eval(experiment).run("./results/tradeoff_report.html")
```

## Multi-tier cascades

When the experiment has two or more LLM models, `TradeoffAnalyzer` also sweeps multi-tier cascades: transformer to the cheapest LLM tier, escalating further to more expensive tiers only when confidence stays low. The cheapest tier that meets the accuracy target at each confidence band is selected automatically, and the cascade results appear alongside the single-LLM sweep in both the HTML and JSON reports.

---

```{rubric} What's next?
```

- Train the transformer this analysis routes to: [Transformer Models](./self-hosting/transformer-models)
- Produce the `ModelEval` run this analysis reads from: [Evaluation API](./classification/evaluation-api)
