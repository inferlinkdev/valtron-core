---
sidebar_position: 4
---

# Transformer Comparison

**File:** [`examples/transformer_comparison.py`](https://github.com/your-org/valtron-core/blob/main/examples/transformer_comparison.py)

Trains a DistilBERT text classifier on a news-topic dataset, then compares it against cloud LLMs in a single evaluation report. The transformer appears as a normal model entry and gets its own accuracy, cost, and latency columns.

## What it demonstrates

- Training a local `TransformerClassifier` from labeled data
- Adding a `"type": "transformer"` model entry to the config
- Cost tracking for local models via `cost_rate`

## Requirements

```bash
pip install torch transformers datasets
```

## Run it

```bash
python examples/transformer_comparison.py
```

## How it works

### Step 1: Train

`TransformerClassifier` fine-tunes `distilbert-base-uncased` on the 16-document dataset and saves the model to `examples/results/transformer/final_model/`.

```python
from valtron_core.transformer_classifier import TransformerClassifier
from valtron_core.models import Document, Label

classifier = TransformerClassifier(
    model_name="distilbert-base-uncased",
    output_dir="examples/results/transformer",
)
train_dataset, test_dataset = classifier.prepare_data(
    documents=documents, labels=labels, test_size=0.2
)
metrics = classifier.train(train_dataset=train_dataset, test_dataset=test_dataset)
```

### Step 2: Evaluate

The transformer is added to the config alongside the cloud models using `"type": "transformer"`:

```python
{
    "type": "transformer",
    "label": "DistilBERT (fine-tuned)",
    "model_path": "examples/results/transformer/final_model",
    "cost_rate": 0.10,          # hourly server cost in USD
    "cost_rate_time_unit": "1hr",
}
```

`model_path` must point to the `final_model/` directory produced by training. That directory contains `label_mapping.json` alongside the model weights.

```python
experiment = ModelEval(config=CONFIG, data=DATA)
report_path = experiment.run(output_dir="examples/results/transformer_comparison")
```

## Key points

- Transformer models only work in label/classification mode. `response_format` must be `None`.
- `cost_rate` tracks effective cost based on actual inference time; with no API calls the baseline cost is zero.
- The small dataset (16 items, 20% test split) is for demonstration. Real training benefits from hundreds of examples.

## What's next

- Train from a larger dataset file: `python -m valtron_core.utilities.train_transformer --data data.json --output_dir ./models`
- Run the trained model standalone with `TransformerModelWrapper`. See [Transformer Models](../transformer-models).
