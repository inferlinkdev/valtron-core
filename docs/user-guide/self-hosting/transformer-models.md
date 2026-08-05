# Transformer Models

Training a classifier uses the same shape of data as everywhere else in Valtron: [`Document`](../../api/document)s paired with [`Label`](../../api/label)s. Hand them to [`TransformerClassifier`](../../api/transformer_classifier) instead of a prompt, and it fine-tunes a small model (DistilBERT by default) that you can then run as a zero-cost model entry alongside cloud LLMs.

See the full example in [Transformer Comparison](../../examples/transformer-comparison).

## Constraints

- Single-label classification only. A transformer learns one label per document, the same shape `ClassificationExperiment` expects, and cannot be trained toward a structured `response_format`.
- Requires labeled training data. Reuse the same documents and labels you'd otherwise evaluate an LLM against.
- Requires the `transformers` extra:

```bash
pip install "valtron-core[transformers]"
```

## Training a classifier

```python
from valtron_core.training import TransformerClassifier
from valtron_core.models import Document, Label

documents = [
    Document(id="1", content="Great product, highly recommend!", metadata={}),
    Document(id="2", content="Terrible quality, waste of money.", metadata={}),
    # ... more documents
]
labels = [
    Label(document_id="1", value="positive", metadata={}),
    Label(document_id="2", value="negative", metadata={}),
    # ...
]

classifier = TransformerClassifier(
    model_name="distilbert-base-uncased",
    output_dir="./transformer_models"
)

train_dataset, test_dataset = classifier.prepare_data(
    documents=documents,
    labels=labels,
    test_size=0.2
)

metrics = classifier.train(
    train_dataset=train_dataset,
    test_dataset=test_dataset
)

print(metrics)  # accuracy, loss, etc.
```

You can also train from the command line:

```bash
python -m valtron_core.utilities.train_transformer \
    --data ./data.json \
    --output_dir ./transformer_models \
    --model distilbert-base-uncased \
    --epochs 3 \
    --batch_size 16
```

`output_dir` then contains:

```
transformer_models/
└── final_model/
    ├── config.json           ← Transformer architecture config
    ├── pytorch_model.bin     ← Model weights
    ├── label_mapping.json    ← {"positive": 0, "negative": 1, ...}
    ├── tokenizer_config.json
    ├── vocab.txt
    └── special_tokens_map.json
```

## Using a transformer in an evaluation

Pass the `final_model/` path to a `"type": "transformer"` entry in the `models` list of a `ModelEval` config, alongside cloud models:

```python
config = {
    "models": [
        {"name": "gpt-4o-mini"},
        {"name": "claude-haiku-4-5-20251001"},
        {
            "type": "transformer",
            "label": "distilbert-sentiment",
            "model_path": "./transformer_models/final_model",
            "cost_rate": 0.50,        # optional: hourly server cost
            "cost_rate_time_unit": "1hr"
        }
    ],
    "prompt": "Classify the sentiment: {content}",
    "output_dir": "./results"
}

experiment = ModelEval(config=config, data=data)
experiment.run()
```

The transformer appears alongside cloud models in the report, with its own accuracy, cost, and latency metrics. A [`TransformerModelConfig`](../../api/transformer_model_config) entry has no `response_format` or `prompt_manipulation` fields: a transformer always predicts one label per document.

### Cost tracking for transformers

Transformer models have no token-based API cost, so Valtron derives one from `cost_rate` and how long each prediction actually took:

```python
{
    "type": "transformer",
    "label": "distilbert-sentiment",
    "model_path": "./transformer_models/final_model",
    "cost_rate": 0.50,
    "cost_rate_time_unit": "1hr",
}
```

`cost_rate` is the cost of running the machine for one `cost_rate_time_unit`, not a per-prediction or per-token price. Valtron divides that rate down to the prediction's actual response time: a `cost_rate` of `0.50` per `"1hr"` costs `0.50 / 3600` per second of inference, so a prediction that takes 0.2 seconds costs about `$0.00003`.

`cost_rate_time_unit` accepts an optional leading number followed by a unit: `s`/`sec`/`second`/`seconds`, `m`/`min`/`minute`/`minutes`, or `h`/`hr`/`hour`/`hours`, case-insensitive. `"30s"`, `"5min"`, and `"2h"` are all valid; a bare unit like `"hour"` implies `1`. `cost_rate_time_unit` defaults to `"1hr"` if omitted.

## Direct inference

You can also use a trained model directly without an evaluation config:

```python
from valtron_core.transformer_wrapper import TransformerModelWrapper

model = TransformerModelWrapper(
    model_path="./transformer_models/final_model",
    model_name="distilbert-sentiment"
)

# Single prediction
label = model.predict("This product is fantastic!")
print(label)  # "positive"

# Batch predictions
labels = model.batch_predict([
    "Great experience",
    "Terrible customer service",
    "Nothing special"
])
print(labels)  # ["positive", "negative", "neutral"]

# Stats
print(model.get_stats())
# {
#   "model_name": "distilbert-sentiment",
#   "model_path": "./transformer_models/final_model",
#   "prediction_count": 3,
#   "total_cost": 0.0,
#   "cost_per_prediction": 0.0
# }
```

```{rubric} What's next?
```

- Point Valtron at a self-hosted LLM instead: [Self-Hosted Models](./self-hosted-models)
- Route between a transformer and an LLM automatically: [Combining Multiple Models](../combining-models)
