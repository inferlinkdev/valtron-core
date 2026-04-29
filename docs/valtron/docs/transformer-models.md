---
sidebar_position: 8
---

# Transformer Models

Valtron can train a local DistilBERT classifier from your evaluation data, then run it as a zero-cost model alongside cloud LLMs. This is useful for:

- **Cost comparison** — see how an on-device model compares to GPT-4o or Claude at a fraction of the price
- **Production deployment** — once trained, the model runs entirely locally with no API calls
- **Privacy** — data never leaves your machine at inference time

## Constraints

- Classification/label mode only. Transformer models do not support `response_format` (structured extraction).
- Requires labeled training data (the same data you use for evaluation)
- Requires the `transformers` extra — install with:

```bash
pip install "valtron-core[transformers]"
```

---

## Step 1: Train a classifier

Use `TransformerClassifier` to train from your documents and labels:

```python
from valtron_core.transformer_classifier import TransformerClassifier
from valtron_core.models import Document, Label

# Prepare data
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

# Train
classifier = TransformerClassifier(
    model_name="distilbert-base-uncased",  # base model
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

### Training CLI

You can also train from the command line:

```bash
python -m valtron_core.utilities.train_transformer \
    --data ./data.json \
    --output_dir ./transformer_models \
    --model distilbert-base-uncased \
    --epochs 3 \
    --batch_size 16
```

---

## Step 2: Model output structure

After training, `output_dir` contains:

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

Pass the `final_model/` path to use the model in a recipe.

---

## Step 3: Use in a recipe

Add a transformer model entry to your config using `"type": "transformer"`:

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

The transformer model appears alongside cloud models in the report, with its own accuracy, cost, and latency metrics.

### Cost tracking for transformers

Transformer models have no token-based API cost. To still track cost fairly (e.g. against cloud models), set `cost_rate` to your server's hourly cost. Valtron will calculate an effective per-prediction cost based on inference time.

---

## Direct inference

You can also use a trained model directly without a recipe:

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
