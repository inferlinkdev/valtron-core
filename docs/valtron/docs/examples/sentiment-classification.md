---
sidebar_position: 2
---

# Sentiment Classification

**File:** [`examples/sentiment_classification.py`](https://github.com/inferlinkdev/valtron-core/blob/main/examples/sentiment_classification.py)

The simplest possible evaluation: two LLMs classifying movie reviews as `positive`, `negative`, or `neutral`.

## What it demonstrates

- Minimal `ModelEval` setup
- Label/classification mode (no `response_format`)
- Accessing results programmatically after evaluation

## Run it

```bash
python examples/sentiment_classification.py
```

## Data file

Data is loaded from [`examples/sentiment_data.json`](https://github.com/inferlinkdev/valtron-core/blob/main/examples/sentiment_data.json). Each record has an `id`, `content`, and `label`:

```json
[
    {"id": "1", "content": "An absolute masterpiece. I was on the edge of my seat the whole time.", "label": "positive"},
    {"id": "2", "content": "Painfully boring. I walked out after thirty minutes.", "label": "negative"},
    {"id": "3", "content": "Decent enough, a few good scenes but nothing memorable.", "label": "neutral"}
]
```

See [Data Format](../data-format) for more details.

## Full code

```python
import json
from pathlib import Path
from valtron_core.recipes import ModelEval

DATA = json.loads((Path(__file__).resolve().parent / "sentiment_data.json").read_text())

CONFIG = {
    "use_case": "movie review sentiment classification",
    "output_formats": ["html"],
    "temperature": 0.0,
    "prompt": (
        "Classify the sentiment of the following movie review as exactly one of: "
        "positive, negative, or neutral. "
        "Reply with only the single word label.\n\n"
        "Review: {content}"
    ),
    "models": [
        {"name": "gpt-4o-mini", "label": "GPT-4o mini"},
        {"name": "claude-haiku-4-5-20251001", "label": "Claude Haiku"},
    ],
}

experiment = ModelEval(config=CONFIG, data=DATA)
report_path = experiment.run(output_dir="examples/results/sentiment")

for result in experiment.results:
    print(f"{result.model}  accuracy={result.metrics.accuracy:.0%}")
```

## Key points

- The prompt must contain `{content}`. That placeholder is replaced with each document's `content` at evaluation time.
- `run()` is synchronous and returns the path to the generated HTML report.
- `experiment.results` is available after `run()` (or `evaluate()`) and contains per-model accuracy, cost, and per-document predictions.

## What's next

- Add a third model or a per-model prompt override. See [Recipes](../evaluation-api).
- Load these results later and extend them. See [Incremental Evaluation](./incremental-evaluation).
- Add few-shot examples to improve accuracy. See [Optimizers](../optimizers).
