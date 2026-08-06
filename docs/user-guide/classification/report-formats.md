# Report Formats

Valtron generates HTML and/or PDF reports after evaluation. Reports are written to `output_dir` alongside the raw JSON result files. See [Evaluation Results](./evaluation-results).

## Valtron outputs

What `run()` outputs repors depending on what `output_formats` sets in the config. By default, it outputs only an html report, but `"pdf"` can be included to create a pdf report.

```python
config = {
    "models": [{"name": "gpt-4o-mini"}, {"name": "gpt-4o"}],
    "prompt": "Classify the sentiment: {content}",
    "output_formats": ["html", "pdf"],  # default is just ["html"]
}

experiment = ModelEval(config=config, data=data)
experiment.run("./results")
```

```text
results/
├── metadata.json
├── models/gpt-4o-mini.json
├── models/gpt-4o.json
├── evaluation_report.html
├── detailed_analysis.html
└── evaluation_report.pdf
```

Reports can also be generated individually, without `output_formats` or re-running the evaluation:

```python
experiment.evaluate()  # no output_formats needed yet

experiment.save_html_report("./results")   # writes evaluation_report.html + detailed_analysis.html
experiment.save_pdf_report("./results")    # writes evaluation_report.pdf
```

## What's in the HTML report

`evaluation_report.html` is the main interactive report. It opens with a header showing the timestamp, use case, and model/document counts, followed by an AI recommendation. Interactive bar charts and distribution histograms compare accuracy, cost, and latency across models. When a [prompt manipulation](../manipulations/index) is in play, a prompt-manipulation breakdown adds a "Base" / "Overridden" toggle per model. When `field_metrics_config` is set, a hierarchical field tree shows per-field precision/recall/F1 bars.

`detailed_analysis.html` is the per-document companion. It shows every document's content, expected label, and each model's prediction, score, cost, and latency side by side.

## What's in the PDF report

`evaluation_report.pdf` carries the same information as the HTML report in a single printable file. Charts are rendered as static images instead of interactive, and generating it requires no additional system dependencies.

## Progress

While `run()`/`evaluate()` is in flight, Valtron writes a `progress.json` to `output_dir`, which can be useful for a dashboard or a separate process that monitors the LLM calls made by Valtron:

```python
import json
from pathlib import Path

progress_path = Path("./results/progress.json")
if progress_path.exists():
    print(json.loads(progress_path.read_text()))
```

```text
{'started_at': '2026-05-29T15:41:48.996Z', 'last_update': '2026-05-29T15:42:10.456Z',
 'models': [{'name': 'gpt-4o', 'docs_done': 12, 'docs_total': 50, 'completed': False},
            {'name': 'gpt-4o-mini', 'docs_done': 50, 'docs_total': 50, 'completed': True}]}
```

Before evaluation begins (during few-shot generation and prompt preparation), the file instead holds just `started_at`/`last_update`/`status_message`.

```{rubric} What's next?
```

- See working end-to-end examples: [Examples](../../examples/index)
- Add a local model with no API cost: [Self-Hosted Models](../self-hosting/self-hosted-models)
- Train a zero-cost local classifier: [Transformer Models](../self-hosting/transformer-models)
