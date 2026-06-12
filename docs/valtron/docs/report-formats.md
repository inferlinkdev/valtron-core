---
sidebar_position: 7
---

# Report Formats

Valtron generates HTML and/or PDF reports after evaluation. Reports are written to `output_dir` alongside the raw JSON result files. For the raw metrics and JSON schema behind the report data, see [Evaluation Results](./evaluation-results).

## Enabling report formats

Set `output_formats` in your config:

```json
{
  "output_formats": ["html"]
}
```

```json
{
  "output_formats": ["html", "pdf"]
}
```

Default is `["html"]`.

---

## HTML report

The HTML report consists of two files:

- **`evaluation_report.html`**: main interactive report
- **`detailed_analysis.html`**: per-document breakdown

### Main report sections

**Header**
- Timestamp, use case label, number of models evaluated, number of documents

**AI recommendation**
- A generated recommendation naming the best model for your use case
- Calculated using an accuracy-to-cost ratio (`accuracy / total_cost`)
- Includes a justification, secondary recommendation if speed is critical, and a cost warning if the highest-accuracy model is significantly more expensive

**Performance overview**
- Interactive bar charts (ECharts) comparing all models on:
  - Accuracy (% correct)
  - Average cost per document
  - Average response time per document
- Distribution histograms showing per-document cost, latency, and score spread across models
- Avg Score per model (mean per-document score across all fields and examples)

**Prompt optimization details** (shown when models have manipulations)
- Lists applied manipulations per model
- "Base" / "Overridden" toggle links showing the exact prompt each model received
- Present when any model uses a per-model prompt override or prompt manipulation

**Field-level metrics** (shown when `field_metrics_config` is set)
- Hierarchical tree view of all schema fields
- Per-field bar charts showing precision, recall, and F1 score across models
- Metric method shown per field (e.g. "Exact Match", "Text Similarity (cosine, threshold: 0.7)")

### Detailed analysis page

`detailed_analysis.html` shows a per-document breakdown:

- Document content and any attachments
- Expected label
- Each model's predicted value, score, cost, and response time
- Field-level results per document (structured mode)

---

## PDF report

The PDF report is a single file `evaluation_report.pdf` containing all the same information as the HTML report, optimized for print:

- **Static charts**: accuracy, cost, and time charts rendered as embedded PNG images (matplotlib, pastel color theme)
- **Generated via ReportLab**: pure Python, no system dependencies required
- Suitable for sharing or archiving results

---

## Output files summary

| File | Always written | Description |
|---|---|---|
| `metadata.json` | Yes | Experiment metadata, documents, prompt |
| `models/<label>.json` | Yes | Per-model metrics and predictions |
| `progress.json` | During run only | Live progress state for external pollers (see below) |
| `evaluation_report.html` | With `"html"` | Main interactive report |
| `detailed_analysis.html` | With `"html"` | Per-document breakdown |
| `evaluation_report.pdf` | With `"pdf"` | Full printable report (charts rendered internally via matplotlib, not written to disk) |

### `progress.json`

`progress.json` is written to `output_dir` while a run is in flight and is removed from consideration once the run completes (it is not cleaned up automatically, but its content becomes stale). It is intended for external systems such as a web dashboard that need to poll for live status without coupling to logs.

The file goes through two phases:

**Setup phase** (few-shot generation, prompt preparation, etc.):

```json
{
  "started_at": "2026-05-29T15:41:48.996Z",
  "last_update": "2026-05-29T15:42:01.123Z",
  "status_message": "Preparing prompts..."
}
```

**Evaluation phase** (once model evaluation begins):

```json
{
  "started_at": "2026-05-29T15:41:48.996Z",
  "last_update": "2026-05-29T15:42:10.456Z",
  "models": [
    {"name": "gpt-4o", "docs_done": 12, "docs_total": 50, "completed": false},
    {"name": "gpt-4o-mini", "docs_done": 50, "docs_total": 50, "completed": true}
  ]
}
```

The `name` field for each model entry is the model's `label` if set, otherwise its `name`. The file is absent before the run starts; consumers should treat a missing file as "initialising" rather than an error. Writes are atomic to avoid partial writes.

---

## Generating reports separately

You can save reports independently after `evaluate()`:

```python
experiment = ModelEval(config=config, data=data)
experiment.evaluate()

# HTML only
experiment.save_html_report("./results")

# PDF only
experiment.save_pdf_report("./results")
```

---

## What's next?

- See working end-to-end examples: [Examples](./examples)
- Add a local model with no API cost: [Self-Hosted Models](./self-hosted-models)
- Train a zero-cost local classifier: [Transformer Models](./transformer-models)
