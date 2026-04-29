---
sidebar_position: 7
---

# Report Formats

Valtron generates HTML and/or PDF reports after evaluation. Reports are written to `output_dir` alongside the raw JSON result files.

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

- **`evaluation_report.html`** — main interactive report
- **`detailed_analysis.html`** — per-document breakdown

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
- Rankings with delta from the top score

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

- **Static charts** — accuracy, cost, and time charts rendered as embedded PNG images (matplotlib, pastel color theme)
- **Generated via ReportLab** — pure Python, no system dependencies required
- Suitable for sharing or archiving results

---

## Output files summary

| File | Always written | Description |
|---|---|---|
| `metadata.json` | Yes | Experiment metadata, documents, prompt |
| `models/<label>.json` | Yes | Per-model metrics and predictions |
| `evaluation_report.html` | With `"html"` | Main interactive report |
| `detailed_analysis.html` | With `"html"` | Per-document breakdown |
| `chart_accuracy.png` | With `"html"` | Accuracy bar chart (used by HTML) |
| `chart_cost.png` | With `"html"` | Cost bar chart |
| `chart_time.png` | With `"html"` | Latency bar chart |
| `evaluation_report.pdf` | With `"pdf"` | Full printable report |

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

## Re-running reports on saved results

Use the `aggregate_reports` utility to regenerate reports from a saved run directory without making any new API calls. This is useful when you want to:

- Re-generate with updated report templates
- Apply a new `field_metrics_config` to previously collected predictions

```bash
# Regenerate HTML report from a saved run
python -m valtron_core.utilities.aggregate_reports \
    --input ./results/my_run \
    --output-dir ./results/my_run

# Re-evaluate with a new field metrics config (no API calls)
python -m valtron_core.utilities.aggregate_reports \
    --input ./results/my_run \
    --output-dir ./results/my_run \
    --field-metrics-config ./new_field_config.json

# Disable the AI recommendation
python -m valtron_core.utilities.aggregate_reports \
    --input ./results/my_run \
    --output-dir ./reports \
    --no-recommendation
```
