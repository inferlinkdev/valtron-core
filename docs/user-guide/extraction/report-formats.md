# Report Formats

Similar to classification, reports land in `output_dir` alongside the raw JSON result files. See [Evaluation Results](./evaluation-results).

## Valtron Outputs

`output_formats` controls what `run()` writes, the same as classification: `["html"]` by default, with `"pdf"` added to also generate a PDF.

```python
config = {
    "prompt": "List all institutions in the following affiliation string.\n\n{content}",
    "models": [{"name": "gpt-4o-mini"}, {"name": "gpt-4o"}],
    "output_formats": ["html", "pdf"],
}

experiment = ExtractionExperiment(config=config, data=data, response_format=AffiliationResult)
experiment.run("./results")
```

Reports can also be generated individually, without `output_formats` or re-running the evaluation, the same way as classification:

```python
experiment.evaluate()

experiment.save_html_report("./results")
experiment.save_pdf_report("./results")
```

## HTML Report

`evaluation_report.html` adds a hierarchical field tree wherever `field_metrics_config` is set: one row per field path (`institutions.name`, `institutions.city`, `institutions.country`, ...), each with its own precision/recall/F1 bar, compared side by side across every model in the run.

```python
config["field_metrics_config"] = {
    "config": {
        "type": "object",
        "fields": {
            "institutions": {
                "type": "list",
                "fields": {
                    "name": {"type": "leaf", "metric_config": {"metric": "text_similarity"}},
                    "city": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
                    "country": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
                },
            }
        },
    }
}
```

Nested fields, an object inside a list like `institutions` above, expand into their children in the tree rather than collapsing to one aggregate score.

`detailed_analysis.html` shows the full predicted JSON and expected `label` for every document instead of a single predicted string, so a mismatched field is visible without decoding the JSON yourself.

## PDF Report

`evaluation_report.pdf` carries the same field tree as a static table instead of interactive bars, same as classification's PDF report otherwise.

## Progress

`progress.json` reports `docs_done`/`docs_total` per model while `run()`/`evaluate()` is in flight, same as classification. See [Classification Report Formats: Progress](../classification/report-formats.md#progress).

```{rubric} What's next?
```

- Read the underlying numbers off a run: [Evaluation Results](./evaluation-results)
- Configure which fields get scored and how: [Field Metrics](./field-metrics/index)
- See working end-to-end examples: [Examples](../../examples/index)
