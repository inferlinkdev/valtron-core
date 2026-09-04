# Report Formats

Reports land in `output_dir` alongside the raw JSON result files. See [Evaluation Results](./evaluation-results).

## Valtron Outputs

`output_formats` controls what `run()` writes, the same way as the other recipes: `["html"]` by default, with `"pdf"` added to also generate a PDF.

```python
config = {
    "models": [{"name": "gpt-4o-mini"}, {"name": "gpt-4o"}],
    "judge_model": "gpt-4o",
    "output_formats": ["html", "pdf"],
}

experiment = SummarizationExperiment(config=config, data=data)
experiment.run("./results")
```

```text
results/
├── metadata.json
├── models/gpt-4o-mini.json
├── models/gpt-4o.json
├── html_report/summarization_report.html
└── summarization_report.pdf
```

Reports can also be generated individually, without `output_formats` or re-running the evaluation, the same as classification and extraction:

```python
experiment.evaluate()

experiment.save_html_report("./results")   # writes html_report/summarization_report.html
experiment.save_pdf_report("./results")    # writes summarization_report.pdf
```

`SummarizationReportGenerator` and `summarization_report.jinja2.html` are their own report generator and template, built around the four axes and the ranking. Classification and extraction's shared report reasons about a correct/incorrect badge and an accuracy-to-cost trade-off; this one reasons about the coverage/precision trade-off and the faithfulness gate instead.

## What's in the HTML report

`summarization_report.html` opens with a header showing model and document counts, the `judge_model`, total cost, total tokens, and the generation timestamp, followed by an AI recommendation that reasons about the coverage/precision trade-off and whether anything failed the faithfulness gate, rather than an accuracy-to-cost ratio.

The ranking table lists every model in tier order, each row showing its tier, a `gated` badge if it failed the faithfulness gate and scored zero as a result, its overall score, each of the four axes (an em dash where an axis was undefined rather than zero), and how many documents it was scored over. A short note beneath the table restates the `gate`/`beta`/`requirement_weight` values the ranking was computed with.

A cost table splits spend three ways: `generation` (candidates writing summaries), `judge_per_candidate` (the judge grading each summary on the four axes), and `judge_shared` (the one-per-document fact extraction and salience marking that every candidate reuses), each with its own call count, token counts, and cost.

Below that, one expandable section per document shows the judge's must-convey facts for that document, and a table of every candidate's axes, how many of the salient facts it hit, its cost, and its full generated summary text (or the error, if that candidate failed on that document). This is the same evidence stored in each prediction's `metadata`; see [Evaluation Results: Reading a model's axes](./evaluation-results.md#reading-a-models-axes).

## What's in the PDF report

`summarization_report.pdf` carries the same ranking table, cost breakdown, recommendation, and per-document detail as the HTML report, laid out as static tables instead of interactive ones. Generating it requires no additional system dependencies.

## Progress

`progress.json` is written to `output_dir` while `run()`/`evaluate()` is in flight, the same mechanism classification and extraction use; see [Classification Report Formats: Progress](../classification/report-formats.md#progress). The per-model `docs_done`/`docs_total` counts advance during candidate evaluation; the earlier shared fact-extraction phase has no per-document counter of its own beyond a `tqdm` progress bar in the terminal, so a slow judge retry there can make a run look stalled for a while even though it's working.

```{rubric} What's next?
```

- Read the underlying axes and ranking off a run: [Evaluation Results](./evaluation-results)
- Reweight or regrade a run after the fact: [Evaluation API: Reweighting and Regrading](./evaluation-api.md#reweighting-and-regrading-a-run)
- See working end-to-end examples: [Examples](../../examples/index)
