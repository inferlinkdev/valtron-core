# Classification

```{toctree}
:hidden:

data-format
config-format
evaluation-api
evaluation-results
report-formats
```

A classification problem is one where you have data and want to map it to some known, existing label. Sentiment ("positive", "negative", "neutral"), topic tags, spam/not-spam: if the correct answer for each document is one label drawn from a fixed set, it's classification.

You express this in Valtron by setting each document's `label` to a simple string, one of your known label values, and running it through `ClassificationExperiment`, which compares that string directly against the model's output.

It is not necessary to indicate a structured response format using `response_format`. By default, Valtron constrains the output of LLMs using a response format generated from the known label values (See [schema inference](./config-format.md#structured-output-schema-inference)).

Accuracy is the primary evaluation criterion in this mode. It is the fraction of documents where the model's output matched the expected label exactly, and it's the first field on each model's [`EvaluationMetrics`](../../api/evaluation_metrics).

To run an evaluation, pass `data` as a list of dicts with `content` (what the model sees) and `label` (the ground truth), along with a `prompt` and one or more `models`.

```{rubric} Sentiment Evaluation Example
```

```python
from valtron_core.evaluation import ClassificationExperiment

data = [
    {"id": "1", "content": "The product arrived damaged and support was unhelpful.", "label": "negative"},
    {"id": "2", "content": "Fast shipping, exactly what I ordered. Very happy!", "label": "positive"},
    {"id": "3", "content": "Average experience, nothing special.", "label": "neutral"},
    {"id": "4", "content": "Outstanding quality. Will definitely buy again.", "label": "positive"},
    {"id": "5", "content": "Wrong item sent. Refund process was painful.", "label": "negative"},
]

config = {
    "prompt": "Classify the sentiment as positive, negative, or neutral.\n\n{content}",
    "models": [{"name": "gpt-4o-mini"}, {"name": "claude-haiku-4-5-20251001"}],
}

experiment = ClassificationExperiment(config=config, data=data)
report_path = experiment.run("./results")

for result in experiment.results:
    print(f"{result.model:<28} accuracy={result.metrics.accuracy:.0%}")
```

```text
gpt-4o-mini                  accuracy=100%
claude-haiku-4-5-20251001    accuracy=80%
```

Each document's `content` fills the `{content}` placeholder in `prompt`, so the model receives a plain text message. For document `"1"` above, the model receives as input:

```text
Classify the sentiment as positive, negative, or neutral.

The product arrived damaged and support was unhelpful.
```

This example only needs one placeholder, but a document's `content` can also be a dict, which lets one prompt take several placeholders at once. See [Data Format: Content Format](./data-format.md#content-format).

`run("./results")` invokes the evaluation pipeline. It validates the data, computes metrics, and writes the results to the `results` directory. Valtron calls model APIs concurrently, and then for classification problems, compares each model's response against the expected `label` by an exact string match to compute the `accuracy`. `run()` outputs a `metadata.json` with the experiment's config and input documents, one `models/<label>.json` per model holding the model's full [`EvaluationResult`](../../api/evaluation_result), and the html reports containing the experiment results. See [Evaluation Results: Output directory layout](./evaluation-results.md#output-directory-layout) for the full layout.

```{rubric} In this chapter
```

Beyond the single-placeholder, plain-string-label example above, documents can also carry `metadata` and file/image attachments, prompts can be automatically improved with few-shot examples, and labels can be graded in more depth than exact string match alone. Each of those is covered in this chapter:

- **[1.1 Data Format](./data-format)**: how to structure input documents and expected labels.
- **[1.2 Config Format](./config-format)**: the full [`ModelEvalConfig`](../../api/model_eval_config) schema, including model definitions, prompt manipulations, and field metrics.
- **[1.3 Evaluation API](./evaluation-api)**: [`ModelEval`](../../api/model_eval) itself, its sync/async methods, and incremental evaluation.
- **[1.4 Evaluation Results](./evaluation-results)**: the [`EvaluationResult`](../../api/evaluation_result)/[`EvaluationMetrics`](../../api/evaluation_metrics) objects a run produces, and the output directory layout.
- **[1.5 Report Formats](./report-formats)**: the HTML and PDF reports generated from those results.
