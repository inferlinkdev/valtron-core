# Data Format

Every model you evaluate, whether that's several LLMs, a locally trained transformer, or both, is scored against the same `data`. Each document in it needs three things: a unique `id`, the `content` the model sees, and the `label` it should have produced. [`ModelEval`](../../api/model_eval) builds each model's input from that `content`, and checks its output against `label`.

Treat this as an experiment. The closer your examples match production, the more meaningful your results will be. Aim for a broad, random sample of real documents and edge cases, not a curated set of easy ones. Prompt manipulation, field scoring, and model selection all build on this data, so time spent getting it right pays off for the rest of the project.

## Basic Structure

In practice, your data is a list of dicts, or a path to a JSON file with the same shape:

```python
from valtron_core.evaluation import ClassificationExperiment

data = [
    {"id": "1", "content": "Absolutely love this product!", "label": "positive"},
    {"id": "2", "content": "Terrible experience, would not recommend.", "label": "negative"},
    {"id": "3", "content": "It's fine, does what it says.", "label": "neutral"},
]

experiment = ClassificationExperiment(
    config={"models": [{"name": "gpt-4o-mini"}], "prompt": "Classify the sentiment of this review: {content}"},
    data=data,
)
experiment.run("./results")
```

The `{content}` placeholder in the prompt gets substituted per-document; for `doc "1"` above, the model actually receives `"Classify the sentiment of this review: Absolutely love this product!"`.

## Content Format

Each document's `content` is the field that gets substituted into your `prompt` template. It can be a plain string, one value for a single `{content}` placeholder, or a dict of multiple named values, one per placeholder.

### String Content

When `content` is plain text, Valtron will insert it into the `{content}` placeholder in the prompt at runtime.

```python
data = [
    {"id": "1", "content": "Absolutely love this product!", "label": "positive"},
]

config = {
    "models": [{"name": "gpt-4o-mini"}],
    "prompt": "Classify the sentiment of this review: {content}",
}
```

For `doc "1"`, the model receives:

```text
Classify the sentiment of this review: Absolutely love this product!
```

### Dict Content

When `content` is an object, Valtron inserts each key's value into the matching placeholder in the prompt at runtime. This is useful when the model needs more than a single blob of text.

```python
data = [
    {
        "id": "1",
        "content": {"text": "The annual rainfall in the Amazon basin exceeds 2,000 mm.", "topic": "climate"},
        "label": "YES",
    },
]

config = {
    "models": [{"name": "gpt-4o-mini"}],
    "prompt": "Text: {text}\nQuestion: Is the topic of this text '{topic}'? Respond with: YES or NO only.",
}
```

For `doc "1"`, the model receives:

```text
Text: The annual rainfall in the Amazon basin exceeds 2,000 mm.
Question: Is the topic of this text 'climate'? Respond with: YES or NO only.
```

If a placeholder in the prompt template is missing from a document's dict, Valtron logs a warning and substitutes an empty string. Extra keys the template doesn't reference are ignored.

## Label Format

Each document's `label` holds its ground truth. It's the correct answer to the prompt provided, and it is what the model's output gets scored against.

For classification tasks, `label` is a plain string compared against the model's output by exact match.

## Attachments

Some tasks may need more than text. A document might reference a scanned form, a screenshot, or a PDF report that the model has to actually see to answer correctly. `attachments` allows you to provide that additional context. Each file is added alongside the `prompt` and `content` when submitted to a model.

See the full example in [Multimodal Molecules](../../examples/multimodal-molecules).

Each entry in `attachments` is a local file path, an HTTP(S) URL, or a `data:` URI, and can be an image (`.png`, `.jpg`/`.jpeg`, `.gif`, `.webp`) or a PDF (`.pdf`):

```python
{
    "id": "doc-003",
    "content": "Summarize the key findings from this report.",
    "label": "cost reduction",
    "attachments": ["https://example.com/report.pdf", "/path/to/local/photo.jpg"],
}
```

The file type is detected from the extension. If the extension is missing or ambiguous, Valtron falls back to the file's magic bytes for local files and downloaded URLs. See `valtron_core.attachments` for the exact detection logic. Anything else raises a `ValueError` before evaluation starts, naming the record whose attachment type couldn't be determined.

Valtron checks that models in an experiment support attachments if they are included. If a model does not support attachments, then a `ValueError` is raised before the experiment begins.

## Passing Data to ModelEval

`data` can be passed as a Python list, as shown above, or as a path to a JSON file with the same shape:

```python
experiment = ClassificationExperiment(config=config, data="./data/reviews.json")
```

The JSON file must be an array at the top level.

```{rubric} What's next?
```

- Each item validates as a [`Document`](../../api/document) paired with a [`Label`](../../api/label); see those pages for the full field reference.
- Next, set up your evaluation in [Config Format](./config-format).
- For structured (extraction-mode) labels, see [Extraction Data Format](../extraction/data-format).
- For structured extraction with field-level scoring, see [Field Metrics](../extraction/field-metrics/index).
- To run your evaluation, see [Evaluation API](./evaluation-api).
