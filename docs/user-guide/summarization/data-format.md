# Data Format

Every model compared in a summarization run reads the same `data`. Unlike in classification and extraction experiments, each document needs only two things: a unique `id` and its `content`, the document to summarize. There's no `label`, because there's no single correct summary to compare a candidate against; see [Summarization](./index) for how a summary gets scored instead.

## Basic Structure

```python
from valtron_core.evaluation import SummarizationExperiment

data = [
    {
        "id": "1",
        "content": "The city council voted 6-3 on Tuesday to approve a $40 million bond measure...",
    },
    {
        "id": "2",
        "content": "Quarterly revenue rose 4% year over year to $1.2 billion, driven by...",
    },
]

experiment = SummarizationExperiment(
    config={
        "models": [{"name": "gpt-4o-mini"}],
        "judge_model": "gpt-4o",
    },
    data=data,
)
experiment.run("./results")
```

`prompt` is left unset above, so it defaults to `SALIENCE_SUMMARY_PROMPT`; see [Config Format](./config-format). Whichever prompt used, its `{content}` placeholder gets substituted per document, the same as in classification and extraction experiments; for `doc "1"` above, the candidate model receives the bond measure text in place of `{content}`.

## Content Format

Each document's `content` can be a plain string or a dict of several named placeholder values, one per `prompt` placeholder. A document with blank or whitespace-only content, string or dict, is rejected before any model is called.

### String Content

The default prompt, `SALIENCE_SUMMARY_PROMPT`, expects a single string:

```python
data = [
    {
        "id": "1",
        "content": "The city council voted 6-3 on Tuesday to approve a $40 million bond measure...",
    },
]
```

For `doc "1"`, both the judge and the candidate model read the same text.

### Dict Content

A dict works with a custom `prompt` that has one placeholder per key:

```python
data = [
    {
        "id": "1",
        "content": {
            "agenda": "Item 4: $40 million bond measure for school construction.",
            "transcript": "MAYOR: The motion carries 6-3. COUNCILMEMBER LEE: For the record...",
        },
    },
]

config = {
    "models": [{"name": "gpt-4o-mini"}],
    "judge_model": "gpt-4o",
    "prompt": "Agenda item:\n{agenda}\n\nTranscript:\n{transcript}\n\nSummarize the discussion.",
}
```

For `doc "1"`, the candidate receives `prompt` with `{agenda}` and `{transcript}` filled in:

```text
Agenda item:
Item 4: $40 million bond measure for school construction.

Transcript:
MAYOR: The motion carries 6-3. COUNCILMEMBER LEE: For the record...

Summarize the discussion.
```

The judge reads every value instead, joined together, regardless of which ones `prompt` shows the candidate:

```text
agenda: Item 4: $40 million bond measure for school construction.

transcript: MAYOR: The motion carries 6-3. COUNCILMEMBER LEE: For the record...
```

A placeholder missing from the dict is substituted with an empty string, the same as classification and extraction.

## Attachments

Some documents need more than text: a scanned page, a chart, a screenshot. `attachments` works the same way as [classification experiments](../classification/data-format.md#attachments), a list of local file paths, HTTP(S) URLs, or `data:` URIs, each an image or a PDF.

```python
data = [
    {
        "id": "1",
        "content": "Board meeting minutes, page 2 of 2.",
        "attachments": ["/path/to/page1.png"],
    },
]
```

Both the judge and every candidate model read a document's attachments: the judge needs them to decompose the facts a summary must convey, and a candidate needs them to write about what's in them. Valtron checks that the judge model and every candidate model support the attachment types present before evaluation starts. If any of them can't, a `ValueError` is raised naming the model and the document.

## Label Format

A `label` present on a document is ignored. Summarization has no ground truth to check a candidate's summary against, so there's nothing for a label to be compared to.

## Passing Data to SummarizationExperiment

`data` can be passed as a Python list, as shown above, or as a path to a JSON file with the same shape, the same as the other recipes:

```python
experiment = SummarizationExperiment(config=config, data="./data/documents.json")
```

The JSON file must be an array at the top level.

```{rubric} What's next?
```

- Each item validates as a [`Document`](../../api/document); a `label`, if present, is dropped rather than validated against anything.
- Set up the judge model, checklist, and scoring parameters in [Config Format](./config-format).
- To run your evaluation, see [Evaluation API](./evaluation-api).
