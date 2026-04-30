---
sidebar_position: 3
---

# Data Format

This page covers input documents: how to structure your data before passing it to `ModelEval`. Once your data is ready, see [Config Format](./config-format) to configure the evaluation run.

Valtron expects input data as a list of documents, each paired with an expected output (label). You can pass data as a Python list of dicts or as a path to a JSON file.

Think of this as an experiment: the more diverse and representative your examples are, the more meaningful your results will be. Aim to include a broad mix of document types and edge cases that reflect real-world variation. This collection is your **random sample**.

## Document schema

Each item in the data list has the following fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | `string` | Yes | Unique identifier for the document |
| `content` | `string \| object` | Yes | Text filled into your [configured prompt](./config-format): either a single string or a `{key: value}` map for multiple placeholders |
| `label` | `string` | Yes | The expected/ground-truth output |
| `metadata` | `object` | No | Arbitrary key-value pairs; displayed in the report |
| `attachments` | `array[string]` | No | URLs or local file paths appended to the prompt |

### How content fills your prompt

**String content** - When `content` is a string, your prompt template is expected to have a `{content}` placeholder:

```json
[
  {
    "id": "doc-001",
    "content": "Absolutely love this product!",
    "label": "positive"
  },
  {
    "id": "doc-002",
    "content": "Terrible experience, would not recommend.",
    "label": "negative"
  }
]
```

Example prompt:

```
Classify the sentiment of this review: {content}
```

What LLM receives as input for `doc-001`

```
Classify the sentiment of this review: Absolutely love this product!
```

**Dict content** - When `content` is a dict, your prompt is expected to have a placeholder for each key in the content object:

```json
[
  {
    "id": "doc-001",
    "content": {
      "text": "The annual rainfall in the Amazon basin exceeds 2,000 mm.",
      "topic": "climate"
    },
    "label": "YES"
  },
  {
    "id": "doc-002",
    "content": {
      "text": "The company reported record profits for Q3.",
      "topic": "climate"
    },
    "label": "NO"
  }
]
```

Example prompt:

```
Text: {text}
Question: Is the topic of this text '{topic}'? Respond with: YES or NO only.
```

What LLM receives as input for `doc-001`

```
"Text: The annual rainfall in the Amazon basin exceeds 2,000 mm.
Question: Is the topic of this text 'climate'? Respond with: YES or NO only."
```

If a placeholder in the template is missing from a document's dict, a warning is logged and an empty string is substituted. Extra keys not referenced by the template are silently ignored.

---

## Label format

The `label` field depends on the evaluation mode:

**Label/classification mode** (no `response_format` in config):
- `label` is a plain string matching one of the known output classes.
- Valtron automatically generates a `Literal` enum from all unique label values in your dataset and uses it as the required output schema. This constrains the LLM to return one of the known classes exactly, reducing hallucinations and making correctness checking unambiguous.
- If your dataset has more than 50 unique label values, Valtron falls back to `label: str` and the LLM returns free text compared against the label by string equality.

```json
{"id": "1", "content": "...", "label": "positive"}
```

For a dataset with labels `"positive"`, `"negative"`, and `"neutral"`, the generated schema looks like:

```python
class ResponseModel(BaseModel):
    label: Literal["negative", "neutral", "positive"]
```

**Structured extraction mode** (with `response_format` in config):
- `label` can be either a **JSON object** or a **JSON string**. Both are accepted.
- Must match the shape of the Pydantic model passed as `response_format`
- If a dict/list is provided, it is serialized to a JSON string internally

```json
// As a JSON object (preferred, more readable)
{
  "id": "1",
  "content": "Apple Inc. was founded in Cupertino, California.",
  "label": {"name": "Apple Inc.", "city": "Cupertino", "state": "California"}
}
```

```json
// As a JSON string (also valid)
{
  "id": "1",
  "content": "Apple Inc. was founded in Cupertino, California.",
  "label": "{\"name\": \"Apple Inc.\", \"city\": \"Cupertino\", \"state\": \"California\"}"
}
```

## Attachments

The `attachments` field allows you to pass supplementary files or URLs alongside the document text. These are automatically fetched and embedded in the prompt before evaluation.

```json
{
  "id": "doc-003",
  "content": "Summarize the key findings from this report.",
  "label": "cost reduction",
  "attachments": [
    "https://example.com/report.pdf",
    "/path/to/local/document.txt"
  ]
}
```

- HTTP/HTTPS URLs are fetched at evaluation time
- Local paths are read from disk
- Attachment content is appended to the document content in the prompt
- In the HTML report, attachments are embedded as base64 (local) or linked (URLs)

## Passing data to ModelEval

**As a Python list:**
```python
data = [
    {"id": "1", "content": "...", "label": "positive"},
    {"id": "2", "content": "...", "label": "negative"},
]
experiment = ModelEval(config=config, data=data)
```

**As a JSON file path:**
```python
experiment = ModelEval(config=config, data="./data/reviews.json")
```

The JSON file must be an array at the top level.

## Tips

- `id` values must be unique across your dataset. They are used as keys in the output files.
- `metadata` fields appear in the detailed analysis page of the HTML report but are not passed to the model
- There is no minimum or maximum document count, but more documents produce more reliable accuracy estimates

---

## What's next?

- Set up your evaluation in [Config Format](./config-format)
- For structured extraction with field-level scoring, see [Field Metrics](../field-metrics)
- Run your evaluation: [Evaluation API](./recipes)
