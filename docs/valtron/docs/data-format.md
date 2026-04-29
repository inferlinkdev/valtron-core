---
sidebar_position: 3
---

# Data Format

Valtron expects input data as a list of documents, each paired with an expected output (label). You can pass data as a Python list of dicts or as a path to a JSON file.

## Document schema

Each item in the data list has the following fields:

| Field | Type | Required | Description |
|---|---|---|---|
| `id` | `string` | Yes | Unique identifier for the document |
| `content` | `string \| object` | Yes | The text the model will evaluate, or a `{key: value}` map of prompt variables (see [Prompt variables](#prompt-variables-dict-content)) |
| `label` | `string` | Yes | The expected/ground-truth output |
| `metadata` | `object` | No | Arbitrary key-value pairs; displayed in the report |
| `attachments` | `array[string]` | No | URLs or local file paths appended to the prompt |

```json
[
  {
    "id": "doc-001",
    "content": "The patient presented with acute chest pain and shortness of breath.",
    "label": "cardiology",
    "metadata": {
      "source": "hospital-a",
      "year": 2024
    }
  },
  {
    "id": "doc-002",
    "content": "Follow-up appointment for knee replacement surgery.",
    "label": "orthopedics"
  }
]
```

## Label format

The `label` field depends on the evaluation mode:

**Label/classification mode** (no `response_format` in config):
- `label` is a plain string. It must be the exact value the model is expected to output.
- Comparison is string equality (or custom comparator if configured)

```json
{"id": "1", "content": "...", "label": "positive"}
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

## Prompt variables (dict content)

When `content` is a `dict[str, str]`, every key becomes a named `{placeholder}` in your prompt template. This lets you pass multiple variables alongside the main document text.

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

Matching prompt template:

```
Text: {text}
Question: Is the topic of this text '{topic}'? Respond with: YES or NO only.
```

Each key in the dict is substituted for its matching `{placeholder}` in the template. If a placeholder referenced in the template is missing from a document's dict, a warning is logged and an empty string is substituted for that document.

Extra keys in the dict that are not referenced by the template are silently ignored.

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
- `content` is inserted at the `{content}` placeholder in your prompt template (string form), or each key is inserted at its matching `{key}` placeholder (dict form)
- `metadata` fields appear in the detailed analysis page of the HTML report but are not passed to the model
- There is no minimum or maximum document count, but more documents produce more reliable accuracy estimates
