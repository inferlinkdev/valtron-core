---
sidebar_position: 3
---

# Affiliation Extraction

**File:** [`examples/affiliation_extraction.py`](https://github.com/your-org/valtron-core/blob/main/examples/affiliation_extraction.py)

Given a single-author affiliation string where a researcher is jointly affiliated with multiple institutions, extract every institution mentioned. Affiliations are semicolon-separated, as commonly found in scientific papers. This is a structured extraction task: each label is a list of institution objects, and scoring uses per-field F1 with fuzzy text matching.

## What it demonstrates

- Structured extraction mode (`response_format` with a Pydantic schema)
- Multi-institution grading across several affiliations
- `field_metrics_config` for unordered list scoring with fuzzy name matching

## Run it

```bash
python examples/affiliation_extraction.py
```

## Schema

```python
from pydantic import BaseModel

class Institution(BaseModel):
    name: str
    city: str
    state: str
    country: str

class AffiliationResult(BaseModel):
    institutions: list[Institution]
```

## Prompt

The prompt is intentionally minimal:

```
List all institutions in the following affiliation string.

{content}
```

## Field metrics config

Institutions are scored as an unordered list. Each extracted name is compared to the ground truth using fuzzy text similarity, so minor wording differences (e.g. `"Stanford University"` vs `"Stanford Univ."`) don't automatically count as wrong. `name` is weighted 3× more than the location fields.

```python
"field_metrics_config": {
    "config": {
        "type": "object",
        "fields": {
            "institutions": {
                "type": "list",
                "metric_config": {
                    "ordered": False,
                    "match_threshold": 0.5,
                    "item_logic": {
                        "type": "object",
                        "metric_config": {"propagation": "weighted_avg"},
                        "fields": {
                            "name": {
                                "metric_config": {
                                    "weight": 3,
                                    "metric": "comparator",
                                    "params": {"element_compare": "text_similarity"},
                                }
                            },
                            "city":    {"metric_config": {"metric": "comparator", "params": {"element_compare": "text_similarity"}}},
                            "state":   {"metric_config": {"metric": "comparator", "params": {"element_compare": "text_similarity"}}},
                            "country": {"metric_config": {"metric": "comparator", "params": {"element_compare": "text_similarity"}}},
                        },
                    },
                },
            }
        },
    }
}
```

## Sample data

Each document is a single affiliation string for one researcher jointly affiliated with multiple institutions, separated by semicolons.

```python
{
    "id": "1",
    "content": "John Smith, Department of Computer Science, Stanford University, Stanford, CA, USA; Google Research, Mountain View, CA, USA",
    "label": {
        "institutions": [
            {"name": "Stanford University", "city": "Stanford",      "state": "CA", "country": "USA"},
            {"name": "Google Research",     "city": "Mountain View", "state": "CA", "country": "USA"},
        ]
    },
}
```

## Key points

- Pass `response_format=AffiliationResult` to enable structured extraction mode.
- Labels can be dicts (Python) or JSON strings; both are supported.
- The list scorer builds a score matrix across all predicted/expected pairs and greedily matches them, so insertion order in the model's output doesn't affect the score.
- Increase `match_threshold` to require stricter name matching.

## What's next

- Try the `decompose` optimizer to call the model once per field. See [Optimizers](../optimizers).
- See [Field Metrics](../field-metrics) for all available comparison strategies.
