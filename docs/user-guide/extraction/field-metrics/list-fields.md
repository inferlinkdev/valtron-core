# List Fields

A `"list"` node scores a predicted array against an expected one. Take a schema that extracts institutions:

```python
from pydantic import BaseModel

class Institution(BaseModel):
    name: str
    city: str

class AffiliationResult(BaseModel):
    institutions: list[Institution]
```

`institutions` is a `"list"` node. Its own `metric_config` is a [`ListMetricConfig`](../../../api/list_metric_config), which controls how predicted and expected items get matched up before each matched pair's fields are scored individually.

## Describing Each Item

Every list node needs `item_logic` inside its `metric_config`. `item_logic` is a `FieldConfig` describing what one item looks like, the same way `fields` describes an object's own children.

If items are objects, `item_logic` is itself an `"object"` node with its own `fields`:

```python
from valtron_core.models import FieldMetricsConfig

field_metrics = FieldMetricsConfig(
    config={
        "type": "object",
        "fields": {
            "institutions": {
                "type": "list",
                "metric_config": {
                    "item_logic": {
                        "type": "object",
                        "fields": {
                            "name": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
                            "city": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
                        },
                    },
                },
            }
        },
    }
)
```

If items are plain scalars instead, with no child object at all, `item_logic` is a `"leaf"` node directly, and `"$item"` is the path used whenever a setting needs to refer to the item itself, for example in `allow_expensive_comparisons_for`:

```python
field_metrics = FieldMetricsConfig(
    config={
        "type": "list",
        "metric_config": {
            "allow_expensive_comparisons_for": ["$item"],
            "item_logic": {"type": "leaf", "metric_config": {"metric": "llm", "params": {}}},
        },
    }
)
```

Every example in the rest of this page builds on one of these two shapes.

## Unordered Matching

Unordered matching is the default. Valtron builds a score matrix across every predicted/expected pair, greedily matches the highest-scoring pairs first, and reports F1 from the resulting precision/recall. Order in the model's output doesn't affect the score:

```python
field_metrics = FieldMetricsConfig(
    config={
        "type": "object",
        "fields": {
            "institutions": {
                "type": "list",
                "metric_config": {
                    "ordered": False,
                    "match_threshold": 0.5,
                    "item_logic": {
                        "type": "object",
                        "fields": {
                            "name": {"type": "leaf", "metric_config": {"metric": "text_similarity", "params": {"threshold": 0.85}}},
                            "city": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
                        },
                    },
                },
            }
        },
    }
)
```

`match_threshold` sets how similar a predicted/expected pair has to be before they even count as a match at all, and defaults to `0.5`.

## Ordered Matching

If your model reliably returns items in the same order as the ground truth (e.g. extracting steps from a numbered procedure), setting `ordered` to `true` compares positionally instead (`expected[i]` against `actual[i]`). It is cheaper but penalizes all items that are shifted by at least 1 position.

```python
field_metrics = FieldMetricsConfig(
    config={
        "type": "object",
        "fields": {
            "institutions": {
                "type": "list",
                "metric_config": {"ordered": True, "item_logic": {"type": "object", "fields": {...}}},
            }
        },
    }
)
```

## Required Fields to Match

`required_fields_to_match` pre-filters candidate pairs on one or more fields before the full comparison runs.

Say two institutions share a near-identical `name` but sit in different countries. Requiring `country` to match first stops them from being paired on `name` alone:

```python
field_metrics = FieldMetricsConfig(
    config={
        "type": "object",
        "fields": {
            "institutions": {
                "type": "list",
                "metric_config": {
                    "required_fields_to_match": ["country"],
                    "item_logic": {
                        "type": "object",
                        "fields": {
                            "name": {"type": "leaf", "metric_config": {"metric": "text_similarity", "params": {"threshold": 0.85}}},
                            "country": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
                        },
                    },
                },
            }
        },
    }
)
```

A predicted/expected pair whose `country` doesn't match is never scored on `name` at all. It's treated as not a match.

## Expensive Comparisons

A metric is considered expensive if it makes a third-party API call per comparison: `"llm"`, `"embedding"`, or `"text_similarity"` with `metric: "cosine"` (see [Leaf Fields](./leaf-fields)).

The expensive designation exists because a list's length is unconstrained. Scoring every predicted/expected pair with an expensive metric, just to work out which items even match, costs N x M API calls for a list of N expected and M actual items, with no cap on how large N and M get. Valtron forces you to acknowledge that cost up front: `allow_expensive_comparisons_for` must list every field path under the list that uses an expensive metric, not just one of them, and Valtron raises rather than silently running an unbounded sweep:

```python
field_metrics = FieldMetricsConfig(
    config={
        "type": "list",
        "metric_config": {
            "allow_expensive_comparisons_for": ["description"],
            "item_logic": {
                "type": "object",
                "fields": {
                    "name": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
                    "description": {
                        "type": "leaf",
                        "metric_config": {"metric": "llm", "params": {"model": "gpt-4o-mini"}},
                    },
                },
            },
        },
    }
)
```

Opting in doesn't mean Valtron actually runs N x M expensive calls, though. The next section covers what it does instead.

## Avoiding N x M Expensive Comparisons

If the expected list has N items and the actual list has M items, comparing every pair with an expensive metric takes N x M calls. Valtron takes a different approach to keep that cost down once a list has an expensive metric anywhere under its items.

`allow_expensive_comparisons_for` is still required here. It gates whether Valtron scores the list at all, not which matching strategy handles it once scoring is allowed.

Instead of scoring every predicted/expected pair with the expensive metric, Valtron aligns items first, then scores only the matches:

1. **First, Valtron decides which fields identify an item.** `match_key_fields` is a setting inside `metric_config`, the same as `lo` or `embed_model` below. Set it explicitly to the fields that identify an item, or leave it unset and Valtron picks automatically: one LLM call (`match_key_model`, default `"gpt-5.4-mini"`) over a handful of sample items. That automatic call happens at most once per list field for the whole evaluation run. Every document scored after the first reuses the same selection.
2. **Then Valtron embeds every item.** Using the identity fields from the previous step, every expected and actual item is embedded in a single batched call (`embed_model`, default `"text-embedding-3-small"`). This produces one cosine-similarity score per predicted/expected pair, instead of one expensive call per pair.
3. **Next, Valtron assigns matches globally.** Rather than the greedy "take the best remaining pair" approach [Unordered Matching](#unordered-matching) uses, Valtron assigns pairs with the [Hungarian algorithm](https://en.wikipedia.org/wiki/Hungarian_algorithm), finding the one-to-one pairing that maximizes total similarity across the whole list at once. Any pair below `lo` (default `0.35`) is left unmatched regardless.
4. **Finally, Valtron scores each matched pair.** Once matches are decided, `item_logic` (see [Describing Each Item](#describing-each-item)) runs once per matched pair the same as any other list, so the expensive metric is called N times, once per expected item, instead of N x M.

```python
field_metrics = FieldMetricsConfig(
    config={
        "type": "object",
        "fields": {
            "institutions": {
                "type": "list",
                "metric_config": {
                    "allow_expensive_comparisons_for": ["description"],
                    "match_key_fields": ["name"],
                    "lo": 0.4,
                    "item_logic": {
                        "type": "object",
                        "fields": {
                            "name": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
                            "description": {
                                "type": "leaf",
                                "metric_config": {"metric": "llm", "params": {"model": "gpt-4o-mini"}},
                            },
                        },
                    },
                },
            }
        },
    }
)
```

`lo` and `match_threshold` are different knobs for different matching strategies. `match_threshold` only applies to the plain greedy matching in [Unordered Matching](#unordered-matching); `lo` only applies to this embedding-based alignment. [Required Fields to Match](#required-fields-to-match) still pre-filters candidate pairs here too, before the Hungarian assignment runs.

```{rubric} What's next?
```

- The metrics available on each item's fields: [Leaf Fields](./leaf-fields)
- Aggregating an object's own children (not a list of them): [Object Fields](./object-fields)
