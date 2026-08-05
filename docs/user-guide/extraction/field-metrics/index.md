# Field Metrics

```{toctree}
:hidden:

leaf-fields
object-fields
list-fields
custom-evaluation
```

Extraction problems have subfields, lists, and multiple values, and each one can come back right, wrong, or partially right. A model that gets `city` right 95% of the time but `country` only 70% of the time looks the same as one that's uniformly mediocre, unless each field is scored on its own.

A classification label either matches the expected value or it doesn't. There's no partial credit to account for, so a single accuracy number is enough.

Field metrics grade each extraction field individually instead of the output as a whole. `field_metrics_config` is part of an experiment's config, set on [`ModelEvalConfig`](../../../api/model_eval_config).

See the full example in [Affiliation Extraction](../../../examples/affiliation-extraction).

## Scoring extraction field by field

Take the affiliation-extraction schema from [this chapter's introduction](../index):

```python
from pydantic import BaseModel
from valtron_core.evaluation import ModelEval

class Institution(BaseModel):
    name: str
    city: str
    country: str

class AffiliationResult(BaseModel):
    institutions: list[Institution]

config = {
    "prompt": "List all institutions in the following affiliation string.\n\n{content}",
    "models": [{"name": "gpt-4o-mini"}],
    "field_metrics_config": {
        "config": {
            "type": "object",
            "fields": {
                "institutions": {
                    "type": "list",
                    "fields": {
                        "name": {"type": "leaf", "metric_config": {"metric": "text_similarity", "params": {"threshold": 0.85}}},
                        "city": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
                        "country": {"type": "leaf", "metric_config": {"metric": "exact_compare"}},
                    },
                }
            },
        }
    },
}

experiment = ModelEval(config=config, data=data, response_format=AffiliationResult)
experiment.run("./results/field_metrics_demo")

for result in experiment.results:
    for path, field_result in result.metrics.aggregated_field_metrics.items():
        print(f"{path:<20} precision={field_result.precision:.0%}  recall={field_result.recall:.0%}")
```

```text
institutions.name    precision=91%  recall=88%
institutions.city    precision=97%  recall=95%
institutions.country precision=99%  recall=99%
```

Each node in `config` above (`institutions`, `name`, `city`, `country`) is a [`FieldConfig`](../../../api/field_config): a `type` (`"leaf"`, `"object"`, or `"list"`), an optional `weight` for how much it counts toward its parent's score, and a `metric_config` whose shape depends on `type`. `result.metrics.aggregated_field_metrics` holds one [`EvalResult`](../../../api/eval_result) per field path, which is what's printed above.

The following sections go over how evaluation works at the leaf, object, and list level:

- **[2.3.1 Leaf Fields](./leaf-fields)**: scalar values (`name`, `city`, `country` above) and the six built-in comparison metrics.
- **[2.3.2 Object Fields](./object-fields)**: grouping child fields and rolling their scores up into one.
- **[2.3.3 List Fields](./list-fields)**: scoring a predicted array against an expected one (`institutions` above), ordered or unordered.
- **[2.3.4 Custom Evaluation](./custom-evaluation)**: registering your own metric or aggregation function when the built-ins don't fit.
