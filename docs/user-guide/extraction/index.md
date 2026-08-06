# Extraction

```{toctree}
:hidden:

data-format
config-format
field-metrics/index
evaluation-api
evaluation-results
report-formats
```

An extraction problem is one where the correct answer is a structured object, not a single label. Pulling every institution out of an author's affiliation string, or the vendor, date, and amount off an invoice: if the correct output has fields, nested objects, or lists rather than one value from a fixed set, it's extraction.

You express this in Valtron by setting each document's `label` to a dict (or JSON string) matching a Pydantic schema, and running it through `ExtractionExperiment`, which checks each field of that schema against the model's output instead of comparing a single string.

Field-level scores, not accuracy, are what matter in this mode. Every field in the schema gets its own precision, recall, and F1 via `field_metrics_config`, reported on [`EvaluationMetrics.aggregated_field_metrics`](../../api/evaluation_metrics). These are reported as [Field Metrics](./field-metrics/index).

```{rubric} Institution Extraction Example
```

```python
from pydantic import BaseModel
from valtron_core.evaluation import ExtractionExperiment

class Institution(BaseModel):
    name: str
    city: str
    country: str

class AffiliationResult(BaseModel):
    institutions: list[Institution]

data = [
    {
        "id": "1",
        "content": "Marie Curie, University of Paris, Paris, France",
        "label": {"institutions": [{"name": "University of Paris", "city": "Paris", "country": "France"}]},
    },
    {
        "id": "2",
        "content": "John Smith, Stanford University, Stanford, USA; Google Research, Mountain View, USA",
        "label": {
            "institutions": [
                {"name": "Stanford University", "city": "Stanford", "country": "USA"},
                {"name": "Google Research", "city": "Mountain View", "country": "USA"},
            ]
        },
    },
]

config = {
    "prompt": "List all institutions in the following affiliation string.\n\n{content}",
    "models": [{"name": "gpt-4o"}],
}

experiment = ExtractionExperiment(config=config, data=data, response_format=AffiliationResult)
report_path = experiment.run("./results")

for result in experiment.results:
    for path, field_result in result.metrics.aggregated_field_metrics.items():
        print(f"{path:<20} precision={field_result.precision:.0%}  recall={field_result.recall:.0%}")
```

```text
institutions.name    precision=95%  recall=100%
institutions.city    precision=100%  recall=100%
institutions.country precision=100%  recall=100%
```

Scoring walks each field of the predicted JSON against the expected `label` per the rules in `field_metrics_config`, instead of comparing the whole document as a unit. `institutions` above is scored as a list: predicted and expected entries are matched up, then each matched pair is scored field by field.

```{rubric} In this chapter
```

- **[2.1 Data Format](./data-format)**: how `label` is shaped for a schema instead of a plain string.
- **[2.2 Config Format](./config-format)**: turning on extraction mode, single-label schemas, and the manipulations it unlocks.
- **[2.3 Field Metrics](./field-metrics/index)**: configuring `field_metrics_config` for per-field precision, recall, and F1.
- **[2.4 Evaluation API](./evaluation-api)**: running `ExtractionExperiment`.
- **[2.5 Evaluation Results](./evaluation-results)**: reading per-field results off a run.
- **[2.6 Report Formats](./report-formats)**: the field-level breakdown in the HTML and PDF reports.
