"""Affiliation string extraction (multi-institution grading).

Given an academic affiliation string, extract every institution mentioned.
Uses a minimal prompt and per-field F1 scoring via field_metrics_config.

Run:
    python examples/affiliation_extraction.py
"""

from pathlib import Path

from pydantic import BaseModel

from valtron_core.recipes import ModelEval


class Institution(BaseModel):
    name: str
    city: str
    state: str
    country: str


class AffiliationResult(BaseModel):
    institutions: list[Institution]


DATA = [
    {
        "id": "1",
        "content": "John Smith, Department of Computer Science, Stanford University, Stanford, CA, USA; Google Research, Mountain View, CA, USA",
        "label": {
            "institutions": [
                {"name": "Stanford University", "city": "Stanford",      "state": "CA", "country": "USA"},
                {"name": "Google Research",     "city": "Mountain View", "state": "CA", "country": "USA"},
            ]
        },
    },
    {
        "id": "2",
        "content": "A. Kumar, Oxford Internet Institute, University of Oxford, Oxford, UK; DeepMind, London, UK; Alan Turing Institute, London, UK",
        "label": {
            "institutions": [
                {"name": "University of Oxford",   "city": "Oxford", "state": "", "country": "UK"},
                {"name": "DeepMind",               "city": "London", "state": "", "country": "UK"},
                {"name": "Alan Turing Institute",  "city": "London", "state": "", "country": "UK"},
            ]
        },
    },
    {
        "id": "3",
        "content": "C. Martinez, Department of Biology, Harvard University, Cambridge, MA, USA; Broad Institute of MIT and Harvard, Cambridge, MA, USA",
        "label": {
            "institutions": [
                {"name": "Harvard University",                   "city": "Cambridge", "state": "MA", "country": "USA"},
                {"name": "Broad Institute of MIT and Harvard",   "city": "Cambridge", "state": "MA", "country": "USA"},
            ]
        },
    },
    {
        "id": "4",
        "content": "E. Nguyen, Max Planck Institute for Intelligent Systems, Tübingen, Germany; ETH Zurich, Zurich, Switzerland",
        "label": {
            "institutions": [
                {"name": "Max Planck Institute for Intelligent Systems", "city": "Tübingen", "state": "", "country": "Germany"},
                {"name": "ETH Zurich",                                   "city": "Zurich",   "state": "", "country": "Switzerland"},
            ]
        },
    },
    {
        "id": "5",
        "content": "G. Brown, Vector Institute, Toronto, ON, Canada; Department of Computer Science, University of Toronto, Toronto, ON, Canada; Canadian Institute for Advanced Research, Toronto, ON, Canada",
        "label": {
            "institutions": [
                {"name": "Vector Institute",                      "city": "Toronto", "state": "ON", "country": "Canada"},
                {"name": "University of Toronto",                 "city": "Toronto", "state": "ON", "country": "Canada"},
                {"name": "Canadian Institute for Advanced Research", "city": "Toronto", "state": "ON", "country": "Canada"},
            ]
        },
    },
]

CONFIG = {
    "use_case": "academic affiliation extraction",
    "output_formats": ["html"],
    "temperature": 0.0,
    "prompt": "List all institutions in the following affiliation string.\n\n{content}",
    "models": [
        {"name": "gpt-4o-mini", "label": "GPT-4o mini"},
        {"name": "claude-haiku-4-5-20251001", "label": "Claude Haiku"},
    ],
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
                                "city": {
                                    "metric_config": {
                                        "metric": "comparator",
                                        "params": {"element_compare": "text_similarity"},
                                    }
                                },
                                "state": {
                                    "metric_config": {
                                        "metric": "comparator",
                                        "params": {"element_compare": "text_similarity"},
                                    }
                                },
                                "country": {
                                    "metric_config": {
                                        "metric": "comparator",
                                        "params": {"element_compare": "text_similarity"},
                                    }
                                },
                            },
                        },
                    },
                }
            },
        }
    },
}

if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "results" / "affiliation"

    experiment = ModelEval(config=CONFIG, data=DATA, response_format=AffiliationResult)
    report_path = experiment.run(output_dir=output_dir)

    print(f"\nReport: {report_path}\n")
    for result in experiment.results:
        print(f"  {result.model:<40}  accuracy={result.metrics.accuracy:.0%}  cost=${result.metrics.total_cost:.4f}")
