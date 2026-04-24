"""Simple sentiment classification.

Compares two LLMs on a small movie-review dataset using a single-line
classification prompt.  Results are saved to examples/results/sentiment/
and reused by incremental_evaluation.py.

Run:
    python examples/sentiment_classification.py
"""

import json
from pathlib import Path

from valtron_core.recipes import ModelEval

DATA = json.loads((Path(__file__).resolve().parent / "sentiment_data.json").read_text())

CONFIG = {
    "use_case": "movie review sentiment classification",
    "output_formats": ["html"],
    "temperature": 0.0,
    "prompt": (
        "Classify the sentiment of the following movie review as exactly one of: "
        "positive, negative, or neutral. "
        "Reply with only the single word label.\n\n"
        "Review: {document}"
    ),
    "models": [
        {"name": "gpt-4o-mini", "label": "GPT-4o mini"},
        {"name": "claude-haiku-4-5-20251001", "label": "Claude Haiku"},
    ],
}

if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "results" / "sentiment"

    experiment = ModelEval(config=CONFIG, data=DATA)
    report_path = experiment.run(output_dir=output_dir)

    print(f"\nReport: {report_path}\n")
    for result in experiment.results:
        print(f"  {result.model:<40}  accuracy={result.metrics.accuracy:.0%}  cost=${result.metrics.total_cost:.4f}")
