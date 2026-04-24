"""Incremental evaluation — extend a completed run with a new model.

Runs an initial evaluation with one model, saves the results, then loads
them and adds a second model.  Only the new model is evaluated; prior
results are preserved and the report is regenerated with all models combined.

Run:
    python examples/incremental_evaluation.py
"""

from pathlib import Path

from valtron_core.recipes import ModelEval

DATA = [
    {"id": "1", "content": "An absolute masterpiece. I was on the edge of my seat the whole time.", "label": "positive"},
    {"id": "2", "content": "Painfully boring. I walked out after thirty minutes.", "label": "negative"},
    {"id": "3", "content": "Decent enough, a few good scenes but nothing memorable.", "label": "neutral"},
    {"id": "4", "content": "The performances were outstanding and the writing was sharp.", "label": "positive"},
    {"id": "5", "content": "Disappointing sequel that fails to live up to the original.", "label": "negative"},
    {"id": "6", "content": "Not great, not terrible. Passes the time.", "label": "neutral"},
]

BASE_PROMPT = (
    "Classify the sentiment of the following movie review as exactly one of: "
    "positive, negative, or neutral. "
    "Reply with only the single word label.\n\n"
    "Review: {document}"
)

INITIAL_CONFIG = {
    "use_case": "incremental evaluation demo",
    "output_formats": ["html"],
    "temperature": 0.0,
    "prompt": BASE_PROMPT,
    "models": [
        {"name": "gpt-4o-mini", "label": "GPT-4o mini"},
    ],
}

if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parent / "results" / "incremental"

    # Step 1: Initial evaluation — one model, results saved to disk
    print("Step 1: Initial evaluation with GPT-4o mini...")
    experiment = ModelEval(config=INITIAL_CONFIG, data=DATA)
    experiment.run(output_dir=results_dir)
    print(f"  Saved to: {results_dir}")

    # Step 2: Load saved results and extend with a new model
    print("\nStep 2: Loading results and adding Claude Haiku...")
    experiment = ModelEval.load_experiment_results(results_dir)
    experiment.add_models([
        {"name": "claude-haiku-4-5-20251001", "label": "Claude Haiku"},
    ])

    # Only Claude Haiku is evaluated; GPT-4o mini results are reused
    experiment.run(output_dir=results_dir)

    print(f"\nUpdated report: {results_dir}\n")
    for result in experiment.results:
        print(f"  {result.model:<40}  accuracy={result.metrics.accuracy:.0%}  cost=${result.metrics.total_cost:.4f}")
