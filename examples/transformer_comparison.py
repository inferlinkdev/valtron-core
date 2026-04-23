"""Transformer vs LLM comparison.

Trains a DistilBERT text classifier on a news-topic dataset, then compares
it side-by-side against cloud LLMs in a single evaluation report.

Requires: torch, transformers, datasets  (pip install torch transformers datasets)

Run:
    python examples/transformer_comparison.py
"""

from pathlib import Path

from valtron_core.models import Document, Label
from valtron_core.recipes import ModelEval
from valtron_core.transformer_classifier import TransformerClassifier

DATA = [
    {"id": "1",  "content": "The court ruled against the defendant in the criminal case.", "label": "legal"},
    {"id": "2",  "content": "Scientists discovered a new species of deep-sea fish.", "label": "science"},
    {"id": "3",  "content": "The central bank raised interest rates by 25 basis points.", "label": "finance"},
    {"id": "4",  "content": "The senator introduced a bill to reform healthcare legislation.", "label": "legal"},
    {"id": "5",  "content": "Researchers published findings on quantum computing error correction.", "label": "science"},
    {"id": "6",  "content": "The stock market fell sharply on disappointing earnings reports.", "label": "finance"},
    {"id": "7",  "content": "The jury delivered a unanimous guilty verdict.", "label": "legal"},
    {"id": "8",  "content": "A new vaccine shows 95% efficacy in clinical trials.", "label": "science"},
    {"id": "9",  "content": "Inflation rose to its highest level in four decades.", "label": "finance"},
    {"id": "10", "content": "The appeals court overturned the lower court's decision.", "label": "legal"},
    {"id": "11", "content": "Astronomers detected gravitational waves from a binary star merger.", "label": "science"},
    {"id": "12", "content": "The Federal Reserve signaled a pause in rate hikes.", "label": "finance"},
    {"id": "13", "content": "A landmark ruling expanded civil liberties protections.", "label": "legal"},
    {"id": "14", "content": "Gene-editing therapy reverses rare genetic disorder in trial.", "label": "science"},
    {"id": "15", "content": "Venture capital funding hit a five-year low last quarter.", "label": "finance"},
    {"id": "16", "content": "The Supreme Court agreed to hear the antitrust case.", "label": "legal"},
]

RESULTS_DIR = Path(__file__).resolve().parent / "results"
TRANSFORMER_PATH = RESULTS_DIR / "transformer" / "final_model"

CONFIG = {
    "use_case": "news topic classification: legal, science, finance",
    "output_formats": ["html"],
    "temperature": 0.0,
    "prompt": (
        "Classify the following news headline into exactly one category: "
        "legal, science, or finance. "
        "Reply with only the category name.\n\n"
        "{document}"
    ),
    "models": [
        {"name": "gpt-4o-mini", "label": "GPT-4o mini"},
        {"name": "claude-haiku-4-5-20251001", "label": "Claude Haiku"},
        {
            "type": "transformer",
            "label": "DistilBERT (fine-tuned)",
            "model_path": str(TRANSFORMER_PATH),
            "cost_rate": 0.10,
            "cost_rate_time_unit": "1hr",
        },
    ],
}


def train_transformer(output_dir: Path) -> None:
    documents = [Document(id=d["id"], content=d["content"], metadata={}) for d in DATA]
    labels = [Label(document_id=d["id"], value=d["label"], metadata={}) for d in DATA]

    classifier = TransformerClassifier(
        model_name="distilbert-base-uncased",
        output_dir=str(output_dir),
    )
    train_dataset, test_dataset = classifier.prepare_data(
        documents=documents,
        labels=labels,
        test_size=0.2,
    )
    metrics = classifier.train(train_dataset=train_dataset, test_dataset=test_dataset)
    print(f"  Training complete — metrics: {metrics}")


if __name__ == "__main__":
    print("Step 1: Training DistilBERT classifier...")
    train_transformer(RESULTS_DIR / "transformer")

    print("\nStep 2: Running evaluation (LLMs + transformer)...")
    experiment = ModelEval(config=CONFIG, data=DATA)
    report_path = experiment.run(output_dir=RESULTS_DIR / "transformer_comparison")

    print(f"\nReport: {report_path}\n")
    for result in experiment.results:
        print(f"  {result.model:<40}  accuracy={result.metrics.accuracy:.0%}  cost=${result.metrics.total_cost:.4f}")
