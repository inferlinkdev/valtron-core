"""Full three-stage workflow: train, evaluate, and analyze routing tradeoffs.

Demonstrates the complete valtron pipeline:

  Stage 1 -- Train a local DistilBERT classifier on labelled data
  Stage 2 -- Evaluate the transformer against cloud LLMs via ClassificationExperiment
  Stage 3 -- Analyze cost/accuracy tradeoffs to find optimal routing thresholds

The tradeoff report answers: "For my dataset, what fraction of predictions can the
transformer handle confidently, and at what confidence threshold should I escalate
to an LLM to hit my target accuracy?"

Requires: torch, transformers (pip install valtron-core[transformers])
API keys: OPENAI_API_KEY (or equivalent for your chosen LLM)

Run:
    python examples/tradeoff_analysis_workflow.py
"""

from pathlib import Path

from valtron_core.analysis import TradeoffAnalyzer
from valtron_core.evaluation import ClassificationExperiment
from valtron_core.models import Document, Label
from valtron_core.training import TransformerClassifier

# ---------------------------------------------------------------------------
# Dataset: customer review sentiment (binary: positive / negative)
# ---------------------------------------------------------------------------

DATA = [
    {
        "id": "1",
        "content": "Absolutely love this product, it exceeded all my expectations.",
        "label": "positive",
    },
    {"id": "2", "content": "Terrible quality, broke after one day of use.", "label": "negative"},
    {
        "id": "3",
        "content": "Fast shipping and the item was exactly as described.",
        "label": "positive",
    },
    {
        "id": "4",
        "content": "Complete waste of money, would not recommend to anyone.",
        "label": "negative",
    },
    {
        "id": "5",
        "content": "Great value for the price, very happy with my purchase.",
        "label": "positive",
    },
    {
        "id": "6",
        "content": "Arrived damaged and customer support was unhelpful.",
        "label": "negative",
    },
    {"id": "7", "content": "Works perfectly, exactly what I was looking for.", "label": "positive"},
    {"id": "8", "content": "Stopped working after a week, very disappointed.", "label": "negative"},
    {
        "id": "9",
        "content": "Exceeded my expectations, will definitely buy again.",
        "label": "positive",
    },
    {
        "id": "10",
        "content": "Poor build quality, feels very cheap and flimsy.",
        "label": "negative",
    },
    {
        "id": "11",
        "content": "Highly recommend this to anyone looking for a reliable option.",
        "label": "positive",
    },
    {
        "id": "12",
        "content": "The instructions were confusing and the product did not work.",
        "label": "negative",
    },
    {"id": "13", "content": "Amazing product, my whole family loves it.", "label": "positive"},
    {
        "id": "14",
        "content": "Not as advertised, very misleading product description.",
        "label": "negative",
    },
    {"id": "15", "content": "Solid construction and looks great in my home.", "label": "positive"},
    {"id": "16", "content": "Returned immediately, this is junk.", "label": "negative"},
    {"id": "17", "content": "Best purchase I have made in years.", "label": "positive"},
    {
        "id": "18",
        "content": "Defective out of the box, very frustrating experience.",
        "label": "negative",
    },
    {"id": "19", "content": "Top quality materials, clearly well-made.", "label": "positive"},
    {
        "id": "20",
        "content": "Cheap and unreliable, fell apart after minimal use.",
        "label": "negative",
    },
]

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "tradeoff_workflow"
TRANSFORMER_DIR = RESULTS_DIR / "transformer"
TRANSFORMER_PATH = TRANSFORMER_DIR / "final_model"

CONFIG = {
    "use_case": "customer review sentiment classification",
    "output_formats": ["html"],
    "temperature": 0.0,
    "prompt": (
        "Classify the sentiment of the following customer review as either "
        "'positive' or 'negative'. Reply with only the label.\n\n"
        "{content}"
    ),
    "models": [
        {
            "type": "transformer",
            "label": "DistilBERT",
            "model_path": str(TRANSFORMER_PATH),
            "cost_rate": 0.085,
            "cost_rate_time_unit": "1hr",
        },
        {"name": "gpt-4o-mini", "label": "GPT-4o Mini"},
        {"name": "gpt-4o", "label": "GPT-4o"},
    ],
}


def stage1_train(output_dir: Path) -> None:
    print("Stage 1: Training DistilBERT classifier...")
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
    print(f"  Training complete  accuracy={metrics.get('eval_accuracy', 0):.0%}")


def stage2_evaluate() -> ClassificationExperiment:
    print("\nStage 2: Evaluating transformer vs LLMs...")
    experiment = ClassificationExperiment(config=CONFIG, data=DATA)
    experiment.run(output_dir=RESULTS_DIR / "eval")

    print()
    for result in experiment.results:
        print(
            f"  {result.model:<30}  "
            f"accuracy={result.metrics.accuracy:.0%}  "
            f"cost=${result.metrics.total_cost:.4f}"
        )
    return experiment


def stage3_analyze(experiment: ClassificationExperiment) -> None:
    print("\nStage 3: Analyzing cost/accuracy tradeoffs...")

    analyzer = TradeoffAnalyzer.from_model_eval(experiment)

    # Compute the sweep once, then save in both formats
    analyzer.analyze()
    html_path = analyzer.save_html_report(RESULTS_DIR / "tradeoff_report.html")
    json_path = analyzer.save_json_report(RESULTS_DIR / "tradeoff_report.json")

    print(f"  HTML report: {html_path}")
    print(f"  JSON report: {json_path}")
    print(
        "  Open the HTML report to see: at what confidence threshold the transformer\n"
        "  can handle predictions autonomously vs. escalating to each LLM tier.\n"
        "  The JSON report contains the full sweep data for custom integrations."
    )


if __name__ == "__main__":
    stage1_train(TRANSFORMER_DIR)
    experiment = stage2_evaluate()
    stage3_analyze(experiment)
