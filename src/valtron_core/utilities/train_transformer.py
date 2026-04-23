"""Utility for training a transformer classifier from recipe-format data.

Usage as a Python API:

    from evaltron_core.utilities.train_transformer import train_transformer

    results = train_transformer(
        data=[{"id": "1", "content": "Great product!", "label": "positive"}, ...],
        output_dir="./my_sentiment_model",
        model_name="distilbert-base-uncased",
        num_epochs=3,
    )
    print(f"Model saved to: {results['model_dir']}")
    # Use results['model_dir'] as model_path in the recipe config:
    # {"type": "transformer", "model_path": results['model_dir']}

Usage as a CLI:

    python -m evaltron_core.utilities.train_transformer \\
        --data path/to/data.json \\
        --output-dir ./my_model \\
        --model-name distilbert-base-uncased \\
        --epochs 3
"""

import argparse
import json
from pathlib import Path
from typing import Any

import structlog

from valtron_core.models import Document, Label
from valtron_core.transformer_classifier import TransformerClassifier

logger = structlog.get_logger()


def train_transformer(
    data: list[dict[str, Any]],
    output_dir: str | Path,
    model_name: str = "distilbert-base-uncased",
    num_epochs: int = 3,
    batch_size: int = 8,
    learning_rate: float = 2e-5,
    warmup_steps: int = 500,
    weight_decay: float = 0.01,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, Any]:
    """Train a transformer classifier and save it to disk.

    Accepts data in the standard recipe format so you can use the same dataset
    for training and for recipe evaluation runs.

    Args:
        data: List of dicts with keys ``id``, ``content``, and ``label``.
              Labels can be any string values — the label set is inferred automatically.
        output_dir: Directory to write training checkpoints and the final model.
                    The final model lands at ``output_dir/final_model/``.
        model_name: Any HuggingFace sequence-classification model
                    (e.g. "distilbert-base-uncased", "roberta-base").
        num_epochs: Number of training epochs.
        batch_size: Per-device batch size for training and evaluation.
        learning_rate: AdamW learning rate.
        warmup_steps: Linear warmup steps.
        weight_decay: L2 regularisation weight.
        test_size: Fraction of data held out for evaluation (0–1).
        random_state: Random seed for reproducible splits.

    Returns:
        Dict with keys:
            - ``model_dir`` (str): Absolute path to the saved model directory.
              Pass this as ``model_path`` in a recipe model config.
            - ``train_loss`` (float)
            - ``eval_accuracy`` (float)
            - ``eval_loss`` (float)
    """
    documents: list[Document] = []
    labels: list[Label] = []

    for idx, item in enumerate(data):
        doc_id = str(item.get("id", f"doc_{idx}"))
        content = str(item.get("content", ""))
        label_raw = item.get("label", "")
        label_value = (
            json.dumps(label_raw) if isinstance(label_raw, (dict, list)) else str(label_raw)
        )
        documents.append(Document(id=doc_id, content=content))
        labels.append(Label(document_id=doc_id, value=label_value))

    classifier = TransformerClassifier(model_name=model_name, output_dir=output_dir)
    train_dataset, test_dataset = classifier.prepare_data(
        documents, labels, test_size=test_size, random_state=random_state
    )
    results = classifier.train(
        train_dataset,
        test_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
    )
    return results


def train_transformer_from_file(
    data_file: str | Path,
    output_dir: str | Path,
    **kwargs: Any,
) -> dict[str, Any]:
    """Load data from a JSON file and train a transformer classifier.

    The JSON file must contain a list of objects with ``id``, ``content``,
    and ``label`` keys — the same format used by recipe data files.

    Args:
        data_file: Path to a JSON data file.
        output_dir: Directory to write the trained model.
        **kwargs: Forwarded to :func:`train_transformer`.

    Returns:
        Same dict as :func:`train_transformer`.
    """
    data_file = Path(data_file)
    with open(data_file) as f:
        data = json.load(f)

    logger.info("loaded_data", path=str(data_file), num_examples=len(data))
    return train_transformer(data=data, output_dir=output_dir, **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m evaltron_core.utilities.train_transformer",
        description="Train a transformer classifier from recipe-format data.",
    )
    parser.add_argument(
        "--data",
        required=True,
        metavar="PATH",
        help="Path to a JSON data file ([{id, content, label}, ...]).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        metavar="DIR",
        help="Directory to save the trained model. Final model lands at DIR/final_model/.",
    )
    parser.add_argument(
        "--model-name",
        default="distilbert-base-uncased",
        metavar="NAME",
        help="HuggingFace model identifier (default: distilbert-base-uncased).",
    )
    parser.add_argument("--epochs", type=int, default=3, metavar="N", help="Training epochs (default: 3).")
    parser.add_argument("--batch-size", type=int, default=8, metavar="N", help="Batch size (default: 8).")
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        metavar="LR",
        help="Learning rate (default: 2e-5).",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=500,
        metavar="N",
        help="Warmup steps (default: 500).",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        metavar="W",
        help="Weight decay (default: 0.01).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        metavar="FRAC",
        help="Fraction of data for evaluation split (default: 0.2).",
    )
    parser.add_argument("--seed", type=int, default=42, metavar="N", help="Random seed (default: 42).")
    return parser


if __name__ == "__main__":
    parser = _build_parser()
    args = parser.parse_args()

    print("=" * 70)
    print("TRANSFORMER CLASSIFIER TRAINING")
    print("=" * 70)
    print(f"  Data:         {args.data}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Model:        {args.model_name}")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Learning rate:{args.learning_rate}")
    print(f"  Test split:   {args.test_size}")
    print()

    results = train_transformer_from_file(
        data_file=args.data,
        output_dir=args.output_dir,
        model_name=args.model_name,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        test_size=args.test_size,
        random_state=args.seed,
    )

    print()
    print("=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Train loss:    {results['train_loss']:.4f}")
    print(f"  Eval accuracy: {results['eval_accuracy']:.4f}")
    print(f"  Eval loss:     {results['eval_loss']:.4f}")
    print(f"  Model saved:   {results['model_dir']}")
    print()
    print("To use in a recipe config:")
    print('  {"type": "transformer", "model_path": "' + results["model_dir"] + '"}')
    print("=" * 70)
