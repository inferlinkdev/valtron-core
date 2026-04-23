"""BERT model evaluation integrated with the evaluation framework.

.. deprecated::
    Use :class:`evaltron_core.transformer_classifier.TransformerClassifier` for
    inference and the recipe layer for evaluation. ``BERTEvaluator`` will be
    removed in a future release.
"""

import warnings

warnings.warn(
    "evaltron_core.bert_evaluator.BERTEvaluator is deprecated. "
    "Use evaltron_core.transformer_classifier.TransformerClassifier with "
    "ModelEval (type='transformer') instead.",
    DeprecationWarning,
    stacklevel=2,
)

import time
from typing import Any

import structlog

from valtron_core.bert_trainer import BERTTrainer
from valtron_core.models import Document, EvaluationInput, EvaluationResult, Label, PredictionResult

logger = structlog.get_logger()


class BERTEvaluator:
    """Evaluate BERT models using the same framework as LLM evaluations."""

    def __init__(self, trainer: BERTTrainer) -> None:
        """
        Initialize BERT evaluator.

        Args:
            trainer: Trained BERTTrainer instance
        """
        self.trainer = trainer

    async def evaluate_single(
        self,
        document: Document,
        label: Label,
    ) -> PredictionResult:
        """
        Evaluate a single document.

        Args:
            document: Document to evaluate
            label: Expected label

        Returns:
            PredictionResult
        """
        start_time = time.time()

        try:
            # Get prediction
            predicted_value = self.trainer.predict_single(document.content)

            end_time = time.time()
            response_time = end_time - start_time

            # Compare with expected
            is_correct = predicted_value.strip().lower() == label.value.strip().lower()

            logger.info(
                "bert_evaluation_single",
                document_id=document.id,
                predicted=predicted_value,
                expected=label.value,
                correct=is_correct,
                time=response_time,
            )

            return PredictionResult(
                document_id=document.id,
                predicted_value=predicted_value,
                expected_value=label.value,
                is_correct=is_correct,
                response_time=response_time,
                original_cost=0.0,
                cost=0.0,  # BERT inference is free
                model="bert-local",
                metadata={"content": document.content},
            )

        except Exception as e:
            end_time = time.time()
            response_time = end_time - start_time

            logger.error(
                "bert_evaluation_error",
                document_id=document.id,
                error=str(e),
                time=response_time,
            )

            return PredictionResult(
                document_id=document.id,
                predicted_value=f"ERROR: {str(e)}",
                expected_value=label.value,
                is_correct=False,
                response_time=response_time,
                original_cost=0.0,
                cost=0.0,
                model="bert-local",
                metadata={"content": document.content, "error": str(e)},
            )

    async def evaluate(
        self,
        eval_input: EvaluationInput,
    ) -> EvaluationResult:
        """
        Evaluate all documents.

        Args:
            eval_input: Evaluation input configuration

        Returns:
            EvaluationResult with all predictions and metrics
        """
        import uuid
        from datetime import datetime

        run_id = str(uuid.uuid4())
        result = EvaluationResult(
            run_id=run_id,
            started_at=datetime.now(),
            prompt_template="BERT local inference (no prompt)",
            model="bert-local",
            status="running",
        )

        # Create label lookup
        label_map = {label.document_id: label for label in eval_input.labels}

        logger.info(
            "bert_evaluation_started",
            run_id=run_id,
            total_documents=len(eval_input.documents),
        )

        try:
            # Evaluate all documents
            for doc in eval_input.documents:
                if doc.id not in label_map:
                    logger.warning("missing_label", document_id=doc.id)
                    continue

                prediction = await self.evaluate_single(
                    document=doc,
                    label=label_map[doc.id],
                )

                result.predictions.append(prediction)

            # Compute metrics
            result.compute_metrics()
            result.completed_at = datetime.now()
            result.status = "completed"

            logger.info(
                "bert_evaluation_completed",
                run_id=run_id,
                accuracy=result.metrics.accuracy if result.metrics else 0,
                total_time=result.metrics.total_time if result.metrics else 0,
            )

        except Exception as e:
            result.status = "failed"
            result.error = str(e)
            result.completed_at = datetime.now()

            logger.error("bert_evaluation_failed", run_id=run_id, error=str(e))

        return result


def create_bert_model_for_comparison(
    documents: list[Document],
    labels: list[Label],
    model_name: str = "bert-base-uncased",
    output_dir: str = "./bert_models",
    num_epochs: int = 3,
    batch_size: int = 8,
) -> BERTTrainer:
    """
    Train a BERT model for comparison with LLMs.

    Args:
        documents: Training documents
        labels: Training labels
        model_name: Pretrained BERT model to use
        output_dir: Directory to save model
        num_epochs: Number of training epochs
        batch_size: Training batch size

    Returns:
        Trained BERTTrainer instance
    """
    logger.info("creating_bert_model", model_name=model_name, num_docs=len(documents))

    # Initialize trainer
    trainer = BERTTrainer(model_name=model_name, output_dir=output_dir)

    # Prepare data
    train_dataset, test_dataset = trainer.prepare_data(documents, labels)

    # Train
    results = trainer.train(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        num_epochs=num_epochs,
        batch_size=batch_size,
    )

    logger.info(
        "bert_model_created",
        accuracy=results["eval_accuracy"],
        model_dir=results["model_dir"],
    )

    return trainer
