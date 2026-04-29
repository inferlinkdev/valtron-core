"""Evaluator for optimized prompts including decomposed/chained prompts."""

import inspect
import time
import uuid
from datetime import datetime
from typing import Any

import structlog

from valtron_core.models import Document, EvaluationResult, Label, PredictionResult
from valtron_core.prompt_optimizer import PromptChainEvaluator, PromptDecomposer

logger = structlog.get_logger()


class OptimizedPromptEvaluator:
    """Evaluate documents using optimized prompts (decomposed, chained, etc.)."""

    def __init__(
        self,
        chain_evaluator: PromptChainEvaluator,
        decomposer: PromptDecomposer,
    ) -> None:
        """
        Initialize optimized prompt evaluator.

        Args:
            chain_evaluator: Chain evaluator instance
            decomposer: Prompt decomposer instance
        """
        self.chain_evaluator = chain_evaluator
        self.decomposer = decomposer

    async def evaluate_with_decomposition(
        self,
        documents: list[Document],
        labels: list[Label],
        original_prompt: str,
        model: str,
        temperature: float = 0.0,
        comparison_fn: Any = None,
    ) -> tuple[EvaluationResult, dict[str, Any]]:
        """
        Evaluate documents using decomposed prompts.

        Args:
            documents: Documents to evaluate
            labels: Expected labels
            original_prompt: Original prompt to decompose
            model: Model to use for evaluation
            temperature: Temperature for generation
            comparison_fn: Optional comparison function

        Returns:
            Tuple of (EvaluationResult, decomposition_info)
        """
        # First, decompose the prompt
        logger.info("decomposing_prompt_for_evaluation", model=model)
        decomposition = await self.decomposer.optimize(original_prompt)

        # Create chained prompts
        chained_prompts = self.decomposer.create_chained_prompts(
            decomposition, document_placeholder="{content}"
        )

        logger.info(
            "evaluating_with_decomposed_prompts",
            num_steps=len(chained_prompts),
            model=model,
        )

        # Create result object
        run_id = str(uuid.uuid4())
        result = EvaluationResult(
            run_id=run_id,
            started_at=datetime.now(),
            prompt_template=f"DECOMPOSED ({len(chained_prompts)} steps): {original_prompt[:100]}...",
            model=f"{model}-decomposed",
            status="running",
        )

        # Create label lookup
        label_map = {label.document_id: label for label in labels}

        # Check comparison_fn arity once before the loop
        comparison_fn_takes_context = False
        if comparison_fn:
            sig = inspect.signature(comparison_fn)
            comparison_fn_takes_context = len(sig.parameters) >= 3

        # Evaluate each document
        for doc in documents:
            if doc.id not in label_map:
                logger.warning("missing_label", document_id=doc.id)
                continue

            start_time = time.time()

            try:
                # Execute chain
                final_output, intermediate_outputs = await self.chain_evaluator.execute_chain(
                    prompts=chained_prompts,
                    document_content=doc.content,
                    model=model,
                    temperature=temperature,
                )

                end_time = time.time()
                response_time = end_time - start_time

                # Compare with expected
                predicted_value = final_output.strip()
                expected_value = label_map[doc.id].value

                if comparison_fn:
                    if comparison_fn_takes_context:
                        is_correct = comparison_fn(predicted_value, expected_value, doc.content)
                    else:
                        is_correct = comparison_fn(predicted_value, expected_value)
                else:
                    is_correct = (
                        predicted_value.lower().strip() == expected_value.lower().strip()
                    )

                logger.info(
                    "decomposed_evaluation_single",
                    document_id=doc.id,
                    predicted=predicted_value,
                    expected=expected_value,
                    correct=is_correct,
                    num_steps=len(chained_prompts),
                )

                # Cost is multiplied by number of steps
                cost_multiplier = len(chained_prompts)

                prediction = PredictionResult(
                    document_id=doc.id,
                    predicted_value=predicted_value,
                    expected_value=expected_value,
                    is_correct=is_correct,
                    response_time=response_time,
                    original_cost=0.0,
                    cost=0.0,  # Will be estimated based on model
                    model=f"{model}-decomposed",
                    metadata={
                        "content": doc.content,
                        "num_steps": len(chained_prompts),
                        "intermediate_outputs": intermediate_outputs,
                        "decomposition_strategy": "chain",
                    },
                )

                result.predictions.append(prediction)

            except Exception as e:
                end_time = time.time()
                response_time = end_time - start_time

                logger.error(
                    "decomposed_evaluation_error",
                    document_id=doc.id,
                    error=str(e),
                )

                prediction = PredictionResult(
                    document_id=doc.id,
                    predicted_value=f"ERROR: {str(e)}",
                    expected_value=label_map[doc.id].value,
                    is_correct=False,
                    response_time=response_time,
                    original_cost=0.0,
                    cost=0.0,
                    model=f"{model}-decomposed",
                    metadata={"content": doc.content, "error": str(e), "num_steps": len(chained_prompts)},
                )

                result.predictions.append(prediction)

        # Compute metrics
        result.compute_metrics()
        result.completed_at = datetime.now()
        result.status = "completed"

        logger.info(
            "decomposed_evaluation_completed",
            run_id=run_id,
            accuracy=result.metrics.accuracy if result.metrics else 0,
            num_steps=len(chained_prompts),
        )

        return result, decomposition


async def compare_original_vs_decomposed(
    documents: list[Document],
    labels: list[Label],
    original_prompt: str,
    model: str,
    optimizer_model: str = "gemini-pro",
    num_sub_prompts: int = 5,
    temperature: float = 0.0,
) -> tuple[EvaluationResult, EvaluationResult, dict[str, Any]]:
    """
    Compare original prompt vs decomposed prompts.

    Args:
        documents: Documents to evaluate
        labels: Expected labels
        original_prompt: Original prompt
        model: Model to test with
        optimizer_model: Model to use for decomposition
        num_sub_prompts: Max number of sub-prompts
        temperature: Temperature for generation

    Returns:
        Tuple of (original_result, decomposed_result, decomposition_info)
    """
    from valtron_core.client import LLMClient
    from valtron_core.evaluator import PromptEvaluator
    from valtron_core.models import EvaluationInput

    client = LLMClient()

    # Evaluate with original prompt
    logger.info("evaluating_original_prompt", model=model)
    evaluator = PromptEvaluator(client=client)

    eval_input = EvaluationInput(
        documents=documents,
        labels=labels,
        prompt_template=original_prompt,
        model=model,
        temperature=temperature,
    )

    original_result = await evaluator.evaluate(eval_input)

    # Evaluate with decomposed prompts
    logger.info("evaluating_decomposed_prompts", model=model)
    decomposer = PromptDecomposer(
        client=client,
        optimizer_model=optimizer_model,
        num_sub_prompts=num_sub_prompts,
    )
    chain_evaluator = PromptChainEvaluator(client=client)
    optimized_evaluator = OptimizedPromptEvaluator(
        chain_evaluator=chain_evaluator,
        decomposer=decomposer,
    )

    decomposed_result, decomposition = await optimized_evaluator.evaluate_with_decomposition(
        documents=documents,
        labels=labels,
        original_prompt=original_prompt,
        model=model,
        temperature=temperature,
    )

    return original_result, decomposed_result, decomposition
