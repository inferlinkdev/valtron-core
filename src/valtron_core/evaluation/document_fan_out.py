"""Shared per-document concurrency bookkeeping, factored out of two duplicate copies.

``PromptEvaluator.evaluate`` (evaluator.py) and
``SummarizationExperiment._evaluate_model_documents`` (summarization.py) each
independently reimplement the same shape: bound concurrency with a semaphore,
run one async callable per document, call ``on_document_complete``/update a
progress bar for whatever comes back, filter out documents that were skipped.
What actually varies between them (call an LLM and score it vs. generate and
grade a summary) lives entirely in the callable each one passes in; this
module holds only the bookkeeping around that call, not the call itself.

Deliberately a plain function, not a class: there is nothing here a recipe
chooses between the way it chooses a Generator or a Scorer, so it does not
belong in evaluation/stages/. It also has no dependency on evaluator.py,
model_eval.py, or runner.py (only valtron_core.models), so both of those
modules, which do form part of a real import chain with each other, can
depend on this one without risk of a cycle.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Awaitable, Callable

from valtron_core.models import Document, PredictionResult

if TYPE_CHECKING:
    from tqdm import tqdm  # type: ignore[import-untyped]


async def fan_out_over_documents(
    documents: "list[Document]",
    max_concurrent: int,
    process_one: "Callable[[Document], Awaitable[PredictionResult | None]]",
    *,
    on_document_complete: "Callable[[PredictionResult], None] | None" = None,
    progress_bar: "tqdm | None" = None,
) -> "list[PredictionResult]":
    """Run ``process_one`` concurrently over ``documents``, bounded by a semaphore.

    ``process_one`` may return ``None`` to skip a document (e.g. one with no
    label); a ``None`` result is filtered out of the return value, and neither
    ``on_document_complete`` nor ``progress_bar`` are called for it. Results
    preserve document order (``asyncio.gather``, not as-completed), which is
    what makes a saved run diffable against another.

    Args:
        documents: Documents to process.
        max_concurrent: Maximum number of ``process_one`` calls in flight at once.
        process_one: Async callable doing the actual generation/scoring for one
            document; this function knows nothing about what it does.
        on_document_complete: Optional callback invoked with each non-``None``
            result, in document order, for live progress reporting.
        progress_bar: Optional progress bar advanced by one per non-``None``
            result.
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def process_with_semaphore(document: "Document") -> "PredictionResult | None":
        async with semaphore:
            result = await process_one(document)
        if result is not None:
            if on_document_complete is not None:
                on_document_complete(result)
            if progress_bar is not None:
                progress_bar.update(1)
        return result

    results = await asyncio.gather(*[process_with_semaphore(doc) for doc in documents])
    return [r for r in results if r is not None]
