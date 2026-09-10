"""Characterization tests for evaluation.document_fan_out.fan_out_over_documents.

Locks in the concurrency/bookkeeping behavior shared by
PromptEvaluator.evaluate and SummarizationExperiment's own document loop,
ahead of migrating either onto this helper.
"""

import asyncio

import pytest

from valtron_core.evaluation.document_fan_out import fan_out_over_documents
from valtron_core.models import Document, PredictionResult


def _doc(doc_id: str) -> Document:
    return Document(id=doc_id, content=f"content-{doc_id}")


def _pred(doc_id: str) -> PredictionResult:
    return PredictionResult(document_id=doc_id, predicted_value="x", response_time=0.0, model="m")


class TestFanOutOverDocuments:
    @pytest.mark.asyncio
    async def test_preserves_document_order_regardless_of_completion_order(self):
        # doc "0" sleeps longest, so it would finish last if order depended on
        # completion time rather than input order.
        delays = {"0": 0.03, "1": 0.02, "2": 0.01}
        documents = [_doc(str(i)) for i in range(3)]

        async def process_one(document):
            await asyncio.sleep(delays[document.id])
            return _pred(document.id)

        results = await fan_out_over_documents(documents, max_concurrent=3, process_one=process_one)

        assert [r.document_id for r in results] == ["0", "1", "2"]

    @pytest.mark.asyncio
    async def test_respects_max_concurrent(self):
        documents = [_doc(str(i)) for i in range(6)]
        in_flight = 0
        max_seen = 0

        async def process_one(document):
            nonlocal in_flight, max_seen
            in_flight += 1
            max_seen = max(max_seen, in_flight)
            await asyncio.sleep(0.01)
            in_flight -= 1
            return _pred(document.id)

        await fan_out_over_documents(documents, max_concurrent=2, process_one=process_one)

        assert max_seen <= 2

    @pytest.mark.asyncio
    async def test_filters_none_and_skips_callbacks_for_it(self):
        documents = [_doc("0"), _doc("1"), _doc("2")]
        completed = []

        async def process_one(document):
            if document.id == "1":
                return None
            return _pred(document.id)

        class FakeProgressBar:
            def __init__(self):
                self.updates = 0

            def update(self, n):
                self.updates += n

        bar = FakeProgressBar()
        results = await fan_out_over_documents(
            documents,
            max_concurrent=3,
            process_one=process_one,
            on_document_complete=completed.append,
            progress_bar=bar,
        )

        assert [r.document_id for r in results] == ["0", "2"]
        assert [p.document_id for p in completed] == ["0", "2"]
        assert bar.updates == 2

    @pytest.mark.asyncio
    async def test_works_with_no_callback_and_no_progress_bar(self):
        documents = [_doc("0"), _doc("1")]

        async def process_one(document):
            return _pred(document.id)

        results = await fan_out_over_documents(documents, max_concurrent=2, process_one=process_one)

        assert [r.document_id for r in results] == ["0", "1"]
