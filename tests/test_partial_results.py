"""Tests for PartialResultStore, compute_prediction_hash, and crash-recovery integration."""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from valtron_core.models import PredictionResult
from valtron_core.partial_results import PartialResultStore, compute_prediction_hash
from valtron_core.runner import _completed_model_labels_on_disk, save_single_model_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_PROMPT = "Classify: {content}"
_PARAMS: dict = {"model": "gpt-4o", "temperature": 0.0}


def _make_pred(doc_id: str, model: str = "gpt-4o", correct: bool = True) -> PredictionResult:
    return PredictionResult(
        document_id=doc_id,
        predicted_value="positive",
        expected_value="positive" if correct else "negative",
        is_correct=correct,
        example_score=1.0 if correct else 0.0,
        response_time=0.5,
        original_cost=0.001,
        llm_cost=0.001,
        evaluation_cost=0.0,
        model=model,
        metadata={"content": f"content of {doc_id}"},
    )


def _hash(doc_id: str, prompt: str = _PROMPT, params: dict = _PARAMS) -> str:
    return compute_prediction_hash(f"content of {doc_id}", prompt, params)


def _record(store: PartialResultStore, model: str, pred: PredictionResult) -> None:
    """Record a prediction using the canonical hash of its metadata content."""
    h = compute_prediction_hash(
        pred.metadata.get("content", ""), _PROMPT, _PARAMS
    )
    store.record(model, pred, h)


def _make_result(model: str, doc_ids: list[str], tmp_path: Path):
    from valtron_core.models import EvaluationResult

    result = EvaluationResult(
        run_id="test-run",
        prompt_template=_PROMPT,
        model=model,
        status="completed",
        predictions=[_make_pred(d, model) for d in doc_ids],
    )
    result.compute_metrics()
    save_single_model_result(tmp_path, result)
    return result


# ---------------------------------------------------------------------------
# compute_prediction_hash
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestComputePredictionHash:
    def test_stable(self):
        h1 = compute_prediction_hash("hello", "prompt", {"model": "gpt-4o"})
        h2 = compute_prediction_hash("hello", "prompt", {"model": "gpt-4o"})
        assert h1 == h2

    def test_content_change(self):
        h1 = compute_prediction_hash("hello", "prompt", {"model": "gpt-4o"})
        h2 = compute_prediction_hash("world", "prompt", {"model": "gpt-4o"})
        assert h1 != h2

    def test_prompt_change(self):
        h1 = compute_prediction_hash("hello", "prompt A", {"model": "gpt-4o"})
        h2 = compute_prediction_hash("hello", "prompt B", {"model": "gpt-4o"})
        assert h1 != h2

    def test_model_params_change(self):
        h1 = compute_prediction_hash("hello", "prompt", {"model": "gpt-4o"})
        h2 = compute_prediction_hash("hello", "prompt", {"model": "gpt-4o-mini"})
        assert h1 != h2

    def test_param_order_invariant(self):
        h1 = compute_prediction_hash("hello", "p", {"model": "gpt-4o", "temperature": 0.0})
        h2 = compute_prediction_hash("hello", "p", {"temperature": 0.0, "model": "gpt-4o"})
        assert h1 == h2

    def test_dict_content_stable(self):
        h1 = compute_prediction_hash({"a": 1, "b": 2}, "p", {})
        h2 = compute_prediction_hash({"b": 2, "a": 1}, "p", {})
        assert h1 == h2

    def test_returns_hex_string(self):
        h = compute_prediction_hash("x", "p", {})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)


# ---------------------------------------------------------------------------
# PartialResultStore
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestPartialResultStore:
    def test_record_creates_jsonl(self, tmp_path: Path):
        store = PartialResultStore(tmp_path)
        _record(store, "gpt-4o", _make_pred("doc-1"))

        partial = tmp_path / "models" / ".gpt-4o.partial.jsonl"
        assert partial.exists()
        lines = [l for l in partial.read_text().splitlines() if l.strip()]
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["document_id"] == "doc-1"
        assert "hash" in data

    def test_record_appends_multiple(self, tmp_path: Path):
        store = PartialResultStore(tmp_path)
        for i in range(3):
            _record(store, "gpt-4o", _make_pred(f"doc-{i}"))

        partial = tmp_path / "models" / ".gpt-4o.partial.jsonl"
        lines = [l for l in partial.read_text().splitlines() if l.strip()]
        assert len(lines) == 3

    def test_get_valid_cached_returns_matching(self, tmp_path: Path):
        store = PartialResultStore(tmp_path)
        for doc_id in ("doc-1", "doc-2", "doc-3"):
            _record(store, "gpt-4o", _make_pred(doc_id))

        doc_content_map = {f"doc-{i}": f"content of doc-{i}" for i in range(1, 4)}
        result = store.get_valid_cached("gpt-4o", doc_content_map, _PROMPT, _PARAMS)
        assert set(result.keys()) == {"doc-1", "doc-2", "doc-3"}
        assert all(isinstance(v, PredictionResult) for v in result.values())

    def test_get_valid_cached_empty_store(self, tmp_path: Path):
        store = PartialResultStore(tmp_path)
        result = store.get_valid_cached("gpt-4o", {"doc-1": "content"}, _PROMPT, _PARAMS)
        assert result == {}

    def test_get_valid_cached_rejects_stale_content(self, tmp_path: Path):
        store = PartialResultStore(tmp_path)
        _record(store, "gpt-4o", _make_pred("doc-1"))

        # doc-1 now has different content
        result = store.get_valid_cached(
            "gpt-4o", {"doc-1": "CHANGED CONTENT"}, _PROMPT, _PARAMS
        )
        assert result == {}

    def test_get_valid_cached_rejects_stale_prompt(self, tmp_path: Path):
        store = PartialResultStore(tmp_path)
        for doc_id in ("doc-1", "doc-2"):
            _record(store, "gpt-4o", _make_pred(doc_id))

        doc_content_map = {f"doc-{i}": f"content of doc-{i}" for i in range(1, 3)}
        result = store.get_valid_cached("gpt-4o", doc_content_map, "DIFFERENT PROMPT", _PARAMS)
        assert result == {}

    def test_get_valid_cached_rejects_stale_model_params(self, tmp_path: Path):
        store = PartialResultStore(tmp_path)
        _record(store, "gpt-4o", _make_pred("doc-1"))

        result = store.get_valid_cached(
            "gpt-4o",
            {"doc-1": "content of doc-1"},
            _PROMPT,
            {"model": "gpt-4o-mini"},  # different model
        )
        assert result == {}

    def test_get_valid_cached_rejects_missing_hash(self, tmp_path: Path):
        """Legacy JSONL lines without a hash field are excluded."""
        store = PartialResultStore(tmp_path)
        partial = store._partial_path("gpt-4o")
        partial.parent.mkdir(parents=True, exist_ok=True)
        # Write a line without a hash field
        with open(partial, "w") as f:
            f.write(json.dumps({"document_id": "doc-1", "predicted_value": "positive",
                                "expected_value": "positive", "is_correct": True,
                                "response_time": 0.5, "model": "gpt-4o"}) + "\n")

        result = store.get_valid_cached(
            "gpt-4o", {"doc-1": "content of doc-1"}, _PROMPT, _PARAMS
        )
        assert result == {}

    def test_get_valid_cached_doc_not_in_current_run(self, tmp_path: Path):
        """Staged predictions for docs not in the current dataset are excluded."""
        store = PartialResultStore(tmp_path)
        _record(store, "gpt-4o", _make_pred("doc-old"))

        # Current run has different docs
        result = store.get_valid_cached(
            "gpt-4o", {"doc-new": "some content"}, _PROMPT, _PARAMS
        )
        assert result == {}

    def test_get_valid_cached_skips_malformed_lines(self, tmp_path: Path):
        store = PartialResultStore(tmp_path)
        _record(store, "gpt-4o", _make_pred("doc-1"))

        partial = store._partial_path("gpt-4o")
        with open(partial, "a") as f:
            f.write("NOT VALID JSON\n")

        _record(store, "gpt-4o", _make_pred("doc-2"))

        doc_content_map = {f"doc-{i}": f"content of doc-{i}" for i in range(1, 3)}
        result = store.get_valid_cached("gpt-4o", doc_content_map, _PROMPT, _PARAMS)
        assert set(result.keys()) == {"doc-1", "doc-2"}

    def test_finalize_removes_jsonl(self, tmp_path: Path):
        store = PartialResultStore(tmp_path)
        _record(store, "gpt-4o", _make_pred("doc-1"))

        partial = store._partial_path("gpt-4o")
        assert partial.exists()

        store.finalize("gpt-4o")
        assert not partial.exists()

    def test_finalize_is_idempotent(self, tmp_path: Path):
        store = PartialResultStore(tmp_path)
        store.finalize("gpt-4o")
        store.finalize("gpt-4o")

    def test_different_models_use_separate_files(self, tmp_path: Path):
        store = PartialResultStore(tmp_path)
        _record(store, "gpt-4o", _make_pred("doc-1", model="gpt-4o"))
        _record(store, "claude-3", _make_pred("doc-1", model="claude-3"))

        assert store._partial_path("gpt-4o") != store._partial_path("claude-3")

    def test_model_name_slash_normalized(self, tmp_path: Path):
        store = PartialResultStore(tmp_path)
        _record(store, "openai/gpt-4o", _make_pred("doc-1"))
        partial = store._partial_path("openai/gpt-4o")
        assert "_" in partial.name
        assert "/" not in partial.name

    def test_concurrent_writes_thread_safe(self, tmp_path: Path):
        store = PartialResultStore(tmp_path)
        errors: list[Exception] = []

        def write(doc_id: str) -> None:
            try:
                _record(store, "gpt-4o", _make_pred(doc_id))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write, args=(f"doc-{i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        doc_content_map = {f"doc-{i}": f"content of doc-{i}" for i in range(20)}
        result = store.get_valid_cached("gpt-4o", doc_content_map, _PROMPT, _PARAMS)
        assert len(result) == 20

    def test_field_metrics_serialized_and_recovered(self, tmp_path: Path):
        from valtron_core.evaluation.json_eval import EvalResult

        store = PartialResultStore(tmp_path)
        pred = _make_pred("doc-1")
        pred.field_metrics = EvalResult(
            path="label",
            score=1.0,
            weight=1.0,
            metric="exact",
            tp=1,
            fp=0,
            fn=0,
            tn=0,
            precision=1.0,
            recall=1.0,
        )
        _record(store, "gpt-4o", pred)

        result = store.get_valid_cached(
            "gpt-4o", {"doc-1": "content of doc-1"}, _PROMPT, _PARAMS
        )
        assert "doc-1" in result
        assert result["doc-1"].field_metrics is not None
        assert result["doc-1"].field_metrics.score == 1.0


# ---------------------------------------------------------------------------
# runner.py helpers
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestRunnerHelpers:
    def test_save_single_model_result_creates_file(self, tmp_path: Path):
        from valtron_core.models import EvaluationResult

        result = EvaluationResult(
            run_id="r1",
            prompt_template=_PROMPT,
            model="gpt-4o",
            status="completed",
            predictions=[_make_pred("doc-1")],
        )
        result.compute_metrics()

        out = save_single_model_result(tmp_path, result, model_prompt=_PROMPT)
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["model"] == "gpt-4o"
        assert data["status"] == "completed"
        assert len(data["predictions"]) == 1

    def test_save_single_model_result_stores_prompt_and_manipulations(self, tmp_path: Path):
        from valtron_core.models import EvaluationResult

        result = EvaluationResult(
            run_id="r1",
            prompt_template="orig",
            model="gpt-4o",
            status="completed",
            predictions=[_make_pred("doc-1")],
        )
        result.compute_metrics()

        save_single_model_result(
            tmp_path,
            result,
            model_prompt="custom prompt",
            prompt_manipulations=["few_shot"],
            model_override_prompt="override",
        )
        data = json.loads((tmp_path / "models" / "gpt-4o.json").read_text())
        assert data["prompt_template"] == "custom prompt"
        assert data["prompt_manipulations"] == ["few_shot"]
        assert data["override_prompt"] == "override"

    def test_completed_model_labels_on_disk_empty_dir(self, tmp_path: Path):
        (tmp_path / "models").mkdir()
        assert _completed_model_labels_on_disk(tmp_path) == set()

    def test_completed_model_labels_on_disk_no_models_dir(self, tmp_path: Path):
        assert _completed_model_labels_on_disk(tmp_path) == set()

    def test_completed_model_labels_on_disk_finds_completed(self, tmp_path: Path):
        _make_result("gpt-4o", ["doc-1", "doc-2"], tmp_path)
        _make_result("claude-3", ["doc-1"], tmp_path)

        labels = _completed_model_labels_on_disk(tmp_path)
        assert labels == {"gpt-4o", "claude-3"}

    def test_completed_model_labels_on_disk_ignores_running(self, tmp_path: Path):
        from valtron_core.models import EvaluationResult

        result = EvaluationResult(
            run_id="r1",
            prompt_template="p",
            model="gpt-4o",
            status="running",
            predictions=[],
        )
        save_single_model_result(tmp_path, result)

        assert _completed_model_labels_on_disk(tmp_path) == set()

    def test_completed_model_labels_skips_partial_jsonl_files(self, tmp_path: Path):
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        (models_dir / ".gpt-4o.partial.jsonl").write_text(
            json.dumps({"model": "gpt-4o", "status": "completed"}) + "\n"
        )
        assert _completed_model_labels_on_disk(tmp_path) == set()


# ---------------------------------------------------------------------------
# Integration: resume skips completed documents
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCrashRecoveryIntegration:
    def test_resume_skips_completed_docs(self, tmp_path: Path):
        """Docs with valid cached predictions are filtered from the evaluator."""
        store = PartialResultStore(tmp_path)
        _record(store, "gpt-4o", _make_pred("doc-1"))
        _record(store, "gpt-4o", _make_pred("doc-2"))

        all_docs = [f"doc-{i}" for i in range(1, 4)]
        doc_content_map = {d: f"content of {d}" for d in all_docs}
        cached = store.get_valid_cached("gpt-4o", doc_content_map, _PROMPT, _PARAMS)
        remaining = [d for d in all_docs if d not in cached]

        assert remaining == ["doc-3"]

    def test_resume_uses_hash_not_just_doc_id(self, tmp_path: Path):
        """A doc with the same ID but changed content is NOT reused."""
        store = PartialResultStore(tmp_path)
        _record(store, "gpt-4o", _make_pred("doc-1"))  # staged with "content of doc-1"

        # Same doc ID, but content has changed
        result = store.get_valid_cached(
            "gpt-4o", {"doc-1": "TOTALLY DIFFERENT CONTENT"}, _PROMPT, _PARAMS
        )
        assert "doc-1" not in result

    def test_resume_merges_predictions(self, tmp_path: Path):
        """Staged predictions merge with newly evaluated ones and metrics recompute."""
        from valtron_core.models import EvaluationResult

        store = PartialResultStore(tmp_path)
        for doc_id in ("doc-1", "doc-2"):
            _record(store, "gpt-4o", _make_pred(doc_id))

        new_result = EvaluationResult(
            run_id="r1",
            prompt_template=_PROMPT,
            model="gpt-4o",
            status="completed",
            predictions=[_make_pred("doc-3")],
        )
        new_result.compute_metrics()

        doc_content_map = {f"doc-{i}": f"content of doc-{i}" for i in range(1, 4)}
        cached = store.get_valid_cached("gpt-4o", doc_content_map, _PROMPT, _PARAMS)
        new_result.predictions = list(cached.values()) + new_result.predictions
        new_result.compute_metrics()

        assert len(new_result.predictions) == 3
        assert {p.document_id for p in new_result.predictions} == {"doc-1", "doc-2", "doc-3"}
        assert new_result.metrics is not None
        assert new_result.metrics.total_documents == 3

    def test_normal_run_cleans_up_staging(self, tmp_path: Path):
        """On successful model completion the JSONL is removed and final JSON exists."""
        from valtron_core.models import EvaluationResult

        store = PartialResultStore(tmp_path)
        preds = [_make_pred(f"doc-{i}") for i in range(3)]
        for p in preds:
            _record(store, "gpt-4o", p)

        assert store._partial_path("gpt-4o").exists()

        result = EvaluationResult(
            run_id="r1",
            prompt_template=_PROMPT,
            model="gpt-4o",
            status="completed",
            predictions=preds,
        )
        result.compute_metrics()
        save_single_model_result(tmp_path, result)
        store.finalize("gpt-4o")

        assert not store._partial_path("gpt-4o").exists()
        assert (tmp_path / "models" / "gpt-4o.json").exists()

    def test_model_level_skip_from_disk(self, tmp_path: Path):
        """Models with a final completed JSON are skipped entirely on re-run."""
        _make_result("gpt-4o", ["doc-1", "doc-2"], tmp_path)

        labels = _completed_model_labels_on_disk(tmp_path)
        assert "gpt-4o" in labels

        all_model_labels = ["gpt-4o", "claude-3"]
        models_to_run = [m for m in all_model_labels if m not in labels]
        assert models_to_run == ["claude-3"]
