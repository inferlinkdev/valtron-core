"""Per-model JSONL staging store for crash-safe prediction persistence."""

from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from valtron_core.models import PredictionResult


def compute_prediction_hash(
    doc_content: Any,
    prompt_template: str,
    model_params: dict[str, Any],
) -> str:
    """Return a stable SHA-256 hex digest over the inputs that determine a prediction.

    A matching hash means: same document content + same prompt + same model params.
    This is the single canonical implementation -- callers must not reimplement it.

    Args:
        doc_content: Raw document content (str or dict). Dicts are serialized with
            sorted keys for stability.
        prompt_template: The effective prompt template used for this model.
        model_params: Model configuration dict (name, temperature, etc.). Key order
            does not affect the hash.
    """
    canonical = json.dumps(
        {
            "content": (
                doc_content
                if isinstance(doc_content, str)
                else json.dumps(doc_content, sort_keys=True, default=str)
            ),
            "prompt": prompt_template,
            "model": json.dumps(model_params, sort_keys=True, default=str),
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


class PartialResultStore:
    """Appends each completed prediction to a per-model JSONL file as it arrives.

    On a normal run the JSONL file is created when the first prediction for a model
    lands and is deleted via ``finalize()`` once the final ``models/{name}.json`` has
    been written.  If the process is killed the JSONL survives and can be used by a
    subsequent run to skip already-completed documents.

    Each line in the JSONL includes a ``hash`` field produced by
    ``compute_prediction_hash()``.  On resume, ``get_valid_cached()`` recomputes the
    hash from the current inputs and only returns entries whose hash still matches,
    automatically invalidating predictions produced with a different prompt, model
    params, or document content.

    Thread-safety: each model has its own ``threading.Lock``, so concurrent document
    threads writing to the same model's file are serialized.  Different models always
    write to different files so there is no cross-model contention.
    """

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _partial_path(self, model_label: str) -> Path:
        safe = model_label.replace("/", "_")
        return self._output_dir / "models" / f".{safe}.partial.jsonl"

    def _get_lock(self, model_label: str) -> threading.Lock:
        with self._global_lock:
            if model_label not in self._locks:
                self._locks[model_label] = threading.Lock()
            return self._locks[model_label]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_valid_cached(
        self,
        model_label: str,
        doc_content_by_id: dict[str, Any],
        prompt_template: str,
        model_params: dict[str, Any],
    ) -> dict[str, PredictionResult]:
        """Return cached predictions whose hash matches the current inputs.

        Recomputes ``compute_prediction_hash()`` for each staged entry using the
        current document content, prompt, and model params.  Only entries where the
        stored hash matches are returned.

        Stale entries (changed content, prompt, or params), entries for document IDs
        not present in ``doc_content_by_id``, and legacy entries written before
        hashing was introduced (no ``"hash"`` field) are all excluded and will be
        re-evaluated.

        Args:
            model_label: Model label used to locate the staging file.
            doc_content_by_id: Mapping of ``{doc_id: content}`` for the current run.
            prompt_template: The effective prompt used for this model.
            model_params: Model configuration dict used for this model.

        Returns:
            ``{doc_id: PredictionResult}`` for every valid cached prediction.
        """
        from valtron_core.models import PredictionResult

        path = self._partial_path(model_label)
        if not path.exists():
            return {}

        valid: dict[str, PredictionResult] = {}
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    stored_hash = data.get("hash")
                    if stored_hash is None:
                        continue
                    doc_id = data["document_id"]
                    content = doc_content_by_id.get(doc_id)
                    if content is None:
                        continue
                    expected = compute_prediction_hash(content, prompt_template, model_params)
                    if expected != stored_hash:
                        continue
                    pred_data = {k: v for k, v in data.items() if k != "hash"}
                    valid[doc_id] = PredictionResult(**pred_data)
                except Exception:
                    pass
        return valid

    def record(self, model_label: str, prediction: PredictionResult, input_hash: str) -> None:
        """Append *prediction* to the model's JSONL staging file.

        Args:
            model_label: Model label used to locate the staging file.
            prediction: The completed prediction to stage.
            input_hash: Hash produced by ``compute_prediction_hash()`` for this
                prediction's inputs.  Must be computed by the caller so that the
                same canonical function is used everywhere.

        The write is serialized via a per-model lock so this is safe to call
        from multiple threads handling the same model concurrently.
        """
        pred_dict: dict[str, Any] = {
            "hash": input_hash,
            "document_id": prediction.document_id,
            "predicted_value": prediction.predicted_value,
            "expected_value": prediction.expected_value,
            "original_cost": prediction.original_cost,
            "llm_cost": prediction.llm_cost,
            "evaluation_cost": prediction.evaluation_cost,
            "response_time": prediction.response_time,
            "is_correct": prediction.is_correct,
            "example_score": prediction.example_score,
            "model": prediction.model,
        }
        if prediction.field_metrics is not None:
            pred_dict["field_metrics"] = prediction.field_metrics.model_dump()

        path = self._partial_path(model_label)
        path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(pred_dict, default=str) + "\n"

        lock = self._get_lock(model_label)
        with lock:
            with open(path, "a") as f:
                f.write(line)

    def finalize(self, model_label: str) -> None:
        """Remove the JSONL staging file after the final model JSON has been written."""
        path = self._partial_path(model_label)
        try:
            path.unlink(missing_ok=True)
        except Exception:
            pass
