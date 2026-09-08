"""Shared run-directory persistence: one writer entry point, one reader.

Every recipe already writes the same on-disk shape (``metadata.json`` plus
``models/<name>.json``) through ``runner.save_run_dir``/``save_single_model_result``,
re-exported here so this module is the one place "how do I read/write a run
directory" is answered, rather than reaching into ``runner.py`` directly.

Reading it back was, before this module existed, three near-identical
hand-rolled loops: ``ModelEval._result_from_model_data`` (defaults
``is_correct``/``example_score`` to ``None`` when a stored prediction lacks
them), ``ReferencedEval.load_experiment_results``'s inline version (defaults to
``False``/``0.0``), and ``EvaluationRunner._load_results_from_run_dir``'s
inline version (also ``False``/``0.0``, via a wholly separate parsing pass).
``RunDirectoryCodec.build_prediction`` collapses those three into one
implementation.

This module deliberately preserves every caller's exact current output,
divergences included: it does not reconcile the ``None`` vs. ``False``/``0.0``
defaulting split, nor the fact that only ``ModelEval``'s reader restored
``error``/``task_scores`` or that only ``EvaluationRunner``'s restored
``confidence_score``. Both are surfaced here as explicit keyword arguments a
caller chooses, rather than silently equalized, so a future commit can
reconcile them deliberately (see ``FORMAT_VERSION``) instead of this
deduplication accidentally doing it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from valtron_core.models import EvaluationResult, PredictionResult
from valtron_core.runner import save_run_dir, save_single_model_result

__all__ = ["RunDirectoryCodec", "save_run_dir", "save_single_model_result"]


class RunDirectoryCodec:
    """Reads the run-directory shape ``runner.save_run_dir`` writes."""

    @staticmethod
    def read_json(path: "str | Path") -> dict[str, Any]:
        """Load one JSON file: a run directory's ``metadata.json`` or a ``models/<name>.json``."""
        with open(path) as f:
            data: dict[str, Any] = json.load(f)
        return data

    @staticmethod
    def build_prediction(
        p: dict[str, Any],
        *,
        model_label: str,
        expected_value_fallback: Any = None,
        legacy_defaults: bool,
        include_error_and_task_scores: bool = False,
        include_confidence_score: bool = False,
    ) -> PredictionResult:
        """One stored prediction dict -> ``PredictionResult``.

        ``legacy_defaults=True`` reproduces ``ReferencedEval``'s and
        ``EvaluationRunner``'s historical behavior: a prediction saved before
        ``is_correct``/``example_score`` existed (or otherwise written without
        them) reloads as ``False``/``0.0``. ``legacy_defaults=False``
        reproduces ``ModelEval``'s: the same case reloads as ``None``, meaning
        "not scored". A caller's choice, not reconciled here.

        ``include_error_and_task_scores``/``include_confidence_score`` are the
        two other small, pre-existing divergences between the three original
        readers (only ``ModelEval`` restored the first pair, only
        ``EvaluationRunner`` restored the second), preserved verbatim as an
        explicit choice per call site rather than silently equalized.
        """
        field_metrics = None
        if p.get("field_metrics"):
            try:
                from valtron_core.scoring.json_eval import EvalResult

                field_metrics = EvalResult.model_validate(p["field_metrics"])
            except Exception:
                pass

        kwargs: dict[str, Any] = {
            "document_id": p["document_id"],
            "predicted_value": p["predicted_value"],
            "expected_value": p.get("expected_value", expected_value_fallback),
            "is_correct": p.get("is_correct", False if legacy_defaults else None),
            "example_score": p.get("example_score", 0.0 if legacy_defaults else None),
            "response_time": p.get("response_time", 0.0),
            "original_cost": p.get("original_cost", 0.0),
            "llm_cost": p.get("llm_cost", p.get("cost", 0.0)),
            "evaluation_cost": p.get("evaluation_cost", 0.0),
            "model": model_label,
            "field_metrics": field_metrics,
        }
        if include_error_and_task_scores:
            kwargs["error"] = p.get("error")
            kwargs["task_scores"] = p.get("task_scores")
        if include_confidence_score:
            kwargs["confidence_score"] = p.get("confidence_score")
        return PredictionResult(**kwargs)

    @staticmethod
    def finalize_evaluation_result(result: EvaluationResult, model_data: dict[str, Any]) -> None:
        """Apply ``started_at``/``completed_at`` overrides; compute metrics if absent.

        The one piece of ``EvaluationResult`` reconstruction that was
        byte-identical across all three original readers, unlike the
        surrounding construction, which differs in which fields each has
        available (see ``build_prediction``).
        """
        if model_data.get("started_at"):
            result.started_at = model_data["started_at"]
        if model_data.get("completed_at"):
            result.completed_at = model_data["completed_at"]
        if not result.metrics and result.predictions:
            result.compute_metrics()
