"""Shared run-directory persistence: one writer entry point, one reader.

Every recipe already writes the same on-disk shape (``metadata.json`` plus
``models/<name>.json``) through ``runner.save_run_dir``/``save_single_model_result``,
re-exported here so this module is the one place "how do I read/write a run
directory" is answered, rather than reaching into ``runner.py`` directly.

Reading it back was, before this module existed, three near-identical
hand-rolled loops: ``ModelEval._result_from_model_data`` (defaulted
``is_correct``/``example_score`` to ``None`` when a stored prediction lacked
them), ``ReferencedEval.load_experiment_results``'s inline version (defaulted
to ``False``/``0.0``), and ``EvaluationRunner._load_results_from_run_dir``'s
inline version (also ``False``/``0.0``, via a wholly separate parsing pass;
``utilities/aggregate_reports.py`` had its own fourth copy of the same
``False``/``0.0`` behavior). ``RunDirectoryCodec.build_prediction`` collapses
all four into one implementation.

That collapse initially preserved every caller's ``None``-vs-``False``/``0.0``
divergence behind a ``legacy_defaults`` flag rather than picking one, exactly
so this reconciliation could ship as its own explicitly labeled, explicitly
tested change. It has now happened: every reader defaults a missing
``is_correct``/``example_score`` to ``None`` (meaning "not scored"), the
semantically correct answer and the one ``ModelEval`` already used, rather
than the misleading ``False``/``0.0`` the other three used to fall back to
(which reads as "scored, and wrong" for a prediction that was never actually
scored at all). ``FORMAT_VERSION`` marks this: every run directory written
from here on records ``metadata.json["format_version"] = FORMAT_VERSION``; a
missing key on load is treated as version 1, today's shape before this field
existed (verified against ``examples/results/summarization/metadata.json``,
which predates it). The reconciled defaulting rule applies uniformly
regardless of version, since there is no way to recover which behavior a
version-1 file's own writer actually intended; the field exists so a future
format change has a real version to branch on, not to gate this one.

The two smaller pre-existing divergences the collapse also surfaced (only
``ModelEval``'s reader restored ``error``/``task_scores``, only
``EvaluationRunner``'s/``aggregate_reports.py``'s restored
``confidence_score``) are deliberately not reconciled here: the plan calls
for reconciling the ``is_correct``/``example_score`` split as its own
isolated, tested change, not for silently equalizing every small difference
these four readers ever had. They stay explicit keyword arguments a caller
opts into.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from valtron_core.models import EvaluationResult, PredictionResult
from valtron_core.runner import save_run_dir, save_single_model_result

#: Bumped whenever the on-disk run-directory shape or its reconstruction rules
#: change in a way a future reader might need to know about. A run directory
#: with no ``format_version`` key predates this field entirely (version 1);
#: see ``RunDirectoryCodec.read_format_version``.
FORMAT_VERSION = 2

__all__ = ["FORMAT_VERSION", "RunDirectoryCodec", "save_run_dir", "save_single_model_result"]


class RunDirectoryCodec:
    """Reads the run-directory shape ``runner.save_run_dir`` writes."""

    @staticmethod
    def read_json(path: "str | Path") -> dict[str, Any]:
        """Load one JSON file: a run directory's ``metadata.json`` or a ``models/<name>.json``."""
        with open(path) as f:
            data: dict[str, Any] = json.load(f)
        return data

    @staticmethod
    def read_format_version(meta: dict[str, Any]) -> int:
        """The format version a loaded ``metadata.json`` was written under.

        ``1`` when the key is absent: every run directory written before
        ``FORMAT_VERSION`` existed as a field at all.
        """
        return int(meta.get("format_version", 1))

    @staticmethod
    def build_prediction(
        p: dict[str, Any],
        *,
        model_label: str,
        expected_value_fallback: Any = None,
        include_error_and_task_scores: bool = False,
        include_confidence_score: bool = False,
    ) -> PredictionResult:
        """One stored prediction dict -> ``PredictionResult``.

        A missing ``is_correct``/``example_score`` (a prediction saved before
        either field existed, or otherwise written without them) reloads as
        ``None``, meaning "not scored", regardless of the run directory's
        ``format_version``; see this module's own docstring for why that
        holds for version-1 files too, not just new ones.

        ``include_error_and_task_scores``/``include_confidence_score`` are two
        smaller, pre-existing divergences between the four original readers
        (only ``ModelEval`` restored the first pair, only ``EvaluationRunner``
        and ``aggregate_reports.py`` restored the second), preserved verbatim
        as an explicit choice per call site rather than silently equalized.
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
            "is_correct": p.get("is_correct"),
            "example_score": p.get("example_score"),
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
