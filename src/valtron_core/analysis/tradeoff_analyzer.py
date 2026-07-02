"""Public entry point for cost/accuracy tradeoff analysis.

The full three-stage workflow::

    from valtron_core.training import TransformerClassifier
    from valtron_core.evaluation import ModelEval
    from valtron_core.analysis import TradeoffAnalyzer

    clf = TransformerClassifier(output_dir="./model")
    clf.train(documents, labels)

    eval = ModelEval(config={...}, data="eval.json")
    eval.run()

    TradeoffAnalyzer.from_model_eval(eval).run("report.html")
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from valtron_core.analysis._analyzer import (
    LLMSpec,
    TradeoffRow,
    select_cascade_tier_order,
    sweep_multitier_cascade,
    sweep_thresholds,
)
from valtron_core.analysis._report import (
    _estimate_cost_per_call,
    _estimate_transformer_cost_per_call,
    _jinja_env,
    compute_tradeoff_sweep
)

if TYPE_CHECKING:
    from valtron_core.evaluation.model_eval import ModelEval

_DEFAULT_INSTANCE_HOURLY: float = 0.085
_DEFAULT_SAMPLES_PER_SECOND: float = 8.0


def _sweep_to_json_dict(sweep: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of sweep with all values JSON-serializable.

    LLMSpec dataclass objects in llm_specs are converted via dataclasses.asdict().
    Everything else in the sweep dict is already a primitive type.
    """
    result = dict(sweep)
    if "llm_specs" in result:
        result["llm_specs"] = [
            dataclasses.asdict(s) if dataclasses.is_dataclass(s) else s
            for s in result["llm_specs"]
        ]
    return result


class TradeoffAnalyzer:
    """Analyze cost/accuracy tradeoffs for routing between a local transformer
    and one or more LLMs. Finds Pareto-optimal confidence thresholds.

    Instantiate via one of two factory methods::

        # Primary path: reuse a completed ModelEval run -- no re-evaluation
        analyzer = TradeoffAnalyzer.from_model_eval(model_eval)

        # Standalone path: provide data + transformer path directly
        analyzer = TradeoffAnalyzer.from_data(
            data="eval.csv",
            transformer_path="./model/final_model",
            llm_specs=["gpt-4o-mini", "gpt-4o"],
        )

    Then call run() to produce an interactive HTML report::

        analyzer.run("report.html")

    Or use the ModelEval-style step-by-step API to save in multiple formats::

        analyzer.analyze()
        analyzer.save_html_report("report.html")
        analyzer.save_json_report("report.json")
    """

    def __init__(
        self,
        _precomputed: dict[str, Any] | None = None,
        _render_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._precomputed = _precomputed
        self._render_kwargs = _render_kwargs
        self._sweep: dict[str, Any] | None = None

    @classmethod
    def from_model_eval(
        cls,
        model_eval: "ModelEval",
        transformer_instance_hourly: float = _DEFAULT_INSTANCE_HOURLY,
        transformer_samples_per_second: float = _DEFAULT_SAMPLES_PER_SECOND,
    ) -> "TradeoffAnalyzer":
        """Build a TradeoffAnalyzer from a completed ModelEval run.

        Reuses the predictions already stored in model_eval.results -- no
        re-evaluation of the transformer or LLMs.

        Args:
            model_eval: A ModelEval instance after evaluate() or run() has been called.
                Must contain exactly one TransformerModelConfig and at least one
                LLMModelConfig.
            transformer_instance_hourly: Hourly instance cost in USD used to compute
                transformer cost-per-call when cost_rate is not set on the transformer
                config (default: $0.085, c5.large spot price).
            transformer_samples_per_second: Throughput used for cost-per-call math
                when cost_rate is not set on the transformer config
                (default: 8.0, typical DistilBERT throughput). When cost_rate is
                configured, cost-per-call is derived from the actual per-prediction
                response times recorded during evaluation.
        """
        from valtron_core.evaluation.config import LLMModelConfig, TransformerModelConfig
        from valtron_core.models import EvaluationResult, PredictionResult

        # --- Separate results by model type ---
        transformer_configs = [m for m in model_eval.config.models if isinstance(m, TransformerModelConfig)]
        llm_labels: set[str] = {
            m.label for m in model_eval.config.models if isinstance(m, LLMModelConfig)
        }
        transformer_labels: set[str] = {m.label for m in transformer_configs}

        transformer_results: list[EvaluationResult] = [
            r for r in model_eval.results if r.model in transformer_labels
        ]
        llm_results: list[EvaluationResult] = [
            r for r in model_eval.results if r.model in llm_labels
        ]

        if len(transformer_results) != 1:
            raise ValueError(
                f"from_model_eval() requires exactly one transformer model in the experiment; "
                f"found {len(transformer_results)}."
            )
        if not llm_results:
            raise ValueError(
                "from_model_eval() requires at least one LLM model in the experiment."
            )

        # --- Transformer predictions -> TradeoffRows + label detection ---
        transformer_config: TransformerModelConfig = transformer_configs[0]
        predictions: list[PredictionResult] = transformer_results[0].predictions

        if any(p.confidence_score is None for p in predictions):
            raise ValueError(
                "Transformer predictions are missing confidence scores. Re-run the "
                "ModelEval with the current version of valtron_core to populate them."
            )

        tradeoff_rows = [
            TradeoffRow(
                pred_label=p.predicted_value,
                confidence=p.confidence_score,  # type: ignore[arg-type]
                ground_truth=p.expected_value,
            )
            for p in predictions
        ]

        label_counts: dict[str, int] = {}
        for row in tradeoff_rows:
            label_counts[row.ground_truth] = label_counts.get(row.ground_truth, 0) + 1
        unique_labels = list(label_counts.keys())
        if len(unique_labels) != 2:
            raise ValueError(
                f"TradeoffAnalyzer supports binary classification only; "
                f"found {len(unique_labels)} labels: {unique_labels}"
            )
        pos_label = min(label_counts, key=lambda k: label_counts[k])
        neg_label = next(lbl for lbl in unique_labels if lbl != pos_label)

        # --- Transformer cost: use cost_rate from config if available, else estimate ---
        if transformer_config.cost_rate is not None and predictions:
            transformer_cost_per_call = sum(p.llm_cost for p in predictions) / len(predictions)
            effective_instance_hourly = transformer_config.cost_rate
        else:
            transformer_cost_per_call = _estimate_transformer_cost_per_call(
                instance_hourly_cost=transformer_instance_hourly,
                samples_per_second=transformer_samples_per_second,
            )
            effective_instance_hourly = transformer_instance_hourly

        # --- LLM specs: align per-example predictions with transformer row order ---
        doc_id_to_idx: dict[str, int] = {p.document_id: i for i, p in enumerate(predictions)}
        n_total = len(tradeoff_rows)

        llm_specs: list[LLMSpec] = []
        for llm_result in llm_results:
            per_example: list[str | None] = [None] * n_total
            for pred in llm_result.predictions:
                idx = doc_id_to_idx.get(pred.document_id)
                if idx is not None:
                    per_example[idx] = pred.predicted_value

            n_correct = sum(
                1 for p in llm_result.predictions if p.predicted_value == p.expected_value
            )
            llm_specs.append(LLMSpec(
                name=llm_result.model,
                cost_per_call=_estimate_cost_per_call(llm_result.model),
                accuracy=n_correct / n_total if n_total > 0 else 1.0,
                predictions=per_example,
            ))

        precomputed = {
            "tradeoff_rows": tradeoff_rows,
            "pos_label": pos_label,
            "neg_label": neg_label,
            "llm_specs": llm_specs,
            "transformer_cost_per_call": transformer_cost_per_call,
            "transformer_instance_hourly": effective_instance_hourly,
            "transformer_samples_per_second": transformer_samples_per_second,
        }
        return cls(_precomputed=precomputed)

    @classmethod
    def from_data(
        cls,
        data: list[dict[str, Any]] | Path | str,
        transformer_path: Path | str,
        llm_specs: list[str] | None = None,
        transformer_instance_hourly: float = _DEFAULT_INSTANCE_HOURLY,
        transformer_samples_per_second: float = _DEFAULT_SAMPLES_PER_SECOND,
    ) -> "TradeoffAnalyzer":
        """Build a TradeoffAnalyzer from raw data and a transformer path.

        Args:
            data: CSV path or list of {text, label} dicts.
            transformer_path: Path to a trained TransformerClassifier model dir.
            llm_specs: LLM model names to use as deferral targets (e.g.
                ["gpt-4o-mini", "gpt-4o"]). Defaults to ["gpt-4o"].
            transformer_instance_hourly: Hourly instance cost in USD.
            transformer_samples_per_second: Throughput for cost-per-call math.
        """
        render_kwargs: dict[str, Any] = {
            "data": data,
            "frontier_model": llm_specs or ["gpt-4o"],
            "transformer_path": transformer_path,
            "transformer_instance_hourly": transformer_instance_hourly,
            "transformer_samples_per_second": transformer_samples_per_second,
        }
        return cls(_render_kwargs=render_kwargs)

    def analyze(self) -> None:
        """Compute the tradeoff sweep and store results in memory.

        Must be called before save_html_report() or save_json_report().
        Equivalent to ModelEval.evaluate() -- separates computation from saving.
        """
        asyncio.run(self.aanalyze())

    async def aanalyze(self) -> None:
        """Async variant of analyze()."""
        if self._render_kwargs is not None:
            self._sweep = compute_tradeoff_sweep(**self._render_kwargs)
            return

        assert self._precomputed is not None
        pc = self._precomputed
        tradeoff_rows: list[TradeoffRow] = pc["tradeoff_rows"]
        llm_specs: list[LLMSpec] = pc["llm_specs"]
        pos_label: str = pc["pos_label"]
        neg_label: str = pc["neg_label"]
        transformer_cost_per_call: float = pc["transformer_cost_per_call"]

        n_steps = 10
        sweep = sweep_thresholds(
            tradeoff_rows,
            positive_label=pos_label,
            negative_label=neg_label,
            n_steps=n_steps,
            llm_specs=llm_specs,
            cost_per_transformer_call=transformer_cost_per_call,
        )
        sweep["positive_label"] = pos_label
        sweep["negative_label"] = neg_label
        sweep["transformer_cost_per_call"] = transformer_cost_per_call
        sweep["transformer_instance_hourly"] = pc["transformer_instance_hourly"]
        sweep["transformer_samples_per_second"] = pc["transformer_samples_per_second"]
        sweep["measured_samples_per_second"] = None
        sweep["used_measured_throughput"] = False
        sweep["default_run_size"] = sweep["n_total"]
        sweep["used_full_llm_eval"] = True
        sweep["llm_measurements"] = {}
        sweep["llm_specs"] = [dataclasses.asdict(s) for s in llm_specs]

        tier_specs, _ = select_cascade_tier_order(llm_specs, have_measurements=True)
        if len(tier_specs) >= 2:
            cascade = sweep_multitier_cascade(
                tradeoff_rows,
                positive_label=pos_label,
                negative_label=neg_label,
                tier_llm_specs=tier_specs,
                n_quantiles=n_steps,
                cost_per_transformer_call=transformer_cost_per_call,
            )
            sweep["cascade"] = cascade
        else:
            sweep["cascade"] = {
                "cells": [],
                "tier_names": ["transformer"] + [s.name for s in tier_specs],
                "tier_specs": [
                    {"name": s.name, "cost_per_call": s.cost_per_call, "accuracy": s.accuracy}
                    for s in tier_specs
                ],
                "pareto_indices": {m: [] for m in ("f1", "precision", "recall", "accuracy")},
                "skipped_reason": "fewer than 2 LLM tiers after selection",
            }

        self._sweep = sweep

    def save_html_report(self, output_path: Path | str) -> Path:
        """Write the tradeoff HTML report from in-memory sweep results.

        Must call analyze() first.

        Args:
            output_path: Path where the HTML file will be written.

        Returns:
            Resolved path to the written HTML file.
        """
        if self._sweep is None:
            raise RuntimeError("Call analyze() before save_html_report().")
        output_path = Path(output_path).resolve()
        template = _jinja_env.get_template("tradeoff_report.jinja2.html")
        html = template.render(data=self._sweep)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(html, encoding="utf-8")
        return output_path

    def save_json_report(self, output_path: Path | str) -> Path:
        """Write the full sweep results as JSON.

        Must call analyze() first. The JSON contains all cells, Pareto indices,
        baselines, and metadata needed for a custom UI to render equivalent visuals.

        Args:
            output_path: Path where the JSON file will be written.

        Returns:
            Resolved path to the written JSON file.
        """
        if self._sweep is None:
            raise RuntimeError("Call analyze() before save_json_report().")
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(_sweep_to_json_dict(self._sweep), indent=2),
            encoding="utf-8",
        )
        return output_path

    def run(self, output_path: Path | str) -> Path:
        """Analyze and write the tradeoff HTML report in one call.

        Args:
            output_path: Path where the HTML file will be written.

        Returns:
            Resolved path to the written HTML file.
        """
        self.analyze()
        return self.save_html_report(output_path)

    async def arun(self, output_path: Path | str) -> Path:
        """Async variant of run()."""
        await self.aanalyze()
        return self.save_html_report(output_path)
