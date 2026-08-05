"""Render the transformer-vs-frontier tradeoff sweep as a self-contained HTML.

End-to-end wrapper that takes a labelled dataset (CSV path or list of dicts),
produces transformer predictions, runs the threshold sweep, and writes a
portable HTML file with an interactive Pareto-frontier chart.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

import litellm
from jinja2 import Environment, FileSystemLoader

from valtron_core.analysis._measurer import (
    DEFAULT_PROMPT_TEMPLATE,
    MeasurementResult,
    measure_llm_accuracies,
)
from valtron_core.analysis._analyzer import (
    LLMSpec,
    TradeoffRow,
    select_cascade_tier_order,
    sweep_multitier_cascade,
    sweep_thresholds,
)

TEMPLATES_DIR = Path(__file__).parent.parent / "templates"
_jinja_env = Environment(loader=FileSystemLoader(TEMPLATES_DIR))


def _estimate_cost_per_call(
    model_name: str,
    *,
    prompt_tokens: int = 500,
    completion_tokens: int = 2,
) -> float:
    """Estimate USD per single LLM call from litellm's pricing data.

    Defaults assume a medium-length classification prompt: ~500 prompt tokens
    (a few hundred-word context + a sentence of instructions) and ~2 completion
    tokens (single-word yes/no style answer). Override either when your prompt
    differs significantly — the cost axis on the report scales linearly.

    Falls back to a small non-zero default when the model isn't in litellm's
    cost table — keeps the chart's cost axis non-degenerate so the user can
    still eyeball relative tradeoffs while flagging that a real cost lookup
    failed.
    """
    try:
        cost_data = litellm.model_cost.get(model_name, {})
        input_rate = float(cost_data.get("input_cost_per_token", 0.0))
        output_rate = float(cost_data.get("output_cost_per_token", 0.0))
        per_call = prompt_tokens * input_rate + completion_tokens * output_rate
        return per_call if per_call > 0 else 0.0001
    except Exception:
        return 0.0001


# Defaults model a DistilBERT-base classifier on a c5.large CPU instance
# (us-east-1, on-demand) with vanilla PyTorch on medium-length text (~400
# tokens, typical of paper abstracts / long-form classification). Pure-CPU
# vanilla-PyTorch throughput on that hardware lands around ~8 samples/sec —
# faster setups (ONNX, INT8 quantization, GPU batch jobs) can be ~5-25× this,
# but require optimization the user has to opt into. Override either knob via
# the render_tradeoff_report() args or the example's CLI flags to model
# different deployments (e.g. g4dn.xlarge GPU @ ~200 samples/sec).
_DEFAULT_INSTANCE_HOURLY_COST = 0.085
_DEFAULT_TRANSFORMER_THROUGHPUT = 8.0  # samples per second


def _estimate_transformer_cost_per_call(
    instance_hourly_cost: float = _DEFAULT_INSTANCE_HOURLY_COST,
    samples_per_second: float = _DEFAULT_TRANSFORMER_THROUGHPUT,
) -> float:
    """Per-sample USD cost of running the transformer locally.

    Pure-Python amortisation: ``hourly_cost / (samples_per_second * 3600)``.
    Defaults to a c5.large instance running DistilBERT-base on short text;
    override either argument when modelling a different deployment.
    """
    if samples_per_second <= 0:
        return 0.0
    return instance_hourly_cost / (samples_per_second * 3600.0)


def _load_csv(path: Path) -> list[dict]:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows or "text" not in rows[0] or "label" not in rows[0]:
        raise ValueError(f"CSV at {path} must have 'text' and 'label' columns.")
    return rows


def _transformer_cache_key(transformer_path: str | Path, rows: list[dict]) -> str:
    """Hash the model weights' size+mtime + the dataset content. Changes to
    either invalidate the cache cleanly."""
    h = hashlib.sha256()
    p = Path(transformer_path).resolve()
    h.update(str(p).encode())
    h.update(b"\0")
    weights = p / "model.safetensors"
    if weights.exists():
        st = weights.stat()
        h.update(f"{st.st_size}:{int(st.st_mtime)}".encode())
    h.update(b"\0")
    for r in rows:
        h.update(r.get("text", "").encode())
        h.update(b"\x1f")
        h.update(r.get("label", "").encode())
        h.update(b"\n")
    return h.hexdigest()[:16]


def _predict_with_real_transformer(
    rows: list[dict],
    transformer_path: str | Path,
    cache_dir: str | Path = "examples/results",
) -> tuple[list[TradeoffRow], float]:
    """Run a trained TransformerClassifier and shape outputs for the analyzer.

    Cached to ``cache_dir/transformer_predictions_{hash}.json`` so re-renders
    (e.g. after a chart-side code change) skip the multi-minute inference.
    Cache key includes the model weights' size+mtime and the dataset's content
    hash, so either changing invalidates cleanly.

    Returns:
        (tradeoff_rows, measured_samples_per_second). When loaded from cache,
        ``measured_samples_per_second`` is the value from the original run.
    """
    from valtron_core.training.transformer_classifier import TransformerClassifier

    cache_dir = Path(cache_dir)
    key = _transformer_cache_key(transformer_path, rows)
    cache_path = cache_dir / f"transformer_predictions_{key}.json"

    if cache_path.exists():
        print(f"  Loaded transformer prediction cache: {cache_path}")
        cached = json.loads(cache_path.read_text())
        tradeoff_rows = [
            TradeoffRow(
                pred_label=p["pred_label"],
                confidence=float(p["confidence"]),
                ground_truth=row["label"],
            )
            for p, row in zip(cached["predictions"], rows)
        ]
        return tradeoff_rows, float(cached.get("samples_per_second", 0.0))

    classifier = TransformerClassifier(output_dir=transformer_path)
    classifier.load_model(transformer_path)

    texts = [r["text"] for r in rows]
    start = time.perf_counter()
    preds = classifier.predict_with_scores(texts)
    elapsed = time.perf_counter() - start
    measured_sps = len(texts) / elapsed if elapsed > 0 else 0.0

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({
        "model_path": str(Path(transformer_path).resolve()),
        "n_total": len(texts),
        "elapsed_sec": elapsed,
        "samples_per_second": measured_sps,
        "predictions": [
            {"pred_label": pl, "confidence": float(s[pl])}
            for pl, s in preds
        ],
    }))
    print(f"  Wrote transformer prediction cache: {cache_path}")

    tradeoff_rows = [
        TradeoffRow(
            pred_label=pred_label,
            confidence=scores[pred_label],
            ground_truth=row["label"],
        )
        for (pred_label, scores), row in zip(preds, rows)
    ]
    return tradeoff_rows, measured_sps


def compute_tradeoff_sweep(
    *,
    data: str | Path | Iterable[dict],
    frontier_model: str | list[str] = "gpt-4o-mini",
    llm_accuracies: dict[str, float] | None = None,
    transformer_path: str | Path | None = None,
    n_steps: int = 10,
    cost_per_call: float | None = None,
    transformer_instance_hourly: float = _DEFAULT_INSTANCE_HOURLY_COST,
    transformer_samples_per_second: float = _DEFAULT_TRANSFORMER_THROUGHPUT,
    transformer_cost_per_call: float | None = None,
    use_measured_throughput: bool = False,
    default_run_size: int | None = None,
    positive_label: str | None = None,
    measure_llm_samples: int = 0,
    measure_llm_full: bool = False,
    llm_prompt_template: str | None = None,
) -> dict[str, Any]:
    """Run the sweep and return the full results dict.

    Args:
        data: CSV path or list of {text, label} dicts.
        frontier_model: Model identifier passed to litellm for cost imputation.
            The model itself is never called; labels in the dataset are treated
            as the frontier-model output.
        transformer_path: Path to a trained TransformerClassifier model dir.
        n_steps: Per-axis threshold-grid resolution.
        cost_per_call: Override the litellm cost lookup with a fixed value.

    Returns:
        Sweep dict containing all metrics, cells, Pareto indices, and metadata.
        Pass to save_html_report() or save_json_report() to persist results.
    """
    if isinstance(data, (str, Path)):
        rows = _load_csv(Path(data))
    else:
        rows = list(data)
        if not rows or "text" not in rows[0] or "label" not in rows[0]:
            raise ValueError("Each row must have 'text' and 'label' keys.")

    label_counts: dict[str, int] = {}
    for r in rows:
        label_counts[r["label"]] = label_counts.get(r["label"], 0) + 1
    unique_labels = list(label_counts.keys())
    if len(unique_labels) != 2:
        raise ValueError(
            f"Binary classification only — found {len(unique_labels)} labels: {unique_labels}"
        )

    if positive_label is not None:
        if positive_label not in label_counts:
            raise ValueError(
                f"--positive-label '{positive_label}' not in dataset labels {unique_labels}"
            )
        pos_label = positive_label
    else:
        # Default: minority class is "positive". Correct for nearly all imbalanced
        # binary classification (the rare event is what you want to detect).
        pos_label = min(label_counts, key=label_counts.get)
    neg_label = next(label for label in unique_labels if label != pos_label)

    out_dir = Path(output_path).resolve().parent

    if not transformer_path:
        raise ValueError("transformer_path is required.")
    measured_sps: float | None = None
    tradeoff_rows, measured_sps = _predict_with_real_transformer(
        rows, transformer_path, cache_dir=out_dir,
    )

    if isinstance(frontier_model, str):
        llm_names = [frontier_model]
    else:
        llm_names = list(frontier_model)
    if not llm_names:
        raise ValueError("At least one frontier_model is required.")

    llm_accuracies = llm_accuracies or {}

    measurements: dict[str, MeasurementResult] = {}
    # Guard: --measure-llm-samples N where N >= dataset size auto-promotes to
    # full-eval / deterministic math. Otherwise we'd silently call the LLM on
    # every example (since stratified sampling caps at dataset size) but still
    # use Bernoulli math instead of the per-example predictions we just bought.
    auto_full = (not measure_llm_full) and measure_llm_samples >= len(rows) > 0
    if auto_full:
        print(f"\n[note] --measure-llm-samples {measure_llm_samples} >= dataset size "
              f"({len(rows)}); upgrading to deterministic math (equivalent to "
              f"--measure-llm-full).")
    use_full = measure_llm_full or auto_full
    effective_samples = len(rows) if use_full else measure_llm_samples
    if effective_samples > 0:
        prompt = llm_prompt_template if llm_prompt_template is not None else DEFAULT_PROMPT_TEMPLATE
        mode_desc = (
            f"all {len(rows)} examples (deterministic cell math)"
            if use_full
            else f"{measure_llm_samples} stratified samples"
        )
        print(f"\nMeasuring LLM accuracies on {mode_desc}...")
        measurements = measure_llm_accuracies(
            rows=rows,
            llm_names=llm_names,
            n_samples=effective_samples,
            prompt_template=prompt,
            cache_dir=out_dir,
        )
        for name, m in measurements.items():
            if m.has_data:
                print(f"  {name}: measured accuracy = {m.accuracy:.4f} "
                      f"(n={m.n_samples}, 95% CI ±{m.ci_half_width:.4f})")

    def _predictions_by_row(meas: MeasurementResult, n: int) -> list[str | None]:
        out: list[str | None] = [None] * n
        for record in meas.predictions:
            idx = int(record["sample_idx"])
            if 0 <= idx < n:
                out[idx] = record["prediction"]
        return out

    llm_specs: list[LLMSpec] = []
    for name in llm_names:
        per_call = (
            cost_per_call if (cost_per_call is not None and name == llm_names[0])
            else _estimate_cost_per_call(name)
        )
        # Precedence: measured > user-provided > 1.0 default
        if name in measurements and measurements[name].has_data:
            acc = measurements[name].accuracy
        else:
            acc = llm_accuracies.get(name, 1.0)
        # In full-eval mode, attach per-example predictions so the analyzer
        # uses deterministic per-cell math instead of Bernoulli with the rate.
        preds = (
            _predictions_by_row(measurements[name], len(rows))
            if use_full and name in measurements and measurements[name].has_data
            else None
        )
        llm_specs.append(LLMSpec(
            name=name,
            cost_per_call=per_call,
            accuracy=acc,
            predictions=preds,
        ))

    if transformer_cost_per_call is None:
        sps_for_cost = (
            measured_sps if (use_measured_throughput and measured_sps is not None and measured_sps > 0)
            else transformer_samples_per_second
        )
        transformer_cost_per_call = _estimate_transformer_cost_per_call(
            instance_hourly_cost=transformer_instance_hourly,
            samples_per_second=sps_for_cost,
        )

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
    sweep["transformer_instance_hourly"] = transformer_instance_hourly
    sweep["transformer_samples_per_second"] = transformer_samples_per_second
    sweep["measured_samples_per_second"] = measured_sps
    sweep["used_measured_throughput"] = bool(
        use_measured_throughput and measured_sps is not None and measured_sps > 0
    )
    sweep["default_run_size"] = default_run_size if default_run_size is not None else sweep["n_total"]
    sweep["used_full_llm_eval"] = use_full
    sweep["llm_measurements"] = {
        name: {
            "n_samples": m.n_samples,
            "n_correct": m.n_correct,
            "accuracy": m.accuracy,
            "ci_half_width": m.ci_half_width,
        }
        for name, m in measurements.items()
        if m.has_data
    }

    # Multi-tier cascade: filter the provided LLMs to a sensible tier ordering,
    # then sweep all monotone cut configurations.
    have_measurements = any(m.has_data for m in measurements.values())
    tier_specs, drop_log = select_cascade_tier_order(llm_specs, have_measurements=have_measurements)
    if drop_log:
        print(f"\nMulti-tier cascade tier selection (kept: {[s.name for s in tier_specs]}):")
        for name, reason in drop_log:
            print(f"  dropped {name}: {reason}")
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

    import dataclasses
    sweep["llm_specs"] = [dataclasses.asdict(s) if dataclasses.is_dataclass(s) else s for s in llm_specs]
    return sweep


def render_tradeoff_report(
    *,
    output_path: str | Path,
    **kwargs: Any,
) -> Path:
    """Compute the sweep and write a self-contained HTML report.

    All keyword arguments are forwarded to ``compute_tradeoff_sweep()``.
    Returns the resolved path to the written HTML file.
    """
    sweep = compute_tradeoff_sweep(**kwargs)
    template = _jinja_env.get_template("tradeoff_report.jinja2.html")
    html = template.render(data=sweep)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html)
    return output_path
