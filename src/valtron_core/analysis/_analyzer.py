"""Threshold-sweep analyzer for transformer-vs-frontier tradeoff visualizations.

Given per-example transformer predictions with confidence scores plus ground-truth
labels (binary classification), sweeps over (positive_threshold, negative_threshold)
pairs and computes f1/precision/recall/accuracy + transformer coverage + estimated
frontier cost at each grid cell. The shape mirrors
``calc_stats_given_thresholds`` in genomenon_analysis but drops the
domain-specific (pmid, gene) short-circuit aggregation — every example is scored
independently here.

Above-threshold examples keep the transformer's label; below-threshold examples
defer to the ground-truth label (which stands in for the frontier model, since
labels are treated as authoritative — no LLM call is actually made during a sweep).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class TradeoffRow:
    """One per-example record fed into the sweep."""
    pred_label: str
    confidence: float
    ground_truth: str


@dataclass
class LLMSpec:
    """Configuration for one LLM used as the deferral target.

    ``accuracy`` is the per-example probability of the LLM producing the correct
    label on the deferred slice. We treat each deferred call as an independent
    draw against this rate. This is the Bernoulli math used when we only have
    a sampled accuracy estimate (the assumed/guessed accuracy or a measurement
    on a stratified subset).

    ``predictions`` (when set) is the LLM's actual predicted label for every
    example in the eval set, in the same row order as the rest of the analyzer.
    Entries may be ``None`` for failed API calls. When present, the analyzer
    uses these deterministically for the cell math — exact per-example
    confusion on the deferred slice, no Bernoulli approximation. The assumed
    accuracy is only used as a fallback for the small fraction of None entries.

    The Bernoulli path implicitly assumes uniform LLM accuracy across all
    examples; the deferred slice is the *hard* subset (low transformer
    confidence) where real LLM accuracy is usually lower than the global rate,
    making Bernoulli optimistic. The deterministic path closes that gap.
    """
    name: str
    cost_per_call: float = 0.0
    accuracy: float = 1.0
    predictions: list[str | None] | None = None


def _quantile_thresholds(values: Sequence[float], n_steps: int) -> list[float]:
    """Pick ``n_steps`` thresholds spread across the empirical confidence range.

    Equal-frequency split so each step roughly moves the same number of examples
    across the threshold (matches the genomenon approach via np.array_split).
    Returns ascending thresholds.
    """
    if not values:
        return [0.0]
    arr = np.sort(np.asarray(values))
    chunks = np.array_split(arr, n_steps)
    return [float(c[0]) for c in chunks if len(c) > 0]


_METRIC_KEYS = ("f1", "precision", "recall", "accuracy")
MAX_LLM_TIERS_IN_CASCADE = 3  # cap; with transformer this is 4 tiers total


def select_cascade_tier_order(
    llm_specs: list[LLMSpec],
    have_measurements: bool,
) -> tuple[list[LLMSpec], list[tuple[str, str]]]:
    """Decide the order of LLM tiers in the multi-tier cascade.

    Tiers are always sorted cost-ascending — the cascade architecture requires
    cheapest LLM first (so low-confidence escalates through cheap to expensive).

    With measurements: also drop any LLM dominated by a cheaper-and-more-accurate
    peer (cost-accuracy Pareto filter) before sorting.

    Caps at MAX_LLM_TIERS_IN_CASCADE entries — if more remain after filtering,
    the cheapest are kept (most useful for cost-saving cascades).

    Returns:
        (selected_tier_specs, drop_log)
        drop_log is a list of ``(llm_name, reason)`` describing what was excluded.
    """
    drop_log: list[tuple[str, str]] = []
    candidates: list[LLMSpec]

    if have_measurements:
        # Pareto filter: keep an LLM iff no peer has both lower-or-equal cost
        # and higher-or-equal accuracy (with at least one strict).
        candidates = []
        for s in llm_specs:
            dominated_by = None
            for other in llm_specs:
                if other is s:
                    continue
                if (
                    other.cost_per_call <= s.cost_per_call
                    and other.accuracy >= s.accuracy
                    and (other.cost_per_call < s.cost_per_call or other.accuracy > s.accuracy)
                ):
                    dominated_by = other
                    break
            if dominated_by is None:
                candidates.append(s)
            else:
                drop_log.append((
                    s.name,
                    f"dominated by {dominated_by.name} "
                    f"(${dominated_by.cost_per_call:.6g}/call @ acc {dominated_by.accuracy:.3f})",
                ))
    else:
        candidates = list(llm_specs)

    candidates.sort(key=lambda s: s.cost_per_call)

    if len(candidates) > MAX_LLM_TIERS_IN_CASCADE:
        for s in candidates[MAX_LLM_TIERS_IN_CASCADE:]:
            drop_log.append((s.name, f"capped — only top {MAX_LLM_TIERS_IN_CASCADE} cheapest kept"))
        candidates = candidates[:MAX_LLM_TIERS_IN_CASCADE]

    return candidates, drop_log


def _multitier_metrics(
    pred_is_pos: np.ndarray,
    pred_is_neg: np.ndarray,
    ground_is_pos: np.ndarray,
    ground_is_neg: np.ndarray,
    tier_assignment: np.ndarray,
    llm_specs: list[LLMSpec],
    positive_label: str,
    negative_label: str,
) -> dict[str, float]:
    """Confusion metrics for a multi-tier cascade configuration.

    Tier 0 = transformer (deterministic from its own predictions).
    Tier k > 0 = ``llm_specs[k-1]``: deterministic from that LLM's actual
    per-example predictions when ``predictions`` is set on the spec, otherwise
    Bernoulli with the spec's accuracy. Handled uniformly via
    ``_llm_confusion_for_subset``.
    """
    t0 = tier_assignment == 0
    tp = float(np.sum(t0 & pred_is_pos & ground_is_pos))
    fp = float(np.sum(t0 & pred_is_pos & ground_is_neg))
    fn = float(np.sum(t0 & pred_is_neg & ground_is_pos))
    tn = float(np.sum(t0 & pred_is_neg & ground_is_neg))

    for k, spec in enumerate(llm_specs, start=1):
        tk = tier_assignment == k
        d_tp, d_fp, d_fn, d_tn = _llm_confusion_for_subset(
            spec, tk, ground_is_pos, ground_is_neg, positive_label, negative_label,
        )
        tp += d_tp
        fp += d_fp
        fn += d_fn
        tn += d_tn

    return _metrics_from_confusion(tp, fp, fn, tn)


def sweep_multitier_cascade(
    rows: list[TradeoffRow],
    positive_label: str,
    negative_label: str,
    tier_llm_specs: list[LLMSpec],
    *,
    n_quantiles: int = 10,
    cost_per_transformer_call: float = 0.0,
) -> dict:
    """Sweep over monotone N-tier cascade configurations.

    For each combination of ``len(tier_llm_specs)`` cuts on the transformer's
    confidence axis (one combination per pos prediction, one per neg, with cut
    values drawn from the empirical quantiles), partition examples into tiers
    (tier 0 = transformer for highest-confidence, tier N = most expensive LLM
    for lowest-confidence) and compute the cell's metrics + cost.

    Returns:
        Dict with:
          - ``cells``: list of cell dicts with pos_cuts/neg_cuts,
            tier_counts, total_cost, n_active_llm_tiers, and per-metric values.
          - ``tier_names``: ``["transformer", llm_specs[0].name, ...]`` in tier order.
          - ``pareto_indices``: per-metric Pareto frontier (same shape as 2-tier).
    """
    if len(tier_llm_specs) < 2:
        return {
            "cells": [],
            "tier_names": ["transformer"] + [s.name for s in tier_llm_specs],
            "tier_specs": [
                {"name": s.name, "cost_per_call": s.cost_per_call, "accuracy": s.accuracy}
                for s in tier_llm_specs
            ],
            "pareto_indices": {m: [] for m in _METRIC_KEYS},
            "skipped_reason": "need at least 2 LLM tiers for a multi-tier cascade",
        }

    pred_labels = np.array([r.pred_label for r in rows])
    confidences = np.array([r.confidence for r in rows])
    ground = np.array([r.ground_truth for r in rows])

    pred_is_pos = pred_labels == positive_label
    pred_is_neg = pred_labels == negative_label
    ground_is_pos = ground == positive_label
    ground_is_neg = ground == negative_label

    n_total = len(rows)
    n_llm_tiers = len(tier_llm_specs)
    n_cuts = n_llm_tiers - 1
    transformer_cost_total = n_total * cost_per_transformer_call

    pos_threshold_pool = _quantile_thresholds(confidences[pred_is_pos].tolist(), n_quantiles)
    neg_threshold_pool = _quantile_thresholds(confidences[pred_is_neg].tolist(), n_quantiles)

    # combinations_with_replacement (not plain combinations) so cuts can be
    # equal, which collapses an intermediate tier to zero examples. Without
    # this the cascade is forced to ALWAYS use every middle tier, which
    # makes it strictly worse than the per-LLM 2-tier sweeps wherever the
    # optimum is "skip the cheap LLM and go straight to a better one". With
    # replacement, the cascade space is a true superset of every per-LLM
    # 2-tier space.
    from itertools import combinations_with_replacement
    pos_cut_combos = list(combinations_with_replacement(pos_threshold_pool, n_cuts)) or [tuple()]
    neg_cut_combos = list(combinations_with_replacement(neg_threshold_pool, n_cuts)) or [tuple()]

    cells: list[dict] = []
    for pos_cuts in pos_cut_combos:
        pos_cuts_arr = np.asarray(pos_cuts)
        pos_tiers = n_cuts - np.searchsorted(pos_cuts_arr, confidences, side="right")
        for neg_cuts in neg_cut_combos:
            neg_cuts_arr = np.asarray(neg_cuts)
            neg_tiers = n_cuts - np.searchsorted(neg_cuts_arr, confidences, side="right")
            tier_assignment = np.where(pred_is_pos, pos_tiers, neg_tiers)

            tier_counts = np.bincount(tier_assignment, minlength=n_llm_tiers + 1).tolist()

            tier_costs: list[float] = [transformer_cost_total]
            for k, spec in enumerate(tier_llm_specs, start=1):
                tier_costs.append(tier_counts[k] * spec.cost_per_call)
            total_cost = sum(tier_costs)

            metrics = _multitier_metrics(
                pred_is_pos, pred_is_neg, ground_is_pos, ground_is_neg,
                tier_assignment, tier_llm_specs,
                positive_label=positive_label, negative_label=negative_label,
            )
            n_active_llm_tiers = sum(1 for k in range(1, n_llm_tiers + 1) if tier_counts[k] > 0)

            cells.append({
                "pos_cuts": [float(c) for c in pos_cuts],
                "neg_cuts": [float(c) for c in neg_cuts],
                "tier_counts": tier_counts,
                "tier_costs": [float(c) for c in tier_costs],
                "transformer_cost": float(transformer_cost_total),
                "cost": float(total_cost),
                "n_active_llm_tiers": n_active_llm_tiers,
                **metrics,
            })

    pareto_indices = {
        metric: _pareto_frontier_indices(cells, x_key="cost", y_key=metric)
        for metric in _METRIC_KEYS
    }

    return {
        "cells": cells,
        "tier_names": ["transformer"] + [s.name for s in tier_llm_specs],
        "tier_specs": [
            {"name": s.name, "cost_per_call": s.cost_per_call, "accuracy": s.accuracy}
            for s in tier_llm_specs
        ],
        "n_total": n_total,
        "pareto_indices": pareto_indices,
    }


def _expected_metrics(
    tp_t: float, fp_t: float, tn_t: float, fn_t: float,
    n_l_pos: float, n_l_neg: float,
    llm_accuracy: float,
) -> dict[str, float]:
    """Compute expected confusion-derived metrics given an LLM accuracy rate.

    Treats each deferred example as an independent Bernoulli draw at the given
    accuracy. Returns the closed-form expectation, which exactly equals the
    average over infinite simulations.
    """
    a = llm_accuracy
    tp = tp_t + a * n_l_pos
    fp = fp_t + (1.0 - a) * n_l_neg
    fn = fn_t + (1.0 - a) * n_l_pos
    tn = tn_t + a * n_l_neg

    return _metrics_from_confusion(tp, fp, fn, tn)


def _metrics_from_confusion(tp: float, fp: float, fn: float, tn: float) -> dict[str, float]:
    n = tp + fp + fn + tn
    accuracy = (tp + tn) / n if n > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    denom = precision + recall
    f1 = 2.0 * precision * recall / denom if denom > 0 else 0.0
    return {"f1": f1, "precision": precision, "recall": recall, "accuracy": accuracy}


def _llm_confusion_for_subset(
    spec: "LLMSpec",
    subset_mask: np.ndarray,
    ground_is_pos: np.ndarray,
    ground_is_neg: np.ndarray,
    positive_label: str,
    negative_label: str,
) -> tuple[float, float, float, float]:
    """LLM's contribution to the confusion matrix for the examples in ``subset_mask``.

    Deterministic when ``spec.predictions`` is set (per-example actual LLM outputs);
    Bernoulli with ``spec.accuracy`` when not. If ``predictions`` is present but
    some entries are None (failed API calls), those entries fall back to the
    Bernoulli rate so the cell math doesn't crash on the rare failure.
    """
    if spec.predictions is None:
        n_pos = float(np.sum(subset_mask & ground_is_pos))
        n_neg = float(np.sum(subset_mask & ground_is_neg))
        a = spec.accuracy
        return a * n_pos, (1 - a) * n_neg, (1 - a) * n_pos, a * n_neg

    pred_arr = np.array(["__missing__" if p is None else p for p in spec.predictions])
    pred_is_pos = pred_arr == positive_label
    pred_is_neg = pred_arr == negative_label
    pred_is_valid = pred_is_pos | pred_is_neg

    valid = subset_mask & pred_is_valid
    tp_det = float(np.sum(valid & pred_is_pos & ground_is_pos))
    fp_det = float(np.sum(valid & pred_is_pos & ground_is_neg))
    fn_det = float(np.sum(valid & pred_is_neg & ground_is_pos))
    tn_det = float(np.sum(valid & pred_is_neg & ground_is_neg))

    missing = subset_mask & (~pred_is_valid)
    n_miss_pos = float(np.sum(missing & ground_is_pos))
    n_miss_neg = float(np.sum(missing & ground_is_neg))
    a = spec.accuracy
    return (
        tp_det + a * n_miss_pos,
        fp_det + (1 - a) * n_miss_neg,
        fn_det + (1 - a) * n_miss_pos,
        tn_det + a * n_miss_neg,
    )


def sweep_thresholds(
    rows: list[TradeoffRow],
    positive_label: str,
    negative_label: str,
    *,
    n_steps: int = 10,
    llm_specs: list[LLMSpec] | None = None,
    cost_per_transformer_call: float = 0.0,
) -> dict:
    """Run a 2D threshold sweep with one or more LLMs and return per-cell metrics.

    Cost model: the transformer runs on every sample, so its contribution is the
    constant ``n_total × cost_per_transformer_call``. The LLM is called only on
    the deferred slice. For each LLM ``X`` with per-call cost ``lc_X`` and
    accuracy ``a_X``, the cell's total cost is
    ``n_total × tc + n_frontier × lc_X`` and the metrics use the expected
    confusion matrix:

        TP = TP_transformer + a × n_LLM_positive
        FP = FP_transformer + (1 - a) × n_LLM_negative
        FN = FN_transformer + (1 - a) × n_LLM_positive
        TN = TN_transformer + a × n_LLM_negative

    The transformer slice's contribution is deterministic (we know what the
    transformer predicted); the LLM slice's contribution is an expectation
    given the user-supplied accuracy.

    Args:
        rows: Per-example (pred_label, confidence, ground_truth) records.
        positive_label: String value for the positive class.
        negative_label: String value for the negative class.
        n_steps: Grid resolution per axis (default 10 → 100 cells).
        llm_specs: One or more LLMs to model. Each contributes its own per-cell
            metrics, baseline, and Pareto sets. Defaults to a single "oracle"
            LLM with cost=0 and accuracy=1 when omitted (back-compat for
            transformer-only views).
        cost_per_transformer_call: USD per transformer inference (always run).

    Returns:
        Dict with:
          - ``cells``: per-cell dicts. Each has shared fields (pos_t, neg_t,
            coverage, n_transformer, n_frontier, transformer_cost) plus a
            ``per_llm`` dict keyed by LLM name with that LLM's metrics +
            llm_cost + total cost.
          - ``baseline_transformer``: same shape as a cell's transformer-only
            view (no LLM contribution).
          - ``baselines_by_llm``: per-LLM 100%-LLM endpoint.
          - ``pareto_indices_by_llm``: dict[llm_name → dict[metric → list[int]]].
          - ``llm_specs``: the list as dicts in the input order.
          - ``primary_llm``: the first LLM's name (UI default-active).
    """
    if not rows:
        raise ValueError("No rows provided.")
    if not llm_specs:
        llm_specs = [LLMSpec(name="LLM", cost_per_call=0.0, accuracy=1.0)]

    pred_labels = np.array([r.pred_label for r in rows])
    confidences = np.array([r.confidence for r in rows])
    ground = np.array([r.ground_truth for r in rows])

    pred_is_pos = pred_labels == positive_label
    pred_is_neg = pred_labels == negative_label
    ground_is_pos = ground == positive_label
    ground_is_neg = ground == negative_label

    pos_thresholds = _quantile_thresholds(confidences[pred_is_pos].tolist(), n_steps)
    neg_thresholds = _quantile_thresholds(confidences[pred_is_neg].tolist(), n_steps)

    n_total = len(rows)
    total_pos = int(ground_is_pos.sum())
    total_neg = int(ground_is_neg.sum())
    transformer_cost_total = n_total * cost_per_transformer_call

    cells: list[dict] = []
    for pos_t in pos_thresholds:
        for neg_t in neg_thresholds:
            keep_transformer = (
                (pred_is_pos & (confidences >= pos_t))
                | (pred_is_neg & (confidences >= neg_t))
            )
            defer = ~keep_transformer

            tp_t = int((keep_transformer & pred_is_pos & ground_is_pos).sum())
            fp_t = int((keep_transformer & pred_is_pos & ground_is_neg).sum())
            fn_t = int((keep_transformer & pred_is_neg & ground_is_pos).sum())
            tn_t = int((keep_transformer & pred_is_neg & ground_is_neg).sum())

            n_l_pos = int((defer & ground_is_pos).sum())
            n_l_neg = int((defer & ground_is_neg).sum())
            n_transformer = int(keep_transformer.sum())
            n_frontier = n_total - n_transformer

            defer_mask = ~keep_transformer
            per_llm: dict[str, dict] = {}
            for spec in llm_specs:
                llm_tp, llm_fp, llm_fn, llm_tn = _llm_confusion_for_subset(
                    spec, defer_mask, ground_is_pos, ground_is_neg, positive_label, negative_label,
                )
                m = _metrics_from_confusion(
                    tp_t + llm_tp, fp_t + llm_fp, fn_t + llm_fn, tn_t + llm_tn,
                )
                llm_cost = n_frontier * spec.cost_per_call
                per_llm[spec.name] = {
                    **m,
                    "llm_cost": llm_cost,
                    "cost": transformer_cost_total + llm_cost,
                }

            cells.append({
                "pos_t": float(pos_t),
                "neg_t": float(neg_t),
                "coverage": n_transformer / n_total,
                "n_transformer": n_transformer,
                "n_frontier": n_frontier,
                "transformer_cost": transformer_cost_total,
                "per_llm": per_llm,
            })

    # transformer-only baseline (no LLM contribution, identical across LLMs)
    tp_full = int((pred_is_pos & ground_is_pos).sum())
    fp_full = int((pred_is_pos & ground_is_neg).sum())
    fn_full = int((pred_is_neg & ground_is_pos).sum())
    tn_full = int((pred_is_neg & ground_is_neg).sum())
    bt = _expected_metrics(tp_full, fp_full, tn_full, fn_full, 0, 0, 1.0)
    baseline_transformer = {
        "label": "100% transformer",
        **bt,
        "coverage": 1.0,
        "n_transformer": n_total,
        "n_frontier": 0,
        "transformer_cost": transformer_cost_total,
        "llm_cost": 0.0,
        "cost": transformer_cost_total,
    }

    baselines_by_llm: dict[str, dict] = {}
    all_mask = np.ones(n_total, dtype=bool)
    for spec in llm_specs:
        llm_tp, llm_fp, llm_fn, llm_tn = _llm_confusion_for_subset(
            spec, all_mask, ground_is_pos, ground_is_neg, positive_label, negative_label,
        )
        m = _metrics_from_confusion(llm_tp, llm_fp, llm_fn, llm_tn)
        baselines_by_llm[spec.name] = {
            "label": f"100% {spec.name}",
            **m,
            "coverage": 0.0,
            "n_transformer": 0,
            "n_frontier": n_total,
            "transformer_cost": 0.0,
            "llm_cost": n_total * spec.cost_per_call,
            "cost": n_total * spec.cost_per_call,
        }

    pareto_indices_by_llm: dict[str, dict[str, list[int]]] = {}
    for spec in llm_specs:
        # Flatten per-LLM view for Pareto computation.
        flat_cells = [
            {**cell["per_llm"][spec.name], "cost": cell["per_llm"][spec.name]["cost"]}
            for cell in cells
        ]
        pareto_indices_by_llm[spec.name] = {
            metric: _pareto_frontier_indices(flat_cells, x_key="cost", y_key=metric)
            for metric in _METRIC_KEYS
        }

    return {
        "cells": cells,
        "pos_thresholds": [float(t) for t in pos_thresholds],
        "neg_thresholds": [float(t) for t in neg_thresholds],
        "baseline_transformer": baseline_transformer,
        "baselines_by_llm": baselines_by_llm,
        "pareto_indices_by_llm": pareto_indices_by_llm,
        "llm_specs": [
            {"name": s.name, "cost_per_call": s.cost_per_call, "accuracy": s.accuracy}
            for s in llm_specs
        ],
        "primary_llm": llm_specs[0].name,
        "n_total": n_total,
    }


def _pareto_frontier_indices(cells: list[dict], *, x_key: str, y_key: str) -> list[int]:
    """Indices of cells where no other cell has both lower x and higher y.

    Sorted by ascending x. Used so the UI can highlight the dominant set.
    """
    indexed = sorted(enumerate(cells), key=lambda kv: (kv[1][x_key], -kv[1][y_key]))
    frontier: list[int] = []
    best_y = -float("inf")
    for idx, cell in indexed:
        if cell[y_key] > best_y:
            frontier.append(idx)
            best_y = cell[y_key]
    return frontier
