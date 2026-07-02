"""Tests for the analysis sub-package: TradeoffAnalyzer and supporting modules."""

import json

import pytest

from valtron_core.analysis._analyzer import (
    LLMSpec,
    TradeoffRow,
    _pareto_frontier_indices,
    sweep_thresholds,
)
from valtron_core.analysis.tradeoff_analyzer import TradeoffAnalyzer


# ---------------------------------------------------------------------------
# Unit: _pareto_frontier_indices
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestParetoFrontierIndices:
    def test_single_cell_is_frontier(self) -> None:
        cells = [{"cost": 1.0, "f1": 0.9}]
        assert _pareto_frontier_indices(cells, x_key="cost", y_key="f1") == [0]

    def test_dominated_cell_excluded(self) -> None:
        # cell 0: cost=1, f1=0.8 -- dominated by cell 1 (same cost, higher f1)
        # cell 1: cost=1, f1=0.9 -- Pareto-optimal
        cells = [{"cost": 1.0, "f1": 0.8}, {"cost": 1.0, "f1": 0.9}]
        result = _pareto_frontier_indices(cells, x_key="cost", y_key="f1")
        assert 1 in result
        assert 0 not in result

    def test_two_independent_frontier_cells(self) -> None:
        # cell 0: cheap, lower f1 -- cell 1: more expensive, higher f1
        # both on Pareto frontier (neither dominates the other)
        cells = [{"cost": 0.5, "f1": 0.7}, {"cost": 2.0, "f1": 0.95}]
        result = _pareto_frontier_indices(cells, x_key="cost", y_key="f1")
        assert set(result) == {0, 1}

    def test_empty_cells(self) -> None:
        assert _pareto_frontier_indices([], x_key="cost", y_key="f1") == []

    def test_all_tied_cost_keeps_best_metric(self) -> None:
        cells = [
            {"cost": 1.0, "f1": 0.6},
            {"cost": 1.0, "f1": 0.9},
            {"cost": 1.0, "f1": 0.7},
        ]
        result = _pareto_frontier_indices(cells, x_key="cost", y_key="f1")
        # Only the cell with f1=0.9 survives (all same cost, pick highest metric)
        assert len(result) == 1
        assert cells[result[0]]["f1"] == 0.9


# ---------------------------------------------------------------------------
# Unit: sweep_thresholds
# ---------------------------------------------------------------------------


def _make_rows(n_correct: int, n_total: int, confidence: float = 0.9) -> list[TradeoffRow]:
    """Create a simple deterministic set of TradeoffRow objects."""
    rows = []
    for i in range(n_total):
        ground_truth = "yes" if i < (n_total // 2) else "no"
        pred_label = ground_truth if i < n_correct else ("no" if ground_truth == "yes" else "yes")
        rows.append(TradeoffRow(pred_label=pred_label, confidence=confidence, ground_truth=ground_truth))
    return rows


@pytest.mark.unit
class TestSweepThresholds:
    def test_returns_expected_keys(self) -> None:
        rows = _make_rows(n_correct=8, n_total=10)
        llm = LLMSpec(name="gpt-4o", cost_per_call=0.01, accuracy=0.95)
        result = sweep_thresholds(
            rows,
            positive_label="yes",
            negative_label="no",
            n_steps=3,
            llm_specs=[llm],
            cost_per_transformer_call=0.001,
        )
        for key in ("cells", "pareto_indices_by_llm", "n_total", "primary_llm"):
            assert key in result, f"Missing key: {key}"

    def test_n_total_matches_input(self) -> None:
        rows = _make_rows(n_correct=8, n_total=10)
        llm = LLMSpec(name="gpt-4o", cost_per_call=0.01, accuracy=0.95)
        result = sweep_thresholds(
            rows,
            positive_label="yes",
            negative_label="no",
            n_steps=3,
            llm_specs=[llm],
            cost_per_transformer_call=0.001,
        )
        assert result["n_total"] == 10

    def test_deterministic_full_eval_predictions(self) -> None:
        rows = _make_rows(n_correct=10, n_total=10, confidence=0.95)
        # Perfect LLM in deterministic mode
        perfect_preds = [r.ground_truth for r in rows]
        llm = LLMSpec(name="gpt-4o", cost_per_call=0.01, accuracy=1.0, predictions=perfect_preds)
        result = sweep_thresholds(
            rows,
            positive_label="yes",
            negative_label="no",
            n_steps=3,
            llm_specs=[llm],
            cost_per_transformer_call=0.001,
        )
        assert result["n_total"] == 10
        assert len(result["cells"]) > 0

    def test_pareto_indices_within_cells_range(self) -> None:
        rows = _make_rows(n_correct=8, n_total=20)
        llm = LLMSpec(name="gpt-4o", cost_per_call=0.01, accuracy=0.95)
        result = sweep_thresholds(
            rows,
            positive_label="yes",
            negative_label="no",
            n_steps=5,
            llm_specs=[llm],
            cost_per_transformer_call=0.001,
        )
        n_cells = len(result["cells"])
        for llm_name, by_metric in result["pareto_indices_by_llm"].items():
            for metric, indices in by_metric.items():
                for idx in indices:
                    assert 0 <= idx < n_cells, f"Pareto index {idx} out of range for {llm_name}/{metric}"


# ---------------------------------------------------------------------------
# Unit: TradeoffAnalyzer analyze/save API
# ---------------------------------------------------------------------------


def _make_precomputed_analyzer() -> TradeoffAnalyzer:
    """Build a TradeoffAnalyzer in precomputed mode using fixture data."""
    rows = _make_rows(n_correct=8, n_total=10)
    llm = LLMSpec(name="gpt-4o", cost_per_call=0.01, accuracy=0.95)
    return TradeoffAnalyzer(
        _precomputed={
            "tradeoff_rows": rows,
            "llm_specs": [llm],
            "pos_label": "yes",
            "neg_label": "no",
            "transformer_cost_per_call": 0.001,
            "transformer_instance_hourly": 0.085,
            "transformer_samples_per_second": 8.0,
        }
    )


@pytest.mark.unit
class TestTradeoffAnalyzerAPI:
    def test_analyze_populates_sweep(self) -> None:
        analyzer = _make_precomputed_analyzer()
        assert analyzer._sweep is None
        analyzer.analyze()
        assert analyzer._sweep is not None
        for key in ("cells", "n_total", "positive_label"):
            assert key in analyzer._sweep, f"Missing key: {key}"

    def test_save_json_report_writes_valid_json(self, tmp_path: "Path") -> None:
        analyzer = _make_precomputed_analyzer()
        analyzer.analyze()
        out = tmp_path / "report.json"
        result = analyzer.save_json_report(out)
        assert result == out
        assert out.exists()
        data = json.loads(out.read_text())
        for key in ("cells", "n_total", "positive_label", "negative_label", "llm_specs"):
            assert key in data, f"Missing key in JSON: {key}"
        assert len(data["cells"]) > 0

    def test_save_html_report_after_analyze(self, tmp_path: "Path") -> None:
        analyzer = _make_precomputed_analyzer()
        analyzer.analyze()
        out = tmp_path / "report.html"
        result = analyzer.save_html_report(out)
        assert result == out
        assert out.exists()
        assert "<html" in out.read_text().lower()

    def test_save_before_analyze_raises(self, tmp_path: "Path") -> None:
        analyzer = _make_precomputed_analyzer()
        with pytest.raises(RuntimeError, match="analyze()"):
            analyzer.save_html_report(tmp_path / "report.html")
        with pytest.raises(RuntimeError, match="analyze()"):
            analyzer.save_json_report(tmp_path / "report.json")

    def test_run_still_works(self, tmp_path: "Path") -> None:
        analyzer = _make_precomputed_analyzer()
        out = tmp_path / "report.html"
        result = analyzer.run(out)
        assert result == out
        assert out.exists()


