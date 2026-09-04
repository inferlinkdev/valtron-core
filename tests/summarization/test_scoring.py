"""Unit tests for :mod:`valtron_core.summarization.scoring` -- the metric and the tiering."""

from __future__ import annotations

import pytest

from valtron_core.summarization.scoring import Axes, mean_axes, rank, score


def _axes(**overrides: float | None) -> Axes:
    """Fully-defined axes that clear the gate, with any field overridden."""
    base: dict[str, float | None] = {
        "correctness": 1.0,
        "salient_coverage": 0.5,
        "salient_precision": 0.5,
        "requirements_met": 0.5,
    }
    base.update(overrides)
    return Axes(**base)


def test_the_faithfulness_gate_zeroes_a_strong_but_unfaithful_summary() -> None:
    # Perfect on every other axis, but below the gate: it must score zero, not
    # merely be discounted.
    unfaithful = _axes(
        correctness=0.49, salient_coverage=1.0, salient_precision=1.0, requirements_met=1.0
    )
    assert score(unfaithful) == 0.0


def test_the_gate_is_inclusive_at_its_boundary() -> None:
    assert score(_axes(correctness=0.5)) > 0.0
    assert score(_axes(correctness=0.5 - 1e-9)) == 0.0


def test_missing_correctness_is_treated_as_failing_the_gate() -> None:
    # An unmeasurable summary must not slip past the gate by default.
    assert score(_axes(correctness=None)) == 0.0


def test_without_a_checklist_the_score_is_the_plain_f_measure() -> None:
    axes = _axes(salient_coverage=0.6, salient_precision=0.3, requirements_met=None)
    expected = 2 * 0.6 * 0.3 / (0.6 + 0.3)  # harmonic mean
    assert score(axes) == pytest.approx(expected)


def test_the_requirements_term_is_a_weighted_blend() -> None:
    axes = _axes(salient_coverage=0.5, salient_precision=0.5, requirements_met=1.0)
    # F(0.5, 0.5) = 0.5, so at w=0.6 the score is 0.4*0.5 + 0.6*1.0.
    assert score(axes, requirement_weight=0.6) == pytest.approx(0.4 * 0.5 + 0.6 * 1.0)


def test_a_zero_requirement_weight_ignores_the_checklist_entirely() -> None:
    with_checklist = _axes(requirements_met=1.0)
    without = _axes(requirements_met=None)
    assert score(with_checklist, requirement_weight=0.0) == score(without, requirement_weight=0.0)


def test_supplying_no_checklist_scores_identically_to_not_using_one() -> None:
    # The checklist is genuinely optional: absent it, the weight cannot change
    # the score, because there is nothing to blend in.
    axes = _axes(requirements_met=None)
    assert score(axes, requirement_weight=0.6) == score(axes, requirement_weight=0.0)


def test_an_undefined_salience_axis_collapses_the_f_measure_to_zero() -> None:
    assert score(_axes(salient_coverage=None, requirements_met=None)) == 0.0
    assert score(_axes(salient_precision=None, requirements_met=None)) == 0.0


def test_beta_above_one_favors_coverage_over_precision() -> None:
    coverage_heavy = _axes(salient_coverage=0.8, salient_precision=0.2, requirements_met=None)
    precision_heavy = _axes(salient_coverage=0.2, salient_precision=0.8, requirements_met=None)
    # At beta=1 the harmonic mean is symmetric; above 1 recall dominates.
    assert score(coverage_heavy, beta=1.0) == pytest.approx(score(precision_heavy, beta=1.0))
    assert score(coverage_heavy, beta=2.0) > score(precision_heavy, beta=2.0)


def test_padding_a_summary_cannot_pay_off() -> None:
    # The point of the harmonic mean: buying coverage at the cost of precision
    # must not raise the score, which is what stops the metric being a length
    # proxy that simply rewards the longest summary.
    focused = _axes(salient_coverage=0.5, salient_precision=0.9, requirements_met=None)
    padded = _axes(salient_coverage=0.6, salient_precision=0.2, requirements_met=None)
    assert score(padded) < score(focused)


class TestMeanAxes:
    """Aggregation averages the axes, and does it before scoring."""

    def test_averages_each_axis_over_the_documents(self) -> None:
        aggregate = mean_axes([_axes(salient_coverage=0.2), _axes(salient_coverage=0.8)])
        assert aggregate is not None
        assert aggregate.salient_coverage == pytest.approx(0.5)

    def test_an_axis_averages_only_where_it_is_defined(self) -> None:
        # A document that yielded no facts must not drag the axis toward zero.
        aggregate = mean_axes([_axes(salient_precision=0.6), _axes(salient_precision=None)])
        assert aggregate is not None
        assert aggregate.salient_precision == pytest.approx(0.6)

    def test_an_axis_undefined_everywhere_stays_undefined(self) -> None:
        aggregate = mean_axes([_axes(requirements_met=None), _axes(requirements_met=None)])
        assert aggregate is not None
        assert aggregate.requirements_met is None

    def test_no_documents_gives_no_axes(self) -> None:
        assert mean_axes([]) is None


class TestRank:
    """Tiering: order by score, split on a drop wider than the gap."""

    def test_orders_models_best_first(self) -> None:
        assert rank({"a": 0.1, "b": 0.9, "c": 0.5}) == [["b"], ["c"], ["a"]]

    def test_exact_ties_share_a_tier(self) -> None:
        assert rank({"a": 0.5, "b": 0.5}) == [["a", "b"]]

    def test_the_default_gap_separates_even_close_scores(self) -> None:
        # tier_gap defaults to 0, so anything but an exact tie is a real ordering.
        assert rank({"a": 0.5000001, "b": 0.5}) == [["a"], ["b"]]

    def test_a_wider_gap_groups_near_neighbors(self) -> None:
        assert rank({"a": 0.90, "b": 0.88, "c": 0.40}, tier_gap=0.05) == [["a", "b"], ["c"]]

    def test_no_models_gives_no_tiers(self) -> None:
        assert rank({}) == []
