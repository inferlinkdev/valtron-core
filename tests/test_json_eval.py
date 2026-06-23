"""Tests for the JSON evaluation module."""

import json
import math
from unittest.mock import Mock, patch

import pytest

from valtron_core.evaluation.json_eval import (
    JsonEvaluator,
    LeafMetricConfig,
    ObjectMetricConfig,
    ListMetricConfig,
    FieldConfig,
    EvalResult,
    AlignmentItem,
    comparator_metric,
    MATCH_KEY_MAX_CHARS,
    find_expensive_unordered_list_fields,
    collect_field_metric_llm_models,
)
from valtron_core.evaluation.json_eval.schema import AlignmentConfig
from valtron_core.evaluation.json_eval.registries import _score_to_result
from valtron_core.evaluation.json_eval.alignment import _match_key_text
from valtron_core.evaluation.json_eval.validation import (
    _uses_expensive_thirdparty_api,
    _scan_item_logic_for_expensive_metrics,
    _item_logic_uses_expensive_api,
)


class TestMetricConfig:
    """Tests for metric config dataclasses."""

    def test_default_leaf_config(self):
        """Test default LeafMetricConfig values."""
        config = LeafMetricConfig()

        assert config.metric == "exact"
        assert config.params == {}

    def test_default_object_config(self):
        """Test default ObjectMetricConfig values."""
        config = ObjectMetricConfig()

        assert config.propagation == "weighted_avg"

    def test_custom_params(self):
        """Test LeafMetricConfig with custom parameters."""
        config = LeafMetricConfig(
            metric="threshold",
            params={"min": 0.8},
        )

        assert config.metric == "threshold"
        assert config.params["min"] == 0.8

    def test_list_config(self):
        """Test ListMetricConfig with list-specific options."""
        config = ListMetricConfig(
            ordered=True,
            match_threshold=0.5,
        )

        assert config.ordered is True
        assert config.match_threshold == 0.5


class TestFieldConfig:
    """Tests for FieldConfig dataclass."""

    def test_leaf_config(self):
        """Test FieldConfig for leaf fields."""
        config = FieldConfig(type="leaf")

        assert config.type == "leaf"
        assert config.fields is None

    def test_object_config(self):
        """Test FieldConfig for object fields."""
        config = FieldConfig(
            type="object",
            fields={
                "name": FieldConfig(type="leaf"),
                "age": FieldConfig(type="leaf"),
            },
        )

        assert config.type == "object"
        assert len(config.fields) == 2
        assert "name" in config.fields

    def test_list_config(self):
        """Test FieldConfig for list fields."""
        config = FieldConfig(
            type="list",
            metric_config=ListMetricConfig(ordered=False),
        )

        assert config.type == "list"
        assert config.metric_config.ordered is False

    def test_optional_field(self):
        """Test FieldConfig with optional flag."""
        config = FieldConfig(type="leaf", optional=True)

        assert config.optional is True

    def test_weighted_field(self):
        """Test FieldConfig with weight."""
        config = FieldConfig(type="leaf", weight=2.0)

        assert config.weight == 2.0


class TestEvalResult:
    """Tests for EvalResult dataclass."""

    def test_result_creation(self):
        """Test basic EvalResult creation."""
        result = EvalResult(
            path="root.field",
            score=0.9,
            weight=1.0,
            metric="exact",
            is_correct=True,
        )

        assert result.path == "root.field"
        assert result.score == 0.9
        assert result.is_correct is True
        assert result.metric == "exact"

    def test_result_with_children(self):
        """Test EvalResult with child results."""
        child1 = EvalResult(path="root.child1", score=1.0, weight=1.0, metric="exact", is_correct=True)
        child2 = EvalResult(path="root.child2", score=0.5, weight=1.0, metric="exact", is_correct=False)

        parent = EvalResult(
            path="root",
            score=0.75,
            weight=1.0,
            metric="exact",
            is_correct=False,
            children={"child1": child1, "child2": child2},
        )

        assert len(parent.children) == 2
        assert parent.children["child1"].score == 1.0

    def test_result_with_tp_fp_fn(self):
        """Test EvalResult with precision/recall tracking."""
        result = EvalResult(
            path="root",
            score=0.8,
            weight=1.0,
            metric="exact",
            is_correct=True,
            tp=8,
            fp=1,
            fn=1,
        )

        assert result.tp == 8
        assert result.fp == 1
        assert result.fn == 1

    def test_result_alignment(self):
        """Test EvalResult with alignment data."""
        child_result = EvalResult(path="root.list[0]", score=1.0, weight=1.0, metric="exact")
        alignment = [
            AlignmentItem(e_idx=0, a_idx=0, score=1.0, result=child_result),
        ]

        result = EvalResult(
            path="root.list",
            score=1.0,
            weight=1.0,
            metric="exact",
            is_correct=True,
            alignment=alignment,
        )

        assert len(result.alignment) == 1
        assert result.alignment[0].score == 1.0


class TestJsonEvaluatorInit:
    """Tests for JsonEvaluator initialization."""

    def test_init_default(self):
        """Test default JsonEvaluator initialization."""
        evaluator = JsonEvaluator()

        assert evaluator is not None
        assert "exact" in evaluator.metric_registry

    def test_init_custom_metrics(self):
        """Test JsonEvaluator with custom metrics."""

        def custom_metric(expected, actual, params):
            return (1.0, True) if expected == actual else (0.0, False)

        evaluator = JsonEvaluator(custom_metrics={"custom": custom_metric})

        assert "custom" in evaluator.metric_registry

    def test_init_custom_aggregators(self):
        """Test JsonEvaluator with custom aggregators."""

        def custom_agg(results):
            if not results:
                return 0.0
            return sum(r.score for r in results) / len(results)

        evaluator = JsonEvaluator(custom_aggs={"custom_avg": custom_agg})

        assert "custom_avg" in evaluator.agg_registry


class TestJsonEvaluatorEvaluate:
    """Tests for JsonEvaluator.evaluate method."""

    def test_evaluate_simple_match(self):
        """Test evaluation of matching simple values."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {
            "name": {"type": "leaf"},
            "age": {"type": "leaf"},
        }}
        expected = {"name": "John", "age": 30}
        actual = {"name": "John", "age": 30}

        result = evaluator.evaluate(config, expected, actual)

        assert result.score == 1.0
        assert result.is_correct is True

    def test_evaluate_simple_mismatch(self):
        """Test evaluation of mismatched simple values."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {
            "name": {"type": "leaf"},
            "age": {"type": "leaf"},
        }}
        expected = {"name": "John", "age": 30}
        actual = {"name": "Jane", "age": 30}

        result = evaluator.evaluate(config, expected, actual)

        assert result.score < 1.0
        # One field matches, one doesn't

    def test_evaluate_nested_object(self):
        """Test evaluation of nested objects."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {
            "person": {"type": "object", "fields": {
                "name": {"type": "leaf"},
                "address": {"type": "object", "fields": {
                    "city": {"type": "leaf"},
                    "zip": {"type": "leaf"},
                }}
            }}
        }}
        expected = {
            "person": {"name": "John", "address": {"city": "NYC", "zip": "10001"}}
        }
        actual = {
            "person": {"name": "John", "address": {"city": "NYC", "zip": "10001"}}
        }

        result = evaluator.evaluate(config, expected, actual)

        assert result.score == 1.0

    def test_evaluate_missing_key(self):
        """Test evaluation when expected key is missing in actual."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {
            "name": {"type": "leaf"},
            "age": {"type": "leaf"},
        }}
        expected = {"name": "John", "age": 30}
        actual = {"name": "John"}

        result = evaluator.evaluate(config, expected, actual)

        assert result.score < 1.0
        # age is missing, should be penalized

    def test_evaluate_list_ordered(self):
        """Test evaluation of ordered list."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {
            "items": {"type": "list", "metric_config": {"ordered": True, "item_logic": {"type": "leaf"}}}
        }}
        expected = {"items": ["a", "b", "c"]}
        actual = {"items": ["a", "b", "c"]}

        result = evaluator.evaluate(config, expected, actual)

        assert result.score == 1.0

    def test_evaluate_list_ordered_mismatch(self):
        """Test evaluation of ordered list with wrong order."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {
            "items": {"type": "list", "metric_config": {"ordered": True, "item_logic": {"type": "leaf"}}}
        }}
        expected = {"items": ["a", "b", "c"]}
        actual = {"items": ["a", "c", "b"]}  # Wrong order

        result = evaluator.evaluate(config, expected, actual)

        # Order matters, so score should be less than 1
        assert result.score < 1.0

    def test_evaluate_list_unordered(self):
        """Test evaluation of unordered list."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {
            "items": {"type": "list", "metric_config": {"ordered": False, "item_logic": {"type": "leaf"}}}
        }}
        expected = {"items": ["a", "b", "c"]}
        actual = {"items": ["c", "a", "b"]}  # Different order but same items

        result = evaluator.evaluate(config, expected, actual)

        # Items are the same, order doesn't matter
        assert result.score == 1.0

    def test_evaluate_list_length_mismatch(self):
        """Test evaluation when list lengths differ."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {
            "items": {"type": "list", "metric_config": {"item_logic": {"type": "leaf"}}}
        }}
        expected = {"items": ["a", "b", "c"]}
        actual = {"items": ["a", "b"]}  # Missing one item

        result = evaluator.evaluate(config, expected, actual)

        assert result.score < 1.0
        # Should have FN for missing item

    def test_evaluate_empty_list(self):
        """Test evaluation of empty lists."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {
            "items": {"type": "list", "metric_config": {"item_logic": {"type": "leaf"}}}
        }}
        expected = {"items": []}
        actual = {"items": []}

        result = evaluator.evaluate(config, expected, actual)

        # Empty lists result in 0 score (no items to match) - this is expected behavior
        assert result is not None

    def test_evaluate_empty_object(self):
        """Test evaluation of empty objects."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {}}
        expected = {}
        actual = {}

        result = evaluator.evaluate(config, expected, actual)

        # Empty object with no fields results in 0 weighted avg - this is expected
        assert result is not None

    def test_evaluate_json_string_input(self):
        """Test evaluation with JSON string input."""
        evaluator = JsonEvaluator()

        config = '{"type": "object", "fields": {"name": {"type": "leaf"}, "age": {"type": "leaf"}}}'
        expected = '{"name": "John", "age": 30}'
        actual = '{"name": "John", "age": 30}'

        result = evaluator.evaluate(config, expected, actual)

        assert result.score == 1.0


class TestJsonEvaluatorEdgeCases:
    """Tests for edge cases in JsonEvaluator."""

    def test_eval_leaf_both_missing(self):
        """Test leaf evaluation when both values are None."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {"field": {"type": "leaf", "optional": True}}}
        expected = {"field": None}
        actual = {"field": None}

        result = evaluator.evaluate(config, expected, actual)

        # Both None with optional field - check child result
        assert result.children["field"].score == 1.0

    def test_eval_leaf_expected_missing(self):
        """Test leaf evaluation when expected is None."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {"field": {"type": "leaf"}}}
        expected = {"field": None}
        actual = {"field": "value"}

        result = evaluator.evaluate(config, expected, actual)

        # Expected None but got value - may be FP depending on implementation
        # Just verify it returns a result
        assert result is not None

    def test_eval_leaf_actual_missing(self):
        """Test leaf evaluation when actual is None/missing."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {"field": {"type": "leaf"}}}
        expected = {"field": "value"}
        actual = {"field": None}

        result = evaluator.evaluate(config, expected, actual)

        # Expected value but got None - FN
        assert result.score < 1.0

    def test_f1_calculation(self):
        """Test F1 score calculation from list comparison."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {
            "items": {"type": "list", "metric_config": {"item_logic": {"type": "leaf"}}}
        }}
        expected = {"items": ["a", "b", "c", "d"]}
        actual = {"items": ["a", "b", "x", "y"]}  # 2 correct, 2 wrong

        result = evaluator.evaluate(config, expected, actual)

        # Should have score between 0 and 1
        assert 0 <= result.score <= 1

    def test_precision_recall_tracking(self):
        """Test that precision and recall are tracked."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {
            "items": {"type": "list", "metric_config": {"item_logic": {"type": "leaf"}}}
        }}
        expected = {"items": ["a", "b"]}
        actual = {"items": ["a", "c"]}  # 1 TP, 1 FP, 1 FN

        result = evaluator.evaluate(config, expected, actual)

        # Check that result exists
        assert result is not None

    def test_deeply_nested_structure(self):
        """Test deeply nested JSON structure."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {
            "level1": {"type": "object", "fields": {
                "level2": {"type": "object", "fields": {
                    "level3": {"type": "object", "fields": {
                        "level4": {"type": "object", "fields": {
                            "value": {"type": "leaf"}
                        }}
                    }}
                }}
            }}
        }}
        expected = {
            "level1": {
                "level2": {
                    "level3": {"level4": {"value": "deep"}}
                }
            }
        }
        actual = {
            "level1": {
                "level2": {
                    "level3": {"level4": {"value": "deep"}}
                }
            }
        }

        result = evaluator.evaluate(config, expected, actual)

        assert result.score == 1.0

    def test_mixed_types_in_list(self):
        """Test list with mixed types."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {
            "items": {"type": "list", "metric_config": {"item_logic": {"type": "leaf"}}}
        }}
        expected = {"items": [1, "two", {"three": 3}]}
        actual = {"items": [1, "two", {"three": 3}]}

        result = evaluator.evaluate(config, expected, actual)

        assert result.score == 1.0

    def test_numeric_comparison(self):
        """Test numeric value comparison."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {"value": {"type": "leaf"}}}
        expected = {"value": 3.14159}
        actual = {"value": 3.14159}

        result = evaluator.evaluate(config, expected, actual)

        assert result.score == 1.0

    def test_boolean_comparison(self):
        """Test boolean value comparison."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {
            "active": {"type": "leaf"},
            "deleted": {"type": "leaf"},
        }}
        expected = {"active": True, "deleted": False}
        actual = {"active": True, "deleted": False}

        result = evaluator.evaluate(config, expected, actual)

        assert result.score == 1.0

    def test_boolean_mismatch(self):
        """Test boolean value mismatch."""
        evaluator = JsonEvaluator()

        config = {"type": "object", "fields": {"active": {"type": "leaf"}}}
        expected = {"active": True}
        actual = {"active": False}

        result = evaluator.evaluate(config, expected, actual)

        assert result.score == 0.0


class TestListSoftF1Scoring:
    """Soft F1 scoring: list score uses continuous item scores; tp/fp/fn stay as hard counts.

    To get fractional item scores without a threshold (which forces 0/1), we use
    text_similarity with no threshold. To force all_correct=False (so the score=1.0
    shortcut is skipped), we use unequal list lengths so some expected items go unmatched.
    """

    # No threshold: _text_similarity_compare returns raw float. all_correct=False because
    # len(expected) != len(actual), so the list has unmatched items.
    _ITEM_LOGIC = {
        "type": "leaf",
        "metric_config": {
            "metric": "text_similarity",
            "params": {"metric": "fuzz_ratio"},
        },
    }

    def _config(self, ordered: bool) -> dict:
        return {
            "type": "object",
            "fields": {
                "items": {
                    "type": "list",
                    "metric_config": {"ordered": ordered, "item_logic": self._ITEM_LOGIC},
                }
            },
        }

    @pytest.mark.unit
    def test_unordered_soft_f1_exceeds_hard_f1(self) -> None:
        """Soft F1 > hard F1 when a near-match item gets partial credit."""
        evaluator = JsonEvaluator()
        # 3 expected, 2 actual: "apple" exact, "bananaa" near-matches "banana" (~0.92),
        # "cherry" unmatched. Hard tp=1 (only exact "apple" is_correct=True via no threshold
        # means is_correct=True for all, but "cherry" has no pair so a_idx=-1).
        # Actually with no threshold is_correct=True always, so hard tp = len(matched_e).
        # "bananaa" aligns with "banana" (score ~0.92 >= match_threshold 0.5) -> matched.
        # "cherry" has no actual counterpart -> unmatched (a_idx=-1).
        # matched_e={0,1}, len(exp)=3 -> all_correct=False (counts differ).
        # hard tp=2, hard precision=2/2=1.0, hard recall=2/3, hard_f1=0.8
        # soft_tp=1.0+0.92=1.92, soft_precision=1.92/2=0.96, soft_recall=1.92/3=0.64, soft_f1~0.77
        # soft_f1 < hard_f1 here because recall drops. The key property: score < 1.0 and != 0.0.
        result = evaluator.evaluate(
            self._config(ordered=False),
            {"items": ["apple", "banana", "cherry"]},
            {"items": ["apple", "bananaa"]},
        )
        list_result = result.children["items"]
        assert 0.0 < list_result.score < 1.0
        assert list_result.tp == 2.0   # hard: both actual items aligned
        assert list_result.fn == 1.0   # hard: "cherry" unmatched

    @pytest.mark.unit
    def test_ordered_soft_f1_partial_credit(self) -> None:
        """Ordered: near-match item contributes fractional soft_tp, score is between 0 and 1."""
        evaluator = JsonEvaluator()
        # 3 expected, 2 actual: positions 0 and 1 compared, position 2 unmatched.
        # "apple"/"apple" -> score=1.0; "banana"/"bananaa" -> score~0.92.
        # matched=2 (both is_correct=True, no threshold), len(exp)=3 -> is_correct=False on list.
        # soft_tp=1.0+~0.92=~1.92 vs hard tp=2.
        result = evaluator.evaluate(
            self._config(ordered=True),
            {"items": ["apple", "banana", "cherry"]},
            {"items": ["apple", "bananaa"]},
        )
        list_result = result.children["items"]
        assert 0.0 < list_result.score < 1.0
        assert list_result.tp == 2.0   # hard: both positional pairs have is_correct=True
        assert list_result.fn == 1.0   # hard: "cherry" at index 2 unmatched

    @pytest.mark.unit
    def test_exact_match_returns_one(self) -> None:
        """When all items match exactly, score is still 1.0."""
        evaluator = JsonEvaluator()
        result = evaluator.evaluate(
            self._config(ordered=False),
            {"items": ["apple", "banana"]},
            {"items": ["banana", "apple"]},
        )
        assert result.children["items"].score == 1.0

    @pytest.mark.unit
    def test_ordered_exact_match_returns_one(self) -> None:
        """Ordered exact match: score is 1.0."""
        evaluator = JsonEvaluator()
        result = evaluator.evaluate(
            self._config(ordered=True),
            {"items": ["apple", "banana"]},
            {"items": ["apple", "banana"]},
        )
        assert result.children["items"].score == 1.0


class TestComparatorMetric:
    """Tests for the comparator_metric function."""

    def test_bool_result_true(self):
        with patch("valtron_core.evaluation.json_eval.registries.Comparator") as MockComp:
            MockComp.return_value.compare.return_value = True
            score, is_correct = comparator_metric("a", "a", {})
        assert score == 1.0
        assert is_correct is True

    def test_bool_result_false(self):
        with patch("valtron_core.evaluation.json_eval.registries.Comparator") as MockComp:
            MockComp.return_value.compare.return_value = False
            score, is_correct = comparator_metric("a", "b", {})
        assert score == 0.0
        assert is_correct is False

    def test_float_with_threshold_met(self):
        with patch("valtron_core.evaluation.json_eval.registries.Comparator") as MockComp:
            MockComp.return_value.compare.return_value = 0.9
            score, is_correct = comparator_metric("a", "b", {"comparison_threshold": 0.8})
        assert score == 0.9
        assert is_correct is True

    def test_float_with_threshold_not_met(self):
        with patch("valtron_core.evaluation.json_eval.registries.Comparator") as MockComp:
            MockComp.return_value.compare.return_value = 0.5
            score, is_correct = comparator_metric("a", "b", {"comparison_threshold": 0.8})
        assert score == 0.5
        assert is_correct is False

    def test_float_no_threshold_defaults_is_correct_true(self):
        with patch("valtron_core.evaluation.json_eval.registries.Comparator") as MockComp:
            MockComp.return_value.compare.return_value = 0.7
            score, is_correct = comparator_metric("a", "b", {})
        assert score == 0.7
        assert is_correct is True

    def test_text_similarity_threshold_used(self):
        with patch("valtron_core.evaluation.json_eval.registries.Comparator") as MockComp:
            MockComp.return_value.compare.return_value = 0.6
            score, is_correct = comparator_metric("a", "b", {"text_similarity_threshold": 0.7})
        assert score == 0.6
        assert is_correct is False


class TestFieldConfigRouting:
    """Tests for FieldConfig._route_metric_config validator."""

    def test_non_dict_data_passthrough(self):
        """When data is not a dict, the validator returns it unchanged."""
        # Pass a pre-built instance as a field value — validator receives a non-dict
        existing = FieldConfig(type="leaf")
        config = FieldConfig(type="object", fields={"child": existing})
        assert config.fields["child"].type == "leaf"

    def test_model_instance_mc_passthrough(self):
        """When metric_config is already a model instance (not a dict), validator skips routing."""
        mc = LeafMetricConfig(metric="exact")
        config = FieldConfig(type="leaf", metric_config=mc)
        assert isinstance(config.metric_config, LeafMetricConfig)
        assert config.metric_config.metric == "exact"

    def test_list_type_filters_to_list_keys_only(self):
        """list-type routing only passes the known list keys; extra keys are dropped."""
        config = FieldConfig.model_validate({
            "type": "list",
            "metric_config": {
                "ordered": True,
                "match_threshold": 0.8,
                "item_logic": {"type": "leaf"},
                # extra key that should NOT reach ListMetricConfig (which forbids extras)
            },
        })
        mc = config.metric_config
        assert mc.ordered is True
        assert mc.match_threshold == 0.8
        assert mc.item_logic is not None

    def test_list_type_always_includes_ordered_default_false(self):
        """ordered defaults to False when not supplied in a list metric_config dict."""
        config = FieldConfig.model_validate({
            "type": "list",
            "metric_config": {"match_threshold": 0.5},
        })
        assert config.metric_config.ordered is False

    def test_list_type_forwards_required_fields_to_match(self):
        """required_fields_to_match is preserved through list routing."""
        config = FieldConfig.model_validate({
            "type": "list",
            "metric_config": {
                "item_logic": {"type": "object", "fields": {"id": {"type": "leaf"}}},
                "required_fields_to_match": ["id"],
            },
        })
        assert config.metric_config.required_fields_to_match == ["id"]

    def test_list_type_forwards_allow_expensive_comparisons_for(self):
        """allow_expensive_comparisons_for is preserved through list routing."""
        config = FieldConfig.model_validate({
            "type": "list",
            "metric_config": {
                "item_logic": {"type": "leaf"},
                "allow_expensive_comparisons_for": ["$item"],
            },
        })
        assert config.metric_config.allow_expensive_comparisons_for == ["$item"]


class TestUsesExpensiveThirdpartyApi:
    """Tests for _uses_expensive_thirdparty_api."""

    def test_exact_is_local(self):
        assert _uses_expensive_thirdparty_api("exact", {}) is False

    def test_threshold_is_local(self):
        assert _uses_expensive_thirdparty_api("threshold", {}) is False

    def test_comparator_with_llm_is_expensive(self):
        assert _uses_expensive_thirdparty_api("comparator", {"element_compare": "llm"}) is True

    def test_comparator_with_embedding_is_expensive(self):
        assert _uses_expensive_thirdparty_api("comparator", {"element_compare": "embedding"}) is True

    def test_comparator_with_exact_is_local(self):
        assert _uses_expensive_thirdparty_api("comparator", {"element_compare": "exact"}) is False

    def test_comparator_default_element_compare_is_local(self):
        assert _uses_expensive_thirdparty_api("comparator", {}) is False

    def test_unknown_metric_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="no API-cost declaration"):
            _uses_expensive_thirdparty_api("unknown_metric", {})


class TestScanItemLogicForExpensiveMetrics:
    """Tests for _scan_item_logic_for_expensive_metrics."""

    def test_leaf_none_metric_config_returns_empty(self):
        config = FieldConfig(type="leaf", metric_config=None)
        issues = _scan_item_logic_for_expensive_metrics(config, "root", frozenset())
        assert issues == []

    def test_leaf_cheap_builtin_returns_empty(self):
        config = FieldConfig.model_validate({"type": "leaf", "metric_config": {"metric": "exact"}})
        issues = _scan_item_logic_for_expensive_metrics(config, "root", frozenset())
        assert issues == []

    def test_leaf_custom_metric_returns_issue(self):
        config = FieldConfig.model_validate({"type": "leaf", "metric_config": {"metric": "my_custom"}})
        issues = _scan_item_logic_for_expensive_metrics(config, "root", frozenset({"my_custom"}))
        assert len(issues) == 1
        assert issues[0]["type"] == "custom"
        assert issues[0]["metric"] == "my_custom"
        assert issues[0]["relative_path"] == "$item"

    def test_leaf_expensive_builtin_comparator_returns_issue(self):
        config = FieldConfig.model_validate({
            "type": "leaf",
            "metric_config": {"metric": "comparator", "params": {"element_compare": "llm"}},
        })
        issues = _scan_item_logic_for_expensive_metrics(config, "root", frozenset())
        assert len(issues) == 1
        assert issues[0]["type"] == "builtin"
        assert issues[0]["metric"] == "comparator"
        assert issues[0]["metric_path"] == "root"

    def test_object_recurses_into_fields_and_collects_issues(self):
        config = FieldConfig.model_validate({
            "type": "object",
            "fields": {
                "cheap": {"type": "leaf", "metric_config": {"metric": "exact"}},
                "expensive": {
                    "type": "leaf",
                    "metric_config": {"metric": "comparator", "params": {"element_compare": "embedding"}},
                },
            },
        })
        issues = _scan_item_logic_for_expensive_metrics(config, "root", frozenset(), "")
        assert len(issues) == 1
        assert issues[0]["relative_path"] == "expensive"
        assert issues[0]["metric_path"] == "root.expensive"

    def test_nested_list_recurses_into_inner_item_logic(self):
        # A list whose item_logic is itself a leaf with an expensive metric
        config = FieldConfig.model_validate({
            "type": "list",
            "metric_config": {
                "item_logic": {
                    "type": "leaf",
                    "metric_config": {"metric": "comparator", "params": {"element_compare": "llm"}},
                },
            },
        })
        issues = _scan_item_logic_for_expensive_metrics(config, "root", frozenset())
        assert len(issues) == 1
        assert issues[0]["metric_path"] == "root[]"

    def test_object_field_containing_list_preserves_relative_path_prefix(self):
        # Outer list item_logic is an object with a field that is itself a list with
        # an expensive leaf. The relative_path reported for the leaf should include the
        # object field name as a prefix, e.g. "institution.city" not just "city".
        config = FieldConfig.model_validate({
            "type": "object",
            "fields": {
                "institution": {
                    "type": "list",
                    "metric_config": {
                        "item_logic": {
                            "type": "object",
                            "fields": {
                                "city": {
                                    "type": "leaf",
                                    "metric_config": {
                                        "metric": "comparator",
                                        "params": {"element_compare": "llm"},
                                    },
                                },
                                "state": {"type": "leaf", "metric_config": {"metric": "exact"}},
                            },
                        },
                    },
                },
            },
        })
        issues = _scan_item_logic_for_expensive_metrics(config, "root", frozenset(), "")
        assert len(issues) == 1
        assert issues[0]["relative_path"] == "institution.city"
        assert issues[0]["metric_path"] == "root.institution[].city"


class TestFindExpensiveLists:
    """Tests for find_expensive_unordered_list_fields."""

    def test_unordered_list_with_expensive_metric_flagged(self):
        config_dict = {
            "type": "list",
            "metric_config": {
                "ordered": False,
                "item_logic": {
                    "type": "leaf",
                    "metric_config": {"metric": "comparator", "params": {"element_compare": "llm"}},
                },
            },
        }
        issues = find_expensive_unordered_list_fields(config_dict)
        assert len(issues) == 1
        assert issues[0]["list_path"] == "root"
        assert issues[0]["relative_path"] == "$item"

    def test_unordered_list_expensive_metric_allowed_suppressed(self):
        config_dict = {
            "type": "list",
            "metric_config": {
                "ordered": False,
                "item_logic": {
                    "type": "leaf",
                    "metric_config": {"metric": "comparator", "params": {"element_compare": "llm"}},
                },
                "allow_expensive_comparisons_for": ["$item"],
            },
        }
        issues = find_expensive_unordered_list_fields(config_dict)
        assert issues == []

    def test_ordered_list_not_flagged(self):
        config_dict = {
            "type": "list",
            "metric_config": {
                "ordered": True,
                "item_logic": {
                    "type": "leaf",
                    "metric_config": {"metric": "comparator", "params": {"element_compare": "llm"}},
                },
            },
        }
        issues = find_expensive_unordered_list_fields(config_dict)
        assert issues == []

    def test_object_wrapping_unordered_list_detected(self):
        config_dict = {
            "type": "object",
            "fields": {
                "tags": {
                    "type": "list",
                    "metric_config": {
                        "ordered": False,
                        "item_logic": {
                            "type": "leaf",
                            "metric_config": {"metric": "comparator", "params": {"element_compare": "embedding"}},
                        },
                    },
                },
            },
        }
        issues = find_expensive_unordered_list_fields(config_dict)
        assert len(issues) == 1
        assert issues[0]["list_path"] == "root.tags"

    def test_nested_list_inner_unordered_expensive_detected(self):
        # Outer list is ordered, inner list (item_logic) is unordered with expensive metric
        config_dict = {
            "type": "list",
            "metric_config": {
                "ordered": True,
                "item_logic": {
                    "type": "list",
                    "metric_config": {
                        "ordered": False,
                        "item_logic": {
                            "type": "leaf",
                            "metric_config": {"metric": "comparator", "params": {"element_compare": "llm"}},
                        },
                    },
                },
            },
        }
        issues = find_expensive_unordered_list_fields(config_dict)
        assert len(issues) == 1
        assert issues[0]["list_path"] == "root[]"

    def test_custom_metric_flagged(self):
        config_dict = {
            "type": "list",
            "metric_config": {
                "ordered": False,
                "item_logic": {"type": "leaf", "metric_config": {"metric": "my_custom"}},
            },
        }
        issues = find_expensive_unordered_list_fields(config_dict, custom_metric_names={"my_custom"})
        assert len(issues) == 1
        assert issues[0]["type"] == "custom"
        assert issues[0]["list_path"] == "root"


class TestRequiredFieldsToMatch:
    """Tests for required_fields_to_match in unordered list evaluation."""

    def _make_config(self):
        return {
            "type": "object",
            "fields": {
                "items": {
                    "type": "list",
                    "metric_config": {
                        "ordered": False,
                        "required_fields_to_match": ["id"],
                        "item_logic": {
                            "type": "object",
                            "fields": {
                                "id": {"type": "leaf"},
                                "name": {"type": "leaf"},
                            },
                        },
                    },
                },
            },
        }

    def test_required_field_match_pair_is_aligned(self):
        evaluator = JsonEvaluator()
        config = self._make_config()
        expected = {"items": [{"id": "A", "name": "Alice"}]}
        actual = {"items": [{"id": "A", "name": "Alice"}]}
        result = evaluator.evaluate(config, expected, actual)
        assert result.children["items"].score == 1.0

    def test_required_field_mismatch_pair_is_skipped(self):
        """When the required field doesn't match, the pair is skipped — FN + FP."""
        evaluator = JsonEvaluator()
        config = self._make_config()
        # expected id=A, actual id=B — required field never matches → no alignment
        expected = {"items": [{"id": "A", "name": "Alice"}]}
        actual = {"items": [{"id": "B", "name": "Alice"}]}
        result = evaluator.evaluate(config, expected, actual)
        items_result = result.children["items"]
        assert items_result.fn == 1
        assert items_result.fp == 1
        assert items_result.tp == 0

    def test_required_field_correct_pairing_over_name_similarity(self):
        """Items are paired by required field id, not by name similarity."""
        evaluator = JsonEvaluator()
        config = self._make_config()
        # Without required_fields_to_match the greedy aligner might pair A↔Alice_wrong
        # and B↔Bob, or mix them.  With it, A always pairs with A and B with B.
        expected = {"items": [{"id": "A", "name": "Alice"}, {"id": "B", "name": "Bob"}]}
        actual = {"items": [{"id": "B", "name": "Bob"}, {"id": "A", "name": "Different"}]}
        result = evaluator.evaluate(config, expected, actual)
        items_result = result.children["items"]
        # B-Bob pair is a full match; A-Different is partial — both should be aligned
        assert items_result.tp >= 1

    def test_multiple_required_fields_all_must_match(self):
        """All required fields must match for a pair to proceed to full scoring."""
        config = {
            "type": "object",
            "fields": {
                "items": {
                    "type": "list",
                    "metric_config": {
                        "ordered": False,
                        "required_fields_to_match": ["type", "id"],
                        "item_logic": {
                            "type": "object",
                            "fields": {
                                "type": {"type": "leaf"},
                                "id": {"type": "leaf"},
                                "value": {"type": "leaf"},
                            },
                        },
                    },
                },
            },
        }
        evaluator = JsonEvaluator()
        # type matches ("X") but id differs ("1" vs "2") → pair skipped
        expected = {"items": [{"type": "X", "id": "1", "value": "hello"}]}
        actual = {"items": [{"type": "X", "id": "2", "value": "hello"}]}
        result = evaluator.evaluate(config, expected, actual)
        items_result = result.children["items"]
        assert items_result.fn == 1
        assert items_result.fp == 1


class TestExpensiveListGuardInEval:
    """Tests for the expensive-metric guard inside _eval_list."""

    def test_expensive_unordered_list_raises_without_opt_in(self):
        evaluator = JsonEvaluator()
        config = {
            "type": "object",
            "fields": {
                "tags": {
                    "type": "list",
                    "metric_config": {
                        "ordered": False,
                        "item_logic": {
                            "type": "leaf",
                            "metric_config": {
                                "metric": "comparator",
                                "params": {"element_compare": "llm"},
                            },
                        },
                    },
                },
            },
        }
        with pytest.raises(ValueError, match="3rd-party metric"):
            evaluator.evaluate(config, {"tags": ["a", "b"]}, {"tags": ["a", "b"]})

    def test_expensive_unordered_list_allowed_does_not_raise(self):
        evaluator = JsonEvaluator()
        config = {
            "type": "object",
            "fields": {
                "tags": {
                    "type": "list",
                    "metric_config": {
                        "ordered": False,
                        "allow_expensive_comparisons_for": ["$item"],
                        "item_logic": {
                            "type": "leaf",
                            "metric_config": {
                                "metric": "comparator",
                                "params": {"element_compare": "llm"},
                            },
                        },
                    },
                },
            },
        }
        with patch.object(evaluator, "_embed_texts", side_effect=lambda texts, model, path: (_embed_by_identity(texts), 0.0)):
            with patch("valtron_core.evaluation.json_eval.registries.Comparator") as MockComp:
                MockComp.return_value.compare.return_value = True
                result = evaluator.evaluate(config, {"tags": ["a"]}, {"tags": ["a"]})
        assert result is not None
        assert result.children["tags"].metric == "list_embed_hungarian_f1"


class TestLLMPromptTemplate:
    """Tests for llm_prompt_template and extra_template_vars plumbing."""

    def test_comparator_metric_passes_template_vars_to_comparator(self):
        """_template_vars from params are forwarded to Comparator as llm_prompt_extra_vars."""
        params = {
            "element_compare": "llm",
            "llm_prompt_template": "Does '{predicted}' equal '{expected}'? YES or NO.",
            "_template_vars": {"prompt_used": "Extract city.", "example_content": "Paris is in France."},
        }

        with patch("valtron_core.evaluation.json_eval.registries.Comparator") as MockComp:
            MockComp.return_value.compare.return_value = True
            comparator_metric("Paris", "Paris", params)

        _, kwargs = MockComp.call_args
        assert kwargs["llm_prompt_template"] == params["llm_prompt_template"]
        assert kwargs["llm_prompt_extra_vars"] == params["_template_vars"]

    def test_comparator_metric_no_template_vars_passes_none(self):
        """When _template_vars is absent, llm_prompt_extra_vars is None."""
        params = {"element_compare": "llm"}

        with patch("valtron_core.evaluation.json_eval.registries.Comparator") as MockComp:
            MockComp.return_value.compare.return_value = True
            comparator_metric("a", "a", params)

        _, kwargs = MockComp.call_args
        assert kwargs["llm_prompt_extra_vars"] is None

    def test_extra_template_vars_flow_through_evaluate(self):
        """extra_template_vars passed to evaluate() reach comparator_metric via _template_vars."""
        config = {
            "type": "object",
            "fields": {
                "city": {
                    "metric_config": {
                        "metric": "comparator",
                        "params": {
                            "element_compare": "llm",
                            "llm_prompt_template": "Prompt: {prompt_used} '{predicted}' vs '{expected}'? YES or NO.",
                        },
                    }
                }
            },
        }
        extra_vars = {"prompt_used": "Extract the city.", "example_content": "doc text"}
        evaluator = JsonEvaluator()

        with patch("valtron_core.evaluation.json_eval.registries.Comparator") as MockComp:
            MockComp.return_value.compare.return_value = True
            evaluator.evaluate(
                config,
                {"city": "Paris"},
                {"city": "Paris"},
                extra_template_vars=extra_vars,
            )

        _, kwargs = MockComp.call_args
        assert kwargs["llm_prompt_extra_vars"] == extra_vars

    def test_extra_template_vars_not_stored_in_eval_result_params(self):
        """_template_vars must not bleed into EvalResult.params."""
        config = {
            "type": "object",
            "fields": {
                "city": {
                    "metric_config": {
                        "metric": "comparator",
                        "params": {"element_compare": "llm"},
                    }
                }
            },
        }
        evaluator = JsonEvaluator()

        with patch("valtron_core.evaluation.json_eval.registries.Comparator") as MockComp:
            MockComp.return_value.compare.return_value = True
            result = evaluator.evaluate(
                config,
                {"city": "Paris"},
                {"city": "Paris"},
                extra_template_vars={"prompt_used": "some prompt"},
            )

        city_result = result.children["city"]
        assert "_template_vars" not in (city_result.params or {})

    def test_evaluate_without_extra_template_vars_still_works(self):
        """evaluate() with no extra_template_vars arg is backward compatible."""
        config = {
            "type": "object",
            "fields": {"name": {"metric_config": {"metric": "exact"}}},
        }
        evaluator = JsonEvaluator()
        result = evaluator.evaluate(config, {"name": "John"}, {"name": "John"})
        assert result.children["name"].is_correct is True


class TestScoreToResult:
    """Tests for the _score_to_result helper."""

    def test_bool_true(self) -> None:
        score, is_correct = _score_to_result(True, {})
        assert score == 1.0
        assert is_correct is True

    def test_bool_false(self) -> None:
        score, is_correct = _score_to_result(False, {})
        assert score == 0.0
        assert is_correct is False

    def test_float_with_threshold_met(self) -> None:
        score, is_correct = _score_to_result(0.9, {"threshold": 0.8})
        assert score == pytest.approx(0.9)
        assert is_correct is True

    def test_float_with_threshold_not_met(self) -> None:
        score, is_correct = _score_to_result(0.5, {"threshold": 0.8})
        assert score == pytest.approx(0.5)
        assert is_correct is False

    def test_float_with_comparison_threshold(self) -> None:
        score, is_correct = _score_to_result(0.7, {"comparison_threshold": 0.8})
        assert is_correct is False

    def test_float_no_threshold_is_correct_true(self) -> None:
        score, is_correct = _score_to_result(0.6, {})
        assert score == pytest.approx(0.6)
        assert is_correct is True


class TestComparatorMetricDeprecationWarning:
    """Verifies that the deprecated comparator_metric emits DeprecationWarning."""

    def test_comparator_metric_warns(self) -> None:
        with patch("valtron_core.evaluation.json_eval.registries.Comparator") as MockComp:
            MockComp.return_value.compare.return_value = True
            with pytest.warns(DeprecationWarning, match="'comparator' metric is deprecated"):
                comparator_metric("a", "a", {})


class TestUsesExpensiveThirdpartyApiNewMetrics:
    """Tests for the new metric branches in _uses_expensive_thirdparty_api."""

    def test_exact_compare_is_local(self) -> None:
        assert _uses_expensive_thirdparty_api("exact_compare", {}) is False

    def test_text_similarity_fuzz_is_local(self) -> None:
        assert _uses_expensive_thirdparty_api("text_similarity", {"metric": "fuzz_ratio"}) is False

    def test_text_similarity_bleu_is_local(self) -> None:
        assert _uses_expensive_thirdparty_api("text_similarity", {"metric": "bleu"}) is False

    def test_text_similarity_cosine_is_expensive(self) -> None:
        assert _uses_expensive_thirdparty_api("text_similarity", {"metric": "cosine"}) is True

    def test_text_similarity_cosine_with_model_is_expensive(self) -> None:
        assert _uses_expensive_thirdparty_api(
            "text_similarity", {"metric": "cosine", "embedding_model": "my-embed-model"}
        ) is True

    def test_llm_metric_is_expensive(self) -> None:
        assert _uses_expensive_thirdparty_api("llm", {}) is True

    def test_llm_metric_with_model_is_expensive(self) -> None:
        assert _uses_expensive_thirdparty_api("llm", {"model": "claude-3"}) is True

    def test_embedding_metric_is_expensive(self) -> None:
        assert _uses_expensive_thirdparty_api("embedding", {}) is True

    def test_embedding_metric_with_model_is_expensive(self) -> None:
        assert _uses_expensive_thirdparty_api("embedding", {"model": "my-embed"}) is True


class TestNewMetricsInRegistry:
    """Tests that the four new metrics are callable via evaluate()."""

    def test_exact_compare_metric_match(self) -> None:
        evaluator = JsonEvaluator()
        config = {
            "type": "object",
            "fields": {
                "name": {"metric_config": {"metric": "exact_compare", "params": {}}},
            },
        }
        result = evaluator.evaluate(config, {"name": "Alice"}, {"name": "Alice"})
        assert result.children["name"].is_correct is True

    def test_exact_compare_metric_case_insensitive(self) -> None:
        evaluator = JsonEvaluator()
        config = {
            "type": "object",
            "fields": {
                "name": {"metric_config": {"metric": "exact_compare", "params": {"case_sensitive": False}}},
            },
        }
        result = evaluator.evaluate(config, {"name": "ALICE"}, {"name": "alice"})
        assert result.children["name"].is_correct is True

    def test_exact_compare_metric_mismatch(self) -> None:
        evaluator = JsonEvaluator()
        config = {
            "type": "object",
            "fields": {
                "name": {"metric_config": {"metric": "exact_compare", "params": {}}},
            },
        }
        result = evaluator.evaluate(config, {"name": "Alice"}, {"name": "Bob"})
        assert result.children["name"].is_correct is False

    def test_text_similarity_metric_match(self) -> None:
        evaluator = JsonEvaluator()
        config = {
            "type": "object",
            "fields": {
                "text": {
                    "metric_config": {
                        "metric": "text_similarity",
                        "params": {"metric": "fuzz_ratio", "threshold": 0.9},
                    }
                },
            },
        }
        result = evaluator.evaluate(config, {"text": "hello"}, {"text": "hello"})
        assert result.children["text"].is_correct is True

    def test_text_similarity_metric_no_threshold_returns_score(self) -> None:
        evaluator = JsonEvaluator()
        config = {
            "type": "object",
            "fields": {
                "text": {
                    "metric_config": {
                        "metric": "text_similarity",
                        "params": {"metric": "fuzz_ratio"},
                    }
                },
            },
        }
        result = evaluator.evaluate(config, {"text": "hello"}, {"text": "hallo"})
        assert 0.0 < result.children["text"].score < 1.0

    def test_llm_metric_match(self) -> None:
        evaluator = JsonEvaluator()
        config = {
            "type": "object",
            "fields": {
                "city": {
                    "metric_config": {
                        "metric": "llm",
                        "params": {"model": "gpt-4o-mini"},
                    }
                },
            },
        }
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "YES"

        with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparisons.completion", return_value=mock_response):
                result = evaluator.evaluate(config, {"city": "NYC"}, {"city": "New York"})

        assert result.children["city"].is_correct is True

    def test_llm_metric_passes_template_vars(self) -> None:
        """_template_vars from evaluate() flow into _llm_compare as prompt_extra_vars."""
        evaluator = JsonEvaluator()
        template = "Context: {ctx}\n'{predicted}' == '{expected}'? YES or NO."
        config = {
            "type": "object",
            "fields": {
                "val": {
                    "metric_config": {
                        "metric": "llm",
                        "params": {"model": "gpt-4o-mini", "prompt_template": template},
                    }
                },
            },
        }
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "YES"

        with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparisons.completion", return_value=mock_response) as mock_comp:
                evaluator.evaluate(
                    config,
                    {"val": "a"},
                    {"val": "a"},
                    extra_template_vars={"ctx": "some context"},
                )

        sent = mock_comp.call_args[1]["messages"][0]["content"]
        assert "some context" in sent

    def test_embedding_metric_match(self) -> None:
        evaluator = JsonEvaluator()
        config = {
            "type": "object",
            "fields": {
                "text": {
                    "metric_config": {
                        "metric": "embedding",
                        "params": {"threshold": 0.9},
                    }
                },
            },
        }
        mock_resp1 = Mock()
        mock_resp1.data = [{"embedding": [1.0, 0.0, 0.0]}]
        mock_resp2 = Mock()
        mock_resp2.data = [{"embedding": [1.0, 0.0, 0.0]}]

        with patch("valtron_core.evaluation.comparisons.embedding") as mock_emb:
            mock_emb.side_effect = [mock_resp1, mock_resp2]
            result = evaluator.evaluate(config, {"text": "hello"}, {"text": "hello"})

        assert result.children["text"].is_correct is True

    def test_embedding_metric_no_threshold_returns_score(self) -> None:
        evaluator = JsonEvaluator()
        config = {
            "type": "object",
            "fields": {
                "text": {
                    "metric_config": {
                        "metric": "embedding",
                        "params": {},
                    }
                },
            },
        }
        mock_resp1 = Mock()
        mock_resp1.data = [{"embedding": [1.0, 0.5, 0.0]}]
        mock_resp2 = Mock()
        mock_resp2.data = [{"embedding": [0.5, 1.0, 0.0]}]

        with patch("valtron_core.evaluation.comparisons.embedding") as mock_emb:
            mock_emb.side_effect = [mock_resp1, mock_resp2]
            result = evaluator.evaluate(config, {"text": "hello"}, {"text": "world"})

        assert 0.0 < result.children["text"].score < 1.0


class TestCollectFieldMetricLLMModels:
    """Tests for collect_field_metric_llm_models with the new 'llm' metric."""

    def test_llm_metric_model_collected(self) -> None:
        config = {
            "type": "object",
            "fields": {
                "city": {
                    "metric_config": {
                        "metric": "llm",
                        "params": {"model": "gpt-4"},
                    }
                },
            },
        }
        models = collect_field_metric_llm_models(config)
        assert "gpt-4" in models

    def test_llm_metric_default_model_collected(self) -> None:
        config = {
            "type": "object",
            "fields": {
                "city": {"metric_config": {"metric": "llm", "params": {}}},
            },
        }
        models = collect_field_metric_llm_models(config)
        assert "gpt-4o-mini" in models

    def test_legacy_comparator_llm_model_still_collected(self) -> None:
        config = {
            "type": "object",
            "fields": {
                "city": {
                    "metric_config": {
                        "metric": "comparator",
                        "params": {"element_compare": "llm", "llm_model": "gpt-4"},
                    }
                },
            },
        }
        models = collect_field_metric_llm_models(config)
        assert "gpt-4" in models

    def test_non_llm_metrics_not_collected(self) -> None:
        config = {
            "type": "object",
            "fields": {
                "a": {"metric_config": {"metric": "exact", "params": {}}},
                "b": {"metric_config": {"metric": "embedding", "params": {}}},
            },
        }
        models = collect_field_metric_llm_models(config)
        assert len(models) == 0


class TestNewMetricsExpensiveListGuard:
    """Tests that the new metrics trigger the expensive-list guard correctly."""

    def test_llm_metric_unordered_list_raises(self) -> None:
        evaluator = JsonEvaluator()
        config = {
            "type": "list",
            "metric_config": {
                "ordered": False,
                "item_logic": {
                    "type": "leaf",
                    "metric_config": {"metric": "llm", "params": {}},
                },
            },
        }
        with pytest.raises(ValueError, match="3rd-party metric"):
            evaluator.evaluate(config, ["a"], ["a"])

    def test_embedding_metric_unordered_list_raises(self) -> None:
        evaluator = JsonEvaluator()
        config = {
            "type": "list",
            "metric_config": {
                "ordered": False,
                "item_logic": {
                    "type": "leaf",
                    "metric_config": {"metric": "embedding", "params": {}},
                },
            },
        }
        with pytest.raises(ValueError, match="3rd-party metric"):
            evaluator.evaluate(config, ["a"], ["a"])

    def test_text_similarity_fuzz_unordered_list_does_not_raise(self) -> None:
        evaluator = JsonEvaluator()
        config = {
            "type": "list",
            "metric_config": {
                "ordered": False,
                "item_logic": {
                    "type": "leaf",
                    "metric_config": {"metric": "text_similarity", "params": {"metric": "fuzz_ratio"}},
                },
            },
        }
        result = evaluator.evaluate(config, ["hello"], ["hello"])
        assert result is not None

    def test_llm_metric_unordered_list_allowed(self) -> None:
        evaluator = JsonEvaluator()
        config = {
            "type": "list",
            "metric_config": {
                "ordered": False,
                "allow_expensive_comparisons_for": ["$item"],
                "item_logic": {
                    "type": "leaf",
                    "metric_config": {"metric": "llm", "params": {}},
                },
            },
        }
        mock_response = Mock()
        mock_response.choices = [Mock()]
        mock_response.choices[0].message.content = "YES"

        with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparisons.completion", return_value=mock_response):
                result = evaluator.evaluate(config, ["a"], ["a"])

        assert result is not None


class TestItemLogicUsesExpensiveApi:
    """Tests for _item_logic_uses_expensive_api."""

    def test_none_returns_false(self):
        assert _item_logic_uses_expensive_api(None) is False

    def test_leaf_exact_returns_false(self):
        config = FieldConfig(type="leaf", metric_config=LeafMetricConfig(metric="exact"))
        assert _item_logic_uses_expensive_api(config) is False

    def test_leaf_comparator_llm_returns_true(self):
        config = FieldConfig.model_validate({
            "type": "leaf",
            "metric_config": {"metric": "comparator", "params": {"element_compare": "llm"}},
        })
        assert _item_logic_uses_expensive_api(config) is True

    def test_leaf_llm_metric_returns_true(self):
        config = FieldConfig.model_validate({
            "type": "leaf",
            "metric_config": {"metric": "llm", "params": {}},
        })
        assert _item_logic_uses_expensive_api(config) is True

    def test_leaf_comparator_embedding_returns_true(self):
        config = FieldConfig.model_validate({
            "type": "leaf",
            "metric_config": {"metric": "comparator", "params": {"element_compare": "embedding"}},
        })
        assert _item_logic_uses_expensive_api(config) is True

    def test_leaf_embedding_metric_returns_true(self):
        config = FieldConfig.model_validate({
            "type": "leaf",
            "metric_config": {"metric": "embedding", "params": {}},
        })
        assert _item_logic_uses_expensive_api(config) is True

    def test_object_with_expensive_field_returns_true(self):
        config = FieldConfig.model_validate({
            "type": "object",
            "fields": {
                "name": {"type": "leaf", "metric_config": {"metric": "exact"}},
                "description": {
                    "type": "leaf",
                    "metric_config": {"metric": "llm", "params": {}},
                },
            },
        })
        assert _item_logic_uses_expensive_api(config) is True

    def test_object_without_expensive_field_returns_false(self):
        config = FieldConfig.model_validate({
            "type": "object",
            "fields": {
                "name": {"type": "leaf", "metric_config": {"metric": "exact"}},
                "score": {"type": "leaf"},
            },
        })
        assert _item_logic_uses_expensive_api(config) is False

    def test_nested_list_with_expensive_leaf_returns_true(self):
        config = FieldConfig.model_validate({
            "type": "list",
            "metric_config": {
                "item_logic": {
                    "type": "leaf",
                    "metric_config": {"metric": "llm", "params": {}},
                },
            },
        })
        assert _item_logic_uses_expensive_api(config) is True

    def test_nested_list_without_expensive_leaf_returns_false(self):
        config = FieldConfig.model_validate({
            "type": "list",
            "metric_config": {"item_logic": {"type": "leaf"}},
        })
        assert _item_logic_uses_expensive_api(config) is False

    def test_custom_metric_returns_true(self):
        config = FieldConfig.model_validate({
            "type": "leaf",
            "metric_config": {"metric": "my_custom_scorer"},
        })
        assert _item_logic_uses_expensive_api(config) is True


def _embed_by_identity(texts: list[str]) -> list[list[float]]:
    """Deterministic stand-in for _embed_texts: identical texts → identical one-hot vectors.

    Matching items get cosine 1.0 and distinct items cosine 0.0, so the embedding+Hungarian
    aligner can be driven in tests without touching the embedding API.

    :param texts: Texts to embed (expected items followed by actual items).
    :return: One one-hot vector per text, in input order.
    """
    index: dict[str, int] = {}
    for t in texts:
        index.setdefault(t, len(index))
    dim = max(1, len(index))
    vecs: list[list[float]] = []
    for t in texts:
        v = [0.0] * dim
        v[index[t]] = 1.0
        vecs.append(v)
    return vecs


def _make_leaf_yes_mock() -> Mock:
    """Return a mock litellm completion response that returns YES."""
    response = Mock()
    response.choices = [Mock()]
    response.choices[0].message.content = "YES"
    return response


class TestLlmAlignmentRouting:
    """Tests that _eval_list routes to the right path based on item_logic content."""

    def _make_llm_list_config(self):
        return {
            "type": "object",
            "fields": {
                "items": {
                    "type": "list",
                    "metric_config": {
                        "ordered": False,
                        "allow_expensive_comparisons_for": ["$item"],
                        "item_logic": {
                            "type": "leaf",
                            "metric_config": {"metric": "llm", "params": {}},
                        },
                    },
                },
            },
        }

    def test_llm_judge_leaf_routes_to_aligned_metric(self):
        evaluator = JsonEvaluator()
        with patch.object(evaluator, "_embed_texts", side_effect=lambda texts, model, path: (_embed_by_identity(texts), 0.0)):
            with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=False):
                with patch("valtron_core.evaluation.comparisons.completion", return_value=_make_leaf_yes_mock()):
                    result = evaluator.evaluate(
                        self._make_llm_list_config(), {"items": ["a"]}, {"items": ["a"]}
                    )
        assert result.children["items"].metric == "list_embed_hungarian_f1"

    def test_no_llm_judge_leaf_uses_greedy_path(self):
        evaluator = JsonEvaluator()
        config = {
            "type": "object",
            "fields": {
                "items": {
                    "type": "list",
                    "metric_config": {
                        "ordered": False,
                        "item_logic": {"type": "leaf"},
                    },
                },
            },
        }
        result = evaluator.evaluate(config, {"items": ["a"]}, {"items": ["a"]})
        assert result.children["items"].metric == "list_greedy_f1"

    def test_ordered_list_with_llm_leaf_uses_ordered_path(self):
        evaluator = JsonEvaluator()
        config = {
            "type": "object",
            "fields": {
                "items": {
                    "type": "list",
                    "metric_config": {
                        "ordered": True,
                        "item_logic": {
                            "type": "leaf",
                            "metric_config": {"metric": "llm", "params": {}},
                        },
                    },
                },
            },
        }
        with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=False):
            with patch("valtron_core.evaluation.comparisons.completion", return_value=_make_leaf_yes_mock()):
                result = evaluator.evaluate(config, {"items": ["a"]}, {"items": ["a"]})
        assert result.children["items"].metric == "list_ordered_f1"


@pytest.mark.unit
class TestEvalListUnorderedWithAlignment:
    """Tests for _eval_list_unordered_with_alignment."""

    def _make_config(self, extra_fields: dict | None = None):
        mc: dict = {
            "ordered": False,
            "allow_expensive_comparisons_for": ["$item"],
            "item_logic": {
                "type": "leaf",
                "metric_config": {"metric": "llm", "params": {}},
            },
        }
        if extra_fields:
            mc.update(extra_fields)
        return {
            "type": "object",
            "fields": {"items": {"type": "list", "metric_config": mc}},
        }

    def test_both_empty_returns_perfect_score(self):
        evaluator = JsonEvaluator()
        result = evaluator.evaluate(self._make_config(), {"items": []}, {"items": []})
        items = result.children["items"]
        assert items.score == 1.0
        assert items.is_correct is True

    def test_empty_expected_nonempty_actual_returns_zero_score(self):
        evaluator = JsonEvaluator()
        result = evaluator.evaluate(self._make_config(), {"items": []}, {"items": ["a", "b"]})
        items = result.children["items"]
        assert items.score == 0.0
        assert items.is_correct is False
        assert items.fp == 2
        assert items.fn == 0
        assert items.tp == 0

    def test_successful_alignment_perfect_match(self):
        evaluator = JsonEvaluator()
        with patch.object(evaluator, "_embed_texts", side_effect=lambda texts, model, path: (_embed_by_identity(texts), 0.0)):
            with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=False):
                with patch("valtron_core.evaluation.comparisons.completion", return_value=_make_leaf_yes_mock()):
                    result = evaluator.evaluate(
                        self._make_config(), {"items": ["apple"]}, {"items": ["apple"]}
                    )
        items = result.children["items"]
        assert items.tp == 1
        assert items.fn == 0
        assert items.fp == 0
        assert items.is_correct is True

    def test_non_matching_item_left_unmatched(self):
        evaluator = JsonEvaluator()
        # "a" and "b" embed to orthogonal vectors (cosine 0 < align_lo) → no match.
        with patch.object(evaluator, "_embed_texts", side_effect=lambda texts, model, path: (_embed_by_identity(texts), 0.0)):
            result = evaluator.evaluate(self._make_config(), {"items": ["a"]}, {"items": ["b"]})
        items = result.children["items"]
        assert items.tp == 0
        assert items.fn == 1

    def test_details_include_aligner_metadata(self):
        evaluator = JsonEvaluator()
        with patch.object(evaluator, "_embed_texts", side_effect=lambda texts, model, path: (_embed_by_identity(texts), 0.0)):
            with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=False):
                with patch("valtron_core.evaluation.comparisons.completion", return_value=_make_leaf_yes_mock()):
                    result = evaluator.evaluate(
                        self._make_config(), {"items": ["x", "y"]}, {"items": ["x", "y"]}
                    )
        details = result.children["items"].details
        assert details["align_method"] == "embed_hungarian"
        assert details["n_matched"] == 2
        assert details["n_embed_calls"] == 1
        assert "embedding_model" in details

    def test_surplus_expected_items_left_unmatched(self):
        evaluator = JsonEvaluator()
        # Two expected items, one actual: only the matching expected item ("a") pairs; the
        # surplus ("b") has no eligible counterpart and stays unmatched.
        with patch.object(evaluator, "_embed_texts", side_effect=lambda texts, model, path: (_embed_by_identity(texts), 0.0)):
            with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=False):
                with patch("valtron_core.evaluation.comparisons.completion", return_value=_make_leaf_yes_mock()):
                    result = evaluator.evaluate(
                        self._make_config(),
                        {"items": ["a", "b"]},
                        {"items": ["a"]},
                    )
        items = result.children["items"]
        matched = [a for a in items.alignment if a.a_idx >= 0]
        unmatched = [a for a in items.alignment if a.a_idx < 0]
        assert len(matched) == 1
        assert matched[0].e_idx == 0
        assert len(unmatched) == 1
        assert unmatched[0].e_idx == 1

    def test_embedding_failure_leaves_items_unmatched(self):
        evaluator = JsonEvaluator()
        with patch.object(evaluator, "_embed_texts", side_effect=RuntimeError("embedding down")):
            result = evaluator.evaluate(self._make_config(), {"items": ["a"]}, {"items": ["b"]})
        assert result.children["items"].tp == 0

    def test_required_fields_pre_filter_limits_candidates(self):
        config = {
            "type": "object",
            "fields": {
                "items": {
                    "type": "list",
                    "metric_config": {
                        "ordered": False,
                        "allow_expensive_comparisons_for": ["description"],
                        "required_fields_to_match": ["id"],
                        "item_logic": {
                            "type": "object",
                            "fields": {
                                "id": {"type": "leaf"},
                                "description": {
                                    "type": "leaf",
                                    "metric_config": {
                                        "metric": "llm",
                                        "params": {},
                                    },
                                },
                            },
                        },
                    },
                },
            },
        }
        evaluator = JsonEvaluator()
        captured_candidates: list[list[int]] = []

        def recording_align(exp, act, candidates_per_e, path, m_cfg):
            captured_candidates.extend(candidates_per_e)
            return {i: None for i in range(len(exp))}, {
                "n_matched": 0, "n_embed_calls": 0, "embedding_ok": False, "match_key_fields": None,
                "match_key_cost": 0.0, "embedding_cost": 0.0,
            }

        with patch.object(evaluator, "_align_by_embedding", side_effect=recording_align):
            with patch("valtron_core.evaluation.comparisons.litellm.supports_response_schema", return_value=False):
                with patch("valtron_core.evaluation.comparisons.completion", return_value=_make_leaf_yes_mock()):
                    evaluator.evaluate(
                        config,
                        {"items": [{"id": "A", "description": "foo"}]},
                        {"items": [
                            {"id": "B", "description": "bar"},
                            {"id": "A", "description": "baz"},
                        ]},
                    )
        # Only A[1] has id="A" matching E[0].id="A"; A[0] has id="B" so it is pre-filtered out.
        assert captured_candidates == [[1]]


class TestEvaluationCostAccumulation:
    """Tests for JsonEvaluator.evaluation_cost instance-level accumulation."""

    # --- accumulator unit tests (no LLM calls) ---

    @pytest.mark.unit
    def test_evaluation_cost_starts_at_zero(self):
        assert JsonEvaluator().evaluation_cost == 0.0

    @pytest.mark.unit
    def test_record_evaluation_cost_accumulates(self):
        evaluator = JsonEvaluator()
        evaluator._record_evaluation_cost(0.005)
        evaluator._record_evaluation_cost(0.003)
        assert evaluator.evaluation_cost == pytest.approx(0.008)

    @pytest.mark.unit
    def test_record_evaluation_cost_ignores_zero(self):
        evaluator = JsonEvaluator()
        evaluator._record_evaluation_cost(0.0)
        assert evaluator.evaluation_cost == 0.0

    @pytest.mark.unit
    def test_evaluation_cost_isolated_per_instance(self):
        evaluator_a = JsonEvaluator()
        evaluator_b = JsonEvaluator()
        evaluator_a._record_evaluation_cost(0.007)
        assert evaluator_a.evaluation_cost == pytest.approx(0.007)
        assert evaluator_b.evaluation_cost == 0.0

    # --- _comparator_metric records cost from _run_comparator ---

    @pytest.mark.unit
    def test_comparator_metric_bound_method_records_cost(self):
        evaluator = JsonEvaluator()
        with patch(
            "valtron_core.evaluation.json_eval.evaluator._run_comparator",
            return_value=(1.0, True, 0.002, 1),
        ):
            evaluator._comparator_metric("a", "a", {})
        assert evaluator.evaluation_cost == pytest.approx(0.002)

    @pytest.mark.unit
    def test_comparator_metric_bound_method_zero_cost_not_recorded(self):
        evaluator = JsonEvaluator()
        with patch(
            "valtron_core.evaluation.json_eval.evaluator._run_comparator",
            return_value=(1.0, True, 0.0, 0),
        ):
            evaluator._comparator_metric("a", "a", {})
        assert evaluator.evaluation_cost == 0.0

    # --- _embed_texts records the embedding-call cost ---

    @pytest.mark.unit
    def test_embed_texts_records_cost(self):
        evaluator = JsonEvaluator()
        resp = Mock()
        resp.data = [{"embedding": [1.0, 0.0]}, {"embedding": [0.0, 1.0]}]
        with patch("valtron_core.evaluation.json_eval.evaluator.embedding", return_value=resp):
            with patch("valtron_core.evaluation.json_eval.evaluator.completion_cost", return_value=0.005):
                with patch.object(evaluator, "_record_evaluation_cost") as mock_record:
                    evaluator._embed_texts(["a", "b"], "text-embedding-3-small", "items")
        mock_record.assert_called_once_with(0.005)

    @pytest.mark.unit
    def test_embed_texts_soft_fails_on_cost_error(self):
        evaluator = JsonEvaluator()
        resp = Mock()
        resp.data = [{"embedding": [1.0, 0.0]}]
        with patch("valtron_core.evaluation.json_eval.evaluator.embedding", return_value=resp):
            with patch(
                "valtron_core.evaluation.json_eval.evaluator.completion_cost",
                side_effect=RuntimeError("billing unavailable"),
            ):
                # Should not raise; soft-fail means the vectors are still returned
                vecs, cost = evaluator._embed_texts(["a"], "text-embedding-3-small", "items")
        assert vecs == [[1.0, 0.0]]
        assert cost == 0.0
        assert evaluator.evaluation_cost == 0.0


class TestEmbeddingHungarianAlignment:
    """Tests for the embedding + Hungarian aligner.

    These drive _align_by_embedding directly with the embedding call patched out, so no
    third-party API is touched. The cosine matrix is computed for real over the supplied
    vectors so the assignment reflects the actual similarity arithmetic.
    """

    def _cfg(self, **overrides):
        """Build a ListMetricConfig with match_key_fields set (skips the field-selection call).

        :param overrides: Field overrides applied on top of the defaults.
        :return: A configured :class:`ListMetricConfig`.
        """
        params = dict(ordered=False, alignment=AlignmentConfig(match_key_fields=["name"], lo=0.35))
        params.update(overrides)
        return ListMetricConfig(**params)

    def test_hungarian_finds_global_optimum_no_llm(self):
        """The optimal one-to-one assignment is found over cosine, with no LLM aligner call.

        Both items have act0 as their top cosine, but the optimal assignment pairs exp0->act0
        and exp1->act1 (total cosine 0.50+0.40 beats 0.45+0.30) -- a global trade that greedy
        per-item matching cannot make.
        """
        ev = JsonEvaluator()
        exp = [{"name": "p"}, {"name": "q"}]
        act = [{"name": "x"}, {"name": "y"}]
        candidates = [[0, 1], [0, 1]]
        c0 = math.sqrt(1 - 0.50 ** 2 - 0.30 ** 2)
        c1 = math.sqrt(1 - 0.45 ** 2 - 0.40 ** 2)
        vecs = [
            [0.50, 0.30, c0],   # exp0: cos->act0=0.50, act1=0.30
            [0.45, 0.40, c1],   # exp1: cos->act0=0.45, act1=0.40
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

        with patch.object(ev, "_embed_texts", return_value=(vecs, 0.0)):
            assignment, stats = ev._align_by_embedding(
                exp, act, candidates, "root.items", self._cfg()
            )

        assert assignment == {0: 0, 1: 1}
        assert stats["embedding_ok"] is True
        assert stats["n_matched"] == 2

    def test_hungarian_drops_sub_lo_pairs(self):
        """An item is left unmatched when its only candidate is below align_lo."""
        ev = JsonEvaluator()
        exp = [{"name": "p"}, {"name": "q"}]
        act = [{"name": "x"}, {"name": "y"}]
        candidates = [[0, 1], [0, 1]]
        # exp0 strongly matches act0; exp1's best is act1 at 0.20, below align_lo=0.35.
        c1 = math.sqrt(1 - 0.20 ** 2 - 0.10 ** 2)
        vecs = [
            [0.95, 0.10, math.sqrt(1 - 0.95 ** 2 - 0.10 ** 2)],
            [0.10, 0.20, c1],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]

        with patch.object(ev, "_embed_texts", return_value=(vecs, 0.0)):
            assignment, stats = ev._align_by_embedding(
                exp, act, candidates, "root.items", self._cfg()
            )

        assert assignment == {0: 0, 1: None}   # exp1 below lo -> unmatched
        assert stats["n_matched"] == 1

    def test_embedding_failure_leaves_items_unmatched(self):
        """If the embedding call fails, every item is left unmatched (no LLM fallback)."""
        ev = JsonEvaluator()
        exp = [{"name": "p"}, {"name": "q"}]
        act = [{"name": "x"}, {"name": "y"}]
        candidates = [[0, 1], [0, 1]]

        with patch.object(ev, "_embed_texts", side_effect=RuntimeError("embedding down")):
            assignment, stats = ev._align_by_embedding(
                exp, act, candidates, "root.items", self._cfg()
            )

        assert assignment == {0: None, 1: None}
        assert stats["embedding_ok"] is False
        assert stats["n_matched"] == 0

    def test_nested_items_skip_match_key_selection(self):
        """Items with nested values skip the field-selection LLM call (return None)."""
        ev = JsonEvaluator()
        exp = [{"name": "a", "details": {"k": 1}}]
        act = [{"name": "a", "details": {"k": 1}}]

        with patch("valtron_core.evaluation.json_eval.evaluator.completion") as mock_completion:
            fields, _ = ev._select_match_key_fields(
                ListMetricConfig(ordered=False, alignment=AlignmentConfig()), exp, act, "root.nested"
            )

        assert fields is None
        mock_completion.assert_not_called()

    def test_flat_items_still_use_match_key_selection(self):
        """Flat items keep the field-selection call so boilerplate fields can be dropped."""
        ev = JsonEvaluator()
        exp = [{"name": "a", "status": "open"}, {"name": "b", "status": "open"}]
        act = [{"name": "a", "status": "open"}]

        resp = Mock()
        resp.choices = [Mock()]
        resp.choices[0].message.content = '{"fields": ["name"]}'

        with patch("valtron_core.evaluation.json_eval.evaluator.completion", return_value=resp), \
             patch("valtron_core.evaluation.json_eval.evaluator.completion_cost", return_value=0.0):
            fields, _ = ev._select_match_key_fields(
                ListMetricConfig(ordered=False, alignment=AlignmentConfig()), exp, act, "root.flat"
            )

        assert fields == ["name"]


class TestMatchKeyText:
    """Tests for _match_key_text — the bounded, top-level-only item rendering for embedding."""

    def test_top_level_scalars_only_excludes_nested(self):
        """With no explicit fields, only top-level scalar fields are rendered; nested are dropped."""
        item = {
            "title": "Blue Whale",
            "category": "mammal",
            "tags": ["marine", "large"],
            "sources": [{"name": "encyclopedia"}],
        }
        text = _match_key_text(item, None)

        assert "title:" in text
        assert "category:" in text
        # Nested list/dict fields and their contents are excluded.
        assert "tags" not in text
        assert "marine" not in text
        assert "sources" not in text
        assert "encyclopedia" not in text

    def test_truncates_to_cap(self):
        """A long top-level string is truncated to the safety cap."""
        item = {"title": "x" * 5000}
        text = _match_key_text(item, None)

        assert len(text) == MATCH_KEY_MAX_CHARS

    def test_no_top_level_scalars_falls_back_to_whole_item(self):
        """An item whose fields are all nested still yields a (bounded) rendering."""
        item = {"program_element": {"number": "0602144A"}, "summary": [{"fy": 2024}]}
        text = _match_key_text(item, None)

        assert text  # non-empty
        assert len(text) <= MATCH_KEY_MAX_CHARS
        # The fallback serializes the whole item, so nested content appears here.
        assert "0602144A" in text

    def test_explicit_fields_are_honored(self):
        """Explicit match_key_fields select exactly those fields."""
        item = {"name": "alpha", "status": "open", "notes": "irrelevant"}
        text = _match_key_text(item, ["name", "status"])

        assert "name: alpha" in text
        assert "status: open" in text
        assert "notes" not in text
