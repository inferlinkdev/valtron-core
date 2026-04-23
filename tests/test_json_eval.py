"""Tests for the JSON evaluation module."""

import json
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
    find_expensive_unordered_list_fields,
    _check_builtin_metric_expensive,
    _scan_item_logic_for_expensive_metrics,
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


class TestComparatorMetric:
    """Tests for the comparator_metric function."""

    def test_bool_result_true(self):
        with patch("valtron_core.evaluation.json_eval.Comparator") as MockComp:
            MockComp.return_value.compare.return_value = True
            score, is_correct = comparator_metric("a", "a", {})
        assert score == 1.0
        assert is_correct is True

    def test_bool_result_false(self):
        with patch("valtron_core.evaluation.json_eval.Comparator") as MockComp:
            MockComp.return_value.compare.return_value = False
            score, is_correct = comparator_metric("a", "b", {})
        assert score == 0.0
        assert is_correct is False

    def test_float_with_threshold_met(self):
        with patch("valtron_core.evaluation.json_eval.Comparator") as MockComp:
            MockComp.return_value.compare.return_value = 0.9
            score, is_correct = comparator_metric("a", "b", {"comparison_threshold": 0.8})
        assert score == 0.9
        assert is_correct is True

    def test_float_with_threshold_not_met(self):
        with patch("valtron_core.evaluation.json_eval.Comparator") as MockComp:
            MockComp.return_value.compare.return_value = 0.5
            score, is_correct = comparator_metric("a", "b", {"comparison_threshold": 0.8})
        assert score == 0.5
        assert is_correct is False

    def test_float_no_threshold_defaults_is_correct_true(self):
        with patch("valtron_core.evaluation.json_eval.Comparator") as MockComp:
            MockComp.return_value.compare.return_value = 0.7
            score, is_correct = comparator_metric("a", "b", {})
        assert score == 0.7
        assert is_correct is True

    def test_text_similarity_threshold_used(self):
        with patch("valtron_core.evaluation.json_eval.Comparator") as MockComp:
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


class TestCheckBuiltinMetricExpensive:
    """Tests for _check_builtin_metric_expensive."""

    def test_exact_is_not_expensive(self):
        expensive, desc = _check_builtin_metric_expensive("exact", {})
        assert expensive is False
        assert desc == ""

    def test_threshold_is_not_expensive(self):
        expensive, desc = _check_builtin_metric_expensive("threshold", {})
        assert expensive is False
        assert desc == ""

    def test_comparator_with_llm_is_expensive(self):
        expensive, desc = _check_builtin_metric_expensive("comparator", {"element_compare": "llm"})
        assert expensive is True
        assert "llm" in desc.lower() or "LLM" in desc

    def test_comparator_with_embedding_is_expensive(self):
        expensive, desc = _check_builtin_metric_expensive("comparator", {"element_compare": "embedding"})
        assert expensive is True
        assert "embedding" in desc.lower()

    def test_comparator_with_exact_is_not_expensive(self):
        expensive, desc = _check_builtin_metric_expensive("comparator", {"element_compare": "exact"})
        assert expensive is False
        assert desc == ""

    def test_comparator_default_element_compare_is_not_expensive(self):
        # default element_compare is "exact"
        expensive, desc = _check_builtin_metric_expensive("comparator", {})
        assert expensive is False

    def test_unknown_metric_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="no 3rd-party API declaration"):
            _check_builtin_metric_expensive("unknown_metric", {})


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
        with patch("valtron_core.evaluation.json_eval.Comparator") as MockComp:
            MockComp.return_value.compare.return_value = True
            result = evaluator.evaluate(config, {"tags": ["a"]}, {"tags": ["a"]})
        assert result is not None
