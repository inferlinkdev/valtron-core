"""
Infer a FieldMetricsConfig from an example JSON value.

Usage (CLI):
    python -m evaltron_core.evaluation.field_config_generator '{"name": "foo", "score": 1}'
    python -m evaltron_core.evaluation.field_config_generator path/to/example.json
    echo '{"name": "foo"}' | python -m evaltron_core.evaluation.field_config_generator
"""

import json
import sys
from pathlib import Path
from typing import Any

from valtron_core.evaluation.json_eval import FieldConfig, LeafMetricConfig, ObjectMetricConfig, ListMetricConfig


def infer_field_config(json_str: str) -> FieldConfig:
    """Infer a FieldConfig from an example JSON string.

    Args:
        json_str: A valid JSON string whose structure defines the evaluation schema.

    Returns:
        A typed FieldConfig describing the inferred metric configuration.

    Raises:
        ValueError: If ``json_str`` is not valid JSON.
    """
    try:
        value = json.loads(json_str)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON: {e}") from e
    return _infer_from_value(value)


def _infer_from_value(value: Any) -> FieldConfig:
    """Recursively infer a FieldConfig from a parsed JSON value."""
    if isinstance(value, dict):
        return FieldConfig(
            type="object",
            metric_config=ObjectMetricConfig(propagation="weighted_avg"),
            fields={k: _infer_from_value(v) for k, v in value.items()},
        )

    if isinstance(value, list):
        item_logic = (
            _infer_from_value(value[0])
            if value
            else FieldConfig(type="leaf", metric_config=LeafMetricConfig(metric="exact"))
        )
        return FieldConfig(
            type="list",
            metric_config=ListMetricConfig(
                ordered=False,
                match_threshold=0.5,
                item_logic=item_logic,
            ),
        )

    if value is None:
        return FieldConfig(
            type="leaf",
            optional=True,
            metric_config=LeafMetricConfig(metric="exact"),
        )

    # str, int, float, bool
    return FieldConfig(
        type="leaf",
        metric_config=LeafMetricConfig(metric="exact"),
    )


if __name__ == "__main__":
    if len(sys.argv) == 2:
        arg = sys.argv[1]
        p = Path(arg)
        raw = p.read_text() if p.exists() else arg
    elif not sys.stdin.isatty():
        raw = sys.stdin.read()
    else:
        print("Usage: provide JSON as an argument, file path, or via stdin.", file=sys.stderr)
        sys.exit(1)

    try:
        config = infer_field_config(raw)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    print(json.dumps(config.model_dump(), indent=2))
