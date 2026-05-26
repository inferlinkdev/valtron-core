"""Synthesize dynamic Pydantic models from litellm-format JSON Schema dicts.

Handles the full OpenAI structured-outputs subset so that dict schemas loaded
from config or from saved experiment metadata can be used the same way as
user-supplied Pydantic classes.
"""

from typing import Any, Literal, Union

import structlog
from pydantic import BaseModel, ConfigDict, Field, create_model

logger = structlog.get_logger()

_JSON_SCALAR_MAP: dict[str, type] = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
}

_JSON_CONSTRAINT_MAP: list[tuple[str, str]] = [
    ("minimum", "ge"),
    ("maximum", "le"),
    ("exclusiveMinimum", "gt"),
    ("exclusiveMaximum", "lt"),
    ("multipleOf", "multiple_of"),
    ("minItems", "min_length"),
    ("maxItems", "max_length"),
]


def synthesize_pydantic_model(schema: dict[str, Any]) -> type[BaseModel] | None:
    """Build a dynamic Pydantic model from a stored litellm-format schema.

    Returns None on failure so callers can fall back gracefully.
    """
    try:
        inner = schema.get("json_schema", {})
        root = inner.get("schema", schema)
        name = inner.get("name", root.get("title", "SynthesizedModel"))
        defs: dict[str, Any] = root.get("$defs", {})
        memo: dict[str, type[BaseModel]] = {}
        model = _build_model(name, root, defs, memo, root)
        for m in memo.values():
            m.model_rebuild(raise_errors=False)
        return model
    except Exception:
        logger.warning(
            "schema_synthesis_failed",
            schema_name=schema.get("json_schema", {}).get("name"),
        )
        return None


def _build_model(
    name: str,
    node: dict[str, Any],
    defs: dict[str, Any],
    memo: dict[str, type[BaseModel]],
    root_schema: dict[str, Any],
) -> type[BaseModel]:
    if name in memo:
        return memo[name]
    memo[name] = create_model(name, __config__=ConfigDict(extra="forbid"))

    props: dict[str, Any] = node.get("properties", {})
    required: set[str] = set(node.get("required", []))
    fields: dict[str, Any] = {}
    for field_name, spec in props.items():
        py_type = _resolve_type(spec, defs, memo, root_schema)
        field_kwargs: dict[str, Any] = {}
        if spec.get("description"):
            field_kwargs["description"] = spec["description"]
        if "pattern" in spec:
            field_kwargs["pattern"] = spec["pattern"]
        for json_kw, pydantic_kw in _JSON_CONSTRAINT_MAP:
            if json_kw in spec:
                field_kwargs[pydantic_kw] = spec[json_kw]
        if field_name not in required:
            fields[field_name] = (py_type | None, Field(None, **field_kwargs))
        else:
            fields[field_name] = (py_type, Field(..., **field_kwargs))
    model = create_model(name, __config__=ConfigDict(extra="forbid"), **fields)
    memo[name] = model
    return model


def _resolve_type(
    spec: dict[str, Any],
    defs: dict[str, Any],
    memo: dict[str, type[BaseModel]],
    root_schema: dict[str, Any],
) -> Any:
    if "$ref" in spec:
        ref = spec["$ref"]
        if ref == "#":
            root_name = root_schema.get("title", "SynthesizedModel")
            return memo.get(root_name, Any)
        ref_name = ref.split("/")[-1]
        return _build_model(ref_name, defs[ref_name], defs, memo, root_schema)

    if "anyOf" in spec:
        non_null = [s for s in spec["anyOf"] if s.get("type") != "null"]
        if len(non_null) == 1:
            inner = _resolve_type(non_null[0], defs, memo, root_schema)
            return inner | None
        resolved = [_resolve_type(s, defs, memo, root_schema) for s in non_null]
        return Union[tuple(resolved)]  # type: ignore[return-value]

    json_type = spec.get("type")

    if isinstance(json_type, list):
        non_null_types = [t for t in json_type if t != "null"]
        if len(non_null_types) == 1:
            inner = _resolve_type({**spec, "type": non_null_types[0]}, defs, memo, root_schema)
            return inner | None
        return Any

    if "enum" in spec:
        return Literal[tuple(spec["enum"])]  # type: ignore[return-value]

    if json_type in _JSON_SCALAR_MAP:
        if json_type == "string":
            from datetime import date, datetime as _dt, time, timedelta
            from uuid import UUID

            _format_map: dict[str, Any] = {
                "date-time": _dt,
                "date": date,
                "time": time,
                "duration": timedelta,
                "uuid": UUID,
            }
            fmt = spec.get("format")
            if fmt in _format_map:
                return _format_map[fmt]
        return _JSON_SCALAR_MAP[json_type]

    if json_type == "array":
        item_type = _resolve_type(spec.get("items", {}), defs, memo, root_schema)
        return list[item_type]  # type: ignore[valid-type]

    if json_type == "object":
        title = spec.get("title", f"Nested_{len(memo)}")
        return _build_model(title, spec, defs, memo, root_schema)

    return Any
