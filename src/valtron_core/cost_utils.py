"""Shared cost-rate utilities used by the evaluator and report aggregation."""

import re
from typing import Any

_TIME_UNIT_RE = re.compile(
    r"^(\d+(?:\.\d+)?)?\s*(s|sec|second|seconds|m|min|minute|minutes|h|hr|hour|hours)$",
    re.IGNORECASE,
)


def _parse_time_unit_to_seconds(time_unit: str) -> float:
    """Parse a time unit string into seconds.

    Accepts an optional leading number (defaults to 1 if omitted).
    Examples: '1hr', '30s', '2h', '5min', 'second', 'hour'.
    """
    m = _TIME_UNIT_RE.match(time_unit.strip())
    if not m:
        raise ValueError(
            f"Unrecognised time_unit: {time_unit!r}. "
            "Examples: '1s', '30s', '1hr', '2h', '5min', 'second', 'hour'."
        )
    value = float(m.group(1)) if m.group(1) else 1.0
    unit = m.group(2).lower()
    if unit in ("s", "sec", "second", "seconds"):
        return value
    if unit in ("m", "min", "minute", "minutes"):
        return value * 60
    return value * 3600  # hours


def _get_fallback_rate_info(model: str | dict[str, Any]) -> dict[str, Any] | None:
    """
    Return the rate that should be used when LiteLLM has no pricing data for
    this model, or None if no default is configured.

    This is the single customization point for imputed costs. Extend it to
    add per-model default rates.

    Llama model sizing guidance — search for the model on Hugging Face or the
    model card (e.g. meta-llama/Llama-2-7b-hf) to confirm parameter count, then
    pick a machine class based on VRAM requirements (FP16 weights ≈ 2 bytes/param):
      - Small  (≤8B params,  ~16 GB VRAM): AWS g5.xlarge    @ $1.006/hr
      - Medium (9B–30B params, ~48 GB VRAM): AWS g5.2xlarge  @ $1.212/hr
      - Large  (>30B params, 80+ GB VRAM): AWS g5.48xlarge  @ $16.288/hr

    Returns:
        Dict with ``cost_rate_imputed`` (float) and ``time_unit_imputed`` (str),
        or None if no default is defined for this model.
    """
    # Maps (max_param_billions, machine_label) -> hourly rate (USD).
    # Entries are checked in order; the first whose threshold covers the model size wins.
    # Use float("inf") as the final sentinel to catch everything else.
    _LLAMA_SIZE_RATES: list[tuple[float, float]] = [
        (8.0,         1.006),   # Small:  ≤8B  — g5.xlarge    @ $1.006/hr
        (30.0,        1.212),   # Medium: ≤30B — g5.2xlarge   @ $1.212/hr
        (float("inf"), 16.288), # Large:  >30B — g5.48xlarge  @ $16.288/hr
    ]

    # Catch-all rate for unrecognised self-hosted models (medium tier).
    _DEFAULT_RATE = 1.212

    model_name = model if isinstance(model, str) else model.get("model", "")
    model_name_lower = model_name.lower()

    if "llama" in model_name_lower:
        param_match = re.search(r"(\d+(?:\.\d+)?)\s*b\b", model_name_lower)
        param_billions = float(param_match.group(1)) if param_match else float("inf")

        for max_b, rate in _LLAMA_SIZE_RATES:
            if param_billions <= max_b:
                return {"cost_rate_imputed": rate, "time_unit_imputed": "1hr"}

    # Catch-all for any other self-hosted / unrecognised model.
    return {"cost_rate_imputed": _DEFAULT_RATE, "time_unit_imputed": "1hr"}


def _fallback_cost(model: str | dict[str, Any], response_time: float) -> float:
    """
    Compute imputed cost for a single prediction using the fallback rate.

    Args:
        model: The model identifier or config dict used for the prediction.
        response_time: Wall-clock seconds the prediction took.

    Returns:
        Imputed cost in dollars, or 0.0 if no fallback rate is configured.
    """
    rate_info = _get_fallback_rate_info(model)
    if rate_info is None:
        return 0.0
    return float(rate_info["cost_rate_imputed"]) * (
        response_time / _parse_time_unit_to_seconds(rate_info["time_unit_imputed"])
    )
