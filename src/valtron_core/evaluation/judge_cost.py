"""Run-level accounting for LLM-as-judge / embedding comparison spend.

valtron_core's field-level comparators call third-party APIs to score model
output whenever ``element_compare`` is ``llm`` or ``embedding`` (and for cosine
text similarity). The per-comparison cost is computed inside
:class:`~valtron_core.evaluation.comparison_functions.Comparator`, but the
comparator is created and discarded per leaf comparison, so the spend is
otherwise lost. This module accumulates it across a run so recipes can report it
alongside the model-under-test cost.

The accumulator is process-global and thread-safe (scoring may run in a thread
pool). It therefore assumes one evaluation run per process at a time: reset it at
the start of a run and read it at the end. Concurrent runs in the same process
would share the counter.
"""
import threading

_lock = threading.Lock()
_state = {"cost_usd": 0.0, "calls": 0}


def reset_judge_cost() -> None:
    """Zero the judge-spend accumulator. Call at the start of a run."""
    with _lock:
        _state["cost_usd"] = 0.0
        _state["calls"] = 0


def record_judge_cost(cost_usd: float, calls: int = 1) -> None:
    """Add one comparison's API spend to the accumulator.

    :param cost_usd: Cost of the comparison's API call(s), in USD.
    :param calls: Number of comparisons represented (default 1).
    """
    with _lock:
        _state["cost_usd"] += float(cost_usd or 0.0)
        _state["calls"] += int(calls or 0)


def get_judge_cost() -> dict:
    """Return the judge spend accumulated since the last reset.

    :return: A dict with ``cost_usd`` (float) and ``calls`` (int).
    """
    with _lock:
        return dict(_state)
