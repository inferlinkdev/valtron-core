"""Valtron Core: LLM call optimization across various providers."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

from valtron_core.evaluation import ModelEval
from valtron_core.analysis import TradeoffAnalyzer

__all__ = ["TransformerClassifier", "ModelEval", "TradeoffAnalyzer"]


def __getattr__(name: str) -> Any:
    if name == "TransformerClassifier":
        from valtron_core.training import TransformerClassifier

        return TransformerClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
