"""Strategy protocols for the shared experiment engine: ingest, generate, score, report.

Each protocol is the seam a new experiment type or a new report format
plugs into, instead of subclassing ModelEval/ReferencedEval and
reimplementing its shared fan-out/persistence machinery. See
evaluation/README.md for the full picture; this package only grows one
stage at a time as each is extracted from where it lives today.
"""

from valtron_core.evaluation.stages.score import (
    ExactMatchScorer,
    FieldMetricsScorer,
    ScoreOutcome,
    Scorer,
)

__all__ = [
    "Scorer",
    "ScoreOutcome",
    "ExactMatchScorer",
    "FieldMetricsScorer",
]
