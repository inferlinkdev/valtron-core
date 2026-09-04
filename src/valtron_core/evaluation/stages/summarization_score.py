"""JudgeScorer: reference-free summary quality via the shared Judge.

Lives outside evaluation/stages/score.py, and is not re-exported from
evaluation/stages/__init__.py, so that importing the shared Scorer protocol
does not pull in the whole summarization subsystem for recipes that never use
it. SummarizationExperiment imports this module directly.

Not typed as Scorer (see score.py's module docstring): JudgeScorer's natural
inputs (an already-generated Summary, the document's shared facts, the
requirements checklist) don't fit Scorer.score()'s current
(predicted_value, expected_value) signature, and its result (a GradeResult,
carrying the per-fact verdicts and judge usage a recipe needs for its
PredictionResult.metadata, not just a score) is richer than ScoreOutcome's
lean shape. The two are expected to unify once the
Generator/RawPrediction/Ingestor split lands and every Scorer is rewired
onto a shape that accommodates both.

Wraps valtron_core.summarization.pipeline.grade_summary, the grading half of
evaluate_candidate. Generation (the other half, generate_summary) is a
separate, not-yet-extracted concern; SummarizationExperiment currently calls
it directly rather than through a Generator.
"""

from __future__ import annotations

from valtron_core.summarization import (
    DocumentFacts,
    GradeResult,
    Judge,
    Requirement,
    Summary,
    grade_summary,
)


class JudgeScorer:
    """Grades an already-generated summary against the document's shared facts."""

    def __init__(self, judge: Judge) -> None:
        self._judge = judge

    async def score(
        self,
        summary: Summary,
        shared: DocumentFacts,
        checklist: "list[Requirement]",
    ) -> GradeResult:
        return await grade_summary(summary, self._judge, shared, checklist)
