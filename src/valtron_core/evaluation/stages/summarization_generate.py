"""JudgeCandidateGenerator: one candidate's summary generation, via generate_summary.

Lives outside evaluation/stages/generate.py, and is not re-exported from
evaluation/stages/__init__.py, for the same reason JudgeScorer lives outside
score.py (see summarization_score.py's module docstring): importing the
generic Generator protocol should not pull in the whole summarization
subsystem for recipes that never use it. SummarizationExperiment imports this
module directly.

Not typed as Generator: its inputs (a Doc, a requirements checklist, an
optional summary_prompt, no prompt_template/model-config-dict/response_format
at all) and its return shape (Summary, Usage, float, not RawPrediction) don't
fit Generator.generate()'s current (document, prompt_template, model, ...) ->
RawPrediction signature. Expected to unify once the fuller
Generator/RawPrediction/Ingestor split lands and every Generator is rewired
onto a shape that accommodates both.

Wraps valtron_core.summarization.pipeline.generate_summary, the generation
half of evaluate_candidate; JudgeScorer (summarization_score.py) wraps the
other half, grade_summary.
"""

from __future__ import annotations

from valtron_core.summarization import (
    Doc,
    Model,
    Prompt,
    Requirement,
    Summary,
    Usage,
    generate_summary,
)


class JudgeCandidateGenerator:
    """Generates one candidate's summary of one document. No grading."""

    async def generate(
        self,
        doc: Doc,
        model: Model,
        checklist: "list[Requirement]",
        *,
        summary_prompt: "Prompt | None" = None,
    ) -> "tuple[Summary, Usage, float]":
        return await generate_summary(doc, model, checklist, summary_prompt=summary_prompt)
