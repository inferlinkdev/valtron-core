"""Reference-free summarization quality: the method behind ``SummarizationExperiment``.

Summarization has no single ground-truth string to diff a prediction against, so
this package scores it a different way. A judge model decomposes the document
and each candidate summary into atomic facts, then one shared entailment call
("is each claim supported by these references?") is reused with the claim and
reference roles swapped to produce three of the four axes:

* **correctness** -- are the summary's facts supported by the document?
* **salient_coverage** -- of the facts a good summary *must* convey, how many does it?
* **salient_precision** -- of what the summary says, how much lands on those facts?
* **requirements_met** -- how much of an optional per-class checklist it satisfies.

What makes it reference free is where importance comes from. Coverage is recall
and recall needs a target set; here that set is the document's own must-convey
facts, marked by the judge in one pass over the source. No human summary and no
panel of frontier models is required.

The scoring scheme, ``salience-f+reqs``, gates on faithfulness and then takes the
harmonic mean of the two salience axes, blending in the checklist when there is
one. Read a ranking at the corpus level: the axes are averaged over documents
*before* scoring, deliberately, because a single document rarely separates two
competent models.

Layout: :mod:`~valtron_core.summarization.judge` and
:mod:`~valtron_core.summarization.prompts` are the LLM calls,
:mod:`~valtron_core.summarization.scoring` the metric,
:mod:`~valtron_core.summarization.pipeline` the flow, and
:mod:`~valtron_core.summarization.client_model` the one adapter that binds all of
it to ``LLMClient``. Everything except that adapter is shared verbatim with the
standalone research package it was ported from, so keep it that way.
"""

from .client_model import ClientModel
from .judge import Judge
from .model import Model, Usage
from .pipeline import (
    CandidateEvaluation,
    DocumentEvaluation,
    DocumentFacts,
    evaluate_candidate,
    evaluate_document,
    extract_document_facts,
)
from .prompts import SALIENCE_SUMMARY_PROMPT, Prompt, TemplatePrompt, render_requirements
from .scoring import (
    DEFAULT_BETA,
    DEFAULT_GATE,
    DEFAULT_REQUIREMENT_WEIGHT,
    DEFAULT_TIER_GAP,
    Axes,
    mean_axes,
    rank,
    score,
)
from .text import Doc, Fact, FactSource, Requirement, Summary

__all__ = [
    "DEFAULT_BETA",
    "DEFAULT_GATE",
    "DEFAULT_REQUIREMENT_WEIGHT",
    "DEFAULT_TIER_GAP",
    "SALIENCE_SUMMARY_PROMPT",
    "Axes",
    "CandidateEvaluation",
    "ClientModel",
    "Doc",
    "DocumentEvaluation",
    "DocumentFacts",
    "Fact",
    "FactSource",
    "Judge",
    "Prompt",
    "Model",
    "Requirement",
    "Summary",
    "TemplatePrompt",
    "Usage",
    "evaluate_candidate",
    "evaluate_document",
    "extract_document_facts",
    "mean_axes",
    "rank",
    "render_requirements",
    "score",
]
