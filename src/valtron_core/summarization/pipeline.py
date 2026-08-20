"""The evaluation flow itself: what is asked, of whom, in what order.

This module holds the method and nothing else. It decides no policy -- not how
many documents run at once, not how a failure is reported, not what a score
means once the axes exist. Callers own all of that, which is what lets the same
flow serve a standalone entry point and a host framework's recipe class without
either shape leaking into the other.

The flow, per document:

1. :func:`extract_document_facts` -- the judge decomposes the document and marks
   which of its facts a good summary *must* convey. Once per document, shared by
   every candidate, which is why the method's cost does not grow with the size
   of the model field the way a per-candidate importance oracle would.
2. :func:`evaluate_candidate` -- one candidate summarizes the document and the
   judge grades it, yielding the four axes.

:func:`evaluate_document` composes the two for the common case.

Every result is returned in full rather than reduced to its axes: the summary,
the facts on both sides, and the per-fact verdicts behind each axis. The axes
are what ranks a model; the verdicts are what lets someone argue with the
ranking, and they cost nothing extra to keep since the judge already returned
them. Spend is likewise split -- generation apart from judging, per-candidate
apart from per-document -- because a shared cost divided across candidates is
the only honest way to compare them.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from .judge import Judge
from .model import Model, Usage
from .prompts import Prompt, SummaryPrompt
from .scoring import Axes
from .text import Doc, Fact, FactSource, Requirement, Summary

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocumentFacts:
    """The judge's per-document work, computed once and shared by every candidate."""

    facts: list[Fact]
    """Every atomic fact the judge found in the document."""

    salient: list[Fact]
    """The subset a good summary must convey -- the reference-free importance set."""

    salience: dict[str, bool]
    """The must-convey verdict for each document fact, keyed by fact id."""

    usage: Usage
    """What extracting and marking these facts cost. Shared across candidates."""


@dataclass(frozen=True)
class CandidateEvaluation:
    """One candidate's showing on one document: the axes, and everything behind them."""

    model: str
    axes: Axes
    summary: Summary
    summary_facts: list[Fact]

    faithful_verdicts: dict[str, bool]
    """Per summary fact: is it supported by the document? Behind ``correctness``."""

    coverage_verdicts: dict[str, bool]
    """Per document fact: does the summary convey it? Masked to the salient set."""

    precision_verdicts: dict[str, bool]
    """Per summary fact: does it land on salient material? Behind ``salient_precision``."""

    requirement_verdicts: dict[str, bool]
    """Per checklist item, keyed ``r0``, ``r1``, ...; empty with no checklist."""

    generation_usage: Usage
    """What the candidate spent writing the summary."""

    judge_usage: Usage
    """What the judge spent grading it, excluding the shared per-document work."""

    generation_seconds: float
    seconds: float


@dataclass(frozen=True)
class DocumentEvaluation:
    """Every candidate's showing on one document, plus the shared work behind them."""

    shared: DocumentFacts
    candidates: dict[str, CandidateEvaluation]

    failures: dict[str, str]
    """Candidates that failed on this document, mapped to why."""


async def extract_document_facts(doc: Doc, judge: Judge) -> DocumentFacts:
    """Decompose the document and mark which of its facts a summary must convey.

    Runs once per document however many candidates are being compared. Calling
    it again for the same document is cheap but not free -- the judge memoizes
    extraction, so the facts come back without a second call, while the salience
    pass is re-asked. Callers that need it more than once should hold the result.
    """
    usage = Usage()
    facts = await judge.facts(doc.text, FactSource.DOCUMENT, usage=usage)
    salience = await judge.mark_salient(facts, usage=usage)
    salient = [fact for fact in facts if salience[fact.id]]
    return DocumentFacts(facts=facts, salient=salient, salience=salience, usage=usage)


async def evaluate_candidate(
    doc: Doc,
    model: Model,
    judge: Judge,
    shared: DocumentFacts,
    checklist: list[Requirement],
    *,
    summary_prompt: Prompt | None = None,
) -> CandidateEvaluation:
    """Summarize one document with one candidate and derive its four axes.

    The four judge calls are independent, so they are issued concurrently; how
    many actually fly at once is the caller's business, not this function's.

    Args:
        doc: The document to summarize.
        model: The candidate being evaluated.
        judge: The judge, shared across candidates so its fact cache is too.
        shared: This document's facts and salience, from
            :func:`extract_document_facts`.
        checklist: The requirements to score against; may be empty.
        summary_prompt: What to ask the candidate. Defaults to
            :class:`SummaryPrompt`, which renders the checklist into the request
            as well as scoring against it. A host whose prompt comes from
            configuration passes its own -- in which case ``checklist`` governs
            scoring alone, and it is the caller's business whether the
            requirements reach the candidate at all.
    """
    started = time.monotonic()
    generation_usage = Usage()
    request = SummaryPrompt(doc, checklist) if summary_prompt is None else summary_prompt
    summary = Summary((await model.run(request, usage=generation_usage)).strip())
    generation_seconds = time.monotonic() - started

    judge_usage = Usage()
    summary_facts = await judge.facts(summary.text, FactSource.GENERATED, usage=judge_usage)

    # Coverage is asked over *all* document facts and then masked down to the
    # salient ones, so reference-free recall costs no judge call of its own.
    (
        (correctness, faithful_verdicts),
        (_, coverage_verdicts),
        (salient_precision, precision_verdicts),
        (requirements_met, requirement_verdicts),
    ) = await asyncio.gather(
        judge.fraction_supported(summary_facts, shared.facts, usage=judge_usage),
        judge.fraction_supported(shared.facts, summary_facts, usage=judge_usage),
        judge.fraction_supported(summary_facts, shared.salient, usage=judge_usage),
        judge.requirements_met(summary, checklist, usage=judge_usage),
    )

    salient_coverage = (
        sum(1 for fact in shared.salient if coverage_verdicts.get(fact.id, False))
        / len(shared.salient)
        if shared.salient
        else None
    )
    return CandidateEvaluation(
        model=model.name,
        axes=Axes(
            correctness=correctness,
            salient_coverage=salient_coverage,
            salient_precision=salient_precision,
            requirements_met=requirements_met,
        ),
        summary=summary,
        summary_facts=summary_facts,
        faithful_verdicts=faithful_verdicts,
        coverage_verdicts=coverage_verdicts,
        precision_verdicts=precision_verdicts,
        requirement_verdicts=requirement_verdicts,
        generation_usage=generation_usage,
        judge_usage=judge_usage,
        generation_seconds=generation_seconds,
        seconds=time.monotonic() - started,
    )


async def evaluate_document(
    doc: Doc,
    judge: Judge,
    candidates: dict[str, Model],
    checklist: list[Requirement],
) -> DocumentEvaluation:
    """Score every candidate on one document, sharing the per-document judge work.

    A candidate that fails is recorded in ``failures`` and left out of
    ``candidates`` rather than propagated: one bad response must not void a run,
    and the candidate is still ranked on the documents it managed.
    """
    shared = await extract_document_facts(doc, judge)

    names = list(candidates)
    results = await asyncio.gather(
        *(evaluate_candidate(doc, candidates[name], judge, shared, checklist) for name in names),
        return_exceptions=True,
    )

    evaluated: dict[str, CandidateEvaluation] = {}
    failures: dict[str, str] = {}
    for name, result in zip(names, results, strict=True):
        if isinstance(result, BaseException):
            _log.warning("candidate %s failed on a document, skipping it: %s", name, result)
            failures[name] = str(result)
            continue
        evaluated[name] = result
    return DocumentEvaluation(shared=shared, candidates=evaluated, failures=failures)
