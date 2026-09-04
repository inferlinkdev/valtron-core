"""The judge: every LLM call that turns text into verdicts.

One model plays the judge for a whole evaluation, and this class is everything
we ask it. Four operations cover the entire method:

* :meth:`Judge.facts` -- decompose a text into atomic facts, memoized so each
  distinct ``(text, source)`` is extracted exactly once per run.
* :meth:`Judge.fraction_supported` -- the shared "what fraction of these claims
  do those references support?" call. Three of the four metric axes are this one
  operation with the claim and reference roles swapped, which is why it is the
  only matching primitive here.
* :meth:`Judge.mark_salient` -- which of a document's own facts a good summary
  must convey. The reference-free importance oracle.
* :meth:`Judge.requirements_met` -- the fraction of the optional checklist the
  summary satisfies.

Two invariants run through all of them. Long claim lists are **chunked** (see
:data:`MAX_CLAIMS_PER_CALL`), because a judge's tendency to silently drop a
verdict scales with how many it must emit in one response. And a dropped verdict
is **re-asked, never guessed**: if a fact is still unjudged after the retries
these methods raise, because inventing a verdict would quietly distort a score
rather than fail visibly.
"""

from __future__ import annotations

import asyncio
import logging

from pydantic import BaseModel

from .model import Model, Usage
from .prompts import (
    DocumentSaliencePrompt,
    FactExtractionPrompt,
    FactMatchPrompt,
    RequirementScoringPrompt,
    normalize_verdict_id,
)
from .text import Fact, FactSource, Requirement, Summary

_log = logging.getLogger(__name__)

# Claims judged per call. Omissions are a function of how many verdicts the judge must
# emit in one structured response: they were routine above ~45 claims, so long claim
# lists are split into chunks judged concurrently. Set well below where omissions were
# observed rather than just under it. Lists at or under this size are still issued as a
# *single* call with the prompt unchanged.
MAX_CLAIMS_PER_CALL = 20

# A judge occasionally returns a well-formed verdict list that silently omits a
# fact. That is not a hard error -- re-ask for just the dropped facts, a few
# times, before giving up. One judge call covers the common (complete) case.
_MAX_VERDICT_ATTEMPTS = 3

# Per-source id prefix, so facts from different sources never collide when
# per-fact verdicts are keyed by id ("g0" generated vs "d0" document).
_ID_PREFIX = {FactSource.GENERATED: "g", FactSource.DOCUMENT: "d"}


class _ExtractedFacts(BaseModel):
    """Structured-output schema: the judge returns a flat list of fact texts."""

    facts: list[str]


class _Verdict(BaseModel):
    id: str
    supported: bool


class _Verdicts(BaseModel):
    verdicts: list[_Verdict]


class _Salience(BaseModel):
    id: str
    required: bool


class _Saliences(BaseModel):
    saliences: list[_Salience]


class _RequirementVerdict(BaseModel):
    id: str
    met: bool


class _RequirementVerdicts(BaseModel):
    verdicts: list[_RequirementVerdict]


class Judge:
    """Wraps the judge model with the four operations an evaluation needs.

    Holds a per-run fact cache, so a document decomposed for one candidate is
    reused by every other candidate summarizing it.
    """

    def __init__(self, model: Model) -> None:
        """Initialize the judge.

        Args:
            model: The judge model; must support structured output.
        """
        self._model = model
        self._fact_tasks: dict[tuple[str, FactSource], asyncio.Task[list[Fact]]] = {}

    async def facts(
        self,
        text: str,
        source: FactSource,
        *,
        attachments: list[str] | None = None,
        usage: Usage | None = None,
    ) -> list[Fact]:
        """Decompose ``text`` into atomic facts, extracting each text only once.

        The first caller for a given ``(text, source)`` starts the extraction; it
        and every later caller await the same task, so the judge decomposes each
        distinct text once even when callers race concurrently. That is something
        the on-disk response cache cannot do, since it dedupes after the call is
        made.

        ``attachments``, if given, are sent alongside ``text`` so the judge can
        decompose facts an image or PDF conveys that the text alone does not.

        Only the first caller's ``usage`` is charged, because only that call is
        actually made. That is the honest accounting: a later caller awaiting the
        same task spends nothing.
        """
        key = (text, source)
        task = self._fact_tasks.get(key)
        if task is None:
            # Store the task before the first ``await`` so concurrent callers
            # find it rather than starting a second extraction.
            task = asyncio.create_task(self._extract(text, source, attachments, usage))
            self._fact_tasks[key] = task
        return await task

    async def _extract(
        self,
        text: str,
        source: FactSource,
        attachments: list[str] | None,
        usage: Usage | None,
    ) -> list[Fact]:
        extracted = await self._model.run_structured(
            FactExtractionPrompt(text), _ExtractedFacts, attachments=attachments, usage=usage
        )
        prefix = _ID_PREFIX[source]
        return [
            Fact(id=f"{prefix}{index}", source=source, text=fact_text)
            for index, fact_text in enumerate(extracted.facts)
        ]

    async def fraction_supported(
        self, claims: list[Fact], references: list[Fact], *, usage: Usage | None = None
    ) -> tuple[float | None, dict[str, bool]]:
        """Fraction of ``claims`` the judge finds supported by ``references``.

        Returns the fraction (``None`` when there are no claims) and the per-claim
        verdicts keyed by fact id -- the caller needs those to mask coverage down
        to the salient facts.

        Raises:
            ValueError: If a claim is still unjudged after the retries.
        """
        if not claims:
            return None, {}
        supported: dict[str, bool] = {}
        for attempt in range(_MAX_VERDICT_ATTEMPTS):
            pending = [fact for fact in claims if fact.id not in supported]
            if not pending:
                break
            self._warn_retry(attempt, pending, "verdict")
            results = await asyncio.gather(
                *(
                    self._model.run_structured(
                        FactMatchPrompt(chunk, references), _Verdicts, usage=usage
                    )
                    for chunk in _chunked(pending)
                )
            )
            known = {fact.id for fact in claims}
            for result in results:
                for verdict in result.verdicts:
                    fact_id = normalize_verdict_id(verdict.id)
                    if fact_id in known:  # ids the judge invents are dropped
                        supported[fact_id] = verdict.supported
        missing = [fact.id for fact in claims if fact.id not in supported]
        if missing:
            raise ValueError(f"judge returned no verdict for fact(s): {missing}")
        verdicts = {fact.id: supported[fact.id] for fact in claims}
        return sum(verdicts.values()) / len(claims), verdicts

    async def mark_salient(
        self, document_facts: list[Fact], *, usage: Usage | None = None
    ) -> dict[str, bool]:
        """For each document fact, whether a good summary must convey it.

        This is the step that removes the need for a reference. Coverage is recall
        and recall needs an important set; every important set the research
        harness had was built from references (a human summary, or a panel of
        frontier models), which is why reference-free ranking sat at chance. One
        judge pass over the document's own facts supplies that set from the source
        alone.

        It runs **per document, not per candidate**, so its cost is amortized over
        every candidate summarizing that document.

        Raises:
            ValueError: If a fact still lacks a verdict after the retries.
        """
        if not document_facts:
            return {}
        salient: dict[str, bool] = {}
        for attempt in range(_MAX_VERDICT_ATTEMPTS):
            pending = [fact for fact in document_facts if fact.id not in salient]
            if not pending:
                break
            self._warn_retry(attempt, pending, "salience verdict")
            results = await asyncio.gather(
                *(
                    self._model.run_structured(
                        DocumentSaliencePrompt(chunk), _Saliences, usage=usage
                    )
                    for chunk in _chunked(pending)
                )
            )
            known = {fact.id for fact in document_facts}
            for result in results:
                for item in result.saliences:
                    fact_id = normalize_verdict_id(item.id)
                    if fact_id in known:
                        salient[fact_id] = item.required
        missing = [fact.id for fact in document_facts if fact.id not in salient]
        if missing:
            raise ValueError(f"judge returned no salience verdict for fact(s): {missing}")
        return {fact.id: salient[fact.id] for fact in document_facts}

    async def requirements_met(
        self,
        summary: Summary,
        requirements: list[Requirement],
        *,
        usage: Usage | None = None,
    ) -> tuple[float | None, dict[str, bool]]:
        """Fraction of ``requirements`` the summary satisfies, and the verdict for each.

        The fraction is ``None`` when there is no checklist -- undefined, which the
        score treats very differently from zero. The per-requirement verdicts are
        returned alongside it because the fraction alone cannot tell a reader
        *which* criterion a summary missed, and that is the useful part when the
        checklist is what a reviewer is arguing about.

        Raises:
            ValueError: If the judge omits a verdict for any requirement.
        """
        if not requirements:
            return None, {}
        result = await self._model.run_structured(
            RequirementScoringPrompt(str(summary), requirements),
            _RequirementVerdicts,
            usage=usage,
        )
        met_by_id = {normalize_verdict_id(verdict.id): verdict.met for verdict in result.verdicts}
        expected = [f"r{index}" for index in range(len(requirements))]
        missing = [rid for rid in expected if rid not in met_by_id]
        if missing:
            raise ValueError(f"judge returned no verdict for requirement(s): {missing}")
        verdicts = {rid: met_by_id[rid] for rid in expected}
        return sum(verdicts.values()) / len(requirements), verdicts

    @staticmethod
    def _warn_retry(attempt: int, pending: list[Fact], kind: str) -> None:
        """Log a re-ask, but only on the retries -- the first pass is the normal path."""
        if attempt:
            _log.warning(
                "judge omitted %d %s(s); re-asking (attempt %d): %s",
                len(pending),
                kind,
                attempt + 1,
                [fact.id for fact in pending],
            )


def _chunked(facts: list[Fact]) -> list[list[Fact]]:
    """Split ``facts`` into judge-sized chunks (see :data:`MAX_CLAIMS_PER_CALL`)."""
    return [
        facts[start : start + MAX_CLAIMS_PER_CALL]
        for start in range(0, len(facts), MAX_CLAIMS_PER_CALL)
    ]
