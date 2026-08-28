"""The five prompts a ``salience-f+reqs`` evaluation issues.

Each subclasses :class:`Prompt` and renders its data through ``__str__``. The
rendered text is the litellm cache key, so **these strings are byte-for-byte
identical to the ones the research harness used**: changing so much as a comma
invalidates every cached response and turns a free re-run into a paid one.
Treat the prose as a fixed interface, not as something to tidy.

The five, in the order an evaluation issues them:

* :class:`SummaryPrompt` -- a candidate summarizes the document.
* :class:`FactExtractionPrompt` -- the judge decomposes a text into atomic facts.
* :class:`DocumentSaliencePrompt` -- the judge marks which document facts a good
  summary *must* convey. This is the reference-free importance oracle, and the
  reason the whole method needs no gold summary and no model panel.
* :class:`FactMatchPrompt` -- the judge decides which claims a reference set
  supports. Three of the four axes are this one prompt with the claim and
  reference roles swapped.
* :class:`RequirementScoringPrompt` -- the judge scores the summary against the
  optional per-class checklist.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod

from .text import Doc, Fact, Requirement


class Prompt(ABC):
    """Base class for a prompt sent to a :class:`~valtron_core.summarization.model.Model`.

    Subclasses carry whatever structured data their use case needs and render it
    to text in ``__str__``, keeping prompt construction next to the data it
    formats rather than scattering string assembly through the pipeline.
    """

    @abstractmethod
    def __str__(self) -> str:
        """Render this prompt to the text sent to the model."""


def normalize_verdict_id(raw: str) -> str:
    """Normalize a judge-returned verdict id to the bare token we keyed on.

    Verdicts come back keyed by the ids rendered into the prompt (``r0``, ``g0``,
    ...). A judge occasionally echoes an id with surrounding brackets (``[r0]``)
    or stray whitespace; strip those so the verdict still matches the bare id its
    requirement or fact was assigned.
    """
    return raw.strip().strip("[]").strip()


SALIENCE_SUMMARY_PROMPT = (
    "Summarize the document below. Write a single concise summary that "
    "captures its most important information.\n"
    "\n"
    "Please be as succinct as possible and only surface facts that are "
    "REALLY, REALLY important.\n"
    "\n"
    "Use ONLY information stated in the document. Do not add facts, "
    "interpretation, or background from outside it.\n"
    "\n"
    "{requirements}"
    'Document:\n"""\n{content}\n"""'
)
"""The summarization prompt, as a template over ``{requirements}`` and ``{content}``.

Exported because a host framework wants the prompt as a *string* it can put in
a config, show in a report, and let a caller override, while this package wants
it as a rendered :class:`SummaryPrompt`. Both come from here, so the two cannot
drift apart -- which matters more than it sounds, since this text is the litellm
cache key and the configuration every published number was derived under.

``{requirements}`` expands to :func:`render_requirements` (empty with no
checklist, trailing blank line included) and ``{content}`` to the document.
"""


def render_requirements(requirements: list[Requirement]) -> str:
    """Render the requirements block, or ``""`` when there is no checklist.

    Carries its own trailing blank line so that the no-checklist case leaves no
    stray separator behind -- the two renderings differ by exactly this block
    and nothing else.
    """
    if not requirements:
        return ""
    lines = ["Your summary must satisfy these requirements:"]
    lines.extend(f"- {requirement.text}" for requirement in requirements)
    return "\n".join(lines) + "\n\n"


class SummaryPrompt(Prompt):
    """Asks a model to summarize a document under closed-world constraints.

    Renders :data:`SALIENCE_SUMMARY_PROMPT` with the document text and an
    optional requirements list. The requirements block is omitted when there
    are none.
    """

    def __init__(self, doc: Doc, requirements: list[Requirement]) -> None:
        self._doc = doc
        self._requirements = requirements

    def __str__(self) -> str:
        # Substituted rather than ``format``ted: a document is arbitrary text and
        # routinely contains braces, which ``format`` would try to interpret.
        return SALIENCE_SUMMARY_PROMPT.replace(
            "{requirements}", render_requirements(self._requirements)
        ).replace("{content}", self._doc.text)


class TemplatePrompt(Prompt):
    """A prompt rendered from a caller-supplied template over ``{content}``.

    For a host application whose prompt arrives as configuration rather than as
    code. :data:`SALIENCE_SUMMARY_PROMPT` is the template to pass to reproduce
    what :class:`SummaryPrompt` sends; anything else is a deliberate deviation
    from the configuration the method was validated under.

    ``content`` is a plain string for a single ``{content}`` placeholder, or a
    dict for a template with several named placeholders, one per key.
    """

    def __init__(self, template: str, content: str | dict[str, str | None]) -> None:
        self._template = template
        self._content = content

    def __str__(self) -> str:
        if isinstance(self._content, str):
            # Substituted rather than ``format``ted, for the same reason as
            # SummaryPrompt: documents routinely contain braces.
            return self._template.replace("{content}", self._content)

        result = self._template
        for key in set(re.findall(r"\{(\w+)\}", result)):
            result = result.replace(f"{{{key}}}", self._content.get(key) or "")
        return result


class FactExtractionPrompt(Prompt):
    """Asks the judge to decompose a piece of text into atomic facts."""

    def __init__(self, text: str) -> None:
        self._text = text

    def __str__(self) -> str:
        return (
            "Decompose the text below into a list of atomic facts.\n\n"
            "Each fact must:\n"
            "- express a single, self-contained piece of information,\n"
            "- be decontextualized: understandable on its own, without pronouns or\n"
            "  references that depend on the surrounding text,\n"
            "- state only what the text says — do not add, infer, or omit information.\n\n"
            f'Text:\n"""\n{self._text}\n"""'
        )


class FactMatchPrompt(Prompt):
    """Asks the judge which claims are supported by a set of reference facts.

    Three axes are this one prompt with the roles swapped: summary facts against
    document facts gives faithfulness; document facts against summary facts
    gives coverage; summary facts against the *salient* document facts gives
    reference-free precision. Claims carry their ids so verdicts come back keyed
    by id.
    """

    def __init__(self, claims: list[Fact], references: list[Fact]) -> None:
        self._claims = claims
        self._references = references

    def __str__(self) -> str:
        references = "\n".join(f"- {fact.text}" for fact in self._references) or "(none)"
        claims = "\n".join(f"- {fact.id}: {fact.text}" for fact in self._claims)
        return (
            "You are checking whether each claim is supported by a set of "
            "reference facts.\n\n"
            f"Reference facts:\n{references}\n\n"
            "For each claim below, decide whether it is supported (entailed) by "
            "the reference facts — i.e. its information is stated by, or follows "
            "from, them.\n\n"
            f"Claims:\n{claims}\n\n"
            "Return a verdict for each claim, identified by its id."
        )


class DocumentSaliencePrompt(Prompt):
    """Asks the judge which of a document's own facts a good summary must convey.

    The *reference-free* importance oracle: it stands in for the reference
    summaries whose facts would otherwise define the important set, which is what
    makes coverage computable with no reference and no requirements list.

    Each fact is judged on an **absolute** test -- would a reader be materially
    misinformed if a summary omitted it -- rather than ranked against the other
    facts. That matters mechanically: an absolute test means a long fact list can
    be judged in independent chunks (see
    :data:`~valtron_core.summarization.judge.MAX_CLAIMS_PER_CALL`) without the question changing
    meaning when the judge sees only part of the list. It deliberately says
    nothing about the document class or its requirements, so the axis it feeds
    stays independent of the optional checklist.
    """

    def __init__(self, facts: list[Fact]) -> None:
        self._facts = facts

    def __str__(self) -> str:
        facts = "\n".join(f"- {fact.id}: {fact.text}" for fact in self._facts)
        return (
            "A good summary is far shorter than the document it summarizes, so most of "
            "a document's facts are detail that a summary rightly leaves out.\n\n"
            "For each document fact below, decide whether a good, self-contained "
            "summary of this document *must* convey it — that is, whether a reader "
            "given a summary that omitted it would be materially misinformed or would "
            "miss the point of the document.\n\n"
            "Mark as required: the document's central subject and purpose; its main "
            "findings, conclusions, or outcomes; and any fact without which the rest "
            "would mislead.\n"
            "Do not mark: supporting detail; individual figures or examples that only "
            "illustrate a point already covered; procedural or background asides; and "
            "anything a reader could do without and still understand the document.\n\n"
            "Judge each fact on its own against that test — do not aim for any "
            "particular number of facts, and do not rank the facts against each "
            "other.\n\n"
            f"Document facts:\n{facts}\n\n"
            "Return a verdict for each fact, identified by its id."
        )


class RequirementScoringPrompt(Prompt):
    """Asks the judge whether the summary satisfies each requirement.

    Requirements are rendered with positional ids (``r0``, ``r1``, ...) so
    verdicts come back keyed by id; the caller assigns the same ids by
    enumerating the requirements in the same order.
    """

    def __init__(self, summary: str, requirements: list[Requirement]) -> None:
        self._summary = summary
        self._requirements = requirements

    def __str__(self) -> str:
        requirements = "\n".join(
            f"- r{index}: {requirement.text}"
            for index, requirement in enumerate(self._requirements)
        )
        return (
            "Score the summary below against each requirement. A summary "
            "satisfies a requirement when it does what the requirement asks.\n\n"
            f"Requirements:\n{requirements}\n\n"
            f'Summary:\n"""\n{self._summary}\n"""\n\n'
            "Return a verdict for each requirement, identified by its id."
        )
