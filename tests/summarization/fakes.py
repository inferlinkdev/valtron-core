"""Deterministic stand-ins for the judge and the candidate models.

The judge is the only component that decides anything, so faking it makes the
whole pipeline predictable without touching the network. These fakes answer by
simple, stated rules rather than by canned scripts, so a test can reason about
what the score *should* be instead of asserting an opaque constant.

The rules:

* **fact extraction** -- one fact per sentence, split on ``". "``.
* **salience** -- a document fact is must-convey when its text contains ``KEY``.
* **fact matching** -- a claim is supported when some reference fact contains it
  as a substring (after normalizing case and the trailing period).
* **requirements** -- a requirement is met when every word of it appears in the
  summary.

Every fake call also charges a fixed :data:`FAKE_CALL_COST` to whatever usage
accumulator it was handed, so a test can check that spend landed in the right
bucket by counting calls rather than by knowing a real price.
"""

from __future__ import annotations

from pydantic import BaseModel

from valtron_core.summarization.model import Model, Usage
from valtron_core.summarization.prompts import (
    DocumentSaliencePrompt,
    FactExtractionPrompt,
    FactMatchPrompt,
    Prompt,
    RequirementScoringPrompt,
    SummaryPrompt,
    TemplatePrompt,
)

SALIENCE_MARKER = "KEY"

# What a fake call "costs". Fixed and non-zero so a test can assert that spend
# was attributed to the right bucket by counting calls, rather than by knowing
# any real price.
FAKE_PROMPT_TOKENS = 10
FAKE_COMPLETION_TOKENS = 5
FAKE_CALL_COST = 0.001


def _record(name: str, usage: Usage | None) -> None:
    """Charge one fake call to ``usage``, if the caller supplied one."""
    if usage is None:
        return
    usage.record(
        name,
        prompt_tokens=FAKE_PROMPT_TOKENS,
        completion_tokens=FAKE_COMPLETION_TOKENS,
        cost_usd=FAKE_CALL_COST,
        cache_hit=False,
    )


def _sentences(text: str) -> list[str]:
    """Split ``text`` into sentence-ish fragments, the fake's notion of a fact."""
    return [part.strip() for part in text.split(". ") if part.strip()]


def _normalize(text: str) -> str:
    return text.strip().rstrip(".").lower()


class FakeJudge(Model):
    """A judge that answers by the rules in the module docstring.

    Records every prompt it renders, so tests can assert on call counts -- which
    is how the "extracted once and shared" and "chunked" behaviors are checked.
    """

    def __init__(self, name: str = "fake-judge") -> None:
        super().__init__(name)
        self.prompts: list[Prompt] = []
        self.attachments_seen: list[list[str] | None] = []

    async def run(
        self,
        prompt: Prompt,
        *,
        attachments: list[str] | None = None,
        usage: Usage | None = None,
    ) -> str:
        self.prompts.append(prompt)
        self.attachments_seen.append(attachments)
        raise NotImplementedError("the judge is only asked for structured output")

    async def run_structured[
        T: BaseModel
    ](
        self,
        prompt: Prompt,
        schema: type[T],
        *,
        attachments: list[str] | None = None,
        usage: Usage | None = None,
    ) -> T:
        self.prompts.append(prompt)
        self.attachments_seen.append(attachments)
        _record(self.name, usage)
        if isinstance(prompt, FactExtractionPrompt):
            return schema.model_validate({"facts": _sentences(_prompt_text(prompt, "_text"))})
        if isinstance(prompt, DocumentSaliencePrompt):
            facts = _prompt_facts(prompt)
            return schema.model_validate(
                {
                    "saliences": [
                        {"id": fact.id, "required": SALIENCE_MARKER in fact.text} for fact in facts
                    ]
                }
            )
        if isinstance(prompt, FactMatchPrompt):
            claims = _prompt_attr(prompt, "_claims")
            references = _prompt_attr(prompt, "_references")
            reference_text = " || ".join(_normalize(fact.text) for fact in references)
            return schema.model_validate(
                {
                    "verdicts": [
                        {
                            "id": claim.id,
                            "supported": _normalize(claim.text) in reference_text,
                        }
                        for claim in claims
                    ]
                }
            )
        if isinstance(prompt, RequirementScoringPrompt):
            summary = _prompt_text(prompt, "_summary").lower()
            requirements = _prompt_attr(prompt, "_requirements")
            return schema.model_validate(
                {
                    "verdicts": [
                        {
                            "id": f"r{index}",
                            "met": all(
                                word in summary for word in requirement.text.lower().split()
                            ),
                        }
                        for index, requirement in enumerate(requirements)
                    ]
                }
            )
        raise AssertionError(f"unexpected prompt type: {type(prompt).__name__}")

    def count(self, prompt_type: type[Prompt]) -> int:
        """How many prompts of ``prompt_type`` this judge has been asked."""
        return sum(1 for prompt in self.prompts if isinstance(prompt, prompt_type))


class OmittingJudge(FakeJudge):
    """A judge that silently drops one item from every response of a given kind.

    Reproduces the failure the retry logic exists for: a well-formed structured
    response that is simply missing a verdict. Since it drops one every time, the
    retries can never converge, which is what makes the "raise rather than guess"
    path observable.
    """

    def __init__(self, prompt_type: type[Prompt], key: str) -> None:
        super().__init__()
        self._prompt_type = prompt_type
        self._key = key

    async def run_structured[
        T: BaseModel
    ](
        self,
        prompt: Prompt,
        schema: type[T],
        *,
        attachments: list[str] | None = None,
        usage: Usage | None = None,
    ) -> T:
        result = await super().run_structured(prompt, schema, attachments=attachments, usage=usage)
        if not isinstance(prompt, self._prompt_type):
            return result
        data = result.model_dump()
        data[self._key] = data[self._key][:-1]
        return schema.model_validate(data)


class FakeSummarizer(Model):
    """A candidate that returns a fixed summary, ignoring the document."""

    def __init__(self, name: str, summary: str) -> None:
        super().__init__(name)
        self._summary = summary
        self.calls = 0
        self.attachments_seen: list[list[str] | None] = []
        self.rendered_prompts: list[str] = []

    async def run(
        self,
        prompt: Prompt,
        *,
        attachments: list[str] | None = None,
        usage: Usage | None = None,
    ) -> str:
        # Either shape of summarization request: built in code, or rendered
        # from a host application's configured template.
        assert isinstance(
            prompt, (SummaryPrompt, TemplatePrompt)
        ), f"expected a summarization prompt, got {type(prompt).__name__}"
        self.calls += 1
        self.attachments_seen.append(attachments)
        self.rendered_prompts.append(str(prompt))
        _record(self.name, usage)
        return self._summary


class FailingSummarizer(Model):
    """A candidate that always fails, for the drop-a-failing-candidate path."""

    def __init__(self, name: str = "fake/raises") -> None:
        super().__init__(name)

    async def run(
        self,
        prompt: Prompt,
        *,
        attachments: list[str] | None = None,
        usage: Usage | None = None,
    ) -> str:
        raise RuntimeError("this model always fails")


class FakeFactory:
    """Stands in for :class:`standalone.litellm_model.ModelFactory`, handing out fakes.

    Accepts and ignores the runtime keyword arguments the real factory takes, so
    a test can swap it in without the pipeline knowing.
    """

    def __init__(self, models: dict[str, Model]) -> None:
        self._models = models
        self._usage = Usage()

    def __call__(self, **_kwargs: object) -> FakeFactory:
        """Allow the class itself to be monkeypatched in as the constructor."""
        return self

    def make(self, model_id: str, *, name: str | None = None) -> Model:
        return self._models[model_id]

    @property
    def usage(self) -> Usage:
        return self._usage


def _prompt_text(prompt: Prompt, attribute: str) -> str:
    value = getattr(prompt, attribute)
    assert isinstance(value, str)
    return value


def _prompt_attr(prompt: Prompt, attribute: str) -> list:
    value = getattr(prompt, attribute)
    assert isinstance(value, list)
    return value


def _prompt_facts(prompt: Prompt) -> list:
    return _prompt_attr(prompt, "_facts")
