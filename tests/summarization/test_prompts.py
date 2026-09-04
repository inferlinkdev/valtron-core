"""The summarization prompt, and the template a host framework configures it by.

:data:`~standalone.prompts.SALIENCE_SUMMARY_PROMPT` and :class:`SummaryPrompt`
must render the same text, because they are the same prompt reached two ways: a
host framework wants a configurable string, this package wants an object. These
tests pin that equivalence and the exact rendered form.

Why the exact form matters: this text is the litellm cache key *and* the
configuration every published number was derived under. Changing so much as a
comma silently invalidates the cache and makes new runs incomparable with old
ones. These assertions are deliberately literal so that a change has to be
made on purpose.
"""

from __future__ import annotations

from valtron_core.summarization.prompts import (
    SALIENCE_SUMMARY_PROMPT,
    SummaryPrompt,
    TemplatePrompt,
    render_requirements,
)
from valtron_core.summarization.text import Doc, Requirement

DOC = Doc("the document text")
REQUIREMENTS = [Requirement("Name the parties."), Requirement("State the outcome.")]


def test_renders_the_instructions_then_the_document() -> None:
    assert str(SummaryPrompt(DOC, [])) == (
        "Summarize the document below. Write a single concise summary that "
        "captures its most important information.\n"
        "\n"
        "Please be as succinct as possible and only surface facts that are "
        "REALLY, REALLY important.\n"
        "\n"
        "Use ONLY information stated in the document. Do not add facts, "
        "interpretation, or background from outside it.\n"
        "\n"
        'Document:\n"""\nthe document text\n"""'
    )


def test_the_checklist_goes_in_before_the_document() -> None:
    assert str(SummaryPrompt(DOC, REQUIREMENTS)) == (
        "Summarize the document below. Write a single concise summary that "
        "captures its most important information.\n"
        "\n"
        "Please be as succinct as possible and only surface facts that are "
        "REALLY, REALLY important.\n"
        "\n"
        "Use ONLY information stated in the document. Do not add facts, "
        "interpretation, or background from outside it.\n"
        "\n"
        "Your summary must satisfy these requirements:\n"
        "- Name the parties.\n"
        "- State the outcome.\n"
        "\n"
        'Document:\n"""\nthe document text\n"""'
    )


def test_the_two_renderings_differ_by_the_checklist_block_alone() -> None:
    # The no-checklist case must leave no stray separator behind, which is the
    # one thing a naive {requirements} slot would get wrong.
    with_list = str(SummaryPrompt(DOC, REQUIREMENTS))
    without = str(SummaryPrompt(DOC, []))
    assert with_list.replace(render_requirements(REQUIREMENTS), "") == without


def test_the_template_renders_what_the_prompt_object_renders() -> None:
    # The equivalence a host framework depends on: put the constant in a config,
    # substitute the same two slots, get the prompt this package would have sent.
    rendered = SALIENCE_SUMMARY_PROMPT.replace(
        "{requirements}", render_requirements(REQUIREMENTS)
    ).replace("{content}", DOC.text)
    assert rendered == str(SummaryPrompt(DOC, REQUIREMENTS))


def test_the_template_carries_the_placeholders_a_config_validator_looks_for() -> None:
    assert "{content}" in SALIENCE_SUMMARY_PROMPT
    assert "{requirements}" in SALIENCE_SUMMARY_PROMPT


def test_braces_in_the_document_survive_rendering() -> None:
    # Substituted rather than formatted: real documents contain braces (code,
    # JSON, math), and str.format would try to interpret them.
    braced = Doc('a dict looks like {"key": value} in the text')
    assert '{"key": value}' in str(SummaryPrompt(braced, []))


def test_an_empty_checklist_renders_nothing_at_all() -> None:
    assert render_requirements([]) == ""


def test_template_prompt_substitutes_a_single_content_placeholder() -> None:
    prompt = TemplatePrompt("Summarize:\n\n{content}", "the document text")
    assert str(prompt) == "Summarize:\n\nthe document text"


def test_template_prompt_fills_named_placeholders_from_a_dict() -> None:
    template = "Transcript:\n{transcript}\n\nSpeaker notes:\n{notes}"
    content = {"transcript": "hello world", "notes": "informal tone"}
    assert str(TemplatePrompt(template, content)) == (
        "Transcript:\nhello world\n\nSpeaker notes:\ninformal tone"
    )


def test_template_prompt_blanks_a_placeholder_missing_from_the_dict() -> None:
    prompt = TemplatePrompt("{a} and {b}", {"a": "present"})
    assert str(prompt) == "present and "


def test_template_prompt_dict_braces_survive_rendering() -> None:
    prompt = TemplatePrompt("{content}", {"content": '{"key": "value"}'})
    assert str(prompt) == '{"key": "value"}'
