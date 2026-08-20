"""The text value types: documents, summaries, requirements, and atomic facts.

All four are frozen dataclasses that carry their text and render it through
``__str__``, so a prompt can interpolate any of them directly. They are
deliberately thin -- this module defines no behavior beyond identity and
rendering, because everything that *interprets* text is a judge call.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True)
class Doc:
    """A source document to be summarized."""

    text: str

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class Summary:
    """A model-generated summary of a document."""

    text: str

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class Requirement:
    """One criterion a summary should satisfy.

    A requirement describes a kind of fact a good summary of this *document
    class* should contain ("name the director and lead actors"). It is authored
    once per class, not per document -- which is exactly why it steadies the
    score from one document to the next.
    """

    text: str

    def __str__(self) -> str:
        return self.text


class FactSource(Enum):
    """Where a fact was extracted from."""

    GENERATED = "generated"
    DOCUMENT = "document"


@dataclass(frozen=True)
class Fact:
    """A single atomic fact extracted from a summary or document.

    What makes a fact *atomic* and *decontextualized* is defined operationally
    by the extraction prompt, not by this type. Here a fact is its text, a tag
    recording where it came from, and a stable ``id`` so per-fact judge verdicts
    key back to it unambiguously rather than by list position -- which matters,
    because long fact lists are judged in several concurrent calls.
    """

    id: str
    source: FactSource
    text: str

    def __str__(self) -> str:
        return self.text
