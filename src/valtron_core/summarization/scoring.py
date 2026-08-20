"""The ``salience-f+reqs`` metric: four axes in, one ranking out.

The scheme in one line::

    score = 0                                             if correctness < gate
          = (1-w) · F(salient_coverage, salient_precision)
            + w · requirements_met                        with a checklist
          = F(salient_coverage, salient_precision)        without one

**Faithfulness is a gate, not a term.** A summary that fails it scores zero
outright rather than trading accuracy off against coverage, because a fluent
fabrication should never outrank a dull, correct summary.

**Importance comes from the document itself.** Recall needs a target set, and
here that set is the document's own must-convey facts, so nothing outside the
source is required -- no human reference, no panel of frontier models. Precision
is the mirror: the share of the summary's facts that land on salient material.
The harmonic mean of the two is what stops the score being a length proxy, since
padding a summary raises recall but costs precision.

**The checklist is optional and enters as its own term.** With no requirements
list the score is exactly the plain salience f-measure, so a caller who has no
checklist loses only the increment it buys. Folding requirements into the
*recall* axis was tried and is destructive: literal slot-counting inverts quality
on document classes where a strong model answers a requirement abstractively,
and that inversion contaminates the whole score. As a separate term it cannot do
that, and it contributes the one thing salience lacks -- steadiness from one
document to the next, since a class-level checklist does not redraw its target
per document the way salience does.

Aggregation is a mean of per-document axes *before* scoring, not a mean of
per-document scores. That is deliberate and it matters: the salience axes carry
roughly a third of the per-document signal-to-noise of reference-based ones, and
averaging the axes over a corpus is exactly what recovers it. Ranking one
document at a time is markedly less reliable.
"""

from __future__ import annotations

from dataclasses import dataclass, fields

# Minimum faithfulness for a summary to score above zero.
DEFAULT_GATE = 0.5

# Weight on the requirements term when a checklist is supplied. From a sweep of
# the whole [0, 1] range: agreement is flat across roughly 0.6-0.95, and
# leave-one-document-set-out cross-validation picks values inside that plateau in
# every fold. Both endpoints are worse -- 0 discards the checklist, 1.0 discards
# the salience term -- which is what tells us both halves are load-bearing.
DEFAULT_REQUIREMENT_WEIGHT = 0.6

# F-measure beta; 1.0 weights coverage and precision equally.
DEFAULT_BETA = 1.0

# Score drop that starts a new tier. Zero by default: these scores are
# fine-grained, and any appreciable gap collapses the candidates into one tie.
DEFAULT_TIER_GAP = 0.0


@dataclass(frozen=True)
class Axes:
    """One summary's four metric axes, on a single document or averaged over many.

    Every field is ``None`` when the axis is undefined rather than zero -- a
    summary with no extracted facts has no precision, which is not the same as
    precision 0. Treating the two alike would make an empty summary look merely
    bad instead of unmeasurable.
    """

    correctness: float | None = None
    salient_coverage: float | None = None
    salient_precision: float | None = None
    requirements_met: float | None = None


def mean_axes(per_document: list[Axes]) -> Axes | None:
    """Average axes field by field, skipping ``None``.

    Each axis averages only the documents where it is defined, so a document that
    yielded no facts does not drag an axis toward zero. Returns ``None`` when
    there is nothing to average.
    """
    if not per_document:
        return None

    def mean(name: str) -> float | None:
        defined = [
            value for value in (getattr(axes, name) for axes in per_document) if value is not None
        ]
        return sum(defined) / len(defined) if defined else None

    return Axes(**{field.name: mean(field.name) for field in fields(Axes)})


def score(
    axes: Axes,
    *,
    gate: float = DEFAULT_GATE,
    beta: float = DEFAULT_BETA,
    requirement_weight: float = DEFAULT_REQUIREMENT_WEIGHT,
) -> float:
    """Score one candidate's axes under ``salience-f+reqs``.

    Args:
        axes: The candidate's axes, typically averaged over the document set.
        gate: Minimum ``correctness`` to score above zero.
        beta: F-measure beta; above 1 weights coverage over precision.
        requirement_weight: Weight on ``requirements_met`` when a checklist was
            supplied. Ignored when it was not, so the score falls back to the
            plain salience f-measure.

    Returns:
        The score, in ``[0, 1]``.
    """
    if axes.correctness is None or axes.correctness < gate:
        return 0.0
    salience = _f_measure(axes.salient_coverage, axes.salient_precision, beta)
    if not requirement_weight or axes.requirements_met is None:
        return salience  # no checklist, or not using one: the plain f-measure
    return (1.0 - requirement_weight) * salience + requirement_weight * axes.requirements_met


def rank(scores: dict[str, float], *, tier_gap: float = DEFAULT_TIER_GAP) -> list[list[str]]:
    """Order models by score, best first, splitting into tiers on a drop > ``tier_gap``.

    Models within a tier are being called equivalent, not merely close: with the
    default gap of zero, only an exact tie shares a tier.
    """
    ordered = sorted(scores, key=lambda model: scores[model], reverse=True)
    tiers: list[list[str]] = []
    previous: float | None = None
    for model in ordered:
        value = scores[model]
        if previous is not None and previous - value <= tier_gap:
            tiers[-1].append(model)
        else:
            tiers.append([model])
        previous = value
    return tiers


def _f_measure(recall: float | None, precision: float | None, beta: float) -> float:
    """The F-measure of the two salience axes, treating an undefined axis as zero."""
    recall = recall if recall is not None else 0.0
    precision = precision if precision is not None else 0.0
    if recall <= 0 or precision <= 0:
        return 0.0
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)
