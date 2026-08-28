"""Reference-free summarization quality: a sibling of ``ReferencedEval``.

``ReferencedEval`` scores a prediction against a known label, exactly or per
field. Summarization has no such label -- a good summary can vary in wording,
compression and structure while still covering everything it owes the reader --
so this recipe extends ``ModelEval`` directly instead, and is the reason that
base class is correctness-agnostic.

What it does per document: a judge decomposes the source into atomic facts and
marks which of them a good summary *must* convey. That happens once and is
shared by every candidate, which is why the cost does not grow with the size of
the model field. Each candidate then summarizes the document and the judge
grades it on four axes -- faithfulness to the source, coverage of the
must-convey facts, precision against them, and an optional per-class checklist.
The method and the metric live in :mod:`valtron_core.summarization`; this module
is the wiring that makes them a recipe.

How it uses the base class's seams:

* ``task_scores`` carries the four axes per document, so
  ``aggregated_task_scores`` gives the corpus-level axes for free -- and those
  are what the score is computed from. Note the ordering: axes are averaged over
  the corpus *first* and scored once, never the mean of per-document scores. A
  single document rarely separates two competent models.
* ``expected_value``, ``is_correct`` and ``example_score`` stay unset. There is
  no ground truth and no binary notion of correct, and faking one would be worse
  than leaving them empty.
* ``compute_task_statistics`` produces the cross-model ranking, which is
  corpus-level and so fits nowhere in the per-model ``EvaluationMetrics``.
"""

import asyncio
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import structlog
from tqdm import tqdm  # type: ignore[import-untyped]

from valtron_core.attachments import check_attachment_support
from valtron_core.cost_utils import _fallback_cost, _parse_time_unit_to_seconds
from valtron_core.evaluation.config import BaseRecipeConfig, SummarizationConfig
from valtron_core.evaluation.model_eval import ModelEval
from valtron_core.models import Document, EvaluationResult, PredictionResult
from valtron_core.summarization import (
    Axes,
    ClientModel,
    Doc,
    DocumentFacts,
    Judge,
    Model,
    Prompt,
    Requirement,
    Summary,
    TemplatePrompt,
    Usage,
    evaluate_candidate,
    extract_document_facts,
    mean_axes,
    rank,
    render_requirements,
    score,
)

if TYPE_CHECKING:
    from valtron_core.reports.generate_summarization_report import (
        SummarizationReportGenerator,
    )

logger = structlog.get_logger()

#: Label the judge's spend is reported under, kept apart from the candidates.
JUDGE_LABEL = "judge"

#: The four axes, in the order a report should read them.
AXIS_NAMES = ("correctness", "salient_coverage", "salient_precision", "requirements_met")


@dataclass(frozen=True)
class SummarizationScore:
    """One candidate's score and the axes it came from."""

    model: str
    score: float
    correctness: float | None
    salient_coverage: float | None
    salient_precision: float | None
    requirements_met: float | None
    documents_scored: int

    def axes(self) -> dict[str, float | None]:
        """The four axes as a mapping, in reading order."""
        return {name: getattr(self, name) for name in AXIS_NAMES}


@dataclass(frozen=True)
class SummarizationRanking:
    """The corpus-level outcome: who won, by how much, and what it cost.

    Returned in full rather than reduced to an ordering because the score alone
    cannot say *why* a model placed where it did -- a zero from a failed
    faithfulness gate means something entirely different from a zero from thin
    coverage.
    """

    tiers: list[list[str]]
    """Model labels in tiers, best first. Sharing a tier means tied, not close."""

    scores: list[SummarizationScore]
    """Per-model scores and axes, best first."""

    parameters: dict[str, float]
    """The scheme's four scalars, so the number can be reproduced."""

    usage: dict[str, Any]
    """Calls, tokens and cost, split by what incurred them."""

    @property
    def best(self) -> list[str]:
        """The top tier -- usually one model, more only on an exact tie."""
        return self.tiers[0] if self.tiers else []

    @property
    def ranked_models(self) -> list[str]:
        """Every model, best first, flattened across tiers."""
        return [model for tier in self.tiers for model in tier]

    def to_dict(self) -> dict[str, Any]:
        """A JSON-serializable view, as ``compute_task_statistics`` returns."""
        return {
            "scheme": "salience-f+reqs",
            "parameters": self.parameters,
            "tiers": self.tiers,
            "best": self.best,
            "models": {
                entry.model: {
                    "score": entry.score,
                    "axes": entry.axes(),
                    "documents_scored": entry.documents_scored,
                }
                for entry in self.scores
            },
            "usage": self.usage,
        }


def _usage_dict(usage: Usage) -> dict[str, Any]:
    """One accumulator as plain data."""
    return {
        "calls": usage.calls,
        "cache_hits": usage.cache_hits,
        "prompt_tokens": usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "total_tokens": usage.total_tokens,
        "cost_usd": usage.cost_usd,
        "by_model": dict(usage.by_model),
    }


class _StoredSummary(Model):
    """Replays an already-generated summary instead of calling an LLM.

    Lets ``evaluate_candidate`` be reused unchanged for a regrade: it still calls
    ``model.run()`` once, but this returns the stored ``predicted_value`` directly and
    records no usage, so regrading pays for judge calls only, never a second generation
    call.
    """

    def __init__(self, name: str, summary: str) -> None:
        super().__init__(name)
        self._summary = summary

    async def run(
        self,
        prompt: Prompt,
        *,
        attachments: list[str] | None = None,
        usage: Usage | None = None,
    ) -> str:
        return self._summary


class SummarizationExperiment(ModelEval):
    """Rank summarization models on a corpus, with no reference summaries.

    Usage mirrors every other recipe::

        from valtron_core.evaluation import SummarizationExperiment

        experiment = SummarizationExperiment(
            config={
                "models": [{"name": "gpt-4o-mini"}, {"name": "gemini/gemini-2.5-flash"}],
                "judge_model": "gemini/gemini-2.5-pro",
                "requirements": ["Name the parties.", "State the outcome."],
                "output_dir": "./results",
            },
            data=[{"id": "0001", "content": "..."}],
        )
        experiment.evaluate()
        print(experiment.ranking.best)

    ``data`` needs only ``id`` and ``content``; a ``label`` is ignored if present.
    ``prompt`` defaults to ``SALIENCE_SUMMARY_PROMPT``; pass your own to deviate
    from the configuration this method was validated under.
    """

    _settings: SummarizationConfig
    _judge: Judge
    _checklist: list[Requirement]
    _document_facts: dict[str, DocumentFacts]
    _ranking: SummarizationRanking | None
    _recommendation: str | None

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    @classmethod
    def _config_model(cls) -> type[BaseRecipeConfig]:
        return SummarizationConfig

    def _post_init(self) -> None:
        if not isinstance(self.config, SummarizationConfig):
            raise TypeError(
                f"{type(self).__name__} requires a SummarizationConfig, "
                f"got {type(self.config).__name__}"
            )
        self._settings = self.config
        self._checklist = [Requirement(text) for text in self._settings.requirements]
        self._document_facts = {}
        self._ranking = None
        self._recommendation = None

        # Three accumulators rather than one, because "what did this cost?" has
        # three different answers worth telling apart: what the candidates spent
        # writing, what the judge spent grading each of them, and the
        # per-document work every candidate shared.
        self._generation_usage = Usage()
        self._candidate_judge_usage = Usage()
        self._shared_judge_usage = Usage()

        self._judge = Judge(
            ClientModel(self._settings.judge_model, client=self.client, name=JUDGE_LABEL)
        )
        # How many models split each document's shared judge cost. Set per pass
        # in _run_evaluations, since add_models() + evaluate() can run a subset.
        self._models_in_pass = max(len(self.models), 1)

    # -------------------------------------------------------------------------
    # Persistence: round-tripping config through load_experiment_results()
    # -------------------------------------------------------------------------

    def _extra_metadata(self) -> dict[str, Any]:
        """Persist the scheme and judge config so a reload can reproduce this run's score.

        Every one of these changes the axes or how they combine into a score, so
        without them a reloaded instance would silently fall back to
        ``SummarizationConfig``'s defaults instead of what this run actually used.
        """
        return {
            "judge_model": self._settings.judge_model,
            "requirements": list(self._settings.requirements),
            "gate": self._settings.gate,
            "beta": self._settings.beta,
            "requirement_weight": self._settings.requirement_weight,
            "tier_gap": self._settings.tier_gap,
            "max_concurrent_documents": self._settings.max_concurrent_documents,
        }

    @classmethod
    def _restore_config(cls, meta: dict[str, Any]) -> dict[str, Any]:
        """Read back whatever ``_extra_metadata()`` wrote.

        A run saved before this existed has no ``task_config`` key, so this
        returns ``{}`` and ``SummarizationConfig``'s field defaults apply --
        the same (imperfect) behavior reloading has always had, not a new failure.
        """
        return dict(meta.get("task_config") or {})

    # -------------------------------------------------------------------------
    # Preflight
    # -------------------------------------------------------------------------

    def _validate_task_data(self) -> None:
        documents, _ = self._load_documents_and_labels()

        blank = [d.id for d in documents if not self._flatten_content(d.content).strip()]
        if blank:
            raise ValueError(f"These documents have no content to summarize: {blank}")

        # A dict document is shown to the candidate through its own named
        # placeholders, so it need not use {content} at all. The single-string
        # case has no other way to reach the candidate, so it is required there.
        has_dict_content = any(isinstance(d.content, dict) for d in documents)
        if not has_dict_content and "{content}" not in self.prompt_template:
            raise ValueError(
                "The prompt must contain a {content} placeholder for the document. "
                "Pass valtron_core.summarization.SALIENCE_SUMMARY_PROMPT to use the "
                "prompt this method was validated under."
            )
        if self._checklist and "{requirements}" not in self.prompt_template:
            logger.warning(
                "requirements_not_in_prompt",
                note=(
                    "requirements will be scored but not shown to the candidates; add a "
                    "{requirements} placeholder to match the validated configuration"
                ),
                count=len(self._checklist),
            )
        if any(getattr(m, "type", "llm") != "llm" for m in self.models):
            raise ValueError(
                "Summarization supports LLM models only; transformer models cannot "
                "generate free text."
            )

        # The judge reads every document (fact extraction, salience marking), and
        # every candidate reads it too (writing the summary), so both need to
        # support whatever attachment types are present.
        check_attachment_support(documents, self._settings.judge_model)
        for model_config in self.models:
            model_name = getattr(model_config, "name", None) or model_config.label
            check_attachment_support(documents, model_name)

    # -------------------------------------------------------------------------
    # Shared per-document work
    # -------------------------------------------------------------------------

    async def _before_evaluation(self, field_metrics_config: Any) -> None:
        """Extract each document's facts and mark which of them a summary must convey.

        Once per document, shared by every candidate. Done here rather than
        lazily so the two phases are visible in the run, and so the shared spend
        is known before it has to be divided.

        Skips any document already in ``self._document_facts``, so a repeat
        call on the same instance (e.g. ``add_models()`` followed by another
        ``evaluate()``) does not re-run salience marking, which is not
        memoized by the judge and would otherwise be paid for twice.
        """
        documents, _ = self._load_documents_and_labels()
        to_extract = [d for d in documents if d.id not in self._document_facts]
        if not to_extract:
            return
        self._status(f"Extracting facts from {len(to_extract)} documents...")
        semaphore = asyncio.Semaphore(self._settings.max_concurrent_documents)

        async def extract(document: Document) -> tuple[str, DocumentFacts]:
            async with semaphore:
                return document.id, await extract_document_facts(
                    Doc(self._document_text(document), attachments=document.attachments),
                    self._judge,
                )

        with tqdm(total=len(to_extract), unit="doc", desc="Reading documents") as bar:
            tasks = [asyncio.create_task(extract(document)) for document in to_extract]
            for finished in asyncio.as_completed(tasks):
                document_id, facts = await finished
                self._document_facts[document_id] = facts
                self._shared_judge_usage.merge(facts.usage)
                bar.update(1)

        logger.info(
            "document_facts_extracted",
            documents=len(self._document_facts),
            salient=sum(len(f.salient) for f in self._document_facts.values()),
            cost=self._shared_judge_usage.cost_usd,
        )

    @staticmethod
    def _flatten_content(content: str | dict[str, str | None]) -> str:
        """Join a document's content into the single text the judge reads.

        A document's ``content`` can be a plain string or a dict of named
        placeholder values. Either way, the judge needs one piece of text to
        decompose into facts, so a dict is flattened into ``key: value``
        lines, skipping blank values. This is the full document regardless of
        which of its keys the prompt actually shows the candidate.
        """
        if isinstance(content, str):
            return content
        return "\n\n".join(f"{key}: {value}" for key, value in content.items() if value)

    @staticmethod
    def _document_text(document: Document) -> str:
        return SummarizationExperiment._flatten_content(document.content)

    # -------------------------------------------------------------------------
    # Prompt preparation
    # -------------------------------------------------------------------------

    async def _prepare_model_prompts(self) -> dict[str, str]:
        """Fill in the ``{requirements}`` block; leave ``{content}`` for evaluation time.

        Done here, rather than when the call is made, so that the checklist is
        part of the prompt that gets persisted and displayed -- what a reader of
        the run sees is then what the candidate was actually asked.
        """
        prompts = await super()._prepare_model_prompts()
        block = render_requirements(self._checklist)
        return {label: prompt.replace("{requirements}", block) for label, prompt in prompts.items()}

    def _cache_key_inputs(self, model_config: Any, prompt: str) -> tuple[str, dict[str, Any]]:
        """Fold the judge and the checklist into what makes a cached prediction reusable.

        Both change the axes, so a prediction produced under a different judge or
        a different checklist is not the same prediction, even though the
        candidate was asked the same question.
        """
        prompt_key, params = super()._cache_key_inputs(model_config, prompt)
        return prompt_key, {
            **params,
            "judge_model": self._settings.judge_model,
            "requirements": list(self._settings.requirements),
        }

    # -------------------------------------------------------------------------
    # Per-model evaluation
    # -------------------------------------------------------------------------

    async def _run_evaluations(
        self,
        model_prompts: dict[str, str],
        field_metrics_config: Any = None,
        models: list[Any] | None = None,
    ) -> tuple[list[EvaluationResult], dict[str, list[Any]]]:
        """Record how many models are in this pass, then run the shared machinery.

        The count is the divisor for each document's shared judge cost, and it is
        the number of models *actually running now* rather than the number
        configured: ``add_models()`` followed by another ``evaluate()`` re-does
        the shared work for the new models alone, and they alone should pay.
        """
        self._models_in_pass = max(len(models if models is not None else self.models), 1)
        return await super()._run_evaluations(model_prompts, field_metrics_config, models)

    async def _evaluate_model_documents(
        self,
        model_config: Any,
        documents: list[Document],
        labels: list[Any],
        prompt: str,
        field_metrics_config: Any,
        on_document_complete: Callable[[PredictionResult], None] | None = None,
        progress_bar: tqdm | None = None,
    ) -> tuple[EvaluationResult, str | None]:
        model_name = getattr(model_config, "name", None) or model_config.label
        model_label = model_config.label or model_name
        model_arg = self._build_model_arg(model_config)
        model = ClientModel(model_arg, client=self.client, name=model_label)

        result = EvaluationResult(
            run_id=str(uuid.uuid4()),
            started_at=datetime.now(),
            prompt_template=prompt,
            model=model_label,
            llm_config=model_arg,
            status="running",
        )

        semaphore = asyncio.Semaphore(self._settings.max_concurrent_documents)

        async def one(document: Document) -> PredictionResult:
            async with semaphore:
                prediction = await self._evaluate_one(document, model, model_config, prompt)
            if on_document_complete is not None:
                on_document_complete(prediction)
            if progress_bar is not None:
                progress_bar.update(1)
            return prediction

        # Gathered so predictions keep the input document order, which is what
        # makes a saved run diffable against another.
        for prediction in await asyncio.gather(*(one(document) for document in documents)):
            result.add_prediction(prediction)

        result.completed_at = datetime.now()
        result.status = "completed"
        if result.predictions:
            result.compute_metrics()
        return result, None

    async def _evaluate_one(
        self,
        document: Document,
        model: ClientModel,
        model_config: Any,
        prompt: str,
    ) -> PredictionResult:
        """Summarize and grade one document, as one ``PredictionResult``."""
        text = self._document_text(document)
        shared = self._document_facts[document.id]

        try:
            evaluation = await evaluate_candidate(
                Doc(text, attachments=document.attachments),
                model,
                self._judge,
                shared,
                self._checklist,
                summary_prompt=TemplatePrompt(prompt, document.content),
            )
        except Exception as error:
            # One document must not void a model's whole run: record the failure
            # and let it be ranked on the documents it managed. An unscored
            # document contributes to no axis, rather than contributing a zero.
            logger.warning(
                "summarization_document_failed",
                model=model.name,
                document=document.id,
                error=str(error),
            )
            return PredictionResult(
                document_id=document.id,
                predicted_value=f"ERROR: {error}",
                error=str(error),
                response_time=0.0,
                model=model.name,
                metadata={"content": document.content, "error": str(error)},
            )

        self._generation_usage.merge(evaluation.generation_usage)
        self._candidate_judge_usage.merge(evaluation.judge_usage)

        original_cost = evaluation.generation_usage.cost_usd
        llm_cost = self._effective_cost(model_config, original_cost, evaluation.generation_seconds)
        # The shared per-document judge work is divided evenly: it belongs to no
        # single candidate, but leaving it out would make the run's total cost
        # understate what was actually spent.
        evaluation_cost = (
            evaluation.judge_usage.cost_usd + shared.usage.cost_usd / self._models_in_pass
        )

        axes = evaluation.axes
        task_scores = {
            name: value for name in AXIS_NAMES if (value := getattr(axes, name)) is not None
        }

        return PredictionResult(
            document_id=document.id,
            predicted_value=evaluation.summary.text,
            # No ground truth, so expected_value/is_correct/example_score stay unset.
            task_scores=task_scores or None,
            response_time=evaluation.seconds,
            original_cost=original_cost,
            llm_cost=llm_cost,
            evaluation_cost=evaluation_cost,
            model=model.name,
            metadata={
                # Required: _run_evaluations hashes this to decide what a resumed
                # run can skip. Without it, partial-result caching silently no-ops.
                "content": document.content,
                "generation_seconds": evaluation.generation_seconds,
                "document_facts": [fact.text for fact in shared.facts],
                "salient_facts": [fact.text for fact in shared.salient],
                "summary_facts": [fact.text for fact in evaluation.summary_facts],
                "faithful_verdicts": evaluation.faithful_verdicts,
                "coverage_verdicts": evaluation.coverage_verdicts,
                "precision_verdicts": evaluation.precision_verdicts,
                "requirement_verdicts": evaluation.requirement_verdicts,
                "requirements": list(self._settings.requirements),
                "generation_usage": _usage_dict(evaluation.generation_usage),
                "judge_usage": _usage_dict(evaluation.judge_usage),
            },
        )

    def _effective_cost(self, model_config: Any, reported: float, seconds: float) -> float:
        """Resolve inference cost: user ``cost_rate`` > litellm pricing > time estimate.

        The same precedence the rest of this codebase uses, so a self-hosted
        model that litellm cannot price is still comparable with a hosted one.
        """
        cost_rate = getattr(model_config, "cost_rate", None)
        if cost_rate is not None:
            unit_seconds = _parse_time_unit_to_seconds(
                getattr(model_config, "cost_rate_time_unit", "1hr")
            )
            return float(cost_rate) * (seconds / unit_seconds)
        if reported == 0.0:
            return _fallback_cost(self._build_model_arg(model_config), seconds)
        return reported

    # -------------------------------------------------------------------------
    # Corpus-level statistics
    # -------------------------------------------------------------------------

    def compute_task_statistics(self, results: list[EvaluationResult]) -> dict[str, Any]:
        """Score the aggregated axes and rank the models.

        Built from each prediction's ``task_scores`` rather than from state held
        on this instance, so it works just as well on a run reloaded from disk
        with ``load_experiment_results()``.

        Note the order of operations: the axes are averaged across the corpus and
        the score computed once from those averages. Scoring each document and
        averaging the scores is a different -- and markedly noisier -- number,
        because the salience axes carry roughly a third of the per-document
        signal of reference-based ones.
        """
        settings = self._settings
        by_model: dict[str, float] = {}
        entries: list[SummarizationScore] = []

        for result in results:
            scored = [p.task_scores for p in result.predictions if p.task_scores]
            aggregate = mean_axes([_axes_from(s) for s in scored]) or Axes()
            value = score(
                aggregate,
                gate=settings.gate,
                beta=settings.beta,
                requirement_weight=settings.requirement_weight,
            )
            by_model[result.model] = value
            entries.append(
                SummarizationScore(
                    model=result.model,
                    score=value,
                    correctness=aggregate.correctness,
                    salient_coverage=aggregate.salient_coverage,
                    salient_precision=aggregate.salient_precision,
                    requirements_met=aggregate.requirements_met,
                    documents_scored=len(scored),
                )
            )

        entries.sort(key=lambda entry: entry.score, reverse=True)
        self._ranking = SummarizationRanking(
            tiers=rank(by_model, tier_gap=settings.tier_gap),
            scores=entries,
            parameters={
                "gate": settings.gate,
                "beta": settings.beta,
                "requirement_weight": settings.requirement_weight,
                "tier_gap": settings.tier_gap,
            },
            usage={
                "generation": _usage_dict(self._generation_usage),
                "judge_per_candidate": _usage_dict(self._candidate_judge_usage),
                "judge_shared": _usage_dict(self._shared_judge_usage),
                "judge_model": settings.judge_model,
            },
        )
        return self._ranking.to_dict()

    # -------------------------------------------------------------------------
    # Reevaluation
    # -------------------------------------------------------------------------

    def reevaluate(  # type: ignore[override]
        self,
        *,
        judge_model: str | None = None,
        requirements: list[str] | None = None,
        gate: float | None = None,
        beta: float | None = None,
        requirement_weight: float | None = None,
        tier_gap: float | None = None,
        output_dir: str | Path | None = None,
    ) -> Path | None:
        """Reweight and/or regrade stored predictions, synchronously (wraps ``areevaluate()``).

        Cannot be called from within a running event loop; call ``areevaluate()``
        directly (with ``await``) there instead -- the same rule ``evaluate()`` follows
        for ``aevaluate()``.
        """
        return asyncio.run(
            self.areevaluate(
                judge_model=judge_model,
                requirements=requirements,
                gate=gate,
                beta=beta,
                requirement_weight=requirement_weight,
                tier_gap=tier_gap,
                output_dir=output_dir,
            )
        )

    async def areevaluate(
        self,
        *,
        judge_model: str | None = None,
        requirements: list[str] | None = None,
        gate: float | None = None,
        beta: float | None = None,
        requirement_weight: float | None = None,
        tier_gap: float | None = None,
        output_dir: str | Path | None = None,
    ) -> Path | None:
        """Reweight and/or regrade stored predictions, at whichever cost the change needs.

        Three tiers, cheapest first:

        * **Reweight** (``gate``/``beta``/``requirement_weight``/``tier_gap``) -- pure
          arithmetic over axes already sitting in ``task_scores``. No LLM calls at all.
        * **Requirements-only regrade** (``requirements`` changes, ``judge_model``
          doesn't) -- only ``requirements_met`` can possibly change, so only that judge
          call reruns; the other three axes are left untouched.
        * **Full regrade** (``judge_model`` changes) -- a different judge has its own
          opinions about salience, so document facts and all four axes are recomputed.

        No tier ever regenerates a candidate's summary: every stored ``predicted_value``
        is replayed rather than requeried, so a regrade pays for judge calls only.

        Args:
            judge_model: New judge model. Triggers a full regrade against every stored
                prediction. Omit to keep the current judge.
            requirements: New checklist. Triggers a requirements-only regrade unless
                ``judge_model`` is also given, in which case it folds into that full
                regrade instead. Omit to keep the current checklist.
            gate: New faithfulness gate for the scoring scheme.
            beta: New F-measure beta for the scoring scheme.
            requirement_weight: New requirements weight for the scoring scheme.
            tier_gap: New tier-boundary gap for the ranking.
            output_dir: If given, writes the re-scored results via
                ``save_experiment_results()``. As with ``ReferencedEval.reevaluate()``,
                ``metadata.json`` is not overwritten if it already exists there -- pass a
                fresh directory to persist an updated ``judge_model``/``requirements``.

        Returns:
            Path to the run directory if ``output_dir`` was given, else ``None``.

        Raises:
            ValueError: No results to reevaluate (call ``evaluate()`` or
                ``load_experiment_results()`` first).
        """
        if self.results is None:
            raise ValueError(
                "No results to reevaluate. Call evaluate() or load_experiment_results() first."
            )

        for name, value in (
            ("gate", gate),
            ("beta", beta),
            ("requirement_weight", requirement_weight),
            ("tier_gap", tier_gap),
        ):
            if value is not None:
                setattr(self._settings, name, value)

        judge_changed = judge_model is not None and judge_model != self._settings.judge_model
        if judge_changed:
            assert judge_model is not None
            await self._regrade_fully(judge_model, requirements)
        elif requirements is not None:
            await self._regrade_requirements(requirements)

        self._task_statistics = self.compute_task_statistics(self.results)

        if output_dir is not None:
            resolved = Path(output_dir)
            if (resolved / "metadata.json").exists():
                logger.warning(
                    "reevaluate_metadata_not_overwritten",
                    output_dir=str(resolved),
                    detail=(
                        "metadata.json already exists and will not be overwritten. "
                        "Only per-model JSON files will be updated. Pass a fresh "
                        "output_dir to preserve the new judge_model/requirements."
                    ),
                )
            return self.save_experiment_results(output_dir)

        return None

    def _document_map(self) -> dict[str, Document]:
        """Every document referenced in ``self.data``, by id."""
        documents, _ = self._load_documents_and_labels()
        return {document.id: document for document in documents}

    async def _regrade_fully(self, judge_model: str | None, requirements: list[str] | None) -> None:
        """Rerun document facts, salience, and all four axes under a new judge.

        A different judge has its own opinions from the ground up, so nothing from the
        old judge is reusable -- not the document facts, not the salience marks, not the
        per-candidate verdicts.
        """
        assert self.results is not None
        if judge_model is not None:
            self._settings.judge_model = judge_model
            self._judge = Judge(ClientModel(judge_model, client=self.client, name=JUDGE_LABEL))
        if requirements is not None:
            self._settings.requirements = requirements
            self._checklist = [Requirement(text) for text in requirements]

        documents_by_id = self._document_map()
        self._document_facts = {}
        semaphore = asyncio.Semaphore(self._settings.max_concurrent_documents)

        async def extract(document_id: str) -> tuple[str, DocumentFacts]:
            async with semaphore:
                document = documents_by_id[document_id]
                return document_id, await extract_document_facts(
                    Doc(self._document_text(document), attachments=document.attachments),
                    self._judge,
                )

        document_ids = {p.document_id for r in self.results for p in r.predictions}
        for document_id, facts in await asyncio.gather(
            *(extract(document_id) for document_id in document_ids)
        ):
            self._document_facts[document_id] = facts

        self._models_in_pass = max(len(self.results), 1)

        async def regrade(
            result: EvaluationResult, prediction: PredictionResult
        ) -> PredictionResult:
            if not isinstance(prediction.predicted_value, str):
                raise TypeError(
                    f"Cannot regrade document {prediction.document_id!r}: predicted_value "
                    f"is a {type(prediction.predicted_value).__name__}, expected a string."
                )
            shared = self._document_facts[prediction.document_id]
            document = documents_by_id[prediction.document_id]
            replay = _StoredSummary(result.model, prediction.predicted_value)
            evaluation = await evaluate_candidate(
                Doc(self._document_text(document), attachments=document.attachments),
                replay,
                self._judge,
                shared,
                self._checklist,
            )
            task_scores = {
                name: value
                for name in AXIS_NAMES
                if (value := getattr(evaluation.axes, name)) is not None
            }
            evaluation_cost = (
                evaluation.judge_usage.cost_usd + shared.usage.cost_usd / self._models_in_pass
            )
            metadata = {
                **prediction.metadata,
                "document_facts": [fact.text for fact in shared.facts],
                "salient_facts": [fact.text for fact in shared.salient],
                "summary_facts": [fact.text for fact in evaluation.summary_facts],
                "faithful_verdicts": evaluation.faithful_verdicts,
                "coverage_verdicts": evaluation.coverage_verdicts,
                "precision_verdicts": evaluation.precision_verdicts,
                "requirement_verdicts": evaluation.requirement_verdicts,
                "requirements": list(self._settings.requirements),
            }
            return prediction.model_copy(
                update={
                    "task_scores": task_scores or None,
                    "evaluation_cost": evaluation_cost,
                    "metadata": metadata,
                }
            )

        for result in self.results:
            result.predictions = list(
                await asyncio.gather(*(regrade(result, p) for p in result.predictions))
            )
            result.compute_metrics()

    async def _regrade_requirements(self, requirements: list[str]) -> None:
        """Rerun only the requirements axis; the other three cannot have changed.

        ``self._judge`` already matches ``self._settings.judge_model`` -- it is rebuilt
        from config in ``_post_init`` regardless of how this instance was constructed --
        so this needs no judge rebuild and no document facts, only the checklist itself.
        """
        assert self.results is not None
        self._settings.requirements = requirements
        self._checklist = [Requirement(text) for text in requirements]
        checklist = self._checklist

        async def regrade(prediction: PredictionResult) -> PredictionResult:
            if not isinstance(prediction.predicted_value, str):
                raise TypeError(
                    f"Cannot regrade document {prediction.document_id!r}: predicted_value "
                    f"is a {type(prediction.predicted_value).__name__}, expected a string."
                )
            usage = Usage()
            fraction, verdicts = await self._judge.requirements_met(
                Summary(prediction.predicted_value), checklist, usage=usage
            )
            task_scores = {
                key: value
                for key, value in (prediction.task_scores or {}).items()
                if key != "requirements_met"
            }
            if fraction is not None:
                task_scores["requirements_met"] = fraction
            metadata = {
                **prediction.metadata,
                "requirement_verdicts": verdicts,
                "requirements": list(requirements),
            }
            return prediction.model_copy(
                update={
                    "task_scores": task_scores or None,
                    "evaluation_cost": prediction.evaluation_cost + usage.cost_usd,
                    "metadata": metadata,
                }
            )

        for result in self.results:
            result.predictions = list(
                await asyncio.gather(*(regrade(p) for p in result.predictions))
            )
            result.compute_metrics()

    # -------------------------------------------------------------------------
    # Reports
    # -------------------------------------------------------------------------

    def save_html_report(self, output_dir: str | Path | None = None) -> Path:
        """Write the HTML report and return its path.

        Overrides the base class's ``NotImplementedError``: the reason it refuses
        by default is that its report assumes a correctness notion, and this
        recipe brings its own report which does not.
        """
        generator, destination = self._report_setup(output_dir, "save_html_report")
        assert self.results is not None
        path, recommendation = generator.generate_html_report(
            self.results,
            self.ranking,
            destination,
            use_case=self.use_case,
            original_prompt=self.prompt_template,
            model_prompts=self._model_prompts,
        )
        # Held so a following save_pdf_report() reuses it rather than paying for
        # a second one; arun() calls both.
        self._recommendation = recommendation
        return path

    def save_pdf_report(self, output_dir: str | Path | None = None) -> Path:
        """Write the PDF report and return its path.

        Reuses the recommendation from a preceding ``save_html_report()`` if there
        was one, and generates none of its own -- the same division of labour the
        classification reports use.
        """
        generator, destination = self._report_setup(output_dir, "save_pdf_report")
        assert self.results is not None
        return generator.generate_pdf_report(
            self.results,
            self.ranking,
            destination,
            use_case=self.use_case,
            recommendation=self._recommendation,
        )

    def _report_setup(
        self, output_dir: str | Path | None, caller: str
    ) -> tuple["SummarizationReportGenerator", Path]:
        """Shared preamble: check there is something to report, and rebuild the ranking."""
        from valtron_core.reports.generate_summarization_report import (
            SummarizationReportGenerator,
        )

        if self.results is None:
            raise RuntimeError(f"Call evaluate() before {caller}().")
        # A run reloaded from disk has predictions but has never been ranked,
        # since compute_task_statistics only runs as part of aevaluate().
        if self._ranking is None:
            self.compute_task_statistics(self.results)
        return SummarizationReportGenerator(client=self.client), self._resolve_output_dir(
            output_dir
        )

    @property
    def ranking(self) -> SummarizationRanking:
        """The corpus-level ranking, as objects rather than a dict.

        Raises:
            RuntimeError: Before ``evaluate()`` has produced one.
        """
        if self._ranking is None:
            raise RuntimeError("Call evaluate() before reading the ranking.")
        return self._ranking


def _axes_from(task_scores: dict[str, float] | None) -> Axes:
    """Rebuild an ``Axes`` from a prediction's ``task_scores``.

    A missing key means the axis was undefined for that document -- a summary
    with no extracted facts has no precision -- which is not the same as zero,
    and averaging must skip it rather than drag the axis down.
    """
    scores = task_scores or {}
    return Axes(**{name: scores.get(name) for name in AXIS_NAMES})
