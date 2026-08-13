"""Abstract base class for multi-model, multi-prompt LLM evaluations.

``ModelEval`` and a planned ``SummarizationTask`` share the same shape: documents go
in, a list of models/prompts get run against them, and the results come back scored
and comparable. What varies -- output shape, whether groundtruth is required, which
statistics matter -- is entirely up to each concrete subclass; this base class makes
no assumption about any of it.

Public contract::

    BaseModelEval(config, data)
    .add_models([...])              # string, dict, or config object; callable again later
    .evaluate() / .run(...)         # sync
    .aevaluate() / .arun(...)       # async; run()/arun() also persist to output_dir
    BaseModelEval.load_experiment_results(dir_path)  # reload a persisted run
    .get_traces(model=None)         # per-call records already collected
    .reevaluate(...)                # optional: rescore without new LLM calls

Every internal seam has a concrete default except one: ``_evaluate_model_documents``,
the method that actually calls a model and scores its output. A subclass need only
implement that to get a working evaluation; every other hook is overridden only to
specialize behavior.

Persistence is symmetric: ``_run_evaluations`` writes each model's results as soon as
it finishes, ``save_experiment_results()`` (from ``BaseRecipe``) writes the full run
directory, and ``load_experiment_results()`` reads it back later. There is no separate
"trace" format -- a trace is a ``PredictionResult``, and ``get_traces()`` just exposes
the ones already in ``self.results``.

This module does not touch ``model_eval.py``; wiring ``ModelEval`` to extend this
class is a follow-up refactor.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Callable

import structlog
from tqdm import tqdm  # type: ignore[import-untyped]

from valtron_core.client import LLMClient
from valtron_core.evaluation.base import BaseRecipe
from valtron_core.evaluation.config import BaseRecipeConfig, LLMModelConfig
from valtron_core.models import Document, PredictionResult
from valtron_core.partial_results import PartialResultStore, compute_prediction_hash
from valtron_core.progress import ProgressTracker, write_status
from valtron_core.runner import EvaluationResult, EvaluationRunner, PreflightError

logger = structlog.get_logger()


class BaseModelEval(BaseRecipe):
    """Shared pipeline for evaluations that run multiple models/prompts and score the results.

    The only required override is ``_evaluate_model_documents``. Everything else has
    a default suitable for a plain "one LLM model, one prompt, optional groundtruth,
    no field-level scoring" evaluation.

    Populated after ``__init__``: ``self.runner``, ``self.client``, ``self.config``,
    ``self.data``, ``self.models``, ``self.prompt_template``, ``self.output_dir``,
    ``self.use_case``, ``self.temperature``.

    Populated after ``aevaluate()`` / ``evaluate()``: ``self.results``,
    ``self._manipulations_applied``, ``self._model_prompts``, ``self._task_statistics``.
    """

    client: LLMClient
    _task_statistics: dict[str, Any] | None

    # -------------------------------------------------------------------------
    # Construction
    # -------------------------------------------------------------------------

    def __init__(
        self,
        config: "BaseRecipeConfig | dict[str, Any] | str | Path",
        data: "list[dict[str, Any]] | str | Path",
    ) -> None:
        """Load config/data and build the model list; task setup happens in ``_post_init``.

        Args:
            config: Config dict, a ``BaseRecipeConfig`` instance (the type returned
                by ``_config_model()``), or a path to a JSON config file.
            data: List of document dicts, or a path to a JSON file with the same
                structure. Each dict needs at least ``content``; ``id``, ``label``,
                ``metadata``, ``attachments`` are optional
                (see ``_load_documents_and_labels``).
        """
        if isinstance(config, (str, Path)):
            with open(config) as f:
                config = json.load(f)
        if isinstance(config, dict):
            config = self._config_model().model_validate(config)
        self.config = config

        if isinstance(data, (str, Path)):
            with open(data) as f:
                data = json.load(f)
        self.data = data

        self.runner = EvaluationRunner()
        self.client = LLMClient()

        self.models: list[Any] = []
        self.add_models(config.models)

        self.prompt_template = config.prompt
        self.output_dir = Path(config.output_dir) if config.output_dir else None
        self.use_case = config.use_case
        self.temperature = config.temperature

        self.results = None
        self._manipulations_applied = None
        self._model_prompts = None
        self._task_statistics = None

        self._post_init()

        logger.info(
            "evaluation_initialized",
            evaluation=type(self).__name__,
            num_models=len(self.models),
            num_documents=len(self.data),
        )

    @classmethod
    def _config_model(cls) -> type[BaseRecipeConfig]:
        """Return the Pydantic config class to validate a dict/JSON-file config against.

        Default is ``BaseRecipeConfig`` itself. Override to return a subclass
        carrying additional task-specific fields.
        """
        return BaseRecipeConfig

    def _post_init(self) -> None:
        """Extension point for setup that needs the fields ``__init__`` just populated.

        Default is a no-op. Must not re-assign ``self.models`` wholesale (use
        ``add_models`` for that).
        """
        return None

    # -------------------------------------------------------------------------
    # Load from disk (the read side of save_experiment_results)
    # -------------------------------------------------------------------------

    @classmethod
    def _restore_config(cls, meta: "dict[str, Any]") -> "dict[str, Any]":
        """Return task-specific extra config fields to restore from a saved ``metadata.json``.

        Default is ``{}``. Override to pull additional config fields out of ``meta``
        so a reloaded instance matches the one that produced the run.
        """
        return {}

    def _post_restore(self, meta: "dict[str, Any]") -> None:
        """Extension point for post-construction fixup during ``load_experiment_results``.

        The reload counterpart to ``_post_init``: called once, right after
        construction (so after ``_post_init`` has already run), before predictions
        are reconstructed. Default is a no-op.
        """
        return None

    @staticmethod
    def _model_data_from_file(model_file: Path) -> "dict[str, Any]":
        """Read one ``models/<name>.json``, returning both its config and result fields.

        The saved-file shape is shared across every task, so this needs no override.
        """
        with open(model_file) as f:
            raw = json.load(f)

        llm_config: dict[str, Any] = raw.get("llm_config") or {}
        model_name = llm_config.get("model") or raw.get("model", "")
        model_label = raw.get("model", model_name)
        params = {k: v for k, v in llm_config.items() if k != "model"}
        manipulations = raw.get("prompt_manipulations") or []

        return {
            # Config fields
            "name": model_name,
            "label": model_label if model_label != model_name else None,
            "params": params,
            "prompt": raw.get("override_prompt"),
            "prompt_manipulation": manipulations,
            # Result fields
            "run_id": raw.get("run_id", ""),
            "started_at": raw.get("started_at"),
            "completed_at": raw.get("completed_at"),
            "status": raw.get("status", "completed"),
            "llm_config": llm_config,
            "metrics": raw.get("metrics"),
            "predictions": raw.get("predictions", []),
            "prompt_template": raw.get("prompt_template", ""),
        }

    @staticmethod
    def _stringify_label(label: Any) -> str:
        """Coerce a raw label/reference into the string ``PredictionResult.expected_value`` requires.

        ``None`` (missing groundtruth) becomes ``""``; anything else is ``str()``'d.
        This makes no assumption about the label's shape -- a task whose labels are
        structured (e.g. dicts) is responsible for serializing them itself before
        they reach this point.
        """
        if label is None:
            return ""
        return str(label)

    @classmethod
    def _config_and_data_from_metadata(
        cls, metadata_path: Path
    ) -> "tuple[dict[str, Any], list[dict[str, Any]]]":
        """Read ``metadata.json`` and return ``(config_dict, data)`` for reconstruction.

        ``models`` is filled in separately by the caller from the model files.
        Task-specific extras come from ``_restore_config``.
        """
        with open(metadata_path) as f:
            meta = json.load(f)

        config_dict: dict[str, Any] = {
            "prompt": meta.get("original_prompt") or "{content}",
            "use_case": meta.get("use_case", "evaluation"),
        }
        config_dict.update(cls._restore_config(meta))

        data: list[dict[str, Any]] = meta.get("documents", [])
        return config_dict, data

    @classmethod
    def load_experiment_results(cls, dir_path: "str | Path") -> "BaseModelEval":
        """Restore a run written by ``save_experiment_results()`` into a live instance.

        Reconstructs an instance in the same state as right after ``aevaluate()``
        (``self.results``, ``self._model_prompts``, ``self._manipulations_applied``
        all populated) -- ready for more ``add_models()`` + ``run()``,
        ``save_html_report()``, or ``reevaluate()``.

        Args:
            dir_path: Directory previously written by ``save_experiment_results()``;
                must contain ``metadata.json`` and a ``models/`` sub-directory.

        Raises:
            FileNotFoundError: ``metadata.json`` is absent.
            ValueError: ``models/`` directory is empty.
        """
        from valtron_core.models import EvaluationMetrics

        dir_path = Path(dir_path)
        metadata_path = dir_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No metadata.json found in {dir_path}. "
                "Pass the directory written by save_experiment_results()."
            )

        config_dict, data = cls._config_and_data_from_metadata(metadata_path)

        model_files = sorted((dir_path / "models").glob("*.json"))
        if not model_files:
            raise ValueError(f"No model result files found in {dir_path / 'models'}.")

        all_model_data = [cls._model_data_from_file(f) for f in model_files]

        config_dict["models"] = [
            {
                k: v
                for k, v in md.items()
                if k in ("name", "label", "params", "prompt", "prompt_manipulation")
                and v is not None
            }
            for md in all_model_data
        ]

        instance = cls(config=config_dict, data=data)
        with open(metadata_path) as f:
            instance._post_restore(json.load(f))

        label_map = {str(d.get("id", "")): cls._stringify_label(d.get("label")) for d in data}

        results: list[EvaluationResult] = []
        model_prompts: dict[str, str] = {}
        manipulations_applied: dict[str, list] = {}

        for md in all_model_data:
            model_label = md["label"] or md["name"]
            model_prompts[model_label] = md["prompt_template"]
            manipulations_applied[model_label] = md["prompt_manipulation"]

            try:
                from valtron_core.scoring.json_eval import EvalResult

                _eval_result_cls: Any = EvalResult
            except ImportError:
                _eval_result_cls = None

            predictions = []
            for p in md.get("predictions", []):
                field_metrics = None
                if p.get("field_metrics") and _eval_result_cls is not None:
                    try:
                        field_metrics = _eval_result_cls.model_validate(p["field_metrics"])
                    except Exception:
                        pass
                predictions.append(
                    PredictionResult(
                        document_id=p["document_id"],
                        predicted_value=p["predicted_value"],
                        expected_value=p.get(
                            "expected_value", label_map.get(p["document_id"], "")
                        ),
                        is_correct=p.get("is_correct", False),
                        example_score=p.get("example_score", 0.0),
                        response_time=p.get("response_time", 0.0),
                        original_cost=p.get("original_cost", 0.0),
                        llm_cost=p.get("llm_cost", p.get("cost", 0.0)),
                        evaluation_cost=p.get("evaluation_cost", 0.0),
                        model=model_label,
                        field_metrics=field_metrics,
                    )
                )

            result = EvaluationResult(
                run_id=md["run_id"],
                model=model_label,
                predictions=predictions,
                metrics=EvaluationMetrics(**md["metrics"]) if md.get("metrics") else None,
                prompt_template=md["prompt_template"],
                llm_config=md.get("llm_config", {}),
                status=md.get("status", "completed"),
            )
            if md.get("started_at"):
                result.started_at = md["started_at"]
            if md.get("completed_at"):
                result.completed_at = md["completed_at"]
            if not result.metrics and result.predictions:
                result.compute_metrics()
            results.append(result)

        instance.results = results
        instance._model_prompts = model_prompts
        instance._manipulations_applied = manipulations_applied
        return instance

    # -------------------------------------------------------------------------
    # Model management
    # -------------------------------------------------------------------------

    def _build_model_config(self, entry: "str | dict[str, Any] | Any") -> Any:
        """Normalize and validate one raw model entry into a model-config object.

        Default accepts a plain model name string, or a dict of ``LLMModelConfig``
        fields; an already-validated config object passes through unchanged.
        Override to accept other model kinds, or to reject entries this task
        doesn't support (raise ``ValueError``).

        Returns:
            A config object exposing at least ``.name`` and ``.label``.
        """
        if isinstance(entry, str):
            return LLMModelConfig(name=entry)
        if isinstance(entry, dict):
            return LLMModelConfig.model_validate(entry)
        return entry

    def add_models(self, models: "list[str | dict[str, Any] | Any]") -> None:
        """Add new models to the experiment; safe to call again after ``evaluate()``.

        Each entry is normalized via ``_build_model_config``, then checked for a
        duplicate label against models already present and other entries in this
        call. Models added later are picked up on the next ``evaluate()``/``run()``
        without re-running models that already have results (see ``aevaluate``).

        Args:
            models: Model name strings, config dicts, or config objects.

        Raises:
            ValueError: Duplicate label, or ``_build_model_config`` rejects an entry.
        """
        normalized = [self._build_model_config(m) for m in models]

        existing_labels = {mc.label or mc.name for mc in self.models}
        seen_in_batch: set[str] = set()
        for mc in normalized:
            label = mc.label or mc.name
            label_source = (
                f"label={mc.label!r}"
                if mc.label
                else f"name={mc.name!r} (label inferred from name)"
            )
            if label in existing_labels or label in seen_in_batch:
                raise ValueError(
                    f"Duplicate model label {label!r} in config ({label_source}). "
                    "Each model entry must have a unique label. "
                    "You can use the same model twice by giving one entry a distinct "
                    "label (e.g. label='gpt-5-mini-v2')."
                )
            seen_in_batch.add(label)

        self.models.extend(normalized)
        self.config.models.extend(normalized)

    # -------------------------------------------------------------------------
    # Preflight
    # -------------------------------------------------------------------------

    def _validate_task_data(self) -> None:
        """Extension point for data validation ahead of running any evaluation.

        Default is a no-op. Raise ``ValueError`` for a fatal problem, or
        ``logger.warning`` for a non-fatal one.
        """
        return None

    def _preflight_check(self) -> None:
        """Run shared checks (``BaseRecipe``) plus ``_validate_task_data``."""
        super()._preflight_check()
        self._validate_task_data()

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def _load_documents_and_labels(self) -> "tuple[list[Document], list[Any]]":
        """Convert ``self.data`` into ``Document`` objects plus a parallel label list.

        No disk I/O. Default builds one ``Document`` per entry (``id``, ``content``,
        ``metadata``, ``attachments``) and takes ``label`` verbatim, defaulting to
        ``None`` when absent -- groundtruth is optional out of the box. Override for
        a richer label shape, or to require groundtruth.

        Returns:
            ``(documents, labels)``, both of length ``len(self.data)`` and
            index-aligned.
        """
        documents: list[Document] = []
        labels: list[Any] = []
        for idx, item in enumerate(self.data):
            doc_id = str(item.get("id", f"doc_{idx}"))
            documents.append(
                Document(
                    id=doc_id,
                    content=item["content"],
                    metadata=item.get("metadata", {}),
                    attachments=item.get("attachments", []),
                )
            )
            labels.append(item.get("label"))
        return documents, labels

    def _get_field_metrics_config(self) -> Any:
        """Return a task-specific scoring config, or ``None`` if this task has none.

        Required override of ``BaseRecipe``'s abstract method; this class doesn't
        assume any particular scoring-config shape, so the type is left as ``Any``.
        Default is ``None``. ``None`` also disables the field-metrics table in
        generated HTML/PDF reports; see ``compute_task_statistics`` for the
        general-purpose alternative.
        """
        return None

    # -------------------------------------------------------------------------
    # Prompt preparation
    # -------------------------------------------------------------------------

    async def _prepare_model_prompts(self) -> "dict[str, str]":
        """Resolve the final prompt to send for each model -- the one actually used.

        Default: each model's own ``prompt`` if set, else ``self.prompt_template`` --
        no further transformation. Override to layer on any prompt-transformation
        pipeline a task wants.

        Returns:
            Mapping of model label -> prompt string, covering every model in
            ``self.models``.
        """
        prompts: dict[str, str] = {}
        for model_config in self.models:
            model_label = model_config.label or model_config.name
            prompts[model_label] = model_config.prompt or self.prompt_template
        return prompts

    # -------------------------------------------------------------------------
    # Per-model evaluation
    # -------------------------------------------------------------------------

    def _cache_key_inputs(self, model_config: Any, prompt: str) -> "tuple[str, dict[str, Any]]":
        """Return the ``(prompt, model_params)`` pair that keys partial-result caching.

        Default: ``(prompt, self._build_model_arg(model_config))``. Two calls with
        the same document content and the same return value here are assumed to
        produce the same prediction, so fold in anything else that affects the
        output (e.g. a task-specific processing parameter) when overriding.
        """
        return prompt, self._build_model_arg(model_config)

    async def _evaluate_model_documents(
        self,
        model_config: Any,
        documents: "list[Document]",
        labels: "list[Any]",
        prompt: str,
        field_metrics_config: Any,
        on_document_complete: "Callable[[PredictionResult], None] | None" = None,
    ) -> "tuple[EvaluationResult, str | None]":
        """Run one model against a set of documents and return a scored result.

        The one method every subclass must implement -- it's the only place a model
        is actually called and its output turned into predictions. Everything else
        in ``_run_evaluations`` (concurrency, partial-result resume, progress
        reporting, persistence) is shared and calls this once per model.

        ``documents``/``labels`` are already filtered to exclude anything resumed
        from a partial prior run. Do not update the shared progress bar or cost
        total directly -- call ``on_document_complete`` once per finished document
        and that happens automatically. Each returned ``PredictionResult`` doubles as
        that document's trace record (see ``get_traces``): put whatever is useful for
        debugging into its ``metadata``.

        Args:
            model_config: This task's model-config type, from ``_build_model_config``.
            documents: Documents still needing a prediction.
            labels: Labels/references index-aligned with ``documents``; entries may
                be ``None`` for optional groundtruth.
            prompt: The resolved prompt for this model, from ``_prepare_model_prompts``.
            field_metrics_config: From ``_get_field_metrics_config()``; may be ``None``.
            on_document_complete: Call once per completed document's
                ``PredictionResult`` to drive progress/cost tracking and crash
                recovery.

        Returns:
            ``(result, updated_prompt)``. ``result`` has ``status="completed"`` and
            ``compute_metrics()`` already called. ``updated_prompt`` overrides what's
            displayed/persisted as this model's prompt in the report; return ``None``
            to keep the prompt passed in.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _evaluate_model_documents()."
        )

    async def _run_evaluations(
        self,
        model_prompts: "dict[str, str]",
        field_metrics_config: Any = None,
        models: "list[Any] | None" = None,
    ) -> "tuple[list[EvaluationResult], dict[str, list[Any]]]":
        """Run the given models concurrently: caching/resume, progress, persistence.

        Shared across every task. Delegates the parts that vary --  actually calling
        a model and scoring it, and what makes a cached prediction reusable -- to
        ``_evaluate_model_documents`` and ``_cache_key_inputs``.

        Args:
            model_prompts: Mapping of model label -> prompt, from
                ``_prepare_model_prompts``.
            field_metrics_config: Re-derived via ``_get_field_metrics_config()`` when
                omitted.
            models: Models to evaluate; defaults to ``self.models``.

        Returns:
            ``(results, manipulations_applied)`` -- the latter maps model label to
            whatever transformation metadata that model applied (empty by default).
        """
        if field_metrics_config is None:
            field_metrics_config = self._get_field_metrics_config()

        effective_models = models if models is not None else self.models
        documents, labels = self._load_documents_and_labels()

        partial_store = (
            PartialResultStore(Path(self.output_dir)) if self.output_dir is not None else None
        )
        if partial_store is not None:
            (Path(self.output_dir) / "models").mkdir(parents=True, exist_ok=True)

        total_docs = len(documents) * len(effective_models)
        running_cost: list[float] = [0.0]
        shared_bar = tqdm(total=total_docs, unit="doc", desc="Evaluating")

        def _on_doc(pred: PredictionResult) -> None:
            running_cost[0] += pred.llm_cost + pred.evaluation_cost
            shared_bar.update(1)
            shared_bar.set_postfix(cost=f"${running_cost[0]:.4f}")

        # Initialise the progress tracker so external pollers (e.g. the valtron web
        # dashboard) can see per-model document progress in real time. The absence of
        # progress.json before this point signals "still initialising".
        progress_tracker = None
        if self.output_dir is not None:
            try:
                progress_model_labels = [
                    (mc.label or getattr(mc, "name", None) or "<unknown>")
                    for mc in effective_models
                ]
                progress_tracker = ProgressTracker(
                    output_dir=self.output_dir,
                    model_names=progress_model_labels,
                    docs_per_model=len(documents),
                )
            except Exception as e:
                logger.warning("progress_tracker_init_failed", error=str(e))
                progress_tracker = None

        def _persist_model_result(
            result: EvaluationResult,
            model_label: str,
            manipulations: list,
            updated_prompt: "str | None",
        ) -> None:
            """Write the completed model result to disk and clean up its staging file."""
            if self.output_dir is None:
                return
            from valtron_core.runner import save_single_model_result

            try:
                effective_prompt = updated_prompt or model_prompts.get(model_label)
                save_single_model_result(
                    self.output_dir,
                    result,
                    model_prompt=effective_prompt,
                    prompt_manipulations=manipulations,
                )
                if partial_store is not None:
                    partial_store.finalize(model_label)
            except Exception as e:
                logger.warning("eager_model_save_failed", model=model_label, error=str(e))

        async def _evaluate_single_model(
            index: int, model_config: Any
        ) -> "tuple[int, EvaluationResult, str, list, str | None]":
            model_name = getattr(model_config, "name", None) or model_config.label
            model_label = model_config.label or model_name
            manipulations = getattr(model_config, "prompt_manipulation", [])
            prompt = model_prompts[model_label]

            hash_prompt, hash_model_params = self._cache_key_inputs(model_config, prompt)

            # Determine which documents have already been persisted for this model so
            # we can skip them and resume from where a prior run left off. Only
            # predictions whose hash matches the current inputs are reused.
            doc_content_map = {d.id: d.content for d in documents}
            cached_preds: dict[str, PredictionResult] = {}
            if partial_store is not None:
                cached_preds = partial_store.get_valid_cached(
                    model_label, doc_content_map, hash_prompt, hash_model_params
                )
            completed_ids = set(cached_preds.keys())
            remaining_docs = [d for d in documents if d.id not in completed_ids]
            remaining_labels = [
                lb for d, lb in zip(documents, labels) if d.id not in completed_ids
            ]

            # Pre-advance the shared progress bar and running cost for documents that
            # were already completed in a prior run and are being reused from cache.
            if cached_preds:
                prior_cost = sum(p.llm_cost + p.evaluation_cost for p in cached_preds.values())
                running_cost[0] += prior_cost
                shared_bar.update(len(cached_preds))
                shared_bar.set_postfix(cost=f"${running_cost[0]:.4f}")
                if progress_tracker is not None:
                    try:
                        for _ in cached_preds:
                            progress_tracker.on_doc_complete(model_label)
                    except Exception:
                        pass

            def _on_doc_with_progress(pred: PredictionResult) -> None:
                _on_doc(pred)
                if progress_tracker is not None:
                    try:
                        progress_tracker.on_doc_complete(model_label)
                    except Exception:
                        pass
                if partial_store is not None:
                    try:
                        h = compute_prediction_hash(
                            pred.metadata.get("content", ""), hash_prompt, hash_model_params
                        )
                        partial_store.record(model_label, pred, h)
                    except Exception:
                        pass

            result, updated_prompt = await self._evaluate_model_documents(
                model_config,
                remaining_docs,
                remaining_labels,
                prompt,
                field_metrics_config,
                on_document_complete=_on_doc_with_progress,
            )

            # Merge predictions from a prior partial run into this result and
            # recompute aggregated metrics over the full document set.
            if cached_preds:
                result.predictions = list(cached_preds.values()) + result.predictions
                result.compute_metrics()

            # Propagate label to result objects when it differs from the model name.
            if model_label != model_name:
                result.model = model_label
                for pred in result.predictions:
                    pred.model = model_label
                if result.metrics:
                    result.metrics.model = model_label

            # Safety net: ensure the model row is marked "done" in progress.json even
            # for implementations that don't emit per-doc callbacks.
            if progress_tracker is not None:
                try:
                    progress_tracker.mark_model_completed(model_label)
                except Exception:
                    pass

            _persist_model_result(result, model_label, manipulations, updated_prompt)
            return index, result, model_label, manipulations, updated_prompt

        indexed_results = await asyncio.gather(
            *[_evaluate_single_model(i, mc) for i, mc in enumerate(effective_models)]
        )
        shared_bar.close()

        results = []
        manipulations_applied: dict[str, list[Any]] = {}
        for _, result, model_label, manipulations, updated_prompt in sorted(
            indexed_results, key=lambda x: x[0]
        ):
            results.append(result)
            manipulations_applied[model_label] = manipulations
            if updated_prompt is not None:
                model_prompts[model_label] = updated_prompt

        self.runner._print_summary_table(
            results, show_field_metrics=field_metrics_config is not None
        )

        return results, manipulations_applied

    # -------------------------------------------------------------------------
    # Traces
    # -------------------------------------------------------------------------

    def get_traces(self, model: "str | None" = None) -> "list[PredictionResult]":
        """Return the per-call trace records (``PredictionResult``) collected so far.

        These are the same objects already persisted per model in
        ``models/{label}.json``; ``metadata`` on each is where request/response
        detail lives. Returns ``[]`` before ``evaluate()`` has run.

        Args:
            model: If given, only return traces for that model label.
        """
        if not self.results:
            return []
        if model is not None:
            return [p for r in self.results if r.model == model for p in r.predictions]
        return [p for r in self.results for p in r.predictions]

    # -------------------------------------------------------------------------
    # Task-specific statistics
    # -------------------------------------------------------------------------

    def compute_task_statistics(self, results: "list[EvaluationResult]") -> dict[str, Any]:
        """Compute aggregate statistics beyond ``EvaluationMetrics``; stored on ``self._task_statistics``.

        Called once from ``aevaluate()`` with the full ``self.results``. Default is
        ``{}`` -- override only when a task's real signal doesn't fit
        ``EvaluationMetrics`` (accuracy, cost, timing) at all. Wiring the result into
        report templates is a follow-up to this abstraction.

        Returns:
            A JSON-serializable dict, keyed however is most useful (e.g. by model
            label).
        """
        return {}

    # -------------------------------------------------------------------------
    # Reevaluation
    # -------------------------------------------------------------------------

    def reevaluate(
        self,
        data: "list[dict[str, Any]] | str | Path | None" = None,
        output_dir: "str | Path | None" = None,
        **kwargs: Any,
    ) -> "Path | None":
        """Re-score stored predictions without new LLM calls, and optionally persist.

        Default raises ``NotImplementedError`` -- rescoring semantics are entirely
        task-defined. Override to support it, accepting ``data`` for updating
        groundtruth in place (matched by document id) and ``output_dir`` for writing
        results back out via ``save_experiment_results()``.

        Args:
            data: Updated groundtruth, same shape as the constructor's ``data``.
            output_dir: If given, writes results via ``save_experiment_results()``.
            **kwargs: Task-specific rescoring parameters.

        Returns:
            Path to the run directory if ``output_dir`` was given, else ``None``.

        Raises:
            NotImplementedError: This task does not support rescoring.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support reevaluate(). Override this "
            "method to support rescoring stored predictions without new LLM calls."
        )

    # -------------------------------------------------------------------------
    # Shared helpers
    # -------------------------------------------------------------------------

    def _build_model_arg(self, model_config: Any) -> dict[str, Any]:
        """Build the litellm kwargs dict for one LLM model config.

        Args:
            model_config: A config with ``.name``, ``.params``, and optionally
                ``.cost_rate`` / ``.cost_rate_time_unit``.
        """
        result: dict[str, Any] = {
            "model": model_config.name,
            "temperature": model_config.params.get("temperature", self.temperature),
        }
        if "max_tokens" in model_config.params:
            result["max_tokens"] = model_config.params["max_tokens"]
        for k, v in model_config.params.items():
            if k not in {"temperature", "max_tokens"}:
                result[k] = v
        if model_config.cost_rate is not None:
            result["cost_rate"] = model_config.cost_rate
            result["cost_rate_time_unit"] = model_config.cost_rate_time_unit
        return result

    def _status(self, message: str) -> None:
        """Best-effort progress status write for external pollers; never raises."""
        if self.output_dir is None:
            return
        try:
            write_status(self.output_dir, message)
        except Exception:
            pass

    async def _before_evaluation(self, field_metrics_config: Any) -> None:
        """Extension point for setup after preflight but before prompt resolution.

        Default is a no-op.
        """
        return None

    # -------------------------------------------------------------------------
    # Main pipeline
    # -------------------------------------------------------------------------

    def evaluate(self) -> None:
        """Run the evaluation pipeline synchronously (wraps ``aevaluate()``).

        Cannot be called from within a running event loop; call ``aevaluate()``
        directly (with ``await``) there instead.
        """
        asyncio.run(self.aevaluate())

    def run(self, output_dir: "str | Path | None" = None) -> Path:
        """Run the full pipeline and save outputs synchronously (wraps ``arun()``).

        Args:
            output_dir: Overrides ``config.output_dir``; one of the two is required.

        Returns:
            Path to the HTML report if generated, else the run directory.
        """
        try:
            return asyncio.run(self.arun(output_dir=output_dir))
        except PreflightError:
            sys.exit(1)

    async def aevaluate(self) -> None:
        """Run the evaluation pipeline and populate results on this instance (async).

        Does not write any files -- call ``save_experiment_results()`` /
        ``save_html_report()`` / ``save_pdf_report()`` afterwards for that. Skips any
        model already in ``self.results`` or already persisted under
        ``output_dir`` from a prior run, which is what makes ``add_models()`` +
        ``aevaluate()`` safe to call again later.
        """
        logger.info("evaluation_pipeline_started", evaluation=type(self).__name__)

        self._status("Preparing run...")
        self._preflight_check()

        field_metrics_config = self._get_field_metrics_config()
        await self._before_evaluation(field_metrics_config)

        self._status("Preparing prompts...")
        self._model_prompts = await self._prepare_model_prompts()

        existing_labels: set[str] = {r.model for r in (self.results or [])}
        if self.output_dir is not None and not self.results:
            from valtron_core.runner import _completed_model_labels_on_disk

            existing_labels |= _completed_model_labels_on_disk(Path(self.output_dir))
        models_to_run = [m for m in self.models if (m.label or m.name) not in existing_labels]

        if models_to_run:
            new_results, new_manipulations = await self._run_evaluations(
                self._model_prompts, field_metrics_config, models=models_to_run
            )
            if self.results:
                self.results = list(self.results) + new_results
                self._manipulations_applied = {
                    **(self._manipulations_applied or {}),
                    **new_manipulations,
                }
            else:
                self.results = new_results
                self._manipulations_applied = new_manipulations

        self._task_statistics = self.compute_task_statistics(self.results or [])

    async def arun(self, output_dir: "str | Path | None" = None) -> Path:
        """Run ``aevaluate()`` then save outputs per ``config.output_formats`` (async).

        Args:
            output_dir: Overrides ``config.output_dir``; one of the two is required.

        Returns:
            Path to the HTML report if ``"html"`` is in ``config.output_formats``,
            else the run directory.
        """
        await self.aevaluate()

        run_dir = self.save_experiment_results(output_dir)
        report_path: Path = run_dir

        if "html" in self.config.output_formats:
            report_path = self.save_html_report(output_dir)
        if "pdf" in self.config.output_formats:
            self.save_pdf_report(output_dir)

        logger.info(
            "evaluation_run_complete", evaluation=type(self).__name__, report_path=str(report_path)
        )
        return report_path
