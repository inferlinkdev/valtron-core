"""Generic base for evaluations that run multiple models/prompts and score the results.

``ModelEval`` and a ``SummarizationExperiment`` (not built yet, but the reason this
class exists in this shape) share the same shape: documents go in, a list of
models/prompts get run against them, and the results come back scored and
comparable. What varies -- output shape, whether groundtruth is required, which
statistics matter -- is entirely up to each concrete subclass; this base class makes
no assumption about any of it.

Public contract::

    ModelEval(config, data)
    .add_models([...])              # string, dict, or config object; callable again later
    .evaluate() / .run(...)         # sync
    .aevaluate() / .arun(...)       # async; run()/arun() also persist to output_dir
    ModelEval.load_experiment_results(dir_path)  # reload a persisted run
    .get_traces(model=None)         # per-call records already collected
    .reevaluate(...)                # optional: rescore without new LLM calls

Every internal seam has a concrete default except one: ``_evaluate_model_documents``,
the method that actually calls a model and scores its output. A subclass need only
implement that to get a working evaluation; every other hook is overridden only to
specialize behavior.

Persistence is symmetric: ``_run_evaluations`` writes each model's results as soon as
it finishes, ``save_experiment_results()`` writes the full run directory (documents +
metrics + predictions), and ``load_experiment_results()`` reads it back later. There is
no separate "trace" format -- a trace is a ``PredictionResult``, and ``get_traces()``
just exposes the ones already in ``self.results``. Rich HTML/PDF reports
(``save_html_report``/``save_pdf_report``) are *not* provided here -- they assume a
correctness/accuracy notion that this class explicitly makes no assumption about;
see ``ReferencedEval`` for the concrete implementation classification/extraction use.
"""

import asyncio
import json
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Sequence

import structlog
from tqdm import tqdm  # type: ignore[import-untyped]

from valtron_core.client import LLMClient
from valtron_core.content_resolution import absolutize_local_path, is_local_path, resolve_content
from valtron_core.evaluation.config import BaseRecipeConfig, LLMModelConfig, ModelEvalConfig
from valtron_core.models import Document, FieldMetricsConfig, PredictionResult
from valtron_core.partial_results import PartialResultStore, compute_prediction_hash
from valtron_core.progress import ProgressTracker, write_status
from valtron_core.runner import EvaluationResult, EvaluationRunner, PreflightError

logger = structlog.get_logger()


def _normalize_label(label: Any) -> Any:
    if isinstance(label, (dict, list)):
        return label
    if isinstance(label, str):
        try:
            parsed = json.loads(label)
            if isinstance(parsed, (dict, list)):
                return parsed
        except (json.JSONDecodeError, ValueError):
            pass
    return str(label)


class ModelEval(ABC):
    """Shared pipeline for evaluations that run multiple models/prompts and score the results.

    The only required override is ``_evaluate_model_documents``. Everything else has
    a default suitable for a plain "one LLM model, one prompt, optional groundtruth,
    no field-level scoring" evaluation.

    Populated after ``__init__``: ``self.runner``, ``self.client``, ``self.config``,
    ``self.data``, ``self._data_base_dir``, ``self.models``, ``self.prompt_template``,
    ``self.output_dir``, ``self.use_case``, ``self.temperature``.

    Populated after ``aevaluate()`` / ``evaluate()``: ``self.results``,
    ``self._manipulations_applied``, ``self._model_prompts``, ``self._task_statistics``.
    """

    client: LLMClient
    config: BaseRecipeConfig
    models: list[Any]
    data: list[dict[str, Any]]
    _data_base_dir: Path
    output_dir: Path | None
    use_case: str
    prompt_template: str
    results: list[EvaluationResult] | None
    _manipulations_applied: dict[str, list[Any]] | None
    _model_prompts: dict[str, str] | None
    _model_override_prompts: dict[str, str] | None
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
                structure. Each dict needs at least ``content`` or ``content_path``;
                ``id``, ``label``, ``metadata``, ``attachments`` are optional
                (see ``_load_documents_and_labels``).
        """
        self.config = self._validate_config(config)

        self._data_base_dir = self._resolve_data_base_dir(data)
        if isinstance(data, (str, Path)):
            with open(data) as f:
                self.data = json.load(f)
        else:
            self.data = data

        self.runner = EvaluationRunner()
        self.client = LLMClient()

        self.models = []
        self.add_models(self.config.models)

        self.prompt_template = self.config.prompt
        self.output_dir = Path(self.config.output_dir) if self.config.output_dir else None
        self.use_case = self.config.use_case
        self.temperature = self.config.temperature

        self.results = None
        self._manipulations_applied = None
        self._model_prompts = None
        self._model_override_prompts = None
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

        Default is ``ModelEvalConfig`` (a no-op subclass of the truly generic
        ``BaseRecipeConfig``, kept for continuity with the config hierarchy's naming).
        Override to return a subclass carrying additional task-specific fields.
        """
        return ModelEvalConfig

    @classmethod
    def _validate_config(
        cls, config: "BaseRecipeConfig | dict[str, Any] | str | Path"
    ) -> BaseRecipeConfig:
        """Normalize config into a validated instance of ``_config_model()``.

        A no-op when ``config`` is already a validated model instance (of any
        subclass) -- safe to call more than once in a constructor chain, e.g. from a
        subclass's own ``__init__`` before it calls ``super().__init__()`` to resolve
        something that must be known earlier (see ``ReferencedEval.__init__``, which
        needs its config validated before ``add_models()`` runs so its own
        ``response_format`` guard has something to check).
        """
        if isinstance(config, (str, Path)):
            with open(config) as f:
                loaded: dict[str, Any] = json.load(f)
            return cls._config_model().model_validate(loaded)
        if isinstance(config, dict):
            return cls._config_model().model_validate(config)
        return config

    @staticmethod
    def _resolve_data_base_dir(data: "list[dict[str, Any]] | str | Path") -> Path:
        """Return the directory a data record's content_path/attachments resolve against.

        Called with the constructor's ``data`` argument before it gets replaced by
        the parsed JSON list, when ``data`` is a path -- otherwise the file's own
        directory is no longer recoverable.
        """
        if isinstance(data, (str, Path)):
            return Path(data).resolve().parent
        return Path.cwd()

    def _post_init(self) -> None:
        """Extension point for setup that needs the fields ``__init__`` just populated.

        Default is a no-op. Must not re-assign ``self.models`` wholesale (use
        ``add_models`` for that).
        """
        return None

    # -------------------------------------------------------------------------
    # Load from disk (the read side of save_experiment_results)
    # -------------------------------------------------------------------------

    def _extra_metadata(self) -> "dict[str, Any]":
        """Task-specific extra config to persist for ``_restore_config()`` to read back.

        Default is ``{}``. Override alongside ``_restore_config()`` -- this is its
        write-side counterpart -- so a reloaded instance matches the one that
        produced the run. Written into ``metadata.json`` by ``save_experiment_results()``.
        """
        return {}

    @classmethod
    def _restore_config(cls, meta: "dict[str, Any]") -> "dict[str, Any]":
        """Return task-specific extra config fields to restore from a saved ``metadata.json``.

        Default is ``{}``. Override to pull additional config fields out of ``meta``
        so a reloaded instance matches the one that produced the run. ``meta`` is
        whatever ``_extra_metadata()`` wrote, read back out.
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
        """Coerce a raw label/reference into a plain string for display/logging.

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
    def load_experiment_results(cls, dir_path: "str | Path") -> "ModelEval":
        """Restore a run written by ``save_experiment_results()`` into a live instance.

        Reconstructs an instance in the same state as right after ``aevaluate()``
        (``self.results``, ``self._model_prompts``, ``self._manipulations_applied``
        all populated) -- ready for more ``add_models()`` + ``run()``,
        ``save_html_report()`` (if implemented), or ``reevaluate()``.

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
        manipulations_applied: dict[str, list[Any]] = {}

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
                        expected_value=p.get("expected_value", label_map.get(p["document_id"])),
                        is_correct=p.get("is_correct"),
                        example_score=p.get("example_score"),
                        error=p.get("error"),
                        task_scores=p.get("task_scores"),
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

    def add_models(self, models: "Sequence[str | dict[str, Any] | Any]") -> None:
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

    def _check_unique_model_labels(self) -> None:
        labels = [m.label or m.name for m in self.models]
        seen: set[str] = set()
        dupes: list[str] = []
        for label in labels:
            if label in seen:
                dupes.append(label)
            else:
                seen.add(label)
        if dupes:
            raise ValueError(
                f"Duplicate model labels: {dupes!r}. "
                "Each model must have a unique label (or a unique name if no label is set)."
            )

    def _validate_task_data(self) -> None:
        """Extension point for data validation ahead of running any evaluation.

        Default is a no-op. Raise ``ValueError`` for a fatal problem, or
        ``logger.warning`` for a non-fatal one.
        """
        return None

    def _preflight_check(self) -> None:
        """Run all pre-flight checks before any evaluation work begins.

        Checks model-label uniqueness and field-metrics cost guards (shared,
        every task gets these for free), then ``_validate_task_data()``
        (task-specific). Add further shared checks here as needed.
        """
        self._check_unique_model_labels()
        field_metrics_config = self._get_field_metrics_config()
        self.runner._preflight_check(
            field_metrics_config, len(self.data), len(self.models), self.models
        )
        self._validate_task_data()

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def _load_documents_and_labels(self) -> "tuple[list[Document], list[Any]]":
        """Convert ``self.data`` into ``Document`` objects plus a parallel label list.

        No disk I/O beyond resolving a document's ``content_path`` if it has one
        (see ``resolve_content``). Default builds one ``Document`` per entry
        (``id``, ``content``, ``metadata``, ``attachments``) and takes ``label``
        verbatim, defaulting to ``None`` when absent -- groundtruth is optional out
        of the box. Override for a richer label shape, or to require groundtruth.

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
                    content=resolve_content(item, self._data_base_dir),
                    metadata=item.get("metadata", {}),
                    attachments=item.get("attachments", []),
                )
            )
            labels.append(item.get("label"))
        return documents, labels

    def _build_save_documents(self) -> list[dict[str, Any]]:
        """Build the document list used when writing the run directory.

        A local (non-URL, non-data-URI) content_path or attachment path is
        written out as an absolute path -- resolved now, while ``_data_base_dir``
        is still known -- so a later ``load_experiment_results()`` from a
        different working directory still finds the same file.
        """
        documents: list[dict[str, Any]] = []
        for item in self.data:
            doc_entry: dict[str, Any] = {
                "id": str(item.get("id", "")),
                "label": _normalize_label(item.get("label", "")),
            }
            if item.get("content_path") is not None:
                doc_entry["content_path"] = str(
                    absolutize_local_path(item["content_path"], self._data_base_dir)
                )
            else:
                doc_entry["content"] = item.get("content", "")
            if item.get("attachments"):
                doc_entry["attachments"] = [
                    (str(absolutize_local_path(a, self._data_base_dir)) if is_local_path(a) else a)
                    for a in item["attachments"]
                ]
            documents.append(doc_entry)
        return documents

    def _get_field_metrics_config(self) -> FieldMetricsConfig | None:
        """Return a task-specific scoring config, or ``None`` if this task has none.

        This class doesn't assume any particular scoring-config shape beyond the
        existing ``FieldMetricsConfig`` (used for the classification/extraction
        field-comparison tree); a task with a fundamentally different notion of
        scoring config can widen this return type in its own override. Default is
        ``None``, which also disables the field-metrics table in HTML/PDF reports;
        see ``compute_task_statistics`` for the general-purpose alternative.
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

    @abstractmethod
    async def _evaluate_model_documents(
        self,
        model_config: Any,
        documents: "list[Document]",
        labels: "list[Any]",
        prompt: str,
        field_metrics_config: Any,
        on_document_complete: "Callable[[PredictionResult], None] | None" = None,
        progress_bar: "tqdm | None" = None,
    ) -> "tuple[EvaluationResult, str | None]":
        """Run one model against a set of documents and return a scored result.

        The one method every subclass must implement -- it's the only place a model
        is actually called and its output turned into predictions. Everything else
        in ``_run_evaluations`` (concurrency, partial-result resume, progress
        reporting, persistence) is shared and calls this once per model.

        ``documents``/``labels`` are already filtered to exclude anything resumed
        from a partial prior run. Do not call ``on_document_complete`` and also move
        ``progress_bar`` for the same document -- ``on_document_complete`` already
        drives cost/progress-tracker/partial-store bookkeeping; ``progress_bar`` is
        handed through only for implementations whose own concurrency shape means
        they must move the shared bar themselves (e.g. a branch with its own
        internal per-document loop). Each returned ``PredictionResult`` doubles as
        that document's trace record (see ``get_traces``): put whatever is useful
        for debugging into its ``metadata``.

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
            progress_bar: The shared ``tqdm`` instance, when an implementation needs
                to advance it itself (see above); otherwise leave untouched.

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

        Shared across every task. Delegates the parts that vary -- actually calling
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
        if self.output_dir is not None:
            (Path(self.output_dir) / "models").mkdir(parents=True, exist_ok=True)

        total_docs = len(documents) * len(effective_models)
        running_cost: list[float] = [0.0]
        shared_bar = tqdm(total=total_docs, unit="doc", desc="Evaluating")

        def _on_doc(pred: PredictionResult) -> None:
            # Cost/postfix accounting only -- moving the bar is each
            # _evaluate_model_documents implementation's own job (it's the only
            # place that knows whether it needs to move it itself, e.g. a branch
            # with its own internal per-document loop, or whether _run_evaluations'
            # caller -- e.g. a shared per-document runner -- already did).
            running_cost[0] += pred.llm_cost + pred.evaluation_cost
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
            manipulations: list[Any],
            updated_prompt: "str | None",
        ) -> None:
            """Write the completed model result to disk and clean up its staging file."""
            if self.output_dir is None:
                return
            from valtron_core.runner import save_single_model_result

            try:
                effective_prompt = updated_prompt or model_prompts.get(model_label)
                override = (self._model_override_prompts or {}).get(model_label)
                save_single_model_result(
                    self.output_dir,
                    result,
                    model_prompt=effective_prompt,
                    prompt_manipulations=manipulations,
                    model_override_prompt=override,
                )
                if partial_store is not None:
                    partial_store.finalize(model_label)
            except Exception as e:
                logger.warning("eager_model_save_failed", model=model_label, error=str(e))

        async def _evaluate_single_model(
            index: int, model_config: Any
        ) -> "tuple[int, EvaluationResult, str, list[Any], str | None]":
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
            remaining_labels = [lb for d, lb in zip(documents, labels) if d.id not in completed_ids]

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
                progress_bar=shared_bar,
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
        ``EvaluationMetrics`` (accuracy, cost, timing) or the per-prediction
        ``task_scores``/``aggregated_task_scores`` bags at all (e.g. a corpus-level
        ranking that isn't shaped as one row per model). Wiring the result into
        report templates is a follow-up to this abstraction.

        Returns:
            A JSON-serializable dict, keyed however is most useful (e.g. by model
            label).
        """
        return {}

    # -------------------------------------------------------------------------
    # Reevaluation
    # -------------------------------------------------------------------------

    async def _score_predictions(
        self,
        predictions: "list[PredictionResult]",
        labels: "list[Any]",
        field_metrics_config: Any,
    ) -> "list[PredictionResult]":
        """Re-derive scoring fields on already-generated predictions. No model calls.

        Optional: only needed to use the default ``reevaluate()`` below. Default
        raises ``NotImplementedError``; a task that wants free rescoring implements
        this one small, pure hook instead of ``reevaluate()`` itself. A task whose
        rescoring needs more than "predictions in, rescored predictions out" (e.g.
        updating groundtruth by document id, or a scoring config that needs to be
        threaded through and persisted) should override ``reevaluate()`` directly
        instead -- this hook is deliberately narrower than that.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement _score_predictions(); "
            "override reevaluate() directly instead, or implement this hook."
        )

    def reevaluate(
        self,
        data: "list[dict[str, Any]] | str | Path | None" = None,
        output_dir: "str | Path | None" = None,
        **kwargs: Any,
    ) -> "Path | None":
        """Re-score stored predictions without new LLM calls, and optionally persist.

        Default implementation re-scores every prediction in ``self.results`` via
        ``_score_predictions()`` and recomputes metrics; raises ``NotImplementedError``
        (from ``_score_predictions``) if that hook isn't implemented. Override either
        hook, or this method directly, to support richer rescoring semantics (e.g.
        updating groundtruth in place).

        Args:
            data: Updated groundtruth, same shape as the constructor's ``data``.
                Not used by the default implementation; accepted for signature
                compatibility with overrides that do use it.
            output_dir: If given, writes results via ``save_experiment_results()``.
            **kwargs: Task-specific rescoring parameters.

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

        field_metrics_config = self._get_field_metrics_config()
        _, labels = self._load_documents_and_labels()

        return asyncio.run(self._reevaluate_async(labels, field_metrics_config, output_dir))

    async def _reevaluate_async(
        self,
        labels: "list[Any]",
        field_metrics_config: Any,
        output_dir: "str | Path | None",
    ) -> "Path | None":
        assert self.results is not None
        for result in self.results:
            result.predictions = await self._score_predictions(
                result.predictions, labels, field_metrics_config
            )
            result.compute_metrics()

        if output_dir is not None:
            return self.save_experiment_results(output_dir)
        return None

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

    def _resolve_output_dir(self, output_dir: "str | Path | None") -> Path:
        """Return the effective output directory, raising if neither source provides one."""
        effective = output_dir or self.output_dir
        if effective is None:
            raise ValueError(
                "output_dir is required. Set it in the config or pass it to the save method."
            )
        return Path(effective)

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

    def save_experiment_results(self, output_dir: "str | Path | None" = None) -> Path:
        """Write the run directory (``metadata.json`` + ``models/*.json``).

        Must be called after ``evaluate()``. Returns the path to the run
        directory that was written. Already correctness-agnostic -- works
        unchanged for a task that never sets ``is_correct``/``accuracy``.

        Args:
            output_dir: Override the output directory for this call. Falls back
                to ``config.output_dir`` if omitted. Raises if neither is set.
        """
        if self.results is None:
            raise RuntimeError("Call evaluate() before save_experiment_results().")

        from valtron_core.runner import save_run_dir

        dest = self._resolve_output_dir(output_dir)

        fmc = self._get_field_metrics_config()

        run_dir = save_run_dir(
            dest,
            self.results,
            self._build_save_documents(),
            use_case=self.use_case,
            original_prompt=self.prompt_template,
            field_config=fmc.config if fmc else None,
            model_prompts=self._model_prompts,
            prompt_manipulations=self._manipulations_applied,
            model_override_prompts=self._model_override_prompts,
            response_format_schema=getattr(self, "_response_format_schema", None),
            task_config=self._extra_metadata(),
        )
        return run_dir

    def save_html_report(self, output_dir: "str | Path | None" = None) -> Path:
        """Generate an HTML report. Not implemented at this level.

        Rich HTML/PDF reports (accuracy charts, correct/incorrect badges) assume a
        correctness notion this generic class makes no assumption about -- a
        capability a task opts into by overriding this, not one forced on it (same
        idiom as ``reevaluate()``). See ``ReferencedEval`` for the real
        implementation classification/extraction use.
        """
        raise NotImplementedError(f"{type(self).__name__} does not implement save_html_report().")

    def save_pdf_report(self, output_dir: "str | Path | None" = None) -> Path:
        """Generate a PDF report. Not implemented at this level; see ``save_html_report``."""
        raise NotImplementedError(f"{type(self).__name__} does not implement save_pdf_report().")

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
