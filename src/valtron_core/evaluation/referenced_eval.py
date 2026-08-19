"""Recipe for classification/extraction: schema-aware, ground-truth-scored evaluation."""

import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Sequence, cast

import litellm
import structlog
from pydantic import BaseModel, ConfigDict, Field, create_model
from tqdm import tqdm  # type: ignore[import-untyped]

from valtron_core.content_resolution import resolve_content
from valtron_core.cost_utils import _parse_time_unit_to_seconds
from valtron_core.decompose import (
    DecomposedEvaluator,
    cleanup_few_shot_sub_prompts,
    create_sub_schemas,
    decompose_few_shot_examples,
    filter_hallucinated_values,
    find_split_point,
    generate_sub_prompts,
    inject_few_shot_into_sub_prompts,
)
from valtron_core.evaluation.config import STRUCTURED_MANIPULATIONS, Manipulation, ModelEvalConfig
from valtron_core.evaluation.model_eval import ModelEval
from valtron_core.evaluator import _score_prediction
from valtron_core.few_shot_training_data_generator import (
    FewShotTrainingDataGenerator,
    LabeledExample,
)
from valtron_core.models import Document, FieldMetricsConfig, Label, PredictionResult
from valtron_core.prompt_optimizer import ExplanationEnhancer
from valtron_core.runner import EvaluationResult
from valtron_core.schema_synthesis import synthesize_pydantic_model
from valtron_core.scoring.json_eval import JsonEvaluator
from valtron_core.utilities.field_config_generator import infer_field_config

logger = structlog.get_logger()


class ReferencedEval(ModelEval):
    """
    Recipe for evaluating and comparing multiple models on a structured task.

    Handles the complete pipeline for both classification and extraction tasks:
    1. Optional: Generate additional training data via few-shot learning
    2. Optimize prompts per model (explanations, few-shot injection, repetition)
    3. Evaluate all models concurrently
    4. Generate a comprehensive report with metrics

    ``ReferencedEval`` itself never guesses a schema and never validates label shape;
    it just uses whatever ``response_format`` it's given (or none). Use
    ``ClassificationExperiment`` for label-mode data with schema auto-inference and
    upfront validation that every label is a plain string, or ``ExtractionExperiment``
    for extraction-mode data with upfront validation that a schema was actually
    given.
    """

    def __init__(
        self,
        config: ModelEvalConfig | dict[str, Any] | str | Path,
        data: list[dict[str, Any]] | str | Path,
        response_format: type[BaseModel] | None = None,
    ):
        """
        Initialize the model evaluation recipe.

        Args:
            config: Configuration dict, ModelEvalConfig, or path (str/Path) to a JSON config file.
                Required keys: ``models``, ``prompt`` (must contain ``{content}``).
                Optional keys: ``output_dir``, ``use_case``, ``temperature``, ``few_shot``,
                ``field_metrics_config``, ``response_format_schema``, ``output_formats``.
                ``response_format_schema`` accepts the litellm format:
                ``{"type": "json_schema", "json_schema": {"name": ..., "strict": true, "schema": {...}}}``.
            data: List of dicts ``[{"id": ..., "content": ..., "label": ...}]``,
                or a path to a JSON file with the same structure.
            response_format: Optional Pydantic model class for structured output validation.
                When provided, enables extraction mode and the structured manipulations
                (``decompose``, ``hallucination_filter``, ``multi_pass``).
                Takes priority over ``config.response_format_schema``.
        """
        # Validated here, before super().__init__(), because the base class's own
        # __init__ calls self.add_models(...) internally (dispatching polymorphically
        # to *this* class's override below), which needs self.response_format already
        # resolved to check the structured-manipulation guard. Passing the
        # already-validated config down means the base's own _validate_config call is
        # just a pass-through, not a second validation. _config_model() isn't
        # overridden here, so this is always a ModelEvalConfig at runtime.
        validated = cast(ModelEvalConfig, self._validate_config(config))
        self.response_format = response_format
        if self.response_format is None and validated.response_format_schema is not None:
            synthesized = synthesize_pydantic_model(validated.response_format_schema)
            if synthesized is not None:
                self.response_format = synthesized

        # Dict schema kept for API calls (synthesis fallback) and metadata serialization.
        # Only capture the config dict when no Pydantic model was passed; aevaluate()
        # will serialize from self.response_format in the user-passed-model case.
        self._response_format_schema: dict[str, Any] | None = (
            validated.response_format_schema if response_format is None else None
        )

        super().__init__(validated, data)

    def _post_init(self) -> None:
        self.enhancer = ExplanationEnhancer()
        # DecomposedEvaluator is only needed in extraction mode
        self.decomposed_evaluator = (
            DecomposedEvaluator(client=self.client) if self.response_format is not None else None
        )
        self.few_shot_config = self.config.few_shot
        self._field_metrics_config_raw = self.config.field_metrics_config
        self.few_shot_examples: list[Any] = []
        self._auto_wrap_string_labels: bool = self._compute_auto_wrap_string_labels()

        logger.info(
            "model_eval_initialized",
            num_models=len(self.models),
            num_documents=len(self.data),
            few_shot_enabled=self.few_shot_config is not None and self.few_shot_config.enabled,
            has_response_format=self.response_format is not None,
        )

    # -------------------------------------------------------------------------
    # Preflight
    # -------------------------------------------------------------------------

    def _validate_task_data(self) -> None:
        """ReferencedEval-specific preflight: model/response_format param support, label shape."""
        self._check_model_param_support()
        self._validate_labels_against_schema()

    def _check_model_param_support(self) -> None:
        from valtron_core.evaluation.config import LLMModelConfig

        wants_response_format = self.response_format is not None

        self._auto_wrap_string_labels = self._compute_auto_wrap_string_labels()

        if wants_response_format and self.data and not self._auto_wrap_string_labels:
            all_plain_string_labels = all(
                not isinstance(item.get("label"), (dict, list)) for item in self.data
            )
            if all_plain_string_labels:
                logger.warning(
                    "plain_string_labels_with_response_format",
                    action="scoring_may_be_incorrect",
                    detail="response_format is set but labels are plain strings -- "
                    "model outputs will be JSON but expected values will not match",
                )

        for mc in self.models:
            if not isinstance(mc, LLMModelConfig):
                continue
            try:
                kwargs: dict[str, Any] = {"model": mc.name}
                provider = mc.params.get("custom_llm_provider")
                if provider:
                    kwargs["custom_llm_provider"] = provider
                supported = litellm.get_supported_openai_params(**kwargs) or []
            except Exception:
                continue

            if "temperature" not in supported:
                logger.warning(
                    "temperature_not_supported",
                    model=mc.name,
                    action="temperature_will_be_dropped",
                )
            if wants_response_format and "response_format" not in supported:
                logger.warning(
                    "response_format_not_supported",
                    model=mc.name,
                    action="structured_output_will_be_skipped",
                )

    def _compute_auto_wrap_string_labels(self) -> bool:
        """Return True if plain string labels should be auto-wrapped as ``{"label": ...}``.

        Requires a resolved response schema with exactly one field named ``label``
        (str or Enum) and every document in ``self.data`` having a plain (non-dict/list)
        label. This is a pure function of ``response_format`` and ``data`` -- called at
        every entry point where either can change (construction, ``load_experiment_results``,
        ``reevaluate``) so the flag never depends on ``_preflight_check`` having run first.
        """
        if self.response_format is None or not self.data:
            return False
        if not self._is_single_label_field_schema():
            return False
        return all(not isinstance(item.get("label"), (dict, list)) for item in self.data)

    def _is_single_label_field_schema(self) -> bool:
        """Return True if the response schema has exactly one field named 'label' (str or Enum)."""
        import enum
        from typing import Literal, get_origin

        if self.response_format is None:
            return False
        fields = self.response_format.model_fields
        if len(fields) != 1 or "label" not in fields:
            return False
        annotation = fields["label"].annotation
        return (
            annotation is str
            or get_origin(annotation) is Literal
            or (isinstance(annotation, type) and issubclass(annotation, enum.Enum))
        )

    def _validate_labels_against_schema(self) -> None:
        """Validate each label against the response schema, raising if any fail.

        Handles both Pydantic response_format (via model_validate_json) and
        response_format_schema dicts (via jsonschema). Auto-wrap is applied before
        validation so plain string labels destined for wrapping are checked in their
        final form. Plain string labels without auto-wrap are skipped -- they are not
        JSON-structured and are already covered by the warning in _check_model_param_support.
        """
        import jsonschema  # type: ignore[import-untyped]

        if self.response_format is None and self._response_format_schema is None:
            return

        json_schema: dict[str, Any] | None = None
        if self.response_format is None:
            try:
                json_schema = self._response_format_schema["json_schema"]["schema"]  # type: ignore[index]
            except (KeyError, TypeError):
                return

        errors: list[str] = []
        for idx, item in enumerate(self.data):
            record_id = item.get("id", f"index {idx}")
            label_raw = item.get("label", "")

            if isinstance(label_raw, (dict, list)):
                label_obj: Any = label_raw
            elif self._auto_wrap_string_labels:
                label_obj = {"label": str(label_raw)}
            else:
                continue

            try:
                if self.response_format is not None:
                    self.response_format.model_validate_json(json.dumps(label_obj))
                elif json_schema is not None:
                    jsonschema.validate(instance=label_obj, schema=json_schema)
            except Exception as exc:
                errors.append(f"  record {record_id!r}: label={label_raw!r} -- {exc}")

        if errors:
            raise ValueError(
                f"Labels failed schema validation ({len(errors)} record(s)):\n" + "\n".join(errors)
            )

    # -------------------------------------------------------------------------
    # Model management
    # -------------------------------------------------------------------------

    def add_models(self, models: "Sequence[str | dict[str, Any] | Any]") -> None:
        """Add new models to the experiment.

        All model validation (uniqueness, structured-manipulation guards) is
        handled here — ``__init__`` delegates to this method for its own model
        initialization.  On the next ``evaluate()`` / ``run()`` call only newly
        added models are evaluated; models that already have results are skipped
        automatically.

        Args:
            models: List of model config dicts or ``ModelConfig`` objects.

        Raises:
            ValueError: Duplicate label or structured manipulation without
                ``response_format``.
        """
        from valtron_core.evaluation.config import LLMModelConfig, TransformerModelConfig

        normalized: list[Any] = []
        for m in models:
            if isinstance(m, dict):
                model_type = m.get("type", "llm")
                if model_type == "transformer":
                    normalized.append(TransformerModelConfig.model_validate(m))
                else:
                    normalized.append(LLMModelConfig.model_validate(m))
            else:
                normalized.append(m)

        existing_labels = {str(mc.label or getattr(mc, "name", None)) for mc in self.models}
        seen_in_batch: set[str] = set()
        for mc in normalized:
            model_name = getattr(mc, "name", None)
            label = str(mc.label or model_name)
            label_source = (
                f"label={mc.label!r}"
                if mc.label
                else f"name={model_name!r} (label inferred from name)"
            )
            if label in existing_labels:
                raise ValueError(
                    f"Duplicate model label {label!r} in config ({label_source}). "
                    "Each model entry must have a unique label. "
                    "You can use the same model twice by giving one entry a distinct label "
                    "(e.g. label='gpt-5-mini-v2')."
                )
            if label in seen_in_batch:
                raise ValueError(
                    f"Duplicate model label {label!r} in config ({label_source}). "
                    "Each model entry must have a unique label. "
                    "You can use the same model twice by giving one entry a distinct label "
                    "(e.g. label='gpt-5-mini-v2')."
                )
            seen_in_batch.add(label)

        structured_requested = [
            (str(mc.label or getattr(mc, "name", None)), manip)
            for mc in normalized
            for manip in getattr(mc, "prompt_manipulation", [])
            if manip in STRUCTURED_MANIPULATIONS
        ]
        has_schema = self.response_format is not None
        if structured_requested and not has_schema:
            bad_models = sorted({name for name, _ in structured_requested})
            bad_manips = sorted({manip.value for _, manip in structured_requested})
            raise ValueError(
                f"Model(s) {bad_models} use structured manipulation(s) {bad_manips}, "
                "which require response_format to be provided."
            )

        self.models.extend(normalized)
        self.config.models.extend(normalized)

    # -------------------------------------------------------------------------
    # Parsing helpers (used by load_experiment_results)
    # -------------------------------------------------------------------------

    @staticmethod
    def _model_data_from_file(model_file: Path) -> "dict[str, Any]":
        """Read one models/<name>.json and return all its data.

        Returns a dict with both model-config fields (for ``add_models``) and
        evaluation-result fields (predictions, metrics, etc.) so that
        ``load_experiment_results`` can reconstruct both from a single parse.
        """
        with open(model_file) as f:
            raw = json.load(f)

        llm_config: dict[str, Any] = raw.get("llm_config") or {}
        model_name = llm_config.get("model") or raw.get("model", "")
        model_label = raw.get("model", model_name)
        params = {k: v for k, v in llm_config.items() if k != "model"}
        override_prompt = raw.get("override_prompt")
        manipulations = raw.get("prompt_manipulations") or []

        return {
            # Config fields
            "name": model_name,
            "label": model_label if model_label != model_name else None,
            "params": params,
            "prompt": override_prompt,
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
            "override_prompt": override_prompt,
        }

    @staticmethod
    def _config_and_data_from_metadata(  # type: ignore[override]
        metadata_path: Path,
    ) -> "tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]":
        """Read metadata.json and return ``(config_dict, data, response_format_schema)``.

        Deliberately not the same shape as the generic base's version (an extra
        response_format_schema element) -- safe in practice because
        load_experiment_results() is also independently overridden below and is
        the only caller of this method.

        ``config_dict`` contains the keys needed to construct a ``ModelEvalConfig``
        (minus ``models``, which the caller fills in).  ``data`` is the raw
        document list ``[{"id": ..., "content": ..., "label": ...}]``.
        ``response_format_schema`` is the Pydantic JSON Schema stored from the
        original run, or ``None`` if absent.
        """
        with open(metadata_path) as f:
            meta = json.load(f)

        original_prompt = meta.get("original_prompt") or "{content}"
        config_dict: dict[str, Any] = {
            "prompt": original_prompt,
            "use_case": meta.get("use_case", "model evaluation"),
        }
        if meta.get("field_metrics_config"):
            config_dict["field_metrics_config"] = meta["field_metrics_config"]

        data: list[dict[str, Any]] = meta.get("documents", [])
        return config_dict, data, meta.get("response_format_schema")

    # -------------------------------------------------------------------------
    # Load from disk
    # -------------------------------------------------------------------------

    @classmethod
    def load_experiment_results(cls, dir_path: "str | Path") -> "ReferencedEval":
        """Restore a previously saved experiment from disk.

        Returns a ``ReferencedEval`` instance in the same state as after
        ``evaluate()`` — ``self.results``, ``self._model_prompts``,
        ``self._manipulations_applied``, and ``self._model_override_prompts``
        are all populated.  The instance is ready for ``save_html_report()``,
        ``add_models()`` + ``run()``, or any other post-evaluate operation.

        Args:
            dir_path: Directory previously written by ``save_experiment_results()``.
                Must contain ``metadata.json`` and a ``models/`` sub-directory.

        Raises:
            FileNotFoundError: ``metadata.json`` is absent.
            ValueError: ``models/`` directory is empty.
        """
        from valtron_core.models import EvaluationMetrics, EvaluationResult

        dir_path = Path(dir_path)
        metadata_path = dir_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"No metadata.json found in {dir_path}. "
                "Pass the directory written by save_experiment_results()."
            )

        config_dict, data, response_format_schema = cls._config_and_data_from_metadata(
            metadata_path
        )
        if response_format_schema:
            config_dict["response_format_schema"] = response_format_schema

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
        if response_format_schema:
            instance._response_format_schema = response_format_schema
            if instance.response_format is None:
                synthesized = synthesize_pydantic_model(response_format_schema)
                if synthesized is not None:
                    instance.response_format = synthesized
                    instance.decomposed_evaluator = DecomposedEvaluator(client=instance.client)
                    instance._auto_wrap_string_labels = instance._compute_auto_wrap_string_labels()

        label_map = {
            str(d.get("id", "")): (
                json.dumps(d["label"])
                if isinstance(d.get("label"), (dict, list))
                else str(d.get("label", ""))
            )
            for d in data
        }

        results: list[EvaluationResult] = []
        model_prompts: dict[str, str] = {}
        manipulations_applied: dict[str, list[Any]] = {}
        model_override_prompts: dict[str, str] = {}

        for md in all_model_data:
            model_label = md["label"] or md["name"]
            model_prompts[model_label] = md["prompt_template"]
            manipulations_applied[model_label] = md["prompt_manipulation"]
            if md.get("override_prompt"):
                model_override_prompts[model_label] = md["override_prompt"]

            try:
                from valtron_core.scoring.json_eval import EvalResult

                _eval_result_cls = EvalResult
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
                        expected_value=p.get("expected_value", label_map.get(p["document_id"], "")),
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
        instance._model_override_prompts = model_override_prompts or None
        return instance

    # -------------------------------------------------------------------------
    # Field metrics
    # -------------------------------------------------------------------------

    def _get_field_metrics_config(self) -> FieldMetricsConfig | None:
        """Return a FieldMetricsConfig, either from explicit config or auto-inferred.

        If ``field_metrics_config`` was provided in the recipe config, that is
        validated and returned. Otherwise the config is inferred from label data:
        JSON labels use the label structure directly; plain-text labels are wrapped
        in the ``{"label": ...}`` shape used by the string-label wrapper. Wrapping is
        only applied when ``_auto_wrap_string_labels`` holds for the whole dataset,
        matching the wrapping actually applied elsewhere during the run.
        """
        if self._field_metrics_config_raw is not None:
            return FieldMetricsConfig.model_validate(self._field_metrics_config_raw)

        if not self.data:
            return None

        first_label = self.data[0].get("label", "")
        if isinstance(first_label, (dict, list)):
            first_label = json.dumps(first_label)

        try:
            json.loads(first_label)
            field_config = infer_field_config(first_label)
            return FieldMetricsConfig(config=field_config.model_dump())
        except (json.JSONDecodeError, TypeError):
            if not self._auto_wrap_string_labels:
                return None
            field_config = infer_field_config(json.dumps({"label": first_label}))
            return FieldMetricsConfig(config=field_config.model_dump())

    # -------------------------------------------------------------------------
    # Reevaluation
    # -------------------------------------------------------------------------

    @staticmethod
    def _rescore_prediction(
        prediction: PredictionResult,
        field_metrics_config: "FieldMetricsConfig | None",
        extra_template_vars: "dict[str, Any] | None" = None,
        json_evaluator: "JsonEvaluator | None" = None,
    ) -> PredictionResult:
        """Re-score a stored prediction without making any LLM calls.

        Delegates to ``_score_prediction`` from the evaluator module, which uses
        JsonEvaluator when a field_metrics_config is provided and falls back to
        case-insensitive string comparison otherwise.
        """
        if not isinstance(prediction.predicted_value, str):
            raise TypeError(
                f"Cannot rescore document {prediction.document_id!r}: predicted_value is a "
                f"{type(prediction.predicted_value).__name__}, expected a string."
            )
        if prediction.expected_value is None:
            raise ValueError(
                f"Cannot rescore document {prediction.document_id!r}: no expected_value is "
                "stored for it."
            )
        field_metrics, example_score, is_correct, evaluation_cost = _score_prediction(
            predicted_value=prediction.predicted_value,
            expected_value=prediction.expected_value,
            field_metrics_config=field_metrics_config,
            extra_template_vars=extra_template_vars,
            document_id=prediction.document_id,
            json_evaluator=json_evaluator,
        )
        return prediction.model_copy(
            update={
                "field_metrics": field_metrics,
                "example_score": example_score,
                "is_correct": is_correct,
                "evaluation_cost": evaluation_cost,
            }
        )

    def reevaluate(  # type: ignore[override]
        self,
        data: "list[dict[str, Any]] | str | Path | None" = None,
        output_dir: "str | Path | None" = None,
        *,
        field_metrics_config: "FieldMetricsConfig | dict[str, Any] | None" = None,
    ) -> "Path | None":
        """Re-score stored predictions with a new field_metrics_config or ground truth.

        No LLM calls are made. Only the scoring step is re-run against the
        ``predicted_value`` / ``expected_value`` pairs already stored in
        ``self.results``.

        Typical workflow::

            me = ReferencedEval.load_experiment_results("path/to/run")
            me.reevaluate(
                field_metrics_config={"config": {...}},
                output_dir="path/to/new_run",
            )

        Args:
            field_metrics_config: New scoring config. Accepts a raw dict (same
                format as the ``field_metrics_config`` config key) or a
                ``FieldMetricsConfig`` instance. When omitted, the existing config
                on this instance is used.
            data: Updated ground truth in the same format as the constructor
                ``data`` argument (``[{"id": ..., "content": ..., "label": ...}]``),
                or a path (str/Path) to a JSON file with that structure.
                Only ``id`` and ``label`` are used; unknown IDs are warned and
                skipped. Documents not present in the new list keep their previous
                expected values.
            output_dir: If provided, writes the re-scored results to this directory
                via ``save_experiment_results()``. Note: ``metadata.json`` is NOT
                overwritten if it already exists in that directory -- only the per-
                model JSON files are updated. Pass a fresh directory to preserve the
                updated field_metrics_config in the saved metadata.

        Returns:
            Path to the run directory if ``output_dir`` was provided, else ``None``.

        Raises:
            ValueError: If ``self.results`` is ``None`` (evaluate or load first).
        """
        if self.results is None:
            raise ValueError(
                "No results to reevaluate. Call evaluate() or load_experiment_results() first."
            )

        # Load data from file path if a string/Path was passed
        resolved_data: list[dict[str, Any]] | None
        if isinstance(data, (str, Path)):
            with open(data) as f:
                resolved_data = json.load(f)
        else:
            resolved_data = data

        # Update ground truth labels from caller-supplied data
        if resolved_data is not None:
            self.data = resolved_data
            self._auto_wrap_string_labels = self._compute_auto_wrap_string_labels()

            new_label_map: dict[str, str] = {}
            for item in resolved_data:
                label_raw = item.get("label", "")
                if isinstance(label_raw, (dict, list)):
                    serialized = json.dumps(label_raw)
                elif self._auto_wrap_string_labels:
                    serialized = json.dumps({"label": str(label_raw)})
                else:
                    serialized = str(label_raw)
                new_label_map[str(item.get("id", ""))] = serialized

            existing_ids: set[str] = {p.document_id for er in self.results for p in er.predictions}
            for doc_id in new_label_map:
                if doc_id not in existing_ids:
                    logger.warning(
                        "reevaluate_unknown_doc_id",
                        document_id=doc_id,
                        action="skipped",
                    )

            for eval_result in self.results:
                eval_result.predictions = [
                    (
                        p.model_copy(update={"expected_value": new_label_map[p.document_id]})
                        if p.document_id in new_label_map
                        else p
                    )
                    for p in eval_result.predictions
                ]

        # Resolve the effective FieldMetricsConfig and update stored config
        effective_fmc: FieldMetricsConfig | None
        if field_metrics_config is not None:
            if isinstance(field_metrics_config, dict):
                effective_fmc = FieldMetricsConfig.model_validate(field_metrics_config)
                raw_dict: dict[str, Any] = field_metrics_config
            else:
                effective_fmc = field_metrics_config
                raw_dict = {"config": field_metrics_config.config}
            self._field_metrics_config_raw = raw_dict
            self.config.field_metrics_config = raw_dict
        else:
            effective_fmc = self._get_field_metrics_config()

        # Build doc_content_map so _rescore_prediction can pass extra_template_vars
        # (prediction.metadata is not persisted after load_experiment_results)
        doc_content_map: dict[str, Any] = {
            str(item.get("id", f"doc_{i}")): item.get("content", "")
            for i, item in enumerate(self.data)
        }

        json_evaluator = (
            JsonEvaluator(
                custom_metrics=effective_fmc.custom_metrics,
                custom_aggs=effective_fmc.custom_aggs,
            )
            if effective_fmc is not None
            else None
        )

        # Re-score all predictions and recompute aggregated metrics
        for eval_result in self.results:
            new_predictions = []
            for p in eval_result.predictions:
                content = doc_content_map.get(p.document_id)
                extra_vars: dict[str, Any] = (
                    {f"example_{k}": v for k, v in content.items()}
                    if isinstance(content, dict)
                    else {}
                )
                new_predictions.append(
                    self._rescore_prediction(p, effective_fmc, extra_vars, json_evaluator)
                )
            eval_result.predictions = new_predictions
            eval_result.compute_metrics()

        if output_dir is not None:
            resolved = Path(output_dir)
            if (resolved / "metadata.json").exists():
                logger.warning(
                    "reevaluate_metadata_not_overwritten",
                    output_dir=str(resolved),
                    detail=(
                        "metadata.json already exists and will not be overwritten. "
                        "Only per-model JSON files will be updated. "
                        "Pass a fresh output_dir to preserve the new field_metrics_config."
                    ),
                )
            return self.save_experiment_results(output_dir)

        return None

    # -------------------------------------------------------------------------
    # Setup before prompt resolution
    # -------------------------------------------------------------------------

    async def _before_evaluation(self, field_metrics_config: "FieldMetricsConfig | None") -> None:
        """Serialize the response schema (if any) and generate few-shot examples (if enabled).

        Runs after preflight, before _prepare_model_prompts() -- both need to have
        happened by the time prompts are resolved, and few-shot generation is
        deliberately gated behind preflight so a fatal config problem fails before
        any API calls are made for it.
        """
        if self._response_format_schema is None and self.response_format is not None:
            self._response_format_schema = self._serialize_response_format_schema(
                self.response_format
            )

        if self.few_shot_config and self.few_shot_config.enabled:
            self._status("Generating few-shot examples...")
            await self._generate_few_shot_data()

    async def _generate_few_shot_data(self) -> None:
        """Generate additional training data using few-shot learning.

        Only ever called from _before_evaluation, already guarded on
        few_shot_config.enabled -- the assert documents that precondition for
        the type checker rather than re-guarding it here.
        """
        assert self.few_shot_config is not None
        logger.info("generating_few_shot_data")
        phase_start = time.perf_counter()

        examples = [
            LabeledExample(document=item["content"], label=item["label"]) for item in self.data
        ]

        generator = FewShotTrainingDataGenerator(
            prompt=self.prompt_template,
            examples=examples[: self.few_shot_config.max_seed_examples],
            max_few_shots=self.few_shot_config.max_few_shots,
            source_data=self.data,
        )

        result = await generator.generate_and_validate_examples(
            generator_model=self.few_shot_config.generator_model,
            num_examples=self.few_shot_config.num_examples,
        )

        correct_examples = [ex for ex in result["examples"] if ex["consensus"] == "correct"]

        # Keep first 5 correct examples for few-shot prompting
        self.few_shot_examples = correct_examples[:5]

        logger.info(
            "few_shot_generation_complete",
            generated=len(result["examples"]),
            correct=len(correct_examples),
            kept_for_few_shot=len(self.few_shot_examples),
            total_cost=result["costs"]["total_cost"],
            duration_s=round(time.perf_counter() - phase_start, 2),
        )

    # -------------------------------------------------------------------------
    # Data loading
    # -------------------------------------------------------------------------

    def _load_documents_and_labels(self) -> tuple[list[Document], list[Label]]:
        """Convert self.data into Document and Label objects (no disk I/O)."""
        documents: list[Document] = []
        labels: list[Label] = []
        for idx, item in enumerate(self.data):
            doc_id = str(item.get("id", f"doc_{idx}"))
            label_raw = item.get("label", "")
            if isinstance(label_raw, (dict, list)):
                label_value = json.dumps(label_raw)
            elif self._auto_wrap_string_labels:
                label_value = json.dumps({"label": str(label_raw)})
            else:
                label_value = str(label_raw)
            documents.append(
                Document(
                    id=doc_id,
                    content=resolve_content(item, self._data_base_dir),
                    metadata=item.get("metadata", {}),
                    attachments=item.get("attachments", []),
                )
            )
            labels.append(Label(document_id=doc_id, value=label_value))
        return documents, labels

    # -------------------------------------------------------------------------
    # Per-model evaluation
    # -------------------------------------------------------------------------

    def _cache_key_inputs(self, model_config: Any, prompt: str) -> tuple[str, dict[str, Any]]:
        """Transformer-vs-LLM cache-key shape; overrides the generic default.

        A transformer's cache validity depends on its model_path, not a prompt
        (transformers don't use one); an LLM's depends on the resolved prompt plus
        its litellm params, same as the generic default would compute.
        """
        model_name = getattr(model_config, "name", None) or model_config.label
        model_label = model_config.label or model_name
        if model_config.type == "transformer":
            hash_prompt = f"transformer:{getattr(model_config, 'model_path', model_label)}"
            hash_model_params: dict[str, Any] = {
                "type": "transformer",
                "model_path": getattr(model_config, "model_path", ""),
            }
        else:
            hash_prompt = prompt
            raw_model_arg = self._build_model_arg(model_config)
            hash_model_params = (
                raw_model_arg if isinstance(raw_model_arg, dict) else {"model": model_name}
            )
        return hash_prompt, hash_model_params

    async def _prepare_model_prompts(self) -> dict[str, str]:
        """Prepare prompts for each model, applying prompt manipulations as configured.

        Returns a dict mapping model label (or name) → final prompt string.
        Transformer models always get an empty string (they do not use prompts).
        """
        model_prompts: dict[str, str] = {}
        override_prompts: dict[str, str] = {}

        for model_config in self.models:
            model_label = model_config.label or model_config.name

            # Transformer models don't use prompts
            if model_config.type == "transformer":
                model_prompts[model_label] = ""
                continue

            manipulations = model_config.prompt_manipulation
            base_prompt = model_config.prompt or self.prompt_template
            if model_config.prompt:
                override_prompts[model_label] = model_config.prompt
            prompt = base_prompt

            # few_shot: skip when decompose is also present (decompose handles its own
            # few-shot injection into sub-prompts)
            if (
                Manipulation.few_shot in manipulations
                and self.few_shot_examples
                and Manipulation.decompose not in manipulations
            ):
                logger.info(
                    "applying_few_shot_manipulation",
                    model=model_label,
                    num_examples=len(self.few_shot_examples),
                )
                prompt = self._inject_few_shot_examples(prompt)

            if Manipulation.explanation in manipulations:
                logger.info("applying_explanation_manipulation", model=model_label)
                result = await self.enhancer.optimize(prompt)
                prompt = result["enhanced_prompt"]

            if Manipulation.prompt_repetition_x3 in manipulations:
                logger.info("applying_prompt_repetition_x3", model=model_label)
                prompt = (
                    prompt
                    + "\n\nLet me repeat that:\n\n"
                    + prompt
                    + "\n\nLet me repeat that one more time:\n\n"
                    + prompt
                )
            elif Manipulation.prompt_repetition in manipulations:
                logger.info("applying_prompt_repetition", model=model_label)
                prompt = prompt + "\n\nLet me repeat that:\n\n" + prompt

            model_prompts[model_label] = prompt

        self._model_override_prompts = override_prompts
        return model_prompts

    def _inject_few_shot_examples(self, prompt: str) -> str:
        """Inject few-shot examples into the prompt before the {content} placeholder."""
        if not self.few_shot_examples:
            return prompt

        examples_text = "\n\nHere are some examples:\n\n"
        for i, example in enumerate(self.few_shot_examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Document: {example['document']}\n"
            examples_text += f"Label: {example['label']}\n\n"

        if "{content}" in prompt:
            parts = prompt.split("{content}", 1)
            # Use task-appropriate suffix depending on mode
            action = "classify" if self.response_format is None else "extract from"
            enhanced_prompt = (
                parts[0] + examples_text + f"Now {action} this document:\n\n{{content}}" + parts[1]
            )
        else:
            enhanced_prompt = prompt + examples_text

        return enhanced_prompt

    # -------------------------------------------------------------------------
    # Response format helpers (label mode)
    # -------------------------------------------------------------------------

    def _has_json_schema_in_prompt(self, prompt: str) -> bool:
        """Check if the prompt contains a JSON schema example."""
        json_pattern = r'\{[^{}]*(?:"[^"]*"[^{}]*)*\}'
        for match in re.finditer(json_pattern, prompt):
            try:
                parsed = json.loads(match.group(0))
                if isinstance(parsed, dict):
                    return True
            except json.JSONDecodeError:
                continue
        return False

    def _serialize_response_format_schema(
        self, rf: type[BaseModel] | None
    ) -> dict[str, Any] | None:
        """Return the litellm-compatible response format dict for a Pydantic model.

        Returns ``None`` when *rf* is ``None``.
        """
        if rf is None:
            return None
        schema = rf.model_json_schema()
        return {
            "type": "json_schema",
            "json_schema": {
                "name": schema.get("title", "ResponseModel"),
                "strict": True,
                "schema": schema,
            },
        }

    # -------------------------------------------------------------------------
    # Response format helpers (extraction mode)
    # -------------------------------------------------------------------------

    def _create_explanation_model(self) -> type[BaseModel] | None:
        """Wrap response_format with an added explanation field.

        Used in extraction mode when the ``explanation`` manipulation is active,
        so the LLM can return its reasoning alongside the structured output.
        """
        if self.response_format is None:
            return None

        field_definitions: dict[str, Any] = {
            "explanation": (str, Field(description="Reasoning explanation")),
        }
        for field_name, field_info in self.response_format.model_fields.items():
            field_definitions[field_name] = (field_info.annotation, field_info)

        return create_model(
            f"{self.response_format.__name__}WithExplanation",
            __config__=ConfigDict(extra="forbid"),
            **field_definitions,
        )

    # -------------------------------------------------------------------------
    # Transformer evaluation (label mode only)
    # -------------------------------------------------------------------------

    async def _evaluate_transformer(
        self,
        model_config: Any,
        documents: list[Document],
        field_metrics_config: FieldMetricsConfig | None,
        on_document_complete: "Any | None" = None,
    ) -> EvaluationResult:
        """Evaluate a local transformer model.

        Args:
            model_config: TransformerModelConfig with ``model_path`` set.
            documents: Pre-loaded Document objects.
            field_metrics_config: Field-level metric config; falls back to exact
                string comparison when ``None``.

        Returns:
            EvaluationResult compatible with LLM evaluation results.
        """
        model_name = model_config.label
        if not model_config.model_path:
            raise ValueError(
                f"Transformer model '{model_name}' requires a 'model_path' in the config. "
                "Set model_path to the directory produced by train_transformer() "
                "(e.g. './my_model/final_model')."
            )
        model_path = model_config.model_path

        logger.info("evaluating_transformer", model=model_name, path=model_path)

        from valtron_core.transformer_wrapper import TransformerModelWrapper

        transformer = TransformerModelWrapper(model_path, model_name)

        # Build label map from self.data (no file I/O)
        label_map: dict[str, str] = {}
        for idx, item in enumerate(self.data):
            doc_id = str(item.get("id", f"doc_{idx}"))
            label_raw = item.get("label", "")
            if isinstance(label_raw, (dict, list)):
                label_map[doc_id] = json.dumps(label_raw)
            elif self._auto_wrap_string_labels:
                label_map[doc_id] = json.dumps({"label": str(label_raw)})
            else:
                label_map[doc_id] = str(label_raw)

        run_id = str(uuid.uuid4())
        result = EvaluationResult(
            run_id=run_id,
            started_at=datetime.now(),
            prompt_template=f"Transformer model: {model_path}",
            model=model_name,
            status="running",
        )

        json_evaluator = (
            JsonEvaluator(
                custom_metrics=field_metrics_config.custom_metrics or None,
                custom_aggs=field_metrics_config.custom_aggs or None,
            )
            if field_metrics_config is not None
            else None
        )

        start_time = time.time()

        for doc in documents:
            expected_label = label_map.get(doc.id, "")

            pred_start = time.time()
            prediction, confidence = transformer.predict_with_confidence(doc.content)
            pred_time = time.time() - pred_start

            if self._auto_wrap_string_labels:
                prediction = json.dumps({"label": prediction})

            if json_evaluator is not None:
                cfg = field_metrics_config.config  # type: ignore[union-attr]
                eval_expected = expected_label
                eval_predicted = prediction
                try:
                    json.loads(expected_label)
                except (json.JSONDecodeError, ValueError):
                    fields = cfg.get("fields") or {} if isinstance(cfg, dict) else {}
                    if isinstance(cfg, dict) and cfg.get("type") == "object" and len(fields) == 1:
                        field_name = next(iter(fields))
                        eval_expected = json.dumps({field_name: expected_label})
                        eval_predicted = json.dumps({field_name: prediction})
                eval_result, _ = json_evaluator.evaluate(cfg, eval_expected, eval_predicted)
                is_correct = eval_result.score == 1.0
            else:
                is_correct = prediction.strip() == expected_label.strip()

            pred_result = PredictionResult(
                document_id=doc.id,
                predicted_value=prediction,
                expected_value=expected_label,
                is_correct=is_correct,
                response_time=pred_time,
                llm_cost=0.0,
                model=model_name,
                metadata={"content": doc.content},
                confidence_score=confidence,
            )
            result.add_prediction(pred_result)
            if on_document_complete is not None:
                on_document_complete(pred_result)

        if model_config.cost_rate is not None:
            unit_seconds = _parse_time_unit_to_seconds(model_config.cost_rate_time_unit)
            for p in result.predictions:
                p.llm_cost = float(model_config.cost_rate) * (p.response_time / unit_seconds)
            result.llm_config = result.llm_config or {}
            result.llm_config["cost_rate"] = model_config.cost_rate
            result.llm_config["cost_rate_time_unit"] = model_config.cost_rate_time_unit

        result.completed_at = datetime.now()
        result.status = "completed"
        result.compute_metrics()

        duration = time.time() - start_time
        logger.info(
            "transformer_evaluation_complete",
            model=model_name,
            run_id=run_id,
            total=len(result.predictions),
            accuracy=result.metrics.accuracy if result.metrics else 0.0,
            duration=duration,
        )

        return result

    # -------------------------------------------------------------------------
    # Evaluation loop
    # -------------------------------------------------------------------------

    async def _evaluate_model_documents(
        self,
        model_config: Any,
        documents: list[Document],
        labels: list[Label],
        prompt: str,
        field_metrics_config: "FieldMetricsConfig | None",
        on_document_complete: "Callable[[PredictionResult], None] | None" = None,
        progress_bar: "tqdm | None" = None,
    ) -> tuple[EvaluationResult, str | None]:
        """Call one model against a batch of documents and score the results.

        The four branches (transformer / decompose / hallucination-filter-wrapped /
        plain) are exactly what the pre-refactor pipeline ran inline per model; this
        is that same logic, now reachable as the one seam a different task type
        would implement instead.
        """
        model_name = getattr(model_config, "name", None) or model_config.label
        manipulations = getattr(model_config, "prompt_manipulation", [])

        # --- Transformer branch (label mode only; guarded at __init__) ---
        if model_config.type == "transformer":

            def _on_doc_transformer(pred: PredictionResult) -> None:
                if on_document_complete is not None:
                    on_document_complete(pred)
                if progress_bar is not None:
                    progress_bar.update(1)

            result = await self._evaluate_transformer(
                model_config,
                documents,
                field_metrics_config,
                on_document_complete=_on_doc_transformer,
            )
            return result, None

        # --- Determine effective response format ---
        effective_rf: type[BaseModel] | dict[str, Any] | None
        if self.response_format is not None:
            # Extraction mode: use provided schema, wrapping with explanation field if needed
            if Manipulation.explanation in manipulations:
                effective_rf = self._create_explanation_model()
            else:
                effective_rf = self.response_format
        elif self._response_format_schema is not None:
            effective_rf = self._response_format_schema
        else:
            effective_rf = None

        # --- Decompose branch (extraction mode only; guarded at __init__) ---
        if Manipulation.decompose in manipulations and self.response_format is not None:
            result, sub_prompts = await self._run_decomposed_evaluation(
                documents=documents,
                labels=labels,
                prompt=prompt,
                model_name=model_name,
                model_config=model_config,
                manipulations=manipulations,
                field_metrics_config=field_metrics_config,
                on_document_complete=on_document_complete,
            )
            # NB: progress_bar is deliberately untouched here -- this faithfully
            # preserves a pre-refactor gap: the decompose branch never advanced the
            # shared bar per-document either. Not this refactor's job to fix.
            return result, self._format_sub_prompts_for_display(sub_prompts)

        post_extraction_filter = None
        if Manipulation.hallucination_filter in manipulations and self.response_format is not None:

            async def _hallucination_filter(
                predicted_json: Any, document: Document, _model: str = model_name
            ) -> Any:
                return await filter_hallucinated_values(
                    predicted_json,
                    document.content,
                    _model,
                    self.client,
                )

            post_extraction_filter = _hallucination_filter

        multi_pass = 2 if Manipulation.multi_pass in manipulations else 1

        result = await self.runner.evaluate(
            documents=documents,
            labels=labels,
            prompt_template=prompt,
            model=self._build_model_arg(model_config),
            # EvaluationRunner.evaluate()'s response_format is typed as
            # type[BaseModel] | None, but it genuinely accepts a raw litellm
            # schema dict too (the response_format_schema fallback path below);
            # pre-existing, not narrowed by this refactor.
            response_format=effective_rf,  # type: ignore[arg-type]
            field_metrics_config=field_metrics_config,
            post_extraction_filter=post_extraction_filter,
            multi_pass=multi_pass,
            _tqdm_bar=progress_bar,
            _on_document_complete=on_document_complete,
        )
        return result, None

    async def _run_decomposed_evaluation(
        self,
        documents: list[Document],
        labels: list[Label],
        prompt: str,
        model_name: str,
        model_config: Any,
        manipulations: list[Any],
        field_metrics_config: FieldMetricsConfig | None,
        on_document_complete: "Callable[[PredictionResult], None] | None" = None,
    ) -> tuple[EvaluationResult, dict[str, str]]:
        """Run evaluation with decomposed sub-prompts for each entity field.

        Returns:
            Tuple of (EvaluationResult, sub_prompts dict).

        Only ever called from _evaluate_model_documents's decompose branch,
        already gated on self.response_format is not None -- the assert documents
        that precondition for the type checker (and implies decomposed_evaluator
        is also set, per _post_init).

        :param on_document_complete: Optional per-document callback for live progress.
        """
        assert self.response_format is not None
        assert self.decomposed_evaluator is not None
        split_info = find_split_point(self.response_format)

        if split_info is None:
            logger.warning(
                "decompose_no_split_point",
                model=model_name,
                msg="No suitable split point found; falling back to normal evaluation.",
            )
            if Manipulation.explanation in manipulations and self.response_format:
                effective_rf = self._create_explanation_model()
            else:
                effective_rf = self.response_format
            result = await self.runner.evaluate(
                documents=documents,
                labels=labels,
                prompt_template=prompt,
                model=self._build_model_arg(model_config),
                response_format=effective_rf,
                field_metrics_config=field_metrics_config,
                _on_document_complete=on_document_complete,
            )
            return result, {}

        include_explanation = Manipulation.explanation in manipulations
        sub_schemas = create_sub_schemas(split_info, self.response_format, include_explanation)

        dc = model_config.decompose_config
        custom_sub_prompts = dc.sub_prompts if dc else None
        rewrite_model = dc.rewrite_model if dc else "gpt-4o-mini"
        sub_prompts = await generate_sub_prompts(
            prompt,
            split_info.list_field_names,
            client=self.client,
            rewrite_model=rewrite_model,
            custom_sub_prompts=custom_sub_prompts,
        )

        if self.few_shot_examples:
            field_examples = decompose_few_shot_examples(self.few_shot_examples, split_info)
            sub_prompts = inject_few_shot_into_sub_prompts(sub_prompts, field_examples)
            sub_prompts = await cleanup_few_shot_sub_prompts(
                sub_prompts,
                client=self.client,
                cleanup_model=rewrite_model,
            )

        params = model_config.params
        multi_pass = 2 if Manipulation.multi_pass in manipulations else 1

        result = await self.decomposed_evaluator.evaluate(
            documents=documents,
            labels=labels,
            sub_prompts=sub_prompts,
            sub_schemas=sub_schemas,
            split_info=split_info,
            model=model_name,
            temperature=params.get("temperature", self.temperature),
            max_tokens=params.get("max_tokens"),
            field_metrics_config=field_metrics_config,
            hallucination_filter=Manipulation.hallucination_filter in manipulations,
            multi_pass=multi_pass,
            on_document_complete=on_document_complete,
        )
        return result, sub_prompts

    @staticmethod
    def _format_sub_prompts_for_display(sub_prompts: dict[str, str]) -> str:
        """Format decomposed sub-prompts into a readable string for the report."""
        separator = "\n\n" + "=" * 60 + "\n\n"
        parts = []
        for field_name, prompt in sub_prompts.items():
            header = f"[DECOMPOSED SUB-PROMPT: {field_name}]"
            parts.append(f"{header}\n{prompt}")
        return separator.join(parts)

    # -------------------------------------------------------------------------
    # Reports
    # -------------------------------------------------------------------------

    def save_html_report(self, output_dir: "str | Path | None" = None) -> Path:
        """Generate the HTML report directly from in-memory results.

        Must be called after ``evaluate()``.
        Returns the path to the generated HTML file.

        Args:
            output_dir: Override the output directory for this call. Falls back
                to ``config.output_dir`` if omitted. Raises if neither is set.
        """
        if self.results is None:
            raise RuntimeError("Call evaluate() before save_html_report().")

        documents = [
            Document(
                id=str(d.get("id", "")),
                content=resolve_content(d, self._data_base_dir),
                metadata={},
                attachments=d.get("attachments", []),
            )
            for d in self.data
        ]
        fmc = self._get_field_metrics_config()
        field_config = fmc.config if fmc else None
        return self.runner.generate_report(
            results=self.results,
            output_dir=self._resolve_output_dir(output_dir),
            use_case=self.use_case,
            include_recommendation=True,
            create_visualizations=True,
            prompt_optimizations=self._manipulations_applied,
            model_prompts=self._model_prompts,
            model_override_prompts=self._model_override_prompts,
            original_prompt=self.prompt_template,
            documents=documents,
            field_config=field_config,
            output_formats=["html"],
        )

    def save_pdf_report(self, output_dir: "str | Path | None" = None) -> Path:
        """Generate the PDF report (and HTML) directly from in-memory results.

        Must be called after ``evaluate()``.
        Returns the path to the generated HTML file; the PDF is written
        alongside it as ``evaluation_report.pdf``.

        Args:
            output_dir: Override the output directory for this call. Falls back
                to ``config.output_dir`` if omitted. Raises if neither is set.
        """
        if self.results is None:
            raise RuntimeError("Call evaluate() before save_pdf_report().")

        documents = [
            Document(
                id=str(d.get("id", "")),
                content=resolve_content(d, self._data_base_dir),
                metadata={},
                attachments=d.get("attachments", []),
            )
            for d in self.data
        ]
        fmc = self._get_field_metrics_config()
        field_config = fmc.config if fmc else None
        return self.runner.generate_report(
            results=self.results,
            output_dir=self._resolve_output_dir(output_dir),
            use_case=self.use_case,
            include_recommendation=True,
            create_visualizations=True,
            prompt_optimizations=self._manipulations_applied,
            model_prompts=self._model_prompts,
            model_override_prompts=self._model_override_prompts,
            original_prompt=self.prompt_template,
            documents=documents,
            field_config=field_config,
            output_formats=["pdf"],
        )
