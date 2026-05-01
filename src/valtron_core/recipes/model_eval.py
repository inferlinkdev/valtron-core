"""Unified recipe for model evaluation tasks (classification and structured extraction)."""

import asyncio
import json
import re
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import structlog
from pydantic import BaseModel, Field, create_model

from valtron_core.client import LLMClient
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
from valtron_core.evaluation.json_eval import JsonEvaluator
from valtron_core.few_shot_training_data_generator import (
    FewShotTrainingDataGenerator,
    LabeledExample,
)
from valtron_core.models import Document, FieldMetricsConfig, Label, PredictionResult
from valtron_core.prompt_optimizer import ExplanationEnhancer
from valtron_core.recipes.base import BaseRecipe
from valtron_core.recipes.config import (
    Manipulation,
    ModelEvalConfig,
    STRUCTURED_MANIPULATIONS,
)
from valtron_core.runner import EvaluationResult, EvaluationRunner

from valtron_core.utilities.field_config_generator import infer_field_config


def _pydantic_type_from_value(value: Any, field_name: str) -> Any:
    """Recursively infer a Pydantic type annotation from a JSON value."""
    if isinstance(value, bool):
        return bool
    if isinstance(value, int):
        return int
    if isinstance(value, float):
        return float
    if isinstance(value, dict):
        nested_fields: dict[str, Any] = {
            k: (_pydantic_type_from_value(v, k), Field(description=f"Field {k}"))
            for k, v in value.items()
        }
        return create_model(field_name.capitalize(), **nested_fields)
    if isinstance(value, list) and value and isinstance(value[0], (dict, list)):
        item_type = _pydantic_type_from_value(value[0], field_name.rstrip("s").capitalize())
        return list[item_type]  # type: ignore[valid-type]
    if isinstance(value, list):
        return list[str]  # type: ignore[valid-type]
    return str

logger = structlog.get_logger()


class ModelEval(BaseRecipe):
    """
    Recipe for evaluating and comparing multiple models on a structured task.

    Handles the complete pipeline for both classification and extraction tasks:
    1. Optional: Generate additional training data via few-shot learning
    2. Optimize prompts per model (explanations, few-shot injection, repetition)
    3. Evaluate all models concurrently
    4. Generate a comprehensive report with metrics

    Behavior depends on whether ``response_format`` is provided:

    - ``response_format=None`` (default) — **label/classification mode**
      Labels are plain strings or simple JSON. The recipe auto-generates Pydantic
      validators from the label schema. Transformer models are supported.

    - ``response_format=SomePydanticModel`` — **structured extraction mode**
      Labels are nested JSON objects. The provided Pydantic schema constrains
      LLM output for strict validation. Enables the structured-only manipulations:
      ``decompose``, ``hallucination_filter``, and ``multi_pass``.
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
                ``field_metrics_config``, ``save_html``, ``save_pdf``.
            data: List of dicts ``[{"id": ..., "content": ..., "label": ...}]``,
                or a path to a JSON file with the same structure.
            response_format: Optional Pydantic model class for structured output validation.
                When provided, enables extraction mode and the structured manipulations
                (``decompose``, ``hallucination_filter``, ``multi_pass``).
                When omitted, the recipe auto-generates validators from label data.
        """
        if isinstance(config, (str, Path)):
            with open(config) as f:
                config = json.load(f)
        if isinstance(config, dict):
            config = ModelEvalConfig.model_validate(config)
        self.config = config

        if isinstance(data, (str, Path)):
            with open(data) as f:
                data = json.load(f)
        self.data = data
        self.response_format = response_format

        self.runner = EvaluationRunner()
        self.enhancer = ExplanationEnhancer()
        self.client = LLMClient()
        # DecomposedEvaluator is only needed in extraction mode
        self.decomposed_evaluator = (
            DecomposedEvaluator(client=self.client) if response_format is not None else None
        )

        # Parse configuration — add_models validates guards and populates self.models
        self.models: list[Any] = []
        self.add_models(config.models)
        self.prompt_template = config.prompt
        self.few_shot_config = config.few_shot
        self.output_dir = Path(config.output_dir) if config.output_dir else None
        self.use_case = config.use_case
        self.temperature = config.temperature
        self._field_metrics_config_raw = config.field_metrics_config

        # Storage
        self.few_shot_examples: list[Any] = []
        self.results: list[Any] | None = None
        self._manipulations_applied: dict[str, list[Any]] | None = None
        self._model_prompts: dict[str, str] | None = None
        self._model_override_prompts: dict[str, str] | None = None
        self._response_format_schema: dict[str, Any] | None = None

        logger.info(
            "model_eval_initialized",
            num_models=len(self.models),
            num_documents=len(self.data),
            few_shot_enabled=self.few_shot_config is not None and self.few_shot_config.enabled,
            has_response_format=response_format is not None,
        )

    # -------------------------------------------------------------------------
    # Preflight
    # -------------------------------------------------------------------------

    def _preflight_check(self) -> None:
        super()._preflight_check()

    # -------------------------------------------------------------------------
    # Model management
    # -------------------------------------------------------------------------

    def add_models(self, models: "list[dict[str, Any] | Any]") -> None:
        """Add new models to the experiment.

        All model validation (uniqueness, structured-manipulation guards) is
        handled here — ``__init__`` delegates to this method for its own model
        initialization.  On the next ``evaluate()`` / ``run()`` call only newly
        added models are evaluated; models that already have results are skipped
        automatically.

        Args:
            models: List of model config dicts or ``ModelConfig`` objects.

        Raises:
            ValueError: Duplicate label, structured manipulation without
                ``response_format``, or transformer model with ``response_format``.
        """
        from valtron_core.recipes.config import LLMModelConfig, TransformerModelConfig

        normalized = []
        for m in models:
            if isinstance(m, dict):
                model_type = m.get("type", "llm")
                if model_type == "transformer":
                    normalized.append(TransformerModelConfig.model_validate(m))
                else:
                    normalized.append(LLMModelConfig.model_validate(m))
            else:
                normalized.append(m)

        existing_labels = {mc.label or mc.name for mc in self.models}
        seen_in_batch: set[str] = set()
        for mc in normalized:
            label = mc.label or mc.name
            if label in existing_labels:
                raise ValueError(f"Model label {label!r} already exists in this experiment.")
            if label in seen_in_batch:
                raise ValueError(f"Duplicate label {label!r} in provided models list.")
            seen_in_batch.add(label)

        structured_requested = [
            (mc.label or mc.name, manip)
            for mc in normalized
            for manip in getattr(mc, "prompt_manipulation", [])
            if manip in STRUCTURED_MANIPULATIONS
        ]
        if structured_requested and self.response_format is None:
            bad_models = sorted({name for name, _ in structured_requested})
            bad_manips = sorted({manip.value for _, manip in structured_requested})
            raise ValueError(
                f"Model(s) {bad_models} use structured manipulation(s) {bad_manips}, "
                "which require response_format to be provided."
            )

        if self.response_format is not None:
            transformer_models = [mc.label for mc in normalized if mc.type == "transformer"]
            if transformer_models:
                raise ValueError(
                    f"Transformer model(s) {transformer_models} cannot be used when "
                    "response_format is provided."
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
    def _config_and_data_from_metadata(
        metadata_path: Path,
    ) -> "tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any] | None]":
        """Read metadata.json and return ``(config_dict, data, response_format_schema)``.

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
    def load_experiment_results(cls, dir_path: "str | Path") -> "ModelEval":
        """Restore a previously saved experiment from disk.

        Returns a ``ModelEval`` instance in the same state as after
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

        config_dict, data, response_format_schema = cls._config_and_data_from_metadata(metadata_path)

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

        label_map = {str(d.get("id", "")): str(d.get("label", "")) for d in data}

        results: list[EvaluationResult] = []
        model_prompts: dict[str, str] = {}
        manipulations_applied: dict[str, list] = {}
        model_override_prompts: dict[str, str] = {}

        for md in all_model_data:
            model_label = md["label"] or md["name"]
            model_prompts[model_label] = md["prompt_template"]
            manipulations_applied[model_label] = md["prompt_manipulation"]
            if md.get("override_prompt"):
                model_override_prompts[model_label] = md["override_prompt"]

            try:
                from valtron_core.evaluation.json_eval import EvalResult

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
                        expected_value=label_map.get(p["document_id"], ""),
                        is_correct=p.get("is_correct", False),
                        example_score=p.get("example_score", 0.0),
                        response_time=p.get("response_time", 0.0),
                        original_cost=p.get("original_cost", 0.0),
                        cost=p.get("cost", 0.0),
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

    def _resolve_response_format(self) -> "type[BaseModel] | None":
        """Return the base response format (no per-model explanation wrapping).

        Used for schema serialisation and field-metrics inference. The per-model
        explanation variation is handled separately in ``_run_evaluations``.
        """
        if self.response_format is not None:
            return self.response_format
        return self._create_response_validator()

    def _get_field_metrics_config(
        self, resolved_rf: "type[BaseModel] | None" = None
    ) -> FieldMetricsConfig | None:
        """Return a FieldMetricsConfig, either from explicit config or auto-inferred.

        If ``field_metrics_config`` was provided in the recipe config, that is
        validated and returned. Otherwise the config is inferred from label data:
        JSON labels use the label structure directly; plain-text labels with an
        auto-inferred response format wrap the label in the ``{"label": ...}``
        shape that ``_create_response_validator`` produces.
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
            pass

        if self.response_format is None and resolved_rf is not None:
            field_config = infer_field_config(json.dumps({"label": first_label}))
            return FieldMetricsConfig(config=field_config.model_dump())

        return None

    # -------------------------------------------------------------------------
    # Main pipeline
    # -------------------------------------------------------------------------

    def evaluate(self) -> None:
        """Run the evaluation pipeline (synchronous).

        Convenience wrapper around ``aevaluate()``. Populates ``self.results``,
        ``self._manipulations_applied``, and ``self._model_prompts``.
        Does not write any files.

        Note: uses ``asyncio.run()`` internally — cannot be called from within a
        running event loop. Use ``await experiment.aevaluate()`` in async contexts
        (e.g. Jupyter notebooks).
        """
        asyncio.run(self.aevaluate())

    def run(self, output_dir: "str | Path | None" = None) -> Path:
        """Run the complete pipeline and save outputs (synchronous).

        Convenience wrapper around ``arun()``. Calls ``aevaluate()``,
        ``save_experiment_results()``, and then ``save_html_report()`` and/or
        ``save_pdf_report()`` based on ``config.save_html`` and ``config.save_pdf``.

        Args:
            output_dir: Override the output directory for this run. Falls back
                to ``config.output_dir`` if omitted. Raises if neither is set.

        Returns:
            Path to the generated HTML report (if save_html is True), otherwise
            the path to the run directory.

        Note: uses ``asyncio.run()`` internally — cannot be called from within a
        running event loop. Use ``await experiment.arun()`` in async contexts
        (e.g. Jupyter notebooks).
        """
        return asyncio.run(self.arun(output_dir=output_dir))

    async def aevaluate(self) -> None:
        """Run the evaluation pipeline and store results on this object (async).

        Populates ``self.results``, ``self._manipulations_applied``, and
        ``self._model_prompts``. Does not write any files.

        Call ``save_experiment_results()`` / ``save_html_report()`` / ``save_pdf_report()``
        afterwards to persist results and generate reports.
        """
        logger.info("starting_model_eval_pipeline")

        self._preflight_check()
        resolved_rf = self._resolve_response_format()
        self._response_format_schema = self._serialize_response_format_schema(resolved_rf)
        field_metrics_config = self._get_field_metrics_config(resolved_rf)

        if self.few_shot_config and self.few_shot_config.enabled:
            await self._generate_few_shot_data()

        self._model_prompts = await self._prepare_model_prompts()

        # Skip models that already have results (supports incremental evaluation)
        existing_labels: set[str] = {r.model for r in (self.results or [])}
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

        logger.info("model_eval_complete", num_models=len(self.results or []))

    async def arun(self, output_dir: "str | Path | None" = None) -> Path:
        """Run the complete pipeline and save outputs according to config flags (async).

        Calls ``aevaluate()``, ``save_experiment_results()``, and then
        ``save_html_report()`` and/or ``save_pdf_report()`` based on
        ``config.output_formats``.

        Args:
            output_dir: Override the output directory for this run. Falls back
                to ``config.output_dir`` if omitted. Raises if neither is set.

        Returns:
            Path to the generated HTML report (if ``"html"`` is in ``output_formats``),
            otherwise the path to the run directory.
        """
        await self.aevaluate()

        run_dir = self.save_experiment_results(output_dir)
        report_path: Path = run_dir

        if "html" in self.config.output_formats:
            report_path = self.save_html_report(output_dir)

        if "pdf" in self.config.output_formats:
            self.save_pdf_report(output_dir)

        logger.info("model_eval_run_complete", report_path=str(report_path))
        return report_path

    # -------------------------------------------------------------------------
    # Few-shot data generation
    # -------------------------------------------------------------------------

    async def _generate_few_shot_data(self) -> None:
        """Generate additional training data using few-shot learning."""
        logger.info("generating_few_shot_data")

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
            label_value = (
                json.dumps(label_raw) if isinstance(label_raw, (dict, list)) else str(label_raw)
            )
            documents.append(
                Document(
                    id=doc_id,
                    content=item["content"],
                    metadata=item.get("metadata", {}),
                    attachments=item.get("attachments", []),
                )
            )
            labels.append(Label(document_id=doc_id, value=label_value))
        return documents, labels

    def _build_model_arg(self, model_config: Any) -> dict[str, Any]:
        """Build the litellm model dict to pass to runner.evaluate."""
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

    # -------------------------------------------------------------------------
    # Prompt preparation
    # -------------------------------------------------------------------------

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

    def _create_response_validator(
        self, include_explanation: bool = False
    ) -> type[BaseModel] | None:
        """Auto-generate a Pydantic validator from label data.

        Only used in label mode (``response_format=None``).

        For JSON-object labels: recursively builds nested Pydantic models matching
        the label structure (lists of objects, nested objects, scalars).
        For plain-text labels: builds a model with a ``Literal[...]`` enum field
            covering every unique label value (up to 50 values); falls back to
            ``label: str`` for larger datasets.
        Returns ``None`` when no validator can be produced.
        """
        if not self.data:
            return None

        first_label = self.data[0].get("label", "")
        if isinstance(first_label, (dict, list)):
            first_label = json.dumps(first_label)
        try:
            label_json = json.loads(first_label)
        except (json.JSONDecodeError, TypeError):
            # Plain-text label path
            unique_values = sorted(
                {str(d.get("label", "")) for d in self.data if str(d.get("label", "")) != ""}
            )
            if not unique_values:
                return None
            field_definitions: dict[str, Any] = {}
            if include_explanation:
                field_definitions["explanation"] = (str, Field(description="Reasoning explanation"))
            if len(unique_values) <= 50:
                literal_type = Literal[tuple(unique_values)]  # type: ignore[valid-type]
                field_definitions["label"] = (literal_type, Field(description="Predicted class label"))
                model_name = "ResponseModelWithExplanation" if include_explanation else "ResponseModel"
                response_model = create_model(model_name, **field_definitions)
                logger.info(
                    "auto_enum_validator_created",
                    model_name=model_name,
                    num_values=len(unique_values),
                )
            else:
                field_definitions["label"] = (str, Field(description="Predicted class label"))
                model_name = "ResponseModelWithExplanation" if include_explanation else "ResponseModel"
                response_model = create_model(model_name, **field_definitions)
                logger.info(
                    "auto_str_validator_created",
                    model_name=model_name,
                    num_values=len(unique_values),
                )
            return response_model

        if not isinstance(label_json, dict):
            return None

        # If explanation is requested but the prompt has no JSON schema, the
        # ExplanationEnhancer uses a text format ("Explanation: ... Answer: ...")
        # rather than JSON — skip the validator in that case.
        if include_explanation and not self._has_json_schema_in_prompt(self.prompt_template):
            logger.info(
                "skipping_validator_for_text_explanation",
                reason="Prompt uses text format (Explanation: ... Answer: ...) not JSON",
            )
            return None

        field_definitions = {}

        if include_explanation:
            field_definitions["explanation"] = (
                str,
                Field(description="Reasoning explanation"),
            )

        for key, value in label_json.items():
            field_definitions[key] = (
                _pydantic_type_from_value(value, key),
                Field(description=f"Field {key}"),
            )

        model_name = "ResponseModelWithExplanation" if include_explanation else "ResponseModel"
        response_model = create_model(model_name, **field_definitions)

        logger.info(
            "response_validator_created",
            model_name=model_name,
            fields=list(field_definitions.keys()),
        )
        return response_model

    def _serialize_response_format_schema(
        self, rf: type[BaseModel] | None
    ) -> dict[str, Any] | None:
        """Return the Pydantic JSON Schema for a model class.

        Returns ``None`` when *rf* is ``None``.
        """
        if rf is None:
            return None
        return rf.model_json_schema()

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
            label_map[doc_id] = (
                json.dumps(label_raw) if isinstance(label_raw, (dict, list)) else str(label_raw)
            )

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
            prediction = transformer.predict(doc.content)
            pred_time = time.time() - pred_start

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
                eval_result = json_evaluator.evaluate(cfg, eval_expected, eval_predicted)
                is_correct = eval_result.score == 1.0
            else:
                is_correct = prediction.strip() == expected_label.strip()

            pred_result = PredictionResult(
                document_id=doc.id,
                predicted_value=prediction,
                expected_value=expected_label,
                is_correct=is_correct,
                response_time=pred_time,
                cost=0.0,
                model=model_name,
                metadata={"content": doc.content},
            )
            result.add_prediction(pred_result)

        if model_config.cost_rate is not None:
            unit_seconds = _parse_time_unit_to_seconds(model_config.cost_rate_time_unit)
            for p in result.predictions:
                p.cost = float(model_config.cost_rate) * (p.response_time / unit_seconds)
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

    async def _run_evaluations(
        self,
        model_prompts: dict[str, str],
        field_metrics_config: "FieldMetricsConfig | None" = None,
        models: "list | None" = None,
    ) -> tuple[list[EvaluationResult], dict[str, list[Any]]]:
        """Run evaluations for the given models concurrently.

        Args:
            model_prompts: Mapping of model label → prompt string.
            field_metrics_config: Optional field-level metrics configuration.
            models: Models to evaluate. Defaults to ``self.models`` when ``None``.

        Returns:
            Tuple of (results, manipulations_applied) where manipulations_applied
            maps model label to the list of Manipulation values that were applied.
        """
        if field_metrics_config is None:
            field_metrics_config = self._get_field_metrics_config(self._resolve_response_format())

        effective_models = models if models is not None else self.models
        documents, labels = self._load_documents_and_labels()

        async def _evaluate_single_model(
            index: int, model_config: Any
        ) -> tuple[int, Any, str, list, str | None]:
            model_name = getattr(model_config, "name", None) or model_config.label
            model_label = model_config.label or model_name
            manipulations = getattr(model_config, "prompt_manipulation", [])

            # --- Transformer branch (label mode only; guarded at __init__) ---
            if model_config.type == "transformer":
                result = await self._evaluate_transformer(
                    model_config, documents, field_metrics_config
                )
                return index, result, model_label, [], None

            prompt = model_prompts[model_label]

            # --- Determine effective response format ---
            if self.response_format is not None:
                # Extraction mode: use provided schema, wrapping with explanation field if needed
                if Manipulation.explanation in manipulations:
                    effective_rf = self._create_explanation_model()
                else:
                    effective_rf = self.response_format
            elif self._response_format_schema is not None:
                # Schema restored from a previous run -- wrap for litellm
                effective_rf = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": self._response_format_schema.get("title", "ResponseModel"),
                        "strict": True,
                        "schema": self._response_format_schema,
                    },
                }
            else:
                # Label mode: auto-generate validator from label data
                include_explanation = Manipulation.explanation in manipulations
                effective_rf = self._create_response_validator(
                    include_explanation=include_explanation
                )

            logger.info(
                "evaluating_model",
                model=model_label,
                manipulations=[m.value if hasattr(m, "value") else m for m in manipulations],
                using_validator=effective_rf is not None,
                using_field_metrics=field_metrics_config is not None,
            )

            updated_prompt = None

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
                )
                updated_prompt = self._format_sub_prompts_for_display(sub_prompts)
            else:
                post_extraction_filter = None
                if (
                    Manipulation.hallucination_filter in manipulations
                    and self.response_format is not None
                ):

                    async def _hallucination_filter(predicted_json, document, _model=model_name):
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
                    response_format=effective_rf,
                    field_metrics_config=field_metrics_config,
                    post_extraction_filter=post_extraction_filter,
                    multi_pass=multi_pass,
                )

            # Propagate label to result objects when it differs from the model name
            if model_label != model_name:
                result.model = model_label
                for pred in result.predictions:
                    pred.model = model_label
                if result.metrics:
                    result.metrics.model = model_label

            return index, result, model_label, manipulations, updated_prompt

        indexed_results = await asyncio.gather(
            *[_evaluate_single_model(i, mc) for i, mc in enumerate(effective_models)]
        )

        results = []
        manipulations_applied: dict[str, list[Any]] = {}
        for _, result, model_label, manipulations, updated_prompt in sorted(
            indexed_results, key=lambda x: x[0]
        ):
            results.append(result)
            manipulations_applied[model_label] = manipulations
            if updated_prompt is not None:
                model_prompts[model_label] = updated_prompt

        return results, manipulations_applied

    # -------------------------------------------------------------------------
    # Decomposed evaluation (extraction mode)
    # -------------------------------------------------------------------------

    async def _run_decomposed_evaluation(
        self,
        documents: list[Document],
        labels: list[Label],
        prompt: str,
        model_name: str,
        model_config: Any,
        manipulations: list,
        field_metrics_config: FieldMetricsConfig | None,
    ) -> tuple[EvaluationResult, dict[str, str]]:
        """Run evaluation with decomposed sub-prompts for each entity field.

        Returns:
            Tuple of (EvaluationResult, sub_prompts dict).
        """
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
