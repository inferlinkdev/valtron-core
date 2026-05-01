"""Abstract base class for evaluation recipes."""

import asyncio
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from valtron_core.models import FieldMetricsConfig
from valtron_core.recipes.config import BaseRecipeConfig, ModelConfig
from valtron_core.runner import EvaluationRunner


class BaseRecipe(ABC):
    """Shared logic for evaluation recipes.

    Subclasses must assign the following in ``__init__``:
        self.runner       : EvaluationRunner
        self.config       : BaseRecipeConfig   (or subclass)
        self.models       : list[ModelConfig]
        self.data         : list[dict[str, Any]]
        self.output_dir   : Path
        self.use_case     : str
        self.prompt_template : str

    After calling ``evaluate()``, the following are populated:
        self.results               : list[EvaluationResult]
        self._manipulations_applied: dict[str, list[str]]
        self._model_prompts        : dict[str, str]
    """

    runner: EvaluationRunner
    config: BaseRecipeConfig
    models: list[ModelConfig]
    data: list[dict[str, Any]]
    output_dir: Path
    use_case: str
    prompt_template: str

    # Populated by evaluate()
    results: list[Any] | None
    _manipulations_applied: dict[str, list[Any]] | None
    _model_prompts: dict[str, str] | None
    _model_override_prompts: dict[str, str] | None
    _response_format_schema: dict[str, Any] | None

    @abstractmethod
    def _get_field_metrics_config(self) -> FieldMetricsConfig | None: ...

    def _preflight_check(self) -> None:
        """Run all pre-flight checks before any evaluation work begins.

        Currently checks:
        - All model labels are unique.
        - No unordered list fields use expensive (3rd-party) metrics without opt-in.

        Add further checks here as needed; every recipe will pick them up automatically.
        """
        self._check_unique_model_labels()
        field_metrics_config = self._get_field_metrics_config()
        self.runner._preflight_check(field_metrics_config, len(self.data), len(self.models))

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

    def _build_save_documents(self) -> list[dict[str, Any]]:
        """Build the document list used when writing the run directory."""
        documents: list[dict[str, Any]] = []
        for item in self.data:
            doc_entry: dict[str, Any] = {
                "id": str(item.get("id", "")),
                "content": item.get("content", ""),
                "label": str(item.get("label", "")),
            }
            if item.get("attachments"):
                doc_entry["attachments"] = item["attachments"]
            documents.append(doc_entry)
        return documents

    def evaluate_sync(self) -> None:
        """Synchronous version of ``evaluate()``.

        Runs the evaluation pipeline and stores results on this object.
        Use this when you are not inside an async context.  Internally calls
        ``asyncio.run(self.evaluate())``.
        """
        asyncio.run(self.evaluate())

    def _resolve_output_dir(self, output_dir: "str | Path | None") -> Path:
        """Return the effective output directory, raising if neither source provides one."""
        effective = output_dir or self.output_dir
        if effective is None:
            raise ValueError(
                "output_dir is required. Set it in the config or pass it to the save method."
            )
        return Path(effective)

    def save_experiment_results(self, output_dir: "str | Path | None" = None) -> Path:
        """Write the run directory (``metadata.json`` + ``models/*.json``).

        Must be called after ``evaluate()``.  Returns the path to the run
        directory that was written.

        Args:
            output_dir: Override the output directory for this call.  Falls back
                to ``config.output_dir`` if omitted.  Raises if neither is set.
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
        )
        return run_dir

    def save_html_report(self, output_dir: "str | Path | None" = None) -> Path:
        """Generate the HTML report directly from in-memory results.

        Must be called after ``evaluate()``.
        Returns the path to the generated HTML file.

        Args:
            output_dir: Override the output directory for this call.  Falls back
                to ``config.output_dir`` if omitted.  Raises if neither is set.
        """
        if self.results is None:
            raise RuntimeError("Call evaluate() before save_html_report().")
        from valtron_core.models import Document

        documents = [
            Document(
                id=str(d.get("id", "")),
                content=d.get("content", ""),
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
            output_dir: Override the output directory for this call.  Falls back
                to ``config.output_dir`` if omitted.  Raises if neither is set.
        """
        if self.results is None:
            raise RuntimeError("Call evaluate() before save_pdf_report().")
        from valtron_core.models import Document

        documents = [
            Document(
                id=str(d.get("id", "")),
                content=d.get("content", ""),
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
