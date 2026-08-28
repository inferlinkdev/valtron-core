"""Recipe for structured extraction: labels are nested JSON objects, scored per field."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel

from valtron_core.evaluation.config import ModelEvalConfig
from valtron_core.evaluation.model_eval import _normalize_label
from valtron_core.evaluation.referenced_eval import ReferencedEval


class ExtractionExperiment(ReferencedEval):
    """Recipe for structured extraction: labels are nested JSON objects, scored per field.

    Requires a schema, either the ``response_format`` constructor argument or
    ``config.response_format_schema``. Unlike ``ReferencedEval``, it fails immediately
    if neither is given rather than running with an unconstrained model.
    """

    def __init__(
        self,
        config: ModelEvalConfig | dict[str, Any] | str | Path,
        data: list[dict[str, Any]] | str | Path,
        response_format: type[BaseModel] | None = None,
    ):
        """
        Initialize the extraction recipe.

        Args:
            config: Configuration dict, ModelEvalConfig, or path (str/Path) to a JSON
                config file. Same keys as ``ModelEvalConfig``.
            data: List of dicts ``[{"id": ..., "content": ..., "label": ...}]`` with
                dict, list, or JSON-string labels, or a path to a JSON file with the
                same structure.
            response_format: Pydantic model class constraining the LLM's structured
                output. Required unless ``config.response_format_schema`` is given instead.
        """
        super().__init__(config=config, data=data, response_format=response_format)
        if self.response_format is not None:
            return

        all_plain_strings = self.data and all(
            not isinstance(_normalize_label(item.get("label", "")), (dict, list))
            for item in self.data
        )
        if all_plain_strings:
            raise ValueError(
                "ExtractionExperiment requires a schema (response_format or "
                "config.response_format_schema), but none was given, and every label "
                "is a plain string. This looks like classification data; use "
                "ClassificationExperiment instead, or pass response_format if you do want "
                "structured output from these labels."
            )
        raise ValueError(
            "ExtractionExperiment requires a schema (response_format or "
            "config.response_format_schema); got neither."
        )
