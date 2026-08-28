"""Recipe for classification-shaped data: plain string labels, compared by exact match."""

from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, create_model

from valtron_core.evaluation.config import BaseRecipeConfig, ClassificationConfig
from valtron_core.evaluation.model_eval import _normalize_label
from valtron_core.evaluation.referenced_eval import ReferencedEval


class ClassificationExperiment(ReferencedEval):
    """Recipe for classification-shaped data: plain string labels, compared by exact match.

    Every label must be a plain string, not a dict, a list, or a string that itself
    parses as a JSON object or array; use ``ExtractionExperiment`` for that kind of data
    instead. A ``response_format`` is optional here (unlike ``ExtractionExperiment``, which
    requires one): when neither it nor ``config.response_format_schema`` is given, a
    single-field ``label`` schema is auto-inferred from the unique label values (see
    ``ClassificationConfig.infer_schema``). Pass ``response_format`` yourself to
    constrain output with your own schema instead, e.g. a single-``label``-field
    schema for structured output over the same plain string labels.
    """

    @classmethod
    def _config_model(cls) -> type[BaseRecipeConfig]:
        return ClassificationConfig

    def __init__(
        self,
        config: ClassificationConfig | dict[str, Any] | str | Path,
        data: list[dict[str, Any]] | str | Path,
        response_format: type[BaseModel] | None = None,
    ):
        """
        Initialize the classification recipe.

        Args:
            config: Configuration dict, ClassificationConfig, or path (str/Path) to a
                JSON config file. Same keys as ``ModelEvalConfig`` plus ``infer_schema``
                (default ``True``): auto-infers a single-field ``label`` schema from the
                unique label values when neither ``response_format`` nor
                ``response_format_schema`` was given. Set to ``False`` to leave the
                model unconstrained (plain text output) in that case.
            data: List of dicts ``[{"id": ..., "content": ..., "label": ...}]`` with
                plain string labels, or a path to a JSON file with the same structure.
            response_format: Optional Pydantic model class for structured output
                validation. Takes priority over ``config.response_format_schema`` and
                schema inference.
        """
        validated = cast(ClassificationConfig, self._validate_config(config))
        super().__init__(config=validated, data=data, response_format=response_format)
        self._validate_plain_string_labels()
        if self.response_format is None and validated.infer_schema:
            self.response_format = self._infer_label_schema()

    def _validate_plain_string_labels(self) -> None:
        for item in self.data:
            label = item.get("label", "")
            if isinstance(_normalize_label(label), (dict, list)):
                raise ValueError(
                    f"ClassificationExperiment requires plain string labels, but record "
                    f"id={item.get('id', '')!r} has a structured label. Use ExtractionExperiment "
                    "for extraction-mode data instead."
                )

    def _infer_label_schema(self) -> type[BaseModel] | None:
        """Build a single-field `label` schema from the unique label values.

        The field is constrained to a `Literal` enum of the unique values seen, up to
        50 distinct values; beyond that the enum would be unwieldy so the field falls
        back to a plain `str`.
        """
        if not self.data:
            return None

        labels = [str(item.get("label", "")) for item in self.data]
        unique_labels = sorted(set(labels))
        annotation: Any = Literal[tuple(unique_labels)] if len(unique_labels) <= 50 else str

        return create_model(
            "ResponseModel",
            __config__=ConfigDict(extra="forbid"),
            label=(annotation, Field(..., description="Predicted class label")),
        )
