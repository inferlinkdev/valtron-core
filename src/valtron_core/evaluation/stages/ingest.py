"""Ingestor: turns raw record dicts into (Document, label) pairs.

No disk I/O beyond resolving a document's ``content_path``/attachments (see
``resolve_content``). Two implementations ship here. ``DefaultIngestor`` is
``ModelEval``'s own historical behavior: a label is optional and taken
verbatim, whatever shape it has; this is what a recipe with no ground
truth (``SummarizationExperiment``) uses, unchanged, by never overriding it.
``StructuredLabelIngestor`` is ``ReferencedEval``'s: a label is required and
coerced into a ``Label`` with a JSON-serialized value, honoring the
plain-string-label auto-wrap that recipe computes from its response schema.

A new ground-truth recipe with a different label shape is a new ``Ingestor``
implementation, wired up in that recipe's own ``_post_init``, never a new
branch here or in ``ModelEval``.
"""

import json
from pathlib import Path
from typing import Any, Protocol

from valtron_core.content_resolution import resolve_content
from valtron_core.models import Document, Label


class Ingestor(Protocol):
    def ingest(
        self, data: list[dict[str, Any]], data_base_dir: Path
    ) -> "tuple[list[Document], list[Any]]":
        """Convert raw record dicts into (documents, labels), index-aligned."""
        ...


class DefaultIngestor:
    """Label-optional: one ``Document`` per record, label taken verbatim (often absent)."""

    def ingest(
        self, data: list[dict[str, Any]], data_base_dir: Path
    ) -> "tuple[list[Document], list[Any]]":
        documents: list[Document] = []
        labels: list[Any] = []
        for idx, item in enumerate(data):
            doc_id = str(item.get("id", f"doc_{idx}"))
            documents.append(
                Document(
                    id=doc_id,
                    content=resolve_content(item, data_base_dir),
                    metadata=item.get("metadata", {}),
                    attachments=item.get("attachments", []),
                )
            )
            labels.append(item.get("label"))
        return documents, labels


def serialize_structured_label(label_raw: Any, *, auto_wrap_string_labels: bool) -> str:
    """The label -> str rule ``ReferencedEval`` scores against.

    A dict/list label is JSON-serialized verbatim. A plain (non-dict/list)
    label is wrapped as ``{"label": ...}`` first when
    ``auto_wrap_string_labels`` holds (a single-field ``{"label": ...}``
    response schema paired with plain string labels; see
    ``ReferencedEval._compute_auto_wrap_string_labels``), otherwise taken as
    a plain string as-is.

    Shared by every place ``ReferencedEval`` needs one record's label
    serialized this same way: full ingestion (``StructuredLabelIngestor``,
    below), ``reevaluate()``'s label refresh, and ``_evaluate_transformer()``'s
    label map, so this rule has exactly one implementation instead of three.
    """
    if isinstance(label_raw, (dict, list)):
        return json.dumps(label_raw)
    if auto_wrap_string_labels:
        return json.dumps({"label": str(label_raw)})
    return str(label_raw)


class StructuredLabelIngestor:
    """Label-required: a ``Label`` per record, value JSON-serialized when structured.

    ``auto_wrap_string_labels`` mirrors ``ReferencedEval``'s own flag of the
    same name and is recomputed the same way, at the same points (initial
    construction, and again ahead of every run, since it depends on
    ``response_format``/``data`` which can both change after construction).
    Mutate this attribute in place when that happens, matching
    ``ReferencedEval._auto_wrap_string_labels``, rather than replacing the
    ingestor, since both name the same fact.
    """

    def __init__(self, auto_wrap_string_labels: bool = False) -> None:
        self.auto_wrap_string_labels = auto_wrap_string_labels

    def ingest(
        self, data: list[dict[str, Any]], data_base_dir: Path
    ) -> "tuple[list[Document], list[Label]]":
        documents: list[Document] = []
        labels: list[Label] = []
        for idx, item in enumerate(data):
            doc_id = str(item.get("id", f"doc_{idx}"))
            documents.append(
                Document(
                    id=doc_id,
                    content=resolve_content(item, data_base_dir),
                    metadata=item.get("metadata", {}),
                    attachments=item.get("attachments", []),
                )
            )
            label_value = serialize_structured_label(
                item.get("label", ""), auto_wrap_string_labels=self.auto_wrap_string_labels
            )
            labels.append(Label(document_id=doc_id, value=label_value))
        return documents, labels
