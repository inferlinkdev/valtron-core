"""Decompose multi-entity extraction prompts into parallel single-entity sub-tasks.

Smaller self-hosted models struggle with complex multi-entity extraction.
This module splits a single prompt that extracts N entity types into N
parallel single-entity prompts, merges the results, and feeds them back
into the existing evaluation pipeline.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, get_args, get_origin

import structlog
from pydantic import BaseModel, Field, create_model
from rapidfuzz import fuzz

from valtron_core.client import LLMClient
from valtron_core.evaluator import PromptEvaluator
from valtron_core.models import (
    Document,
    EvaluationResult,
    FieldMetricsConfig,
    Label,
    PredictionResult,
)

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# filter_hallucinated_values
# ---------------------------------------------------------------------------

async def filter_hallucinated_values(
    predicted_json: str,
    document_text: str,
    model: str,
    client: LLMClient,
) -> str:
    """Verify each extracted value against the source document and remove hallucinations.

    For every leaf value in the predicted JSON, asks the same *model* whether the
    value appears in *document_text*.  Values the model says "no" to are removed.
    On any error (parse failure, LLM failure) the function fails open and returns
    the input unchanged.

    Args:
        predicted_json: Raw JSON string from the extraction step.
        document_text: The original source document.
        model: Model identifier (same model used for extraction).
        client: An ``LLMClient`` instance for making verification calls.

    Returns:
        Filtered JSON string with hallucinated values removed.
    """
    try:
        parsed = json.loads(predicted_json)
    except (json.JSONDecodeError, TypeError):
        return predicted_json

    # Collect all (json_path, value) pairs from dicts nested inside lists.
    values_with_paths: list[tuple[list[str | int], str]] = []
    _collect_values(parsed, [], values_with_paths)

    if not values_with_paths:
        return predicted_json

    # Pre-check: skip LLM verification for values found in the document text.
    document_lower = document_text.lower()
    needs_llm: list[tuple[list[str | int], str]] = []
    pre_verified: list[tuple[list[str | int], bool]] = []
    for path, value in values_with_paths:
        if value.lower() in document_lower:
            pre_verified.append((path, True))
        else:
            needs_llm.append((path, value))

    # Verify remaining values in parallel via LLM.
    async def _verify(path: list[str | int], value: str) -> tuple[list[str | int], bool]:
        try:
            prompt = (
                "Does the EXACT name or phrase below appear word-for-word in the text? "
                "Look carefully. If you cannot find it written in the text, answer \"no\".\n"
                "Answer ONLY \"yes\" or \"no\". Default to \"no\" if uncertain.\n\n"
                f"Name/phrase to find: \"{value}\"\n\n"
                f"Text:\n{document_text}"
            )
            response = await client.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
            )
            answer = response.choices[0].message.content.strip().lower()
            return path, answer.startswith("yes")
        except Exception:
            # Fail open – keep the value on LLM error.
            return path, True

    llm_results = await asyncio.gather(
        *[_verify(p, v) for p, v in needs_llm]
    ) if needs_llm else []

    results = pre_verified + list(llm_results)

    # Build set of paths to remove (where model said "no").
    paths_to_remove: set[tuple[str | int, ...]] = set()
    for path, keep in results:
        if not keep:
            paths_to_remove.add(tuple(path))

    if not paths_to_remove:
        logger.debug("hallucination_filter_no_removals", model=model)
        return predicted_json

    removed_values = [
        {"path": list(p), "value": v}
        for p, v in zip(
            [path for path, keep in results if not keep],
            [v for (path, v) in needs_llm if tuple(path) in paths_to_remove],
        )
    ]
    logger.info(
        "hallucination_filter_removed",
        model=model,
        removed=removed_values,
        before=predicted_json[:500],
    )

    # Remove hallucinated values from the parsed structure.
    _remove_paths(parsed, [], paths_to_remove)

    filtered = json.dumps(parsed)
    logger.info(
        "hallucination_filter_result",
        model=model,
        after=filtered[:500],
    )
    return filtered


def _collect_values(
    obj: Any,
    current_path: list[str | int],
    out: list[tuple[list[str | int], str]],
) -> None:
    """Recursively collect leaf string values with their JSON paths."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            _collect_values(value, current_path + [key], out)
    elif isinstance(obj, list):
        for idx, item in enumerate(obj):
            _collect_values(item, current_path + [idx], out)
    elif isinstance(obj, str) and obj.strip():
        out.append((current_path, obj))


def _remove_paths(
    obj: Any,
    current_path: list[str | int],
    paths_to_remove: set[tuple[str | int, ...]],
) -> None:
    """Remove items at *paths_to_remove* from a nested structure in-place.

    When a dict inside a list has all its leaf values removed, the dict itself
    is removed from the parent list.
    """
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            child_path = current_path + [key]
            if tuple(child_path) in paths_to_remove:
                del obj[key]
            else:
                _remove_paths(obj[key], child_path, paths_to_remove)
    elif isinstance(obj, list):
        # Collect indices to remove (in reverse order to preserve indices).
        indices_to_remove: set[int] = set()
        for idx in range(len(obj)):
            child_path = current_path + [idx]
            if tuple(child_path) in paths_to_remove:
                indices_to_remove.add(idx)
            else:
                _remove_paths(obj[idx], child_path, paths_to_remove)

        # Remove marked indices and empty dicts (reverse to preserve order).
        obj[:] = [
            item for idx, item in enumerate(obj)
            if idx not in indices_to_remove
            and not (isinstance(item, dict) and len(item) == 0)
        ]


# ---------------------------------------------------------------------------
# Multi-pass helpers
# ---------------------------------------------------------------------------


def _deduplicate_list(items: list[dict], threshold: float = 0.85) -> list[dict]:
    """Remove near-duplicate dicts from *items* using fuzzy string matching.

    For each pair of dicts, all string values are compared with
    ``rapidfuzz.fuzz.ratio``.  If the average similarity across string fields
    is >= *threshold*, only the first occurrence is kept.
    """
    if not items:
        return items

    keep: list[dict] = []
    for item in items:
        is_dup = False
        for kept in keep:
            if _dicts_similar(item, kept, threshold):
                is_dup = True
                break
        if not is_dup:
            keep.append(item)
    return keep


def _dicts_similar(a: dict, b: dict, threshold: float) -> bool:
    """Return ``True`` when two dicts are similar above *threshold*."""
    str_scores: list[float] = []
    all_keys = set(a.keys()) | set(b.keys())
    for key in all_keys:
        va, vb = a.get(key, ""), b.get(key, "")
        if isinstance(va, str) and isinstance(vb, str):
            str_scores.append(fuzz.ratio(va.lower(), vb.lower()) / 100.0)
    if not str_scores:
        return a == b
    return (sum(str_scores) / len(str_scores)) >= threshold


def _multi_pass_merge(responses: list[str]) -> str:
    """Merge multiple JSON extraction responses into one.

    * Parses each JSON string; skips responses that fail to parse.
    * For every list field (recursively), unions items from all passes.
    * Deduplicates each list using :func:`_deduplicate_list`.
    * For non-list fields, keeps the value from the first (lowest-temperature) pass.
    * Returns the merged JSON string.
    """
    parsed: list[dict] = []
    for r in responses:
        try:
            obj = json.loads(r)
            if isinstance(obj, dict):
                parsed.append(obj)
        except (json.JSONDecodeError, TypeError):
            continue

    if not parsed:
        return responses[0] if responses else "{}"

    merged = _deep_merge_dicts(parsed)
    return json.dumps(merged)


def _deep_merge_dicts(dicts: list[dict]) -> dict:
    """Recursively merge a list of dicts, unioning list fields and deduplicating."""
    if not dicts:
        return {}

    base = dicts[0]
    result: dict = {}

    all_keys: list[str] = list(dict.fromkeys(k for d in dicts for k in d))

    for key in all_keys:
        values = [d[key] for d in dicts if key in d]
        if not values:
            continue

        if all(isinstance(v, list) for v in values):
            # Union all list items across passes
            combined: list = []
            for v in values:
                combined.extend(v)
            # Deduplicate dicts; leave non-dicts as-is (dedup via set-like logic)
            dict_items = [x for x in combined if isinstance(x, dict)]
            non_dict_items = list(dict.fromkeys(
                x for x in combined if not isinstance(x, dict)
            ))
            result[key] = _deduplicate_list(dict_items) + non_dict_items
        elif all(isinstance(v, dict) for v in values):
            result[key] = _deep_merge_dicts(values)
        else:
            # Non-list, non-dict: keep first pass value
            result[key] = values[0]

    return result


# ---------------------------------------------------------------------------
# SplitPointInfo
# ---------------------------------------------------------------------------

@dataclass
class SplitPointInfo:
    """Describes where a Pydantic model can be split into per-field sub-schemas.

    Attributes:
        path_from_root: Field names from the root model down to the model that
            contains the list fields (e.g. ``["entities"]``).
        split_model: The Pydantic class at the split point.
        list_field_names: Names of the ``list``-typed fields on *split_model*
            (e.g. ``["people", "pathogens"]``).
        list_field_annotations: Mapping from field name to its full type
            annotation (e.g. ``{"people": list[Entity]}``).
    """

    path_from_root: list[str]
    split_model: type[BaseModel]
    list_field_names: list[str]
    list_field_annotations: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# find_split_point
# ---------------------------------------------------------------------------

def find_split_point(response_format: type[BaseModel]) -> SplitPointInfo | None:
    """BFS over Pydantic model tree to find a model with >= 2 list fields.

    Returns ``None`` when no suitable split point exists.
    """
    from collections import deque

    queue: deque[tuple[type[BaseModel], list[str]]] = deque()
    queue.append((response_format, []))

    while queue:
        model, path = queue.popleft()

        list_fields: dict[str, Any] = {}
        child_models: list[tuple[str, type[BaseModel]]] = []

        for fname, finfo in model.model_fields.items():
            annotation = finfo.annotation
            if get_origin(annotation) is list:
                list_fields[fname] = annotation
            elif isinstance(annotation, type) and issubclass(annotation, BaseModel):
                child_models.append((fname, annotation))

        if len(list_fields) >= 2:
            return SplitPointInfo(
                path_from_root=path,
                split_model=model,
                list_field_names=list(list_fields.keys()),
                list_field_annotations=list_fields,
            )

        for fname, child_model in child_models:
            queue.append((child_model, path + [fname]))

    return None


# ---------------------------------------------------------------------------
# create_sub_schemas
# ---------------------------------------------------------------------------

def create_sub_schemas(
    split_info: SplitPointInfo,
    response_format: type[BaseModel],
    include_explanation: bool = False,
) -> dict[str, type[BaseModel]]:
    """Create one sub-schema per list field, wrapped in the original nesting.

    For a schema like::

        ExtractionSchema(entities: Entities(people: list[Entity], pathogens: list[Pathogen]))

    this returns::

        {
            "people": ExtractionSchema_people(entities: Entities_people(people: list[Entity])),
            "pathogens": ExtractionSchema_pathogens(entities: Entities_pathogens(pathogens: list[Pathogen])),
        }

    If *include_explanation* is ``True``, an ``explanation: str`` field is added
    at the root level of each sub-schema.
    """
    sub_schemas: dict[str, type[BaseModel]] = {}

    for field_name in split_info.list_field_names:
        annotation = split_info.list_field_annotations[field_name]
        original_field_info = split_info.split_model.model_fields[field_name]

        # Build the inner model containing only this one list field
        inner_fields: dict[str, Any] = {
            field_name: (annotation, original_field_info),
        }
        inner_model = create_model(
            f"{split_info.split_model.__name__}_{field_name}",
            **inner_fields,
        )

        # Re-wrap in the nesting above the split point
        current_model = inner_model
        for ancestor_field in reversed(split_info.path_from_root):
            # Find the ancestor's original field info from the root model
            ancestor_info = _resolve_field_info(response_format, split_info.path_from_root, ancestor_field)
            desc = ancestor_info.description if ancestor_info and ancestor_info.description else None
            wrapper_fields: dict[str, Any] = {
                ancestor_field: (current_model, Field(description=desc)) if desc else (current_model, Field()),
            }
            current_model = create_model(
                f"{response_format.__name__}_{field_name}_wrap_{ancestor_field}",
                **wrapper_fields,
            )

        # If there's no nesting (split at root level), current_model is already inner_model
        root_model = current_model

        # Optionally add explanation field at root level
        if include_explanation:
            root_fields: dict[str, Any] = {
                "explanation": (str, Field(description="Reasoning explanation")),
            }
            for fname, finfo in root_model.model_fields.items():
                root_fields[fname] = (finfo.annotation, finfo)
            root_model = create_model(
                f"{response_format.__name__}_{field_name}",
                **root_fields,
            )
        else:
            # Rename to a cleaner name
            root_model = create_model(
                f"{response_format.__name__}_{field_name}",
                **{fname: (finfo.annotation, finfo) for fname, finfo in root_model.model_fields.items()},
            )

        sub_schemas[field_name] = root_model

    return sub_schemas


def _resolve_field_info(
    root_model: type[BaseModel],
    path: list[str],
    target_field: str,
) -> Any | None:
    """Walk *path* from *root_model* to find the FieldInfo for *target_field*."""
    current = root_model
    for step in path:
        if step == target_field:
            return current.model_fields.get(step)
        finfo = current.model_fields.get(step)
        if finfo is None:
            return None
        ann = finfo.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            current = ann
        else:
            return None
    return current.model_fields.get(target_field)


# ---------------------------------------------------------------------------
# generate_sub_prompts
# ---------------------------------------------------------------------------

async def generate_sub_prompts(
    original_prompt: str,
    field_names: list[str],
    client: LLMClient | None = None,
    rewrite_model: str = "gpt-4o-mini",
    custom_sub_prompts: dict[str, str] | None = None,
) -> dict[str, str]:
    """Create one focused prompt per entity field by rewriting the original.

    Uses an LLM to rewrite the original multi-entity prompt into a clean,
    focused single-entity prompt for each field.  If *custom_sub_prompts* is
    provided it must cover every field name and is used as-is.
    """
    if custom_sub_prompts is not None:
        missing = set(field_names) - set(custom_sub_prompts)
        if missing:
            raise ValueError(
                f"custom_sub_prompts is missing entries for: {sorted(missing)}"
            )
        return {name: custom_sub_prompts[name] for name in field_names}

    client = client or LLMClient()

    async def _rewrite_for_field(field_name: str) -> tuple[str, str]:
        display_name = field_name.replace("_", " ")
        rewrite_instruction = (
            f"Rewrite the following extraction prompt so that it ONLY extracts "
            f"\"{display_name}\". Remove all references to other entity types, "
            f"their definitions, and their formatting instructions. "
            f"Keep the same tone, style, and level of detail. "
            f"The output schema should only contain the \"{field_name}\" field. "
            f"Return ONLY the rewritten prompt text, nothing else.\n\n"
            f"--- ORIGINAL PROMPT ---\n{original_prompt}"
        )

        response = await client.complete(
            model=rewrite_model,
            messages=[{"role": "user", "content": rewrite_instruction}],
            temperature=0.0,
        )
        rewritten = response.choices[0].message.content.strip()
        logger.info(
            "sub_prompt_rewritten",
            field=field_name,
            model=rewrite_model,
        )
        return field_name, rewritten

    results = await asyncio.gather(
        *[_rewrite_for_field(name) for name in field_names]
    )
    return dict(results)


# ---------------------------------------------------------------------------
# merge_sub_results
# ---------------------------------------------------------------------------

def merge_sub_results(
    sub_results: dict[str, str],
    split_info: SplitPointInfo,
) -> str:
    """Merge per-field JSON sub-results into one combined JSON string.

    Each value in *sub_results* is a raw JSON string from the LLM.  This
    function parses each, navigates down ``split_info.path_from_root`` to
    extract the list for that field, then reassembles everything into a
    single JSON object with the full nesting restored.

    Parse errors are handled gracefully by defaulting to an empty list.
    """
    merged_at_split: dict[str, Any] = {}

    for field_name, raw_json in sub_results.items():
        try:
            parsed = json.loads(raw_json)
        except (json.JSONDecodeError, TypeError):
            logger.warning("decompose_parse_error", field=field_name)
            merged_at_split[field_name] = []
            continue

        # Navigate down path_from_root to reach the split-point object
        current = parsed
        for step in split_info.path_from_root:
            if isinstance(current, dict) and step in current:
                current = current[step]
            else:
                current = {}
                break

        # Extract the list for this field
        if isinstance(current, dict) and field_name in current:
            merged_at_split[field_name] = current[field_name]
        else:
            merged_at_split[field_name] = []

    # Rebuild the full nested structure
    result = merged_at_split
    for step in reversed(split_info.path_from_root):
        result = {step: result}

    return json.dumps(result)


# ---------------------------------------------------------------------------
# DecomposedEvaluator
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# decompose_few_shot_examples / inject_few_shot_into_sub_prompts
# ---------------------------------------------------------------------------

def decompose_few_shot_examples(
    examples: list[dict[str, str]],
    split_info: SplitPointInfo,
) -> dict[str, list[dict[str, str]]]:
    """Split full few-shot examples into per-field examples.

    Each entry in *examples* has ``{"document": str, "label": str}`` where
    *label* is a JSON string covering all entity fields.  This function
    returns a dict mapping each field name to a list of examples whose
    labels contain only that single field.

    Examples where the field's list is empty are skipped.
    """
    field_examples: dict[str, list[dict[str, str]]] = {
        name: [] for name in split_info.list_field_names
    }

    for example in examples:
        try:
            label_obj = json.loads(example["label"])
        except (json.JSONDecodeError, TypeError, KeyError):
            continue

        # Navigate down path_from_root to reach the split-point object
        split_obj = label_obj
        for step in split_info.path_from_root:
            if isinstance(split_obj, dict) and step in split_obj:
                split_obj = split_obj[step]
            else:
                split_obj = None
                break

        if not isinstance(split_obj, dict):
            continue

        for field_name in split_info.list_field_names:
            field_value = split_obj.get(field_name, [])
            if not field_value:
                continue

            # Rebuild the nested label with only this one field
            single_field_obj: dict[str, Any] = {field_name: field_value}
            rebuilt = single_field_obj
            for step in reversed(split_info.path_from_root):
                rebuilt = {step: rebuilt}

            field_examples[field_name].append({
                "document": example["document"],
                "label": json.dumps(rebuilt),
            })

    return field_examples


def inject_few_shot_into_sub_prompts(
    sub_prompts: dict[str, str],
    field_examples: dict[str, list[dict[str, str]]],
) -> dict[str, str]:
    """Inject per-field few-shot examples into already-rewritten sub-prompts.

    For each field with examples, builds an examples text block and inserts it
    before the ``{content}`` placeholder (or appends if the placeholder is
    absent).  Sub-prompts with no matching examples are left unchanged.
    """
    result: dict[str, str] = {}

    for field_name, prompt in sub_prompts.items():
        examples = field_examples.get(field_name, [])
        if not examples:
            result[field_name] = prompt
            continue

        examples_text = "\n\nHere are some examples:\n\n"
        for i, ex in enumerate(examples, 1):
            examples_text += f"Example {i}:\n"
            examples_text += f"Document: {ex['document']}\n"
            examples_text += f"Label: {ex['label']}\n\n"

        caveat = "Only extract values from the text, not the provided examples.\n\n"

        if "{content}" in prompt:
            parts = prompt.split("{content}", 1)
            result[field_name] = (
                parts[0] + examples_text + caveat
                + "Now extract from this document:\n\n{content}" + parts[1]
            )
        else:
            result[field_name] = prompt + examples_text + caveat

    return result


async def cleanup_few_shot_sub_prompts(
    sub_prompts: dict[str, str],
    client: LLMClient | None = None,
    cleanup_model: str = "gpt-4o-mini",
) -> dict[str, str]:
    """Use an LLM to clean up sub-prompts after few-shot example injection.

    The mechanical injection of examples can leave formatting artifacts
    (e.g. a dangling ``Text:`` label before the examples block).  This
    function asks a fast model to tidy each prompt while preserving the
    exact content — examples, instructions, and the ``{content}``
    placeholder.
    """
    client = client or LLMClient()

    async def _cleanup(field_name: str, prompt: str) -> tuple[str, str]:
        instruction = (
            "The following extraction prompt had few-shot examples injected "
            "into it mechanically. Clean up the formatting so it reads "
            "naturally:\n"
            "- Remove any dangling labels that now precede the examples "
            "section instead of the document (e.g. a bare \"Text:\" line "
            "right before \"Here are some examples\"). These labels were "
            "originally meant to introduce the document but are now misplaced.\n"
            "- Fix awkward transitions between the instructions and the "
            "examples section, and between the examples and the document "
            "placeholder.\n"
            "- Do NOT change the meaning, the examples, or the output schema.\n"
            "- The placeholder {content} must remain exactly as-is.\n"
            "Return ONLY the cleaned-up prompt text, nothing else.\n\n"
            f"--- PROMPT TO CLEAN UP ---\n{prompt}"
        )
        response = await client.complete(
            model=cleanup_model,
            messages=[{"role": "user", "content": instruction}],
            temperature=0.0,
        )
        cleaned = response.choices[0].message.content.strip()
        logger.info("few_shot_sub_prompt_cleaned", field=field_name, model=cleanup_model)
        return field_name, cleaned

    results = await asyncio.gather(
        *[_cleanup(name, prompt) for name, prompt in sub_prompts.items()]
    )
    return dict(results)


# ---------------------------------------------------------------------------
# DecomposedEvaluator
# ---------------------------------------------------------------------------

class DecomposedEvaluator:
    """Orchestrates decomposed per-field evaluation across documents."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()
        self.evaluator = PromptEvaluator(client=self.client)

    async def evaluate(
        self,
        documents: list[Document],
        labels: list[Label],
        sub_prompts: dict[str, str],
        sub_schemas: dict[str, type[BaseModel]],
        split_info: SplitPointInfo,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        comparison_fn: Callable[..., bool] | None = None,
        max_concurrent: int = 5,
        field_metrics_config: FieldMetricsConfig | None = None,
        hallucination_filter: bool = False,
        multi_pass: int = 1,
    ) -> EvaluationResult:
        """Evaluate all documents using decomposed sub-prompts.

        For each document, N sub-prompts are run in parallel (one per entity
        field), then merged and graded against the full label.
        """
        run_id = str(uuid.uuid4())
        result = EvaluationResult(
            run_id=run_id,
            started_at=datetime.now(),
            prompt_template=next(iter(sub_prompts.values()), ""),
            model=model,
            status="running",
        )

        label_map = {label.document_id: label for label in labels}
        semaphore = asyncio.Semaphore(max_concurrent)

        async def eval_doc(doc: Document) -> PredictionResult | None:
            label = label_map.get(doc.id)
            if label is None:
                return None
            async with semaphore:
                return await self._evaluate_single_decomposed(
                    document=doc,
                    label=label,
                    sub_prompts=sub_prompts,
                    sub_schemas=sub_schemas,
                    split_info=split_info,
                    model=model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    comparison_fn=comparison_fn,
                    field_metrics_config=field_metrics_config,
                    hallucination_filter=hallucination_filter,
                    multi_pass=multi_pass,
                )

        predictions = await asyncio.gather(
            *[eval_doc(doc) for doc in documents]
        )
        result.predictions = [p for p in predictions if p is not None]

        result.compute_metrics()
        result.completed_at = datetime.now()
        result.status = "completed"

        logger.info(
            "decomposed_evaluation_completed",
            run_id=run_id,
            accuracy=result.metrics.accuracy if result.metrics else 0,
            total_cost=result.metrics.total_cost if result.metrics else 0,
        )
        return result

    async def _evaluate_single_decomposed(
        self,
        document: Document,
        label: Label,
        sub_prompts: dict[str, str],
        sub_schemas: dict[str, type[BaseModel]],
        split_info: SplitPointInfo,
        model: str,
        temperature: float = 0.0,
        max_tokens: int | None = None,
        comparison_fn: Callable[..., bool] | None = None,
        field_metrics_config: FieldMetricsConfig | None = None,
        hallucination_filter: bool = False,
        multi_pass: int = 1,
    ) -> PredictionResult:
        """Run all sub-prompts for a single document, merge, and grade."""
        start_time = time.time()
        sub_results: dict[str, str] = {}
        total_cost = 0.0
        sub_metadata: dict[str, Any] = {}

        # Dummy label — we only care about the raw LLM output per sub-task
        dummy_label = Label(document_id=document.id, value="{}")

        # Run sub-prompts in parallel
        async def run_sub(field_name: str) -> tuple[str, PredictionResult]:
            pred = await self.evaluator.evaluate_single(
                document=document,
                label=dummy_label,
                prompt_template=sub_prompts[field_name],
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format=sub_schemas.get(field_name),
                multi_pass=multi_pass,
            )
            return field_name, pred

        field_predictions = await asyncio.gather(
            *[run_sub(fn) for fn in sub_prompts]
        )

        for field_name, pred in field_predictions:
            sub_results[field_name] = pred.predicted_value
            total_cost += pred.cost
            sub_metadata[field_name] = pred.predicted_value

        # Merge sub-results
        merged_json = merge_sub_results(sub_results, split_info)

        # Apply hallucination filter to the merged result
        if hallucination_filter:
            merged_json = await filter_hallucinated_values(
                merged_json, document.content, model, self.client,
            )

        # Grade the merged result against the real label
        from valtron_core.evaluation.json_eval import JsonEvaluator

        is_correct = False
        example_score = 0.0
        field_metrics = None

        if comparison_fn:
            is_correct = comparison_fn(merged_json, label.value, document.content)
            example_score = 1.0 if is_correct else 0.0

        if field_metrics_config:
            try:
                evaluator = JsonEvaluator(
                    custom_metrics=field_metrics_config.custom_metrics,
                    custom_aggs=field_metrics_config.custom_aggs,
                )
                field_metrics = evaluator.evaluate(
                    field_metrics_config.config,
                    label.value,
                    merged_json,
                )
                example_score = field_metrics.score
                is_correct = field_metrics.is_correct
            except Exception as e:
                logger.warning(
                    "decomposed_field_metrics_error",
                    document_id=document.id,
                    error=str(e),
                )

        response_time = time.time() - start_time
        model_name = model if isinstance(model, str) else model.get("model", "unknown")

        return PredictionResult(
            document_id=document.id,
            predicted_value=merged_json,
            expected_value=label.value,
            is_correct=is_correct,
            example_score=example_score,
            response_time=response_time,
            cost=total_cost,
            model=model_name,
            field_metrics=field_metrics,
            metadata={
                "content": document.content,
                "decomposed": True,
                "sub_results": sub_metadata,
            },
        )
