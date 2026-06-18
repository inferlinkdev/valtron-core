from __future__ import annotations
from typing import Any
import json
import numpy as np

from valtron_core.evaluation.json_eval.schema import MATCH_KEY_MAX_CHARS


def _cosine_matrix(exp_vecs: list[list[float]], act_vecs: list[list[float]]) -> np.ndarray:
    """Cosine-similarity matrix between two sets of vectors via one normalized matrix multiply.

    Row ``i`` / column ``j`` is the cosine similarity of ``exp_vecs[i]`` and ``act_vecs[j]``.
    Computing it as a single BLAS matmul (rather than pairwise in Python) keeps alignment cheap
    even for long lists. A zero-magnitude vector yields ``0.0`` similarity, matching a direct
    cosine.

    :param exp_vecs: Expected-item embedding vectors (n_exp × d).
    :param act_vecs: Actual-item embedding vectors (n_act × d).
    :return: An ``n_exp × n_act`` cosine-similarity matrix.
    """
    E = np.asarray(exp_vecs, dtype=float)
    A = np.asarray(act_vecs, dtype=float)
    # Clip the norm so a zero vector normalizes to zero (→ 0 similarity) instead of dividing by 0.
    E /= np.clip(np.linalg.norm(E, axis=1, keepdims=True), 1e-12, None)
    A /= np.clip(np.linalg.norm(A, axis=1, keepdims=True), 1e-12, None)

    return E @ A.T


def _match_key_text(item: Any, fields: list[str] | None) -> str:
    """Build the text embedded to represent ``item`` when aligning candidates.

    An item is characterized by its **top-level elements only** — nested lists/dicts are not
    recursed into, since identity almost always lives at the top level and the nested content
    is mostly noise (and bulk) for matching. Resolution:

    * explicit ``fields`` (dict item): serialize just those fields (a nested field named
      explicitly is still honored);
    * no ``fields`` (dict item): serialize only the top-level *scalar* fields;
    * neither yields anything (e.g. an item whose identity is entirely nested), or a
      non-dict item: fall back to a whole-item rendering.

    The result is truncated to :data:`MATCH_KEY_MAX_CHARS` as a safety bound so a batched
    embedding over a long list cannot grow into an oversized request.

    :param item: The list item (dict or primitive).
    :param fields: Explicit identity field names to embed, or None to use top-level scalars.
    :return: A compact, length-bounded text representation suitable for embedding.
    """
    if not isinstance(item, dict):
        return json.dumps(item, default=str, ensure_ascii=False, sort_keys=True)[:MATCH_KEY_MAX_CHARS]

    if fields:
        keys = [f for f in fields if f in item and item[f] is not None]
    else:
        # Top-level (non-recursive) scalar fields only.
        keys = [k for k, v in item.items() if v is not None and isinstance(v, (str, int, float, bool))]

    parts: list[str] = []
    for k in keys:
        val = item[k]
        val_str = val if isinstance(val, str) else json.dumps(val, default=str, ensure_ascii=False)
        parts.append(f"{k}: {val_str}")

    if not parts:
        return json.dumps(item, default=str, ensure_ascii=False, sort_keys=True)[:MATCH_KEY_MAX_CHARS]

    return "\n".join(parts)[:MATCH_KEY_MAX_CHARS]


def _truncate_for_prompt(value: Any, limit: int = 200) -> str:
    """Serialize a field value and truncate it for inclusion in a sample prompt.

    :param value: The field value to render.
    :param limit: Maximum number of characters to keep.
    :return: A truncated string representation.
    """
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    if len(text) > limit:
        return text[:limit] + "…"

    return text
