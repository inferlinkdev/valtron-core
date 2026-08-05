"""Measure LLM accuracy on a labelled dataset via structured outputs.

Used by the tradeoff report when the user wants empirical (rather than guessed)
accuracy estimates for each LLM tier under consideration as a deferral target.

Calls each LLM on a stratified sample of the eval set, constrains the response
to a JSON schema enum over the dataset's actual label set (no regex parsing,
no failure modes from chatty responses), and returns measured accuracy plus
the per-example predictions. Results are cached to disk keyed by
``(llm_name, n_samples, prompt_hash)`` so re-renders are free.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import litellm

DEFAULT_PROMPT_TEMPLATE = (
    "Classify the following text into exactly one of the categories: {labels}.\n\n"
    "Text:\n{text}"
)


@dataclass
class MeasurementResult:
    """One LLM's measured accuracy plus the per-sample evidence."""
    llm_name: str
    n_samples: int
    n_correct: int
    accuracy: float
    ci_half_width: float
    predictions: list[dict] = field(default_factory=list)  # sample dicts, see _measure_one

    @property
    def has_data(self) -> bool:
        return self.n_samples > 0


def _build_response_schema(labels: list[str]) -> dict:
    """Strict JSON schema with the label field constrained to the dataset's enum."""
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "Classification",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "label": {"type": "string", "enum": list(labels)},
                },
                "required": ["label"],
                "additionalProperties": False,
            },
        },
    }


def _stratified_sample(
    rows: list[dict],
    n_samples: int,
    seed: int = 42,
) -> list[tuple[int, dict]]:
    """Return ``[(original_index, row), ...]`` stratified by row['label']."""
    by_label: dict[str, list[tuple[int, dict]]] = {}
    for i, r in enumerate(rows):
        by_label.setdefault(r["label"], []).append((i, r))

    rng = random.Random(seed)
    n_total = sum(len(v) for v in by_label.values())
    sampled: list[tuple[int, dict]] = []
    for label, group in by_label.items():
        # Proportional allocation; minimum 1 per stratum if the group is non-empty
        n_for_label = max(1, round(n_samples * len(group) / n_total))
        n_for_label = min(n_for_label, len(group))
        sampled.extend(rng.sample(group, n_for_label))
    # If rounding under-/over-shot, trim or top up
    if len(sampled) > n_samples:
        rng.shuffle(sampled)
        sampled = sampled[:n_samples]
    elif len(sampled) < n_samples:
        leftover = [
            (i, r) for i, r in enumerate(rows)
            if (i, r) not in set(sampled)
        ]
        rng.shuffle(leftover)
        sampled.extend(leftover[: n_samples - len(sampled)])
    return sampled


def _prompt_hash(prompt_template: str, labels: list[str]) -> str:
    h = hashlib.sha256()
    h.update(prompt_template.encode("utf-8"))
    h.update(b"\0")
    h.update("|".join(sorted(labels)).encode("utf-8"))
    return h.hexdigest()[:12]


def _cache_path(cache_dir: Path, llm_name: str, n_samples: int, prompt_hash: str) -> Path:
    safe = llm_name.replace("/", "__")
    return cache_dir / f"llm_predictions_{safe}_n{n_samples}_{prompt_hash}.jsonl"


def _load_cache(path: Path) -> list[dict] | None:
    if not path.exists():
        return None
    out: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _save_cache(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


async def _call_one(
    sem: asyncio.Semaphore,
    llm_name: str,
    prompt: str,
    schema: dict,
    max_retries: int = 2,
) -> str | None:
    """Returns the predicted label string, or None if the call ultimately failed.

    We intentionally do NOT pass ``temperature``: reasoning-capable models
    (o1/o3/gpt-5.x) reject any explicit value and require the model default.
    Determinism is anyway guaranteed by the response_format enum constraint —
    the label MUST be one of the dataset's actual labels.

    ``max_tokens=512`` gives reasoning models enough budget to think before
    emitting the answer. Non-reasoning models stop once the JSON is complete,
    so the budget doesn't inflate their cost.
    """
    async with sem:
        for attempt in range(max_retries + 1):
            try:
                response = await litellm.acompletion(
                    model=llm_name,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=schema,
                    max_tokens=512,
                )
                content = response.choices[0].message.content
                if not content:
                    raise ValueError("empty response content")
                parsed = json.loads(content)
                return parsed.get("label")
            except Exception:
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * (attempt + 1))
        return None


async def _measure_one(
    llm_name: str,
    samples: list[tuple[int, dict]],
    labels: list[str],
    prompt_template: str,
    max_concurrency: int,
) -> list[dict]:
    """Returns per-sample dicts with sample_idx, ground_truth, prediction."""
    sem = asyncio.Semaphore(max_concurrency)
    schema = _build_response_schema(labels)
    labels_csv = ", ".join(f'"{lbl}"' for lbl in labels)

    async def go(idx: int, row: dict) -> dict:
        prompt = prompt_template.format(text=row["text"], labels=labels_csv)
        pred = await _call_one(sem, llm_name, prompt, schema)
        return {
            "sample_idx": idx,
            "ground_truth": row["label"],
            "prediction": pred,
        }

    return await asyncio.gather(*(go(i, r) for i, r in samples))


def measure_llm_accuracies(
    rows: list[dict],
    llm_names: Iterable[str],
    n_samples: int,
    *,
    prompt_template: str = DEFAULT_PROMPT_TEMPLATE,
    cache_dir: str | Path = "examples/results",
    max_concurrency: int = 20,
    seed: int = 42,
) -> dict[str, MeasurementResult]:
    """Measure each LLM's accuracy on a stratified sample of the data.

    Args:
        rows: List of ``{"text", "label"}`` dicts (full eval set).
        llm_names: LLM identifiers to measure (litellm names).
        n_samples: How many examples to sample. 0 returns empty results.
        prompt_template: Must contain ``{text}`` and may contain ``{labels}``.
        cache_dir: Where to read/write per-LLM prediction cache files.
        max_concurrency: Max simultaneous API calls per LLM.
        seed: Sampling RNG seed.

    Returns:
        Dict ``{llm_name: MeasurementResult}``. Always populated even when
        cached predictions are loaded from disk.
    """
    if n_samples <= 0:
        return {name: MeasurementResult(name, 0, 0, 0.0, 0.0) for name in llm_names}

    labels = sorted({r["label"] for r in rows})
    if len(labels) < 2:
        raise ValueError(f"Need at least 2 distinct labels in data, got {labels}")

    samples = _stratified_sample(rows, n_samples, seed=seed)
    ph = _prompt_hash(prompt_template, labels)
    cache_dir = Path(cache_dir)

    results: dict[str, MeasurementResult] = {}
    for name in llm_names:
        cache_file = _cache_path(cache_dir, name, len(samples), ph)
        records = _load_cache(cache_file)
        if records is None:
            print(f"  Measuring {name} on {len(samples)} samples...")
            records = asyncio.run(
                _measure_one(name, samples, labels, prompt_template, max_concurrency)
            )
            _save_cache(cache_file, records)
            print(f"    Wrote cache: {cache_file}")
        else:
            print(f"  Loaded {name} cache: {cache_file}")
        results[name] = _summarize(name, records)
    return results


def _summarize(llm_name: str, records: list[dict]) -> MeasurementResult:
    valid = [r for r in records if r["prediction"] is not None]
    n_valid = len(valid)
    n_failed = len(records) - n_valid
    n_correct = sum(1 for r in valid if r["prediction"] == r["ground_truth"])
    acc = n_correct / n_valid if n_valid > 0 else 0.0
    # 95% CI half-width via normal approximation
    ci = 1.96 * math.sqrt(acc * (1 - acc) / n_valid) if n_valid > 0 else 0.0
    if n_failed > 0:
        print(f"    [{llm_name}] WARNING: {n_failed}/{len(records)} calls failed; "
              f"accuracy computed over {n_valid} successful calls only")
    return MeasurementResult(
        llm_name=llm_name,
        n_samples=n_valid,
        n_correct=n_correct,
        accuracy=acc,
        ci_half_width=ci,
        predictions=records,
    )
