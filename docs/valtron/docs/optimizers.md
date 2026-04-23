---
sidebar_position: 6
---

# Prompt Optimizers

Valtron includes seven built-in prompt manipulation strategies that can be applied per model. They are applied before evaluation and modify the prompt each model receives, without changing your base prompt or data.

## How to use them

Add a `prompt_manipulation` list to any model in your config:

```json
{
  "models": [
    {
      "name": "gpt-4o-mini",
      "prompt_manipulation": ["few_shot", "explanation"]
    },
    {
      "name": "claude-haiku-4-5-20251001",
      "prompt_manipulation": ["prompt_repetition"]
    },
    {
      "name": "gpt-4o"
    }
  ]
}
```

Each model gets its own modified prompt. The base model (no manipulations) acts as a control.

## Application order

When multiple manipulations are listed, they are applied in this order:

1. `few_shot` — prepend examples
2. `explanation` — rewrite with chain-of-thought
3. `prompt_repetition` / `prompt_repetition_x3` — append repeated text
4. Structured-only manipulations (`decompose`, `hallucination_filter`, `multi_pass`) — applied during evaluation

---

## Universal manipulations

These work in both label/classification mode and structured extraction mode.

### `few_shot`

Generates synthetic document+label examples from your seed data and prepends them into the prompt before `{document}`.

**How it works:**
1. Seeds from the first `max_seed_examples` real documents in your dataset
2. Uses `generator_model` to generate `num_examples` synthetic document+label pairs
3. Validates each example via a consensus check (re-runs the model to confirm the label)
4. The top `max_few_shots` validated examples are stored and injected into prompts for any model with `"few_shot"` in its manipulation list

**Config required:** Add `few_shot` to the top-level config:

```json
{
  "few_shot": {
    "enabled": true,
    "generator_model": "gpt-4o-mini",
    "num_examples": 30,
    "max_seed_examples": 10,
    "max_few_shots": 5
  }
}
```

**Cost note:** Few-shot generation runs before evaluation and incurs one-time API calls to `generator_model`. Evaluation runs after generation completes.

---

### `explanation`

Rewrites the prompt to elicit chain-of-thought reasoning from the model.

**How it works:**
- In label mode: the prompt is rewritten to ask the model to reason step-by-step before giving its final answer
- In structured extraction mode: the response schema is automatically extended with an `explanation: str` field so the model can output its reasoning alongside the structured result

**Config:** None. The rewrite is deterministic.

**When to use:** Tasks where reasoning through the problem improves accuracy, such as classification with ambiguous categories or entity extraction with edge cases.

---

### `prompt_repetition`

Appends the prompt text a second time (the model receives the prompt twice).

**How it works:** The full prompt is appended again at the end of the message.

**When to use:** Some models perform better when the instruction is reinforced. Useful as a cheap alternative to `explanation` for models that tend to ignore single-pass instructions.

**Cost note:** Approximately doubles the input token count.

---

### `prompt_repetition_x3`

Appends the prompt text two additional times (the model receives the prompt three times total).

**How it works:** Same as `prompt_repetition` but triples the instruction.

**Cost note:** Approximately triples the input token count.

---

## Structured-only manipulations

These require a `response_format` to be passed to `ModelEval`. They will raise a validation error if used in label/classification mode.

```python
experiment = ModelEval(config=config, data=data, response_format=MySchema)
```

### `decompose`

Splits a multi-field schema into per-field sub-calls, then merges the results back into the full schema before scoring.

**How it works:**
1. Finds the list field in your response schema (the "split point")
2. Creates sub-schemas, one per entity field
3. Generates focused sub-prompts for each field (via `rewrite_model`, or from your `sub_prompts` overrides)
4. Runs one LLM call per field per document
5. Merges sub-results back into the full schema

**Config required:** Add `decompose_config` to the model:

```json
{
  "name": "gpt-4o",
  "prompt_manipulation": ["decompose"],
  "decompose_config": {
    "rewrite_model": "gpt-4o-mini",
    "sub_prompts": {
      "institutions": "Extract only the list of institution names from: {document}"
    }
  }
}
```

If omitted, Valtron auto-generates sub-prompts using `rewrite_model`.

**When to use:** Complex schemas with multiple fields where a single prompt struggles to extract everything accurately. Trades cost (more calls) for accuracy.

**Cost note:** One LLM call per entity field per document. If your schema has 5 fields, each document requires 5 calls instead of 1.

---

### `hallucination_filter`

Post-processes the model's structured output by dropping any predicted values that don't appear in the source document.

**How it works:** After the model returns a structured response, each predicted string value is checked against the original document text. Values not found in the source are set to `null` or removed from lists.

**Config:** None.

**When to use:** Tasks where the model tends to invent or infer values not present in the document. Improves precision at the cost of recall (some valid predictions may be filtered if phrasing doesn't exactly match the document).

---

### `multi_pass`

Calls the model twice with the same prompt, then deduplicates and reconciles the two structured outputs.

**How it works:** Runs two independent LLM calls per document, then merges results by deduplicating items and resolving conflicts between the two responses.

**Config:** None.

**When to use:** Tasks where the model occasionally misses items (e.g., entity extraction with variable-length lists). The second pass often catches items the first pass missed.

**Cost note:** Doubles inference cost and latency.

---

## Combining manipulations

Manipulations are applied independently per model and can be combined:

```json
{
  "name": "gpt-4o",
  "prompt_manipulation": ["few_shot", "explanation", "decompose"],
  "decompose_config": { "rewrite_model": "gpt-4o-mini" }
}
```

Running the same model with different manipulation sets is a common pattern for ablation testing:

```json
{
  "models": [
    {"name": "gpt-4o-mini", "label": "gpt-4o-mini (baseline)"},
    {"name": "gpt-4o-mini", "label": "gpt-4o-mini + few_shot", "prompt_manipulation": ["few_shot"]},
    {"name": "gpt-4o-mini", "label": "gpt-4o-mini + explanation", "prompt_manipulation": ["explanation"]},
    {"name": "gpt-4o-mini", "label": "gpt-4o-mini + both", "prompt_manipulation": ["few_shot", "explanation"]}
  ]
}
```

## Summary table

| Manipulation | Mode | Requires config | Cost impact |
|---|---|---|---|
| `few_shot` | Any | `few_shot` block in config | Pre-eval generation cost |
| `explanation` | Any | None | Small input token increase |
| `prompt_repetition` | Any | None | ~2× input tokens |
| `prompt_repetition_x3` | Any | None | ~3× input tokens |
| `decompose` | Structured only | `decompose_config` on model | N× calls (N = field count) |
| `hallucination_filter` | Structured only | None | No additional calls |
| `multi_pass` | Structured only | None | 2× calls |
