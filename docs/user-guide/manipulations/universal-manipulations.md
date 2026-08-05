# Universal Manipulations

A manipulation is "universal" when it only rewrites the prompt or the response schema itself, so it works the same way in either [classification](../classification/index) or [extraction](../extraction/index) mode. Turn one on per model by adding its name to that model's `prompt_manipulation` list.

## Few-Shot Examples

Enable `few_shot` once at the top level, then opt individual models in via `prompt_manipulation`:

```python
config = {
    "few_shot": {
        "enabled": True,
        "generator_model": "gpt-4o-mini",
        "num_examples": 30,
        "max_seed_examples": 10,
        "max_few_shots": 5,
    },
    "models": [{"name": "gpt-4o-mini", "prompt_manipulation": ["few_shot"]}],
}
```

Before evaluation starts, Valtron seeds from the first `max_seed_examples` real documents in your dataset, asks `generator_model` to generate `num_examples` synthetic document+label pairs, validates each one by re-running the model to confirm the label matches, and keeps the top `max_few_shots`; those are what get prepended into the prompt of any model that lists `"few_shot"`. This is a one-time cost paid before evaluation, not per model. Full field list: [`FewShotConfig`](../../api/few_shot_config).

## Chain-of-Thought Explanations

`explanation` needs no config; the rewrite is deterministic. In classification mode it asks the model to reason step by step before answering:

```python
{"name": "gpt-4o-mini", "prompt_manipulation": ["explanation"]}
```

In [extraction mode](../extraction/index), the same manipulation instead extends your response schema with an `explanation: str` field, so the model reasons alongside the structured output rather than instead of it. No prompt rewrite is needed since the schema itself carries the request, and the token cost stays small either way.

Best for ambiguous categories or extraction edge cases where reasoning through the problem measurably helps; the [worked ablation](./index) shows one way to check whether it actually does for your task.

## Prompt Repetition

```python
{"name": "claude-haiku-4-5-20251001", "prompt_manipulation": ["prompt_repetition"]}
```

`prompt_repetition` appends the full prompt again at the end of the message. `prompt_repetition_x3` appends it twice more, three copies total, at roughly 3x the input tokens instead of 2x. Some models respond better when the instruction is repeated; it's a cheap alternative to `explanation` for models that tend to ignore single-pass instructions.

```{rubric} What's next?
```

- The manipulations that require a response schema: [Extraction-Based Manipulations](./extraction-manipulations)
- Compare variants side by side in one run: [back to the ablation example](./index)
