# Prompt Manipulations

```{toctree}
:hidden:

universal-manipulations
extraction-manipulations
```

Valtron includes seven built-in prompt manipulation strategies. Each one rewrites the prompt a specific model receives before evaluation, without touching your base prompt or your data. They're most useful as an ablation: run several variants of the same model side by side in a single report and see which one actually moves accuracy.

```{rubric} Comparing manipulations side by side
```

Say `gpt-4o-mini` is landing at 80% on an ambiguous sentiment task and you want to know whether few-shot examples or chain-of-thought reasoning helps. List the same model four times with different `prompt_manipulation` sets and different `label`s; [`ModelEval`](../../api/model_eval) runs all four in one call and reports them side by side:

```python
from valtron_core.evaluation import ModelEval

config = {
    "prompt": "Classify the sentiment as positive, negative, or neutral.\n\n{content}",
    "few_shot": {"enabled": True, "generator_model": "gpt-4o-mini", "num_examples": 30},
    "models": [
        {"name": "gpt-4o-mini", "label": "baseline"},
        {"name": "gpt-4o-mini", "label": "+ few_shot", "prompt_manipulation": ["few_shot"]},
        {"name": "gpt-4o-mini", "label": "+ explanation", "prompt_manipulation": ["explanation"]},
        {"name": "gpt-4o-mini", "label": "+ both", "prompt_manipulation": ["few_shot", "explanation"]},
    ],
}

experiment = ModelEval(config=config, data=data)
experiment.run("./results/manipulation_ablation")

for result in experiment.results:
    print(f"{result.model:<16} accuracy={result.metrics.accuracy:.0%}")
```

```text
baseline         accuracy=80%
+ few_shot       accuracy=87%
+ explanation    accuracy=84%
+ both           accuracy=91%
```

Because `few_shot` is enabled once at the top level, Valtron generates the synthetic examples a single time before evaluation starts, then injects them into any model whose `prompt_manipulation` list includes `"few_shot"`; the baseline and `+ explanation` variants don't get them, since they don't ask for it.

Open `./results/manipulation_ablation/evaluation_report.html` and each variant shows up as its own row, with a "Base" / "Overridden" toggle so you can inspect the exact prompt each one actually received.

```{rubric} Application order
```

When a model lists more than one manipulation, they're applied in this order regardless of how you wrote the list:

1. `few_shot`: prepend examples
2. `explanation`: rewrite with chain-of-thought
3. `prompt_repetition` / `prompt_repetition_x3`: append repeated text
4. Extraction-based manipulations (`decompose`, `hallucination_filter`, `multi_pass`): applied during evaluation, after the prompt is built

`prompt_manipulation` values are members of the [`Manipulation`](../../api/manipulation) enum.

```{rubric} In this chapter
```

- **[3.1 Universal Manipulations](./universal-manipulations)**: `few_shot`, `explanation`, `prompt_repetition`, `prompt_repetition_x3`. Work in either classification or extraction mode.
- **[3.2 Extraction-Based Manipulations](./extraction-manipulations)**: `decompose`, `hallucination_filter`, `multi_pass`. Require `response_format`.
