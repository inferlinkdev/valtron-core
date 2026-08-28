# Choosing the Right Approach

## What shape is your output?

- **A single label from a closed or open set of strings** (sentiment, topic, category) → [Chapter 1: Classification](./classification/index). No `response_format` needed.
- **A nested JSON object** (multiple fields, lists of entities, sub-objects) → [Chapter 2: Extraction](./extraction/index). Pass a Pydantic `response_format`, and use [Field Metrics](./extraction/field-metrics/index) for per-field scoring instead of a single pass/fail.
- **Free-form text with no single correct answer** (a document summary) → [Chapter 3: Summarization](./summarization/index). No `label` needed; an LLM judge decides which facts a good summary must convey and scores candidates against that instead of a ground truth.

## Trying to improve accuracy on a single model?

Reach for [Chapter 4: Prompt Manipulations](./manipulations/index) before reaching for a different model or more data:

- Ambiguous categories or tasks that benefit from reasoning → `explanation`
- A model that ignores single-pass instructions → `prompt_repetition` (or `prompt_repetition_x3`)
- Limited labeled data → `few_shot` (generates synthetic examples from a handful of seeds)
- Complex multi-field extraction schemas → `decompose` (extraction mode only)
- A model that invents values not in the source document → `hallucination_filter` (extraction mode only)
- Extraction tasks with variable-length lists where items get missed → `multi_pass` (extraction mode only)

## Trying to cut cost at scale?

- **Want a single free/cheap model in the mix, no cost-optimal routing logic**: train and evaluate a [local transformer](./self-hosting/transformer-models) alongside your LLMs (classification/label mode only).
- **Want the cheapest model that still hits a target accuracy, automatically, with cheaper LLMs escalating to more expensive ones only when needed**: [Chapter 6: Combining Multiple Models](./combining-models). Requires a trained transformer plus at least one LLM in a completed `ModelEval` run. Currently supports binary classification only.
- **Want to keep inference in your own infrastructure without training anything**: [Self-Hosted Models](./self-hosting/self-hosted-models), which points Valtron at Ollama, vLLM, LM Studio, or HuggingFace TGI instead of a cloud provider.

```{rubric} What's next?
```

- Already know your chapter? Jump back to the [User Guide](./index).
- Not sure why a run behaved a certain way? [Common Pitfalls](./common-pitfalls) covers the gotchas that come up most often.
