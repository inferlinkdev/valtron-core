# Self-Hosting and Local Models

```{toctree}
:hidden:

transformer-models
self-hosted-models
```

Every model in a [`ModelEval`](../../api/model_eval) run, cloud or local, is scored the same way: accuracy against your labels, cost, and latency, in one report. Self-hosting extends that same comparison to models that don't run against a provider's API at all.

Two situations call for it. Some data can't leave your network, so the model has to come to the data instead of the data going to a provider. And at high enough volume, a model with no per-call charge, only the fixed cost of the hardware it runs on, can end up cheaper than any cloud model.

Valtron supports both routes into the same evaluation. Point it at any OpenAI-compatible local server as an ordinary model entry, or train a small transformer classifier on your own labeled data and run it as a zero-cost model entry alongside cloud models. A trained transformer can also feed [Combining Multiple Models](../combining-models), which escalates only the cases it's least confident about to a paid LLM.

- **[4.1 Transformer Models](./transformer-models)**: train and evaluate a local DistilBERT classifier ([`TransformerClassifier`](../../api/transformer_classifier)) from your own data.
- **[4.2 Self-Hosted Models](./self-hosted-models)**: run any OpenAI-compatible local provider (Ollama, vLLM, LM Studio, HuggingFace TGI) as a normal [`LLMModelConfig`](../../api/llm_model_config) entry.
