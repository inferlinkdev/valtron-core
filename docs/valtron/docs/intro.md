---
sidebar_position: 1
---

# Valtron


Valtron is a Python framework for evaluating and optimizing LLM calls. It lets you run the same task across multiple models simultaneously, then compare them on accuracy, cost, and speed. An interactive HTML/PDF report summarizes the results and recommends the best model for your use case.

Valtron is proudly built and backed by [InferLink](https://inferlink.com).

## What it does

- **Multi-model comparison**: run GPT-4o, Claude, Gemini, Llama, and any other LiteLLM-compatible model side-by-side on the same dataset
- **Detailed evaluation**: label classification (string match) and structured extraction (nested JSON with field-level precision/recall/F1)
- **Prompt optimization**: seven built-in strategies (few-shot generation, chain-of-thought, decomposition, hallucination filtering, and more) that can be applied per model
- **Cost and latency tracking**: per-document and aggregate cost, response time, and accuracy metrics
- **HTML and PDF reports**: interactive charts, per-document breakdowns, and an AI-generated recommendation
- **Transformer training**: train a local DistilBERT classifier from your evaluation data for zero-cost inference
- **Self-hosted model support**: run any LiteLLM-compatible local provider (Ollama, vLLM, LM Studio, and more) for free, private inference

## How it works

```
Input data (documents + expected labels)
    ↓
Config (models, prompt, optimizations)
    ↓
ModelEval.run()
    ├── Generate few-shot examples (optional)
    ├── Prepare per-model prompts (apply manipulations)
    ├── Evaluate all models concurrently
    ├── Compute metrics (accuracy, cost, time, field scores)
    └── Generate report (HTML + optional PDF)
    ↓
evaluation_report.html  ·  models/*.json  ·  metadata.json
```

## Quick navigation

- New here? Start with [Installation](./getting-started/installation) and [Quick Start](./getting-started/quick-start)
- Understand inputs: [Data Format](./data-format) and [Config Format](./config-format)
- Understand outputs: [Evaluation Results](./evaluation-results) and [Report Formats](./report-formats)
- Improve accuracy: [Prompt Optimizers](./optimizers)
- Run evaluations: [Evaluation API](./recipes)
- Zero-cost inference: [Transformer Models](./transformer-models)
- Run models locally: [Self-Hosted Models](./self-hosted-models)
- See working code: [Examples](./examples/)
