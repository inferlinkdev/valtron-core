<div align="center">

# Valtron Core

[Documentation](docs/valtron/) · [Quick Start](#quick-start) · [Examples](#examples) · [Contributing](#contributing)

A Python framework for evaluating and optimizing LLM calls across any provider.

Proudly built and backed by [InferLink](https://inferlink.com).

</div>

---

Valtron lets you run the same task across multiple LLMs simultaneously, then compare them on accuracy, cost, and speed. Define a prompt, a labeled dataset, and the models you want to test — Valtron evaluates each one, scores the results, and produces an interactive HTML/PDF report with an AI recommendation.

## Features

- **Multi-model comparison** — run GPT-4o, Claude, Gemini, Llama, and any LiteLLM-compatible model side-by-side on the same dataset
- **Detailed evaluation** — label classification (string match) and structured extraction (nested JSON with field-level precision/recall/F1)
- **Prompt optimization** — seven built-in strategies (few-shot generation, chain-of-thought, decomposition, hallucination filtering, and more) applied per model
- **Cost and latency tracking** — per-document and aggregate cost, response time, and accuracy metrics
- **HTML and PDF reports** — interactive charts, per-document breakdowns, and an AI-generated recommendation
- **Transformer training** — train a local DistilBERT classifier from your evaluation data for zero-cost inference
- **Self-hosted model support** — Ollama, vLLM, LM Studio, and more for free, private inference

## Quick Start

### Prerequisites

- Python 3.12+
- [Poetry](https://python-poetry.org/) for dependency management
- Docker (optional)

### Install

```bash
poetry install
```

Copy `.env.example` to `.env` and add at least one provider key:

```env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

### Run an evaluation

```python
from valtron_core.recipes import ModelEval

data = [
    {"id": "1", "content": "Fast shipping, exactly what I ordered.", "label": "positive"},
    {"id": "2", "content": "Wrong item sent. Refund process was painful.", "label": "negative"},
    {"id": "3", "content": "Average experience, nothing special.", "label": "neutral"},
]

config = {
    "models": [
        {"name": "gpt-4o-mini"},
        {"name": "claude-haiku-4-5-20251001"},
    ],
    "prompt": "Classify the sentiment of the following review as positive, negative, or neutral.\n\nReview: {document}\n\nSentiment:",
    "output_dir": "./results",
}

experiment = ModelEval(config=config, data=data)
report_path = experiment.run()
print(f"Report: {report_path}")
```

Open `./results/evaluation_report.html` to see accuracy, cost, latency, and a recommendation.

## Configuration

The config accepts a dict, a JSON file path, or a typed `ModelEvalConfig` object. Key fields:

| Field | Required | Description |
|---|---|---|
| `models` | Yes | Models to evaluate; each has a `name` and optional `prompt_manipulation` list |
| `prompt` | Yes | Prompt template containing `{document}` |
| `output_dir` | No | Where to write results (can also be passed to `run()`) |
| `few_shot` | No | Settings for few-shot example generation |
| `output_formats` | No | `["html"]` by default; add `"pdf"` for a PDF report |

**Prompt manipulation options** (set per model via `"prompt_manipulation": [...]`):

- `"explanation"` — add chain-of-thought reasoning before the answer
- `"few_shot"` — inject generated few-shot examples into the prompt
- `"decompose"` — split the prompt into sequential sub-calls *(structured extraction only)*
- `"hallucination_filter"` — drop extracted values not grounded in the source text *(structured extraction only)*
- `"multi_pass"` — call the model twice and merge results *(structured extraction only)*

See [Config Format](docs/valtron/docs/config-format.md) for the full reference, including field-level metrics, structured extraction, and few-shot settings.

**Prefer a guided setup?** The Configuration Wizard is a browser-based UI that builds your config file step by step:

```bash
poetry run python -m valtron_core.utilities.config_wizard
# or with Docker
docker compose run --rm -p 5000:5000 server poetry run python -m valtron_core.utilities.config_wizard
```

Open `http://localhost:5000` in your browser.

## Examples

| Script | What it demonstrates |
|---|---|
| `examples/sentiment_classification.py` | Label classification across two models |
| `examples/affiliation_extraction.py` | Structured extraction with field-level metrics |
| `examples/transformer_comparison.py` | Train a local transformer and compare against cloud LLMs |
| `examples/multimodal_molecules.py` | Multimodal classification with image attachments |
| `examples/incremental_evaluation.py` | Load a prior run and add new models without re-evaluating |

```bash
# Run any example
poetry run python examples/sentiment_classification.py

# With Docker
docker compose run --rm server poetry run python examples/sentiment_classification.py
```

See [Examples](docs/valtron/docs/examples/) for detailed walkthroughs.

## Docker

```bash
# Build the image (first run is slow; installs system dependencies including LaTeX)
docker compose build server

# Run an example
docker compose run --rm server poetry run python examples/sentiment_classification.py

# Interactive shell
docker compose run --rm --service-ports server bash

# Run the test suite
docker compose run --rm pipeline-test
```

> **Using Ollama with Docker?** Set `OLLAMA_API_BASE=http://host.docker.internal:11434` in `.env` so the container can reach Ollama running on your host.

## Development

```bash
# Install dependencies
poetry install

# Run tests
poetry run pytest

# Run with coverage
poetry run pytest --cov=src/valtron_core --cov-report=html

# Format and lint
poetry run black src/ tests/
poetry run ruff check src/ tests/
poetry run mypy src/
```

The test suite covers the evaluation pipeline, report generation, prompt optimizers, few-shot generation, transformer training, and the `ModelEval` recipe. The Docker Compose `pipeline-test` service runs the full suite with coverage and JUnit XML output.

## Architecture

Valtron is built on [LiteLLM](https://github.com/BerriAI/litellm), which provides a unified interface to 100+ LLM providers. On top of that, Valtron adds the evaluation pipeline, prompt optimization strategies, report generation, and transformer training:

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

Full documentation lives in [docs/valtron/](docs/valtron/). To run the docs site locally:

```bash
cd docs/valtron && docker compose up
```

Then open `http://localhost:3000`.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes and run the test suite
4. Submit a pull request

## License

TBD
