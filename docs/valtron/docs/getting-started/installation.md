---
sidebar_position: 1
---

# Installation

## Requirements

- Python 3.10+
- An API key for at least one LLM provider (or [Ollama](../self-hosted-models) for local models)

## Install

```bash
pip install valtron-core
```

This installs the core package including reporting, the configuration wizard, and evaluation utilities.

## Optional extras

To train or run inference on local transformer models (DistilBERT, etc.), install the `transformers` extra:

```bash
pip install "valtron-core[transformers]"
```

This adds `torch`, `transformers`, `scikit-learn`, and `datasets`. See [Transformer Models](../transformer-models) for details.

## Configure API keys

Valtron reads credentials from environment variables or a `.env` file in your working directory.

Create a `.env` file:

```env
# --- LLM Providers (add keys for the providers you use) ---
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
COHERE_API_KEY=...
AZURE_API_KEY=...
HUGGINGFACE_API_KEY=...
REPLICATE_API_KEY=...
TOGETHER_API_KEY=...

# AWS Bedrock
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...

# Self-hosted Ollama (default: http://localhost:11434)
OLLAMA_API_BASE=http://localhost:11434

# --- Optional settings ---

# Retry failed requests with exponential backoff
MAX_RETRIES=3           # attempts per call (default: 3)
RETRY_DELAY=1.0         # base delay in seconds; doubles each retry (default: 1.0)

# Rate limit API calls (disabled by default, no limit)
# Set only if your provider enforces a cap
# REQUESTS_PER_MINUTE=60

# Logging level: DEBUG, INFO, WARNING, ERROR
LOG_LEVEL=INFO
```

You only need keys for the providers you actually use. Unused keys can be omitted.

## Verify setup

```python
from valtron_core.client import LLMClient
import asyncio

async def test():
    client = LLMClient()
    response = await client.complete(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "Say hello"}]
    )
    print(response.choices[0].message.content)

asyncio.run(test())
```

If this runs without error you're ready to go. Continue to [Quick Start](./quick-start).
