---
sidebar_position: 10
---

# Self-Hosted Models

Valtron supports self-hosted LLMs through any OpenAI-compatible backend. Self-hosted models use the exact same interface as cloud LLMs and have no inference costs, though you still incur hosting and infrastructure costs (electricity, cloud compute, etc.).

**Benefits:**
- No inference costs
- No rate limits
- Data stays on your machine
- Works offline

---

## Supported Providers

| Provider | Status | Notes |
|---|---|---|
| [Ollama](#configuration-for-ollama) | Supported | Easy local setup, good for development |
| [vLLM](#configuration-for-vllm) | Supported | High-throughput, GPU-optimized server |
| [LM Studio](#configuration-for-lm-studio) | Supported | GUI-first local model runner |
| [HuggingFace TGI](#configuration-for-huggingface-tgi) | Supported | HuggingFace's production inference server |
---

## Available Models

Popular open models you can run locally:

| Model | Pull name | Size | Min RAM | Notes |
|---|---|---|---|---|
| LLama 3.1 | `llama3.1` | 8B | 8 GB | Latest, best 8B performance |
| LLama 3 | `llama3` | 8B | 8 GB | Improved reasoning |
| LLama 2 | `llama2` | 7B | 8 GB | General purpose |
| LLama 3 (large) | `llama3:70b` | 70B | 64 GB | Maximum accuracy |
| Mistral | `mistral` | 7B | 8 GB | Fast and efficient |
| Mixtral | `mixtral` | 8x7B | 48 GB | High quality MoE model |
| Phi-3 | `phi3` | 3.8B | 4 GB | Lightweight, very fast |
| Gemma | `gemma` | 7B | 8 GB | Google's open model |
| CodeLLama | `codellama` | 7B | 8 GB | Code generation |

---

## Walkthrough

The general flow for any self-hosted provider:

### 1. Install and start your model server

Choose a provider from the list above and follow its installation guide. Each provider section below has setup instructions and links.

### 2. Configure Valtron to point at your server

Set the appropriate environment variable in your `.env` file. Each provider uses a different variable (see the provider sections below).

### 3. Add the model to your config

Use the format `provider/model-name` anywhere Valtron accepts a model name:

```python
config = {
    "models": [
        {"name": "gpt-4o-mini"},
        {"name": "ollama/llama3.1", "label": "LLama 3.1 (local)"},
        {"name": "vllm/mistral", "label": "Mistral (vLLM)"},
    ],
    "prompt": "Classify: {document}",
}
```

### 4. Track infrastructure costs (optional)

Self-hosted models have no per-call API cost, but you can track effective cost (electricity, cloud compute) using `cost_rate`:

```json
{
  "name": "ollama/llama3.1",
  "cost_rate": 0.05,
  "cost_rate_time_unit": "1hr"
}
```

This lets the report calculate and compare per-document costs alongside cloud models.

#### Cost comparison example (10,000 calls)

| Model | Approximate cost |
|---|---|
| GPT-4o | $30-60 |
| Claude Sonnet | $10-20 |
| GPT-4o-mini | $1-3 |
| Self-hosted | Varies by hardware/cloud spend |

---

## Configuration for Ollama

[Ollama](https://ollama.com) runs models locally on your machine and exposes an OpenAI-compatible API. Best for local development and experimentation.

**Install Ollama:**

macOS:
```bash
brew install ollama
```

Linux:
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Windows: Download from [ollama.com/download](https://ollama.com/download)

**Start the server:**
```bash
ollama serve
```

**Pull a model:**
```bash
ollama pull llama3.1
ollama pull mistral
ollama pull phi3
```

**Set the API base** (only needed if Ollama is not on localhost):
```env
OLLAMA_API_BASE=http://localhost:11434
```

**Model naming:**
```python
{"name": "ollama/llama3.1", "label": "LLama 3.1 (local)"}
```

**Quantized models** (reduces RAM requirements):
```python
{"name": "ollama/llama3.1:q4"}   # 4-bit quantization
{"name": "ollama/llama3.1:q8"}   # 8-bit quantization
```

**Custom GGUF models:**
```bash
echo "FROM ./my-model.gguf" > Modelfile
ollama create my-custom-model -f Modelfile
# Use as: {"name": "ollama/my-custom-model"}
```

See the [Ollama Modelfile docs](https://docs.ollama.com/modelfile) for details.

**Performance notes:**
- The model loads into RAM on the first request and stays there. Subsequent calls are fast.
- Keep `ollama serve` running between evaluation runs to avoid reload time
- Ollama auto-detects GPU; verify it's using your GPU if available

**Troubleshooting:**

Connection refused: confirm Ollama is running (`ollama list`), then start it (`ollama serve`) and check the port matches `OLLAMA_API_BASE`.

Model not found: pull the model (`ollama pull llama3.1`) and confirm the name matches exactly (`ollama/llama3.1`).

Out of memory: try a smaller model (`phi3`, `gemma`) or a quantized version (`ollama/llama3.1:q4`). Stop unused models with `ollama stop <model>`.

---

## Configuration for vLLM

vLLM is a high-throughput inference server optimized for GPU deployments. Best for production or multi-user setups where request volume is high.

See the [vLLM installation guide](https://docs.vllm.ai/en/latest/getting_started/installation/) for setup instructions.

```env
VLLM_API_BASE=http://localhost:8000
```

```python
{"name": "vllm/meta-llama/Llama-3.1-8B-Instruct"}
```

---

## Configuration for LM Studio

LM Studio is a desktop app for running models locally with a GUI. Best for exploring and testing models before integrating them into a pipeline.

See the [LM Studio download page](https://lmstudio.ai/docs/app/basics) for installation instructions. Once installed, enable the local server from within the app.

```env
LM_STUDIO_API_BASE=http://localhost:1234
```

```python
{"name": "lm_studio/llama3.1"}
```

---

## Configuration for HuggingFace TGI

HuggingFace Text Generation Inference (TGI) is a production-grade server that supports any model on the HuggingFace Hub.

See the [HuggingFace TGI documentation](https://huggingface.co/docs/text-generation-inference/index) for setup instructions.

```env
HUGGINGFACE_API_BASE=http://localhost:8080
```

```python
{"name": "huggingface/meta-llama/Llama-3.1-8B-Instruct"}
```
