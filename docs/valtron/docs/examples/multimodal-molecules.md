---
sidebar_position: 5
---

# Multimodal Molecules

**File:** [`examples/multimodal_molecules.py`](https://github.com/your-org/valtron-core/blob/main/examples/multimodal_molecules.py)

Each document has a 2D molecular structure diagram attached as an image URL from PubChem. Vision-capable LLMs are asked to count the number of carbon atoms from the image alone.

## What it demonstrates

- Multimodal documents using the `attachments` field
- Vision-capable models (`gpt-4o-mini`, `claude-haiku`) processing image URLs
- Label/classification mode with numeric string labels

## Run it

```bash
python examples/multimodal_molecules.py
```

## Data format

Images are served directly from PubChem by compound ID (CID). The `content` field is empty because the question is entirely in the prompt; the model receives the image via `attachments`.

```python
DATA = [
    {
        "id": "methanol",
        "content": "",
        "label": "1",   # number of carbon atoms
        "attachments": ["https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=887&width=300&height=300"],
    },
    {
        "id": "benzene",
        "content": "",
        "label": "6",
        "attachments": ["https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=241&width=300&height=300"],
    },
    # ...
]
```

| Molecule | CID | Carbon atoms |
|---|---|---|
| Methanol | 887 | 1 |
| Ethanol | 702 | 2 |
| Acetone | 180 | 3 |
| Benzene | 241 | 6 |
| Caffeine | 2519 | 8 |
| Aspirin | 2244 | 9 |

## Prompt

```python
"prompt": (
    "Count the number of carbon atoms in the molecular structure shown in the image. "
    "Reply with only the integer.\n\n{content}"
)
```

When `content` is empty, `{content}` resolves to an empty string, so the prompt ends with a blank line, which is harmless.

## Key points

- `attachments` accepts a list of image URLs. The framework forwards them to the LLM as multimodal message parts.
- Only models with vision support can process image attachments. Both `gpt-4o-mini` and `claude-haiku-4-5-20251001` support images.
- Labels are plain strings: `"6"`, not `6`. The comparison is exact string match after whitespace normalization.
- To swap in a different question (e.g. molecule name identification), change the `prompt` and the `label` values.

## What's next

- Swap `content` for a short textual description if you want to combine text and image context.
- Use `response_format` with a Pydantic schema to extract structured answers (e.g. molecular formula + carbon count together). See [Evaluation Results](../evaluation-results).
