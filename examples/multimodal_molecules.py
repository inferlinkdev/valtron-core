"""Multimodal evaluation — count carbons in molecule images.

Each document has a 2D structure diagram attached as an image URL from PubChem.
The model is asked to count the number of carbon atoms from the image alone.
Uses vision-capable LLMs (gpt-4o-mini and claude-haiku support images).

Run:
    python examples/multimodal_molecules.py
"""

from pathlib import Path

from valtron_core.recipes import ModelEval

# Images are 2D structure diagrams served directly from PubChem.
# The `content` field is empty — the question is fully in the prompt.
DATA = [
    {
        "id": "methanol",
        "content": "",
        "label": "1",
        "attachments": ["https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=887&width=300&height=300"],
    },
    {
        "id": "ethanol",
        "content": "",
        "label": "2",
        "attachments": ["https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=702&width=300&height=300"],
    },
    {
        "id": "acetone",
        "content": "",
        "label": "3",
        "attachments": ["https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=180&width=300&height=300"],
    },
    {
        "id": "benzene",
        "content": "",
        "label": "6",
        "attachments": ["https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=241&width=300&height=300"],
    },
    {
        "id": "caffeine",
        "content": "",
        "label": "8",
        "attachments": ["https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=2519&width=300&height=300"],
    },
    {
        "id": "aspirin",
        "content": "",
        "label": "9",
        "attachments": ["https://pubchem.ncbi.nlm.nih.gov/image/imagefly.cgi?cid=2244&width=300&height=300"],
    },
]

CONFIG = {
    "use_case": "multimodal: count carbon atoms in molecule structure diagrams",
    "output_formats": ["html"],
    "temperature": 0.0,
    "prompt": (
        "Count the number of carbon atoms in the molecular structure shown in the image. "
        "Reply with only the integer.\n\n{content}"
    ),
    "models": [
        {"name": "gpt-4o-mini", "label": "GPT-4o mini"},
        {"name": "claude-haiku-4-5-20251001", "label": "Claude Haiku"},
    ],
}

if __name__ == "__main__":
    output_dir = Path(__file__).resolve().parent / "results" / "multimodal_molecules"

    experiment = ModelEval(config=CONFIG, data=DATA)
    report_path = experiment.run(output_dir=output_dir)

    print(f"\nReport: {report_path}\n")
    for result in experiment.results:
        print(f"  {result.model:<40}  accuracy={result.metrics.accuracy:.0%}  cost=${result.metrics.total_cost:.4f}")
