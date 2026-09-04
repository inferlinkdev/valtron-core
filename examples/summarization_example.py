#!/usr/bin/env python3
# Copyright 2026 InferLink
# SPDX-License-Identifier: Apache-2.0
"""Reference-free summarization demo.

Ranks gpt-4.1 against gpt-4.1-mini on three public-domain Congressional bills
(see examples/summarization/README.md), using an LLM judge that decomposes
each bill into must-convey facts and grades both candidates against them.

Makes real, billed calls to OpenAI for both candidates and the judge.
Requires OPENAI_API_KEY (see .env).

Run:
    python examples/summarization_example.py
"""

from pathlib import Path

from valtron_core.evaluation import SummarizationExperiment

DATA_DIR = Path(__file__).resolve().parent / "summarization"
DOCUMENT_IDS = ["0001", "0003", "0006"]

DATA = [
    {"id": doc_id, "content": (DATA_DIR / f"{doc_id}.txt").read_text()} for doc_id in DOCUMENT_IDS
]

CONFIG = {
    "models": [{"name": "gpt-4.1"}, {"name": "gpt-4.1-mini"}],
    "judge_model": "gpt-5.4-mini",
}

if __name__ == "__main__":
    output_dir = Path.cwd() / "results" / "summarization"

    experiment = SummarizationExperiment(config=CONFIG, data=DATA)
    report_path = experiment.run(output_dir=output_dir)

    print(f"\nReport: {report_path}\n")
    print("Best model(s):", experiment.ranking.best)
    for candidate in experiment.ranking.scores:
        axes = ", ".join(
            f"{name}={value:.0%}" if value is not None else f"{name}=n/a"
            for name, value in candidate.axes().items()
        )
        print(f"  {candidate.model:<20}  score={candidate.score:.0%}  {axes}")
