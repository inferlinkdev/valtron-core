#!/usr/bin/env python3
# Copyright 2026 InferLink
# SPDX-License-Identifier: Apache-2.0
"""Cost-based model spread.

Print a uniform linear sweep, by expected dollar cost, from the cheapest
eligible model up to gpt-5.5 — for a representative 1k-in / 500-out token
workload. Uses live LiteLLM pricing data (no mocking).

The catalog is restricted to chat-completion models (LiteLLM ``mode == "chat"``),
so embedding, rerank, image, audio and legacy text-completion models never
appear in the spread. We narrow further with an include filter so the spread
only uses first-party OpenAI or Anthropic models. The filter is matched against
each model's ``"{provider}/{name}"`` string, so anchoring on the provider prefix
keeps Azure-/Bedrock-/etc.-hosted copies out.

Run:
    python examples/cost_model_example.py
"""

from valtron_core.model_catalog import ModelCatalog

ANCHOR_MODEL = "gpt-5.5"
NUM_MODELS = 10
INPUT_TOKENS = 1_000
OUTPUT_TOKENS = 500
# Only first-party OpenAI or Anthropic models: match the provider at the start
# of the "<provider>/<model>" filter target.
INCLUDE_FILTER = r"^(openai|anthropic)/"


if __name__ == "__main__":
    spread = ModelCatalog.get_spread(
        ANCHOR_MODEL,
        NUM_MODELS,
        INPUT_TOKENS,
        OUTPUT_TOKENS,
        include_filter=INCLUDE_FILTER,
    )

    print(
        f"\nCost spread up to {ANCHOR_MODEL}  "
        f"[{INPUT_TOKENS} in / {OUTPUT_TOKENS} out tokens, provider/model matching "
        f"/{INCLUDE_FILTER}/]:\n"
    )
    for rank, name in enumerate(spread, start=1):
        # Show the "<provider>/<model>" filter target so the vendor is explicit —
        # litellm model names alone (e.g. "chatgpt-4o-latest") don't reveal it.
        model = ModelCatalog.get_model_cost(name)
        cost = model.expected_cost(INPUT_TOKENS, OUTPUT_TOKENS)
        print(f"  {rank:>2}. {model.filter_target:<48}  ${cost:.6f}")

    if len(spread) < NUM_MODELS:
        print(f"\n  (asked for {NUM_MODELS}; only {len(spread)} distinct models in range)")
    print()
