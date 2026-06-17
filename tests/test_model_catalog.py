# Copyright 2026 InferLink
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ModelCatalog.

Following the existing test convention (see test_client.py / test_config_wizard.py),
LiteLLM is mocked rather than hit live: ``litellm.model_cost`` is patched at the
module boundary with a small, deterministic fake pricing table.
"""

from unittest.mock import patch

import pytest

from valtron_core import model_catalog
from valtron_core.model_catalog import ModelCatalog, ModelCost


@pytest.fixture
def fake_model_cost() -> dict[str, object]:
    """A small, deterministic stand-in for ``litellm.model_cost``.

    Five chat models priced 1e-6 .. 5e-6 per token (input == output rate), plus
    entries that must be excluded from the token-cost catalog.
    """
    return {
        "model-1": {
            "input_cost_per_token": 1e-6,
            "output_cost_per_token": 1e-6,
            "litellm_provider": "openai",
            "mode": "chat",
        },
        "model-2": {
            "input_cost_per_token": 2e-6,
            "output_cost_per_token": 2e-6,
            "litellm_provider": "openai",
            "mode": "chat",
        },
        "model-3": {
            "input_cost_per_token": 3e-6,
            "output_cost_per_token": 3e-6,
            "litellm_provider": "anthropic",
            "mode": "chat",
        },
        "model-4": {
            "input_cost_per_token": 4e-6,
            "output_cost_per_token": 4e-6,
            "litellm_provider": "anthropic",
            "mode": "chat",
        },
        "model-5": {
            "input_cost_per_token": 5e-6,
            "output_cost_per_token": 5e-6,
            "litellm_provider": "vertex_ai",
            "mode": "chat",
        },
        # Excluded by mode: a legacy text-completion model, despite having both rates.
        "legacy-completion": {
            "input_cost_per_token": 5e-7,
            "output_cost_per_token": 5e-7,
            "mode": "completion",
        },
        # Excluded by mode: a rerank model, despite having both rates.
        "reranker": {
            "input_cost_per_token": 1e-9,
            "output_cost_per_token": 1e-9,
            "mode": "rerank",
        },
        # Excluded: an embedding model exposes only an input rate.
        "embed-only": {"input_cost_per_token": 1e-7, "mode": "embedding"},
        # Excluded: an image model exposes neither per-token rate.
        "image-gen": {"litellm_provider": "openai", "mode": "image_generation"},
        # Excluded by name: the schema spec entry that ships in litellm.model_cost.
        "sample_spec": {
            "input_cost_per_token": 0.0,
            "output_cost_per_token": 0.0,
            "mode": "chat",
        },
        # Excluded defensively: non-dict entries are ignored.
        "bogus": "not-a-dict",
    }


@pytest.fixture(autouse=True)
def _patch_model_cost(fake_model_cost: dict[str, object]) -> object:
    """Patch litellm's pricing table and isolate the cached catalog per test."""
    ModelCatalog._catalog = None
    with patch.object(model_catalog.litellm, "model_cost", fake_model_cost):
        yield
    ModelCatalog._catalog = None


class TestModelCost:
    """The per-model cost value object."""

    def test_expected_cost_uses_both_rates(self) -> None:
        mc = ModelCost(name="m", input_cost_per_token=1e-6, output_cost_per_token=2e-6)
        assert mc.expected_cost(1000, 500) == pytest.approx(1000 * 1e-6 + 500 * 2e-6)

    def test_expected_cost_zero_tokens(self) -> None:
        mc = ModelCost(name="m", input_cost_per_token=1e-6, output_cost_per_token=2e-6)
        assert mc.expected_cost(0, 0) == 0.0

    def test_filter_target_combines_provider_and_name(self) -> None:
        mc = ModelCost(
            name="gpt-5.5", input_cost_per_token=5e-6, output_cost_per_token=3e-5, provider="openai"
        )
        assert mc.filter_target == "openai/gpt-5.5"

    def test_filter_target_handles_missing_provider(self) -> None:
        mc = ModelCost(name="gpt-5.5", input_cost_per_token=5e-6, output_cost_per_token=3e-5)
        assert mc.filter_target == "/gpt-5.5"


class TestCatalogBuild:
    """Distilling litellm.model_cost into the catalog."""

    def test_available_models_keeps_only_chat_with_rates_and_sorts(self) -> None:
        # Only chat-completion models with both rates survive: the legacy
        # completion, rerank, embedding, image and spec entries are all dropped.
        assert ModelCatalog.available_models() == [
            "model-1",
            "model-2",
            "model-3",
            "model-4",
            "model-5",
        ]

    def test_non_chat_modes_are_excluded(self) -> None:
        # legacy-completion and reranker have both per-token rates but aren't chat.
        for name in ("legacy-completion", "reranker", "embed-only"):
            with pytest.raises(ValueError, match="token-cost catalog"):
                ModelCatalog.get_model_cost(name)

    def test_cost_model_captures_provider_and_mode(self) -> None:
        mc = ModelCatalog.get_model_cost("model-3")
        assert mc.input_cost_per_token == pytest.approx(3e-6)
        assert mc.output_cost_per_token == pytest.approx(3e-6)
        assert mc.provider == "anthropic"
        assert mc.mode == "chat"

    def test_get_model_cost_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="not in the LiteLLM token-cost catalog"):
            ModelCatalog.get_model_cost("does-not-exist")

    def test_refresh_picks_up_new_pricing(self) -> None:
        assert "model-9" not in ModelCatalog.available_models()
        new_table = {
            "model-9": {
                "input_cost_per_token": 1e-6,
                "output_cost_per_token": 1e-6,
                "mode": "chat",
            }
        }
        with patch.object(model_catalog.litellm, "model_cost", new_table):
            ModelCatalog.refresh()
            assert ModelCatalog.available_models() == ["model-9"]


class TestGetExpectedCost:
    """ModelCatalog.get_expected_cost."""

    def test_combines_input_and_output(self) -> None:
        # model-2 is priced at 2e-6 for both input and output tokens.
        cost = ModelCatalog.get_expected_cost("model-2", 1000, 500)
        assert cost == pytest.approx(1000 * 2e-6 + 500 * 2e-6)

    def test_input_only(self) -> None:
        assert ModelCatalog.get_expected_cost("model-1", 1000, 0) == pytest.approx(1000 * 1e-6)

    def test_output_only(self) -> None:
        assert ModelCatalog.get_expected_cost("model-5", 0, 200) == pytest.approx(200 * 5e-6)

    def test_zero_tokens_is_free(self) -> None:
        assert ModelCatalog.get_expected_cost("model-3", 0, 0) == 0.0

    def test_unknown_model_raises(self) -> None:
        with pytest.raises(ValueError, match="token-cost catalog"):
            ModelCatalog.get_expected_cost("ghost-model", 100, 100)

    def test_excluded_model_raises(self) -> None:
        # embed-only is dropped during catalog build, so it is unknown here.
        with pytest.raises(ValueError, match="token-cost catalog"):
            ModelCatalog.get_expected_cost("embed-only", 100, 100)

    @pytest.mark.parametrize("num_in,num_out", [(-1, 0), (0, -5), (-3, -3)])
    def test_negative_tokens_raise(self, num_in: int, num_out: int) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ModelCatalog.get_expected_cost("model-1", num_in, num_out)


class TestGetSpread:
    """ModelCatalog.get_spread.

    With ``num_output_tokens=0`` and 1e6 input tokens the five models cost
    exactly 1.0 .. 5.0 dollars, so a uniform linear cost spread lands on them
    one-for-one — convenient for asserting exact sequences.
    """

    def test_full_spread_hits_every_model(self) -> None:
        spread = ModelCatalog.get_spread("model-5", 5, 1_000_000, 0)
        assert spread == ["model-1", "model-2", "model-3", "model-4", "model-5"]

    def test_coarse_spread_samples_evenly(self) -> None:
        # Targets linspace(1, 5, 3) = [1, 3, 5] -> cheapest, mid, current.
        spread = ModelCatalog.get_spread("model-5", 3, 1_000_000, 0)
        assert spread == ["model-1", "model-3", "model-5"]

    def test_excludes_models_above_current(self) -> None:
        # Anchored at model-3, models 4 and 5 are out of range.
        spread = ModelCatalog.get_spread("model-3", 3, 1_000_000, 0)
        assert spread == ["model-1", "model-2", "model-3"]

    def test_single_model_returns_current(self) -> None:
        assert ModelCatalog.get_spread("model-4", 1, 1_000_000, 0) == ["model-4"]

    def test_endpoints_are_cheapest_and_current(self) -> None:
        spread = ModelCatalog.get_spread("model-4", 3, 1_000_000, 0)
        assert spread[0] == "model-1"
        assert spread[-1] == "model-4"

    @pytest.mark.parametrize("num_models", [1, 2, 3, 4, 5])
    def test_returns_exactly_num_models_when_catalog_is_rich_enough(self, num_models: int) -> None:
        # Five evenly priced models, so requests up to 5 are filled exactly.
        spread = ModelCatalog.get_spread("model-5", num_models, 1000, 1000)
        assert len(spread) == num_models
        assert len(set(spread)) == num_models  # distinct

    def test_returns_fewer_than_requested_when_catalog_too_sparse(self) -> None:
        # Only 5 models exist in range, so a request for more returns just those 5.
        spread = ModelCatalog.get_spread("model-5", 10, 1_000_000, 0)
        assert spread == ["model-1", "model-2", "model-3", "model-4", "model-5"]

    def test_current_is_cheapest_collapses_to_single_model(self) -> None:
        # model-1 is the global cheapest, so the spread collapses onto it.
        assert ModelCatalog.get_spread("model-1", 3, 1_000_000, 0) == ["model-1"]

    def test_spread_is_workload_dependent(self) -> None:
        # Equal rates here means ordering is stable, but the call must still
        # price every model with the supplied token mix without error.
        spread = ModelCatalog.get_spread("model-5", 5, 10, 10)
        assert spread == ["model-1", "model-2", "model-3", "model-4", "model-5"]

    def test_invalid_num_models_raises(self) -> None:
        with pytest.raises(ValueError, match="num_models must be >= 1"):
            ModelCatalog.get_spread("model-5", 0, 100, 100)

    def test_unknown_current_model_raises(self) -> None:
        with pytest.raises(ValueError, match="token-cost catalog"):
            ModelCatalog.get_spread("ghost-model", 3, 100, 100)

    def test_negative_tokens_raise(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ModelCatalog.get_spread("model-5", 3, -1, 100)


class TestGetSpreadFilters:
    """include_filter / exclude_filter regex narrowing of the candidate pool."""

    def test_include_filter_narrows_pool(self) -> None:
        # Only models 1, 2, 5 are eligible; 3 and 4 are filtered out even though
        # they fall within the cost range.
        spread = ModelCatalog.get_spread("model-5", 3, 1_000_000, 0, include_filter="model-[125]")
        assert spread == ["model-1", "model-2", "model-5"]

    def test_exclude_filter_drops_matches(self) -> None:
        spread = ModelCatalog.get_spread("model-5", 5, 1_000_000, 0, exclude_filter="model-3")
        assert spread == ["model-1", "model-2", "model-4", "model-5"]

    def test_include_and_exclude_combine(self) -> None:
        # include keeps 1-4; exclude drops 2 -> eligible pool is {1, 3, 4}.
        spread = ModelCatalog.get_spread(
            "model-4", 3, 1_000_000, 0, include_filter="model-[1-4]", exclude_filter="model-2"
        )
        assert spread == ["model-1", "model-3", "model-4"]

    def test_order_of_filters_is_irrelevant(self) -> None:
        # Same predicate however you read it: matches include AND not exclude.
        a = ModelCatalog.get_spread(
            "model-5", 4, 1_000_000, 0, include_filter="model-", exclude_filter="model-[34]"
        )
        assert a == ["model-1", "model-2", "model-5"]

    def test_filter_collapsing_to_only_current(self) -> None:
        spread = ModelCatalog.get_spread("model-5", 3, 1_000_000, 0, include_filter="model-5")
        assert spread == ["model-5"]

    def test_current_excluded_by_exclude_filter_raises(self) -> None:
        with pytest.raises(ValueError, match="does not pass the include/exclude filters"):
            ModelCatalog.get_spread("model-5", 3, 1_000_000, 0, exclude_filter="model-5")

    def test_current_not_matching_include_filter_raises(self) -> None:
        with pytest.raises(ValueError, match="does not pass the include/exclude filters"):
            ModelCatalog.get_spread("model-5", 3, 1_000_000, 0, include_filter="model-[12]")

    def test_include_filter_matches_provider(self) -> None:
        # "anthropic" appears only in the provider field (names are "model-N"),
        # so this proves the filter sees "{provider}/{name}", not just the name.
        spread = ModelCatalog.get_spread("model-4", 2, 1_000_000, 0, include_filter="anthropic")
        assert spread == ["model-3", "model-4"]

    def test_exclude_filter_matches_provider(self) -> None:
        spread = ModelCatalog.get_spread("model-4", 5, 1_000_000, 0, exclude_filter="openai")
        assert spread == ["model-3", "model-4"]

    def test_provider_prefix_anchor_selects_first_party(self) -> None:
        # "^openai/" anchors on the provider, keeping only provider == "openai".
        spread = ModelCatalog.get_spread("model-2", 2, 1_000_000, 0, include_filter=r"^openai/")
        assert spread == ["model-1", "model-2"]

    def test_current_rejected_by_provider_filter_raises(self) -> None:
        # model-3 is an anthropic model, so an openai-only filter excludes the anchor.
        with pytest.raises(ValueError, match="does not pass the include/exclude filters"):
            ModelCatalog.get_spread("model-3", 3, 1_000_000, 0, include_filter=r"^openai/")

    def test_invalid_include_regex_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid include_filter regex"):
            ModelCatalog.get_spread("model-5", 3, 1_000_000, 0, include_filter="model-[")

    def test_invalid_exclude_regex_raises(self) -> None:
        with pytest.raises(ValueError, match="Invalid exclude_filter regex"):
            ModelCatalog.get_spread("model-5", 3, 1_000_000, 0, exclude_filter="(")
