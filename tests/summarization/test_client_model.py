"""The one adapter binding the ported summarization logic to ``LLMClient``.

Everything else in the summarization package is exercised against fake models.
This is the seam where it meets the real client, so these tests mock at the
``LLMClient`` boundary -- the same place the rest of this suite mocks -- and
check the three things the adapter exists to add: structured-output validation,
per-call usage attribution, and a fixed temperature.
"""

from typing import Any
from unittest.mock import AsyncMock, Mock

import pytest
from litellm.types.utils import Choices
from litellm.utils import ModelResponse  # type: ignore[attr-defined]
from pydantic import BaseModel

from valtron_core.client import LLMClient
from valtron_core.summarization.client_model import ClientModel
from valtron_core.summarization.model import Usage
from valtron_core.summarization.prompts import Prompt


class _Schema(BaseModel):
    facts: list[str]


class _Fixed(Prompt):
    """A prompt that renders to a known string."""

    def __init__(self, text: str = "render me") -> None:
        self._text = text

    def __str__(self) -> str:
        return self._text


def _response(
    content: str | None,
    *,
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
    cost: float | None = 0.002,
    cache_hit: bool = False,
) -> Any:
    response = Mock(spec=ModelResponse)
    choice = Mock(spec=Choices)
    choice.message = Mock()
    choice.message.content = content
    response.choices = [choice]
    response.usage = Mock(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
    response._hidden_params = {"cache_hit": cache_hit, "response_cost": cost}
    return response


def _client(response: Any) -> Mock:
    client = Mock(spec=LLMClient)
    client.complete = AsyncMock(return_value=response)
    return client


class TestRun:
    """Plain text generation."""

    async def test_returns_the_message_content(self) -> None:
        model = ClientModel("gpt-4o-mini", client=_client(_response("a summary")))
        assert await model.run(_Fixed()) == "a summary"

    async def test_sends_the_rendered_prompt_as_one_user_message(self) -> None:
        client = _client(_response("ok"))
        await ClientModel("gpt-4o-mini", client=client).run(_Fixed("render me"))
        kwargs = client.complete.await_args.kwargs
        assert kwargs["messages"] == [{"role": "user", "content": "render me"}]

    async def test_pins_temperature_to_zero_by_default(self) -> None:
        # LLMClient.complete defaults to 0.7; an evaluation wants a re-run to be
        # comparable, which is the whole reason this default is overridden.
        client = _client(_response("ok"))
        await ClientModel("gpt-4o-mini", client=client).run(_Fixed())
        assert client.complete.await_args.kwargs["temperature"] == 0.0

    async def test_a_response_with_no_content_is_an_error(self) -> None:
        model = ClientModel("gpt-4o-mini", client=_client(_response(None)))
        with pytest.raises(ValueError, match="returned no content"):
            await model.run(_Fixed())


class TestRunStructured:
    """The judge's verdicts, which must come back as validated objects."""

    async def test_validates_the_response_into_the_schema(self) -> None:
        client = _client(_response('{"facts": ["one", "two"]}'))
        result = await ClientModel("gpt-4o-mini", client=client).run_structured(_Fixed(), _Schema)
        assert result.facts == ["one", "two"]

    async def test_passes_the_schema_through_as_the_response_format(self) -> None:
        client = _client(_response('{"facts": []}'))
        await ClientModel("gpt-4o-mini", client=client).run_structured(_Fixed(), _Schema)
        assert client.complete.await_args.kwargs["response_format"] is _Schema

    async def test_names_the_model_when_the_response_is_not_valid_json(self) -> None:
        # The likeliest operational failure: the model does not really support
        # structured output and litellm's drop_params quietly removed it. A bare
        # ValidationError does not say which model misbehaved.
        client = _client(_response("Certainly! Here are the facts: ..."))
        model = ClientModel("gpt-4o-mini", client=client, name="judge")
        with pytest.raises(ValueError, match="'judge' did not return valid _Schema JSON"):
            await model.run_structured(_Fixed(), _Schema)


class TestUsageAccounting:
    """Per-call attribution, which LLMClient's process-wide totals cannot give."""

    async def test_records_tokens_and_cost_into_the_call_scoped_accumulator(self) -> None:
        usage = Usage()
        client = _client(_response("ok", prompt_tokens=11, completion_tokens=7, cost=0.002))
        await ClientModel("gpt-4o-mini", client=client).run(_Fixed(), usage=usage)
        assert usage.calls == 1
        assert usage.prompt_tokens == 11
        assert usage.completion_tokens == 7
        assert usage.total_tokens == 18
        assert usage.cost_usd == pytest.approx(0.002)

    async def test_records_into_the_models_own_accumulator_too(self) -> None:
        # Both at once: the run-wide total and the per-candidate slice are the
        # same calls counted twice, not two different sets of calls.
        run_wide = Usage()
        scoped = Usage()
        client = _client(_response("ok"))
        model = ClientModel("gpt-4o-mini", client=client, usage=run_wide)
        await model.run(_Fixed(), usage=scoped)
        assert run_wide.calls == 1
        assert scoped.calls == 1
        assert run_wide.by_model == {"gpt-4o-mini": 1}

    async def test_a_cache_hit_is_free(self) -> None:
        usage = Usage()
        client = _client(_response("ok", cost=0.002, cache_hit=True))
        await ClientModel("gpt-4o-mini", client=client).run(_Fixed(), usage=usage)
        assert usage.cache_hits == 1
        assert usage.cost_usd == 0.0

    async def test_a_response_without_a_recorded_price_costs_nothing_rather_than_failing(
        self,
    ) -> None:
        # completion_cost cannot price a mock; the adapter must degrade to zero
        # rather than take the whole evaluation down over an accounting detail.
        usage = Usage()
        client = _client(_response("ok", cost=None))
        await ClientModel("gpt-4o-mini", client=client).run(_Fixed(), usage=usage)
        assert usage.calls == 1
        assert usage.cost_usd == 0.0


class TestModelArgument:
    """Both shapes LLMClient.complete accepts."""

    async def test_a_dict_of_litellm_params_is_passed_through(self) -> None:
        params = {"model": "gpt-4o-mini", "temperature": 0.3, "cost_rate": 2.5}
        client = _client(_response("ok"))
        await ClientModel(params, client=client).run(_Fixed())
        assert client.complete.await_args.kwargs["model"] == params

    async def test_the_name_comes_from_the_dicts_model_key(self) -> None:
        model = ClientModel({"model": "gpt-4o-mini"}, client=_client(_response("ok")))
        assert model.name == "gpt-4o-mini"

    async def test_an_explicit_name_wins(self) -> None:
        # The judge is registered under its own label so its spend is reported
        # apart from the candidates being ranked.
        model = ClientModel("gemini/gemini-2.5-pro", client=_client(_response("ok")), name="judge")
        assert model.name == "judge"
