"""LLM client wrapper using LiteLLM for unified access across providers."""

import asyncio
import collections
import logging
import time
from typing import Any, AsyncIterator, Literal

from pydantic import BaseModel
import structlog
import litellm
from litellm import acompletion, completion, completion_cost
from litellm.utils import ModelResponse

from valtron_core.config import config

logger = structlog.get_logger()

# Automatically drop unsupported parameters instead of raising errors
# This allows us to pass response_format to all models, and litellm will
# automatically drop it for models that don't support it (like gpt-4)
litellm.drop_params = True


class LLMClient:
    """Unified LLM client with optimization features."""

    def __init__(self) -> None:
        """Initialize the LLM client with configuration."""
        self.config = config
        self._setup_logging()
        self._call_count = 0
        self._total_cost = 0.0
        self._rate_lock = asyncio.Lock()
        self._rate_window: collections.deque[float] = collections.deque()

    async def _rate_limit(self) -> None:
        """Enforce requests_per_minute if configured; no-op when None."""
        rpm = self.config.optimization.requests_per_minute
        if rpm is None:
            return
        async with self._rate_lock:
            now = time.monotonic()
            while self._rate_window and now - self._rate_window[0] >= 60.0:
                self._rate_window.popleft()
            if len(self._rate_window) >= rpm:
                wait = 60.0 - (now - self._rate_window[0])
                if wait > 0:
                    await asyncio.sleep(wait)
                    now = time.monotonic()
                    while self._rate_window and now - self._rate_window[0] >= 60.0:
                        self._rate_window.popleft()
            self._rate_window.append(time.monotonic())

    def _setup_logging(self) -> None:
        """Configure logging based on settings."""
        # Convert string log level to logging level constant
        log_level_str = self.config.optimization.log_level.upper()
        log_level = getattr(logging, log_level_str, logging.INFO)

        structlog.configure(
            wrapper_class=structlog.make_filtering_bound_logger(log_level),
        )

    async def complete(
        self,
        model: str | dict[str, Any],
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
        response_format: "type[BaseModel] | dict | None" = None,
        **kwargs: Any,
    ) -> ModelResponse | AsyncIterator[ModelResponse]:
        """
        Generate a completion using the specified model.

        Args:
            model: Model identifier (e.g., "gpt-4", "claude-3-opus-20240229", "ollama/llama2")
                   or dict with LiteLLM parameters (e.g., {"model": "gpt-4", "api_base": "...", "api_key": "..."})
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Sampling temperature (0.0 to 2.0)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            response_format: Optional pydantic model for response parsing
            **kwargs: Additional provider-specific parameters

        Returns:
            ModelResponse object or async iterator if streaming

        Example:
            >>> client = LLMClient()
            >>> messages = [{"role": "user", "content": "Hello!"}]
            >>> response = await client.complete("gpt-3.5-turbo", messages)
            >>> print(response.choices[0].message.content)

            >>> # Using a dict with custom parameters
            >>> model_config = {"model": "gpt-4", "api_base": "https://custom.api"}
            >>> response = await client.complete(model_config, messages)
        """
        await self._rate_limit()
        self._call_count += 1

        # Extract model name for logging
        model_name = model if isinstance(model, str) else model.get("model", "unknown")

        logger.info(
            "llm_request",
            model=model_name,
            message_count=len(messages),
            temperature=temperature,
            stream=stream,
        )

        # Prepare completion arguments
        completion_args: dict[str, Any] = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            "response_format": response_format,
            **kwargs,
        }

        # If model is a dict, merge its parameters (strip evaltron_core-only keys)
        if isinstance(model, dict):
            _EVALTRON_KEYS = {"cost_rate", "cost_rate_time_unit"}
            completion_args.update({k: v for k, v in model.items() if k not in _EVALTRON_KEYS})
        else:
            completion_args["model"] = model

        # Drop params that the model doesn't support
        try:
            param_check_kwargs: dict[str, Any] = {"model": model_name}
            if isinstance(model, dict) and "custom_llm_provider" in model:
                param_check_kwargs["custom_llm_provider"] = model["custom_llm_provider"]
            supported_params = litellm.get_supported_openai_params(**param_check_kwargs) or []

            if "temperature" not in supported_params:
                logger.warning(
                    "temperature_not_supported",
                    model=model_name,
                    action="dropping_temperature",
                )
                completion_args.pop("temperature", None)

            if response_format is not None and not litellm.supports_response_schema(**param_check_kwargs):
                logger.warning(
                    "response_format_not_supported",
                    model=model_name,
                    action="dropping_response_format",
                )
                completion_args["response_format"] = None
        except Exception:
            pass  # unknown model; proceed and let litellm.drop_params handle it

        max_retries = self.config.optimization.max_retries
        retry_delay = self.config.optimization.retry_delay

        for attempt in range(max_retries + 1):
            if attempt > 0:
                wait = retry_delay * (2 ** (attempt - 1))
                logger.warning("llm_retry", model=model_name, attempt=attempt, wait=wait)
                await asyncio.sleep(wait)
            try:
                response = await acompletion(**completion_args)
                break
            except Exception as e:
                if attempt == max_retries:
                    logger.error("llm_error", model=model_name, error=str(e))
                    raise
                logger.warning("llm_attempt_failed", model=model_name, attempt=attempt, error=str(e))

        if not stream:
            try:
                cost = completion_cost(completion_response=response)
                self._total_cost += cost
                logger.info("llm_response", model=model_name, cost=cost, total_cost=self._total_cost)
            except Exception as e:
                logger.warning("cost_tracking_failed", model=model_name, error=str(e))

        return response

    def complete_sync(
        self,
        model: str,
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        stream: bool = False,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Synchronous version of complete().

        Args:
            model: Model identifier
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            **kwargs: Additional provider-specific parameters

        Returns:
            ModelResponse object
        """
        self._call_count += 1

        try:
            logger.info(
                "llm_request_sync",
                model=model,
                message_count=len(messages),
                temperature=temperature,
            )

            response = completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs,
            )

            # Track costs using LiteLLM's completion_cost function
            try:
                cost = completion_cost(completion_response=response)
                self._total_cost += cost
                logger.info("llm_response_sync", model=model, cost=cost, total_cost=self._total_cost)
            except Exception as e:
                logger.warning("cost_tracking_failed_sync", model=model, error=str(e))

            return response

        except Exception as e:
            logger.error("llm_error_sync", model=model, error=str(e))
            raise

    async def complete_with_fallback(
        self,
        models: list[str | dict[str, Any]],
        messages: list[dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """
        Try multiple models in order until one succeeds.

        Args:
            models: List of model identifiers to try in order. Each can be a string or dict with parameters
            messages: List of message dictionaries
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            ModelResponse from the first successful model

        Raises:
            Exception: If all models fail
        """
        last_error = None

        for model in models:
            try:
                model_name = model if isinstance(model, str) else model.get("model", "unknown")
                logger.info("attempting_model", model=model_name)
                response = await self.complete(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    **kwargs,
                )
                logger.info("model_success", model=model_name)
                return response
            except Exception as e:
                last_error = e
                logger.warning("model_failed", model=model_name, error=str(e))
                continue

        raise Exception(f"All models failed. Last error: {last_error}")

    def get_stats(self) -> dict[str, Any]:
        """
        Get client statistics.

        Returns:
            Dictionary with call count and total cost
        """
        return {
            "total_calls": self._call_count,
            "total_cost": self._total_cost,
        }

    def reset_stats(self) -> None:
        """Reset statistics counters."""
        self._call_count = 0
        self._total_cost = 0.0
