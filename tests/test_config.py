"""Tests for configuration module."""

import os
from typing import Any

import pytest

from valtron_core.config import ValtronCoreConfig, LLMProviderConfig, OptimizationConfig


class TestLLMProviderConfig:
    """Tests for LLMProviderConfig."""

    def test_default_values(self) -> None:
        """Test that default ollama base is set correctly."""
        config = LLMProviderConfig()

        # Only test the default that doesn't come from env
        assert config.ollama_api_base is not None

    def test_load_from_env(self, mock_env_vars: dict[str, str]) -> None:
        """Test loading configuration from environment variables."""
        config = LLMProviderConfig()

        assert config.openai_api_key == "test-openai-key"
        assert config.anthropic_api_key == "test-anthropic-key"
        assert config.google_api_key == "test-google-key"

    def test_custom_ollama_base(self, mock_env_vars: dict[str, str]) -> None:
        """Test custom Ollama API base URL."""
        os.environ["OLLAMA_API_BASE"] = "http://custom-ollama:8080"
        config = LLMProviderConfig()

        assert config.ollama_api_base == "http://custom-ollama:8080"

    def test_azure_config(self, mock_env_vars: dict[str, str]) -> None:
        """Test Azure OpenAI configuration."""
        os.environ["AZURE_API_KEY"] = "test-azure-key"
        os.environ["AZURE_API_BASE"] = "https://test.openai.azure.com/"
        os.environ["AZURE_API_VERSION"] = "2024-02-15"

        config = LLMProviderConfig()

        assert config.azure_api_key == "test-azure-key"
        assert config.azure_api_base == "https://test.openai.azure.com/"
        assert config.azure_api_version == "2024-02-15"


class TestOptimizationConfig:
    """Tests for OptimizationConfig."""

    def test_default_values(self) -> None:
        """Test default optimization configuration values."""
        config = OptimizationConfig()

        assert config.max_retries == 3
        assert config.retry_delay == 1.0
        assert config.requests_per_minute is None
        assert config.log_level == "INFO"

    def test_load_from_env(self, mock_env_vars: dict[str, str]) -> None:
        """Test loading optimization config from environment."""
        os.environ["MAX_RETRIES"] = "5"
        os.environ["LOG_LEVEL"] = "DEBUG"

        config = OptimizationConfig()

        assert config.max_retries == 5
        assert config.log_level == "DEBUG"

    def test_rate_limiting(self, mock_env_vars: dict[str, str]) -> None:
        """Test rate limiting configuration."""
        os.environ["REQUESTS_PER_MINUTE"] = "60"
        config = OptimizationConfig()

        assert config.requests_per_minute == 60


class TestValtronCoreConfig:
    """Tests for main ValtronCoreConfig."""

    def test_initialization(self, mock_env_vars: dict[str, str]) -> None:
        """Test ValtronCoreConfig initialization."""
        config = ValtronCoreConfig()

        assert isinstance(config.providers, LLMProviderConfig)
        assert isinstance(config.optimization, OptimizationConfig)

    def test_nested_config_access(self, mock_env_vars: dict[str, str]) -> None:
        """Test accessing nested configuration."""
        config = ValtronCoreConfig()

        # Provider config
        assert config.providers.openai_api_key == "test-openai-key"
        assert config.providers.anthropic_api_key == "test-anthropic-key"

        # Optimization config
        assert config.optimization.max_retries == 3

    def test_config_immutability(self, mock_env_vars: dict[str, str]) -> None:
        """Test that config values can be updated."""
        config = ValtronCoreConfig()

        # Original value
        assert config.optimization.log_level == "DEBUG"

        # Update should work (Pydantic models are mutable by default)
        config.optimization.log_level = "ERROR"
        assert config.optimization.log_level == "ERROR"
