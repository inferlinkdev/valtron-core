"""Configuration management for evaltron_core."""

from typing import Any

from dotenv import find_dotenv, load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProviderConfig(BaseSettings):
    """Configuration for LLM providers."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI
    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    openai_api_base: str | None = Field(default=None, validation_alias="OPENAI_API_BASE")

    # Anthropic
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")

    # Google (Gemini)
    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")

    # Cohere
    cohere_api_key: str | None = Field(default=None, validation_alias="COHERE_API_KEY")

    # Azure OpenAI
    azure_api_key: str | None = Field(default=None, validation_alias="AZURE_API_KEY")
    azure_api_base: str | None = Field(default=None, validation_alias="AZURE_API_BASE")
    azure_api_version: str | None = Field(default=None, validation_alias="AZURE_API_VERSION")

    # Hugging Face
    huggingface_api_key: str | None = Field(
        default=None, validation_alias="HUGGINGFACE_API_KEY"
    )

    # Replicate
    replicate_api_key: str | None = Field(default=None, validation_alias="REPLICATE_API_KEY")

    # Together AI
    together_api_key: str | None = Field(default=None, validation_alias="TOGETHER_API_KEY")

    # Ollama (local)
    ollama_api_base: str = Field(
        default="http://localhost:11434", validation_alias="OLLAMA_API_BASE"
    )

    # AWS SageMaker
    aws_access_key_id: str | None = Field(default=None, validation_alias="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: str | None = Field(
        default=None, validation_alias="AWS_SECRET_ACCESS_KEY"
    )
    aws_region_name: str | None = Field(default=None, validation_alias="AWS_REGION_NAME")


class OptimizationConfig(BaseSettings):
    """Configuration for optimization strategies."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Retry
    max_retries: int = Field(default=3, validation_alias="MAX_RETRIES")
    retry_delay: float = Field(default=1.0, validation_alias="RETRY_DELAY")

    # Rate limiting (None = no limit)
    requests_per_minute: int | None = Field(
        default=None, validation_alias="REQUESTS_PER_MINUTE"
    )

    # Logging
    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")

    @field_validator("requests_per_minute", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        """Convert empty strings to None for optional numeric fields."""
        if v == "" or v is None:
            return None
        return v


class ValtronCoreConfig(BaseSettings):
    """Main configuration for evaltron_core."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def __init__(self, **data: Any) -> None:
        super().__init__(**data)
        self.providers = LLMProviderConfig()
        self.optimization = OptimizationConfig()

    providers: LLMProviderConfig = Field(default_factory=LLMProviderConfig)
    optimization: OptimizationConfig = Field(default_factory=OptimizationConfig)


# Global config instance
load_dotenv(find_dotenv(usecwd=True))
config = ValtronCoreConfig()
