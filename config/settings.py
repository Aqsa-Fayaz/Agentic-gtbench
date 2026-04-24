"""
Central configuration and environment loading.

Usage:
    from config.settings import settings
    print(settings.openai_api_key)
"""

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    default_model: str = Field(default="gpt-4o-mini", alias="DEFAULT_MODEL")
    default_temperature: float = Field(default=0.3, alias="DEFAULT_TEMPERATURE")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")

    results_dir: Path = Field(default=PROJECT_ROOT / "results", alias="RESULTS_DIR")
    log_dir: Path = Field(default=PROJECT_ROOT / "logs", alias="LOG_DIR")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    def ensure_dirs(self) -> None:
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def get_openai_client_kwargs(self, model: str = None, temperature: float = None) -> dict:
        return {
            "model": model or self.default_model,
            "temperature": self.default_temperature if temperature is None else temperature,
            "api_key": self.openai_api_key,
        }

    def get_groq_client_kwargs(self, model: str = "llama-3-8b-8192", temperature: float = None) -> dict:
        return {
            "model": model,
            "temperature": self.default_temperature if temperature is None else temperature,
            "api_key": self.groq_api_key,
            "base_url": "https://api.groq.com/openai/v1",
        }


settings = Settings()
settings.ensure_dirs()
