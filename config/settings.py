"""
Central configuration and environment loading.

Usage:
    from config.settings import settings
    print(settings.openai_api_key)
"""

import itertools
import os
import threading
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from pydantic import Field, PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _collect_groq_keys() -> List[str]:
    """
    Collect Groq API keys from the environment in priority order:
        1. GROQ_API_KEY_1, GROQ_API_KEY_2, ... GROQ_API_KEY_N (any contiguous range)
        2. GROQ_API_KEYS=comma,separated,list
        3. GROQ_API_KEY (legacy single key)
    Blank / placeholder values are filtered out. Order is preserved and
    duplicates removed.
    """
    keys: List[str] = []

    # Numbered slots — scan a generous range so users can leave gaps.
    for i in range(1, 21):
        val = os.environ.get(f"GROQ_API_KEY_{i}", "").strip()
        if val:
            keys.append(val)

    # Comma-separated list.
    csv_val = os.environ.get("GROQ_API_KEYS", "").strip()
    if csv_val:
        for part in csv_val.split(","):
            part = part.strip()
            if part:
                keys.append(part)

    # Legacy single key.
    legacy = os.environ.get("GROQ_API_KEY", "").strip()
    if legacy:
        keys.append(legacy)

    # Drop placeholders and duplicates while preserving order.
    seen: set = set()
    cleaned: List[str] = []
    for k in keys:
        if k.startswith("gsk_your-") or k in seen:
            continue
        seen.add(k)
        cleaned.append(k)
    return cleaned


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    groq_api_key: str = Field(default="", alias="GROQ_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")

    default_provider: str = Field(default="groq", alias="DEFAULT_PROVIDER")
    default_model: str = Field(default="llama-3.1-8b-instant", alias="DEFAULT_MODEL")
    default_groq_model: str = Field(default="llama-3.1-8b-instant", alias="DEFAULT_GROQ_MODEL")
    default_openai_model: str = Field(default="gpt-4o-mini", alias="DEFAULT_OPENAI_MODEL")
    default_temperature: float = Field(default=0.3, alias="DEFAULT_TEMPERATURE")
    default_max_tokens: int = Field(default=1024, alias="DEFAULT_MAX_TOKENS")
    max_retries: int = Field(default=3, alias="MAX_RETRIES")

    results_dir: Path = Field(default=PROJECT_ROOT / "results", alias="RESULTS_DIR")
    log_dir: Path = Field(default=PROJECT_ROOT / "logs", alias="LOG_DIR")

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    # Private state for Groq round-robin (not loaded from env).
    _groq_keys: Optional[List[str]] = PrivateAttr(default=None)
    _groq_cycle: Optional[object] = PrivateAttr(default=None)
    _groq_pool_lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

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

    # ─────────────────────────────────────────────
    # Groq key rotation
    # ─────────────────────────────────────────────

    @property
    def groq_keys(self) -> List[str]:
        """All configured Groq keys, in rotation order."""
        if self._groq_keys is None:
            with self._groq_pool_lock:
                if self._groq_keys is None:
                    keys = _collect_groq_keys()
                    self._groq_keys = keys
                    self._groq_cycle = itertools.cycle(keys) if keys else None
        return list(self._groq_keys)

    def next_groq_key(self) -> str:
        """
        Return the next Groq API key in round-robin order.
        Falls back to the legacy single key if no rotation pool is configured.
        Raises RuntimeError if no Groq key is available at all.
        """
        # Trigger lazy init of the pool.
        _ = self.groq_keys
        with self._groq_pool_lock:
            if self._groq_cycle is not None:
                return next(self._groq_cycle)
        if self.groq_api_key:
            return self.groq_api_key
        raise RuntimeError(
            "No Groq API key configured. Set GROQ_API_KEY_1.. / GROQ_API_KEYS / "
            "GROQ_API_KEY in your .env."
        )

    def reset_groq_rotation(self) -> None:
        """Reload Groq keys from env (useful for tests / hot reloads)."""
        with self._groq_pool_lock:
            self._groq_keys = None
            self._groq_cycle = None

    # ─────────────────────────────────────────────
    # Client kwargs
    # ─────────────────────────────────────────────

    def resolve_model(self, provider: str, model: str = None) -> str:
        """
        Pick a sensible default model for `provider` if none was supplied.
        Honors any explicit user-provided model first.
        """
        if model:
            return model
        if provider == "groq":
            return self.default_groq_model
        if provider == "openai":
            return self.default_openai_model
        return self.default_model

    def get_openai_client_kwargs(self, model: str = None, temperature: float = None) -> dict:
        return {
            "model": self.resolve_model("openai", model),
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_tokens": self.default_max_tokens,
            "api_key": self.openai_api_key,
        }

    def get_groq_client_kwargs(self, model: str = None, temperature: float = None) -> dict:
        """
        Returns kwargs for an OpenAI-compatible client pointed at Groq.
        The api_key is drawn from the round-robin pool on every call, so
        successive PlayerAgent instantiations distribute load across keys.
        """
        return {
            "model": self.resolve_model("groq", model),
            "temperature": self.default_temperature if temperature is None else temperature,
            "max_tokens": self.default_max_tokens,
            "api_key": self.next_groq_key(),
            "base_url": "https://api.groq.com/openai/v1",
        }


settings = Settings()
settings.ensure_dirs()
