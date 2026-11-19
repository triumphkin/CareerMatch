"""Centralized environment-backed model, database, and budget settings."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Load process environment overrides and optional local .env settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    DATABASE_URL: str = Field(
        default="postgresql+psycopg://careermatch:careermatch@localhost:5432/careermatch",
        min_length=1,
        repr=False,
    )
    GEMINI_API_KEY: str = Field(default="", repr=False)
    # CHAT_MODEL_ID: str = Field(default="gemini-3.8-flash", min_length=1)
    CHAT_MODEL_ID: str = Field(default="gemini-3.5-flash-lite", min_length=1)
    CHAT_MODEL_ID_FAST: str = Field(default="gemini-3.5-flash-lite", min_length=1)
    EMBEDDING_MODEL_ID: str = Field(default="gemini-embedding-2", min_length=1)
    EMBEDDING_DIM: int = Field(default=768, gt=0)
    LLM_RPM_LIMIT: int = Field(default=15, gt=0)
    LLM_CACHE_PATH: str = Field(default=".cache/llm_cache.sqlite", min_length=1)
    MAX_DISCOVERY_SEARCHES: int = Field(default=2, ge=0)
    MAX_CANDIDATE_POOL: int = Field(default=10, gt=0)
    MAX_SHORTLIST: int = Field(default=5, ge=0)
    MAX_VERIFICATION_CORRECTIONS: int = Field(default=1, ge=0)
    MAX_MODEL_CALLS_PER_RUN: int = Field(default=8, gt=0)
    EVAL_MAX_COST_USD: float = Field(default=2.00, ge=0, allow_inf_nan=False)


settings = Settings()
