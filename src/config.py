"""Validated environment configuration; deployment choices have no code defaults."""

import hashlib
import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", hide_input_in_errors=True)

    database_url: SecretStr
    redis_url: SecretStr
    storage_root: Path = Path("data/uploads")
    results_root: Path = Path("data/evaluations")
    llm_provider: str
    llm_model: str
    llm_base_url: str
    llm_api_key: SecretStr = SecretStr("")
    llm_same_provider_fallback_model: str = ""
    llm_fallback_provider: str = ""
    llm_fallback_model: str = ""
    llm_fallback_base_url: str = ""
    llm_fallback_api_key: SecretStr = SecretStr("")
    embedding_provider: str
    embedding_model: str
    embedding_dimension: int = Field(default=384, ge=1, le=2000)
    embedding_base_url: str = ""
    embedding_api_key: SecretStr = SecretStr("")
    reranker_provider: str
    reranker_model: str
    reranker_passage_tokens: int = Field(default=180, ge=64, le=256)
    reranker_passage_overlap: int = Field(default=40, ge=0, le=128)
    reranker_max_passages: int = Field(default=16, ge=1, le=32)
    tokenizer_encoding: str
    model_device: str = "cpu"
    observability_enabled: bool = False
    langfuse_host: str = ""
    langfuse_public_key: SecretStr = SecretStr("")
    langfuse_secret_key: SecretStr = SecretStr("")
    llm_timeout_seconds: float = Field(default=30, gt=0, le=120)
    llm_max_attempts: int = Field(default=3, ge=1, le=3)
    circuit_failure_threshold: int = Field(default=3, ge=1)
    circuit_cooldown_seconds: int = Field(default=60, ge=1)
    max_upload_mb: int = Field(default=20, ge=1, le=100)
    max_pages: int = Field(default=100, ge=1, le=1000)
    max_extracted_characters: int = Field(default=2_000_000, ge=1000)
    chunk_tokens: int = Field(default=600, ge=450, le=700)
    chunk_overlap: int = Field(default=100, ge=80, le=120)
    vector_top_k: int = Field(default=20, ge=1, le=20)
    keyword_top_k: int = Field(default=20, ge=1, le=20)
    rerank_top_k: int = Field(default=10, ge=1, le=10)
    context_top_k: int = Field(default=5, ge=4, le=6)
    min_rerank_score: float = Field(default=0.55, ge=0, le=1)
    max_context_tokens: int = Field(default=3500, ge=700, le=4200)
    max_output_tokens: int = Field(default=1200, ge=128, le=4096)
    chat_history_messages: int = Field(default=6, ge=0, le=10)
    answer_cache_ttl: int = Field(default=3600, ge=1)
    embedding_cache_ttl: int = Field(default=604800, ge=1)
    chat_rate_limit: int = Field(default=20, ge=1)
    upload_rate_limit: int = Field(default=10, ge=1)
    rate_window_seconds: int = Field(default=60, ge=1)
    eval_max_questions: int = Field(default=15, ge=1, le=200)
    eval_dataset_path: Path = Path("evals/dataset.json")
    eval_request_delay_seconds: float = Field(default=1.0, ge=0.0, le=10.0)
    eval_enable_ragas: bool = False
    eval_llm_model: str = ""
    eval_llm_base_url: str = ""
    eval_llm_api_key: SecretStr = SecretStr("")

    @model_validator(mode="after")
    def validate_providers(self) -> "Settings":
        if self.reranker_passage_overlap >= self.reranker_passage_tokens:
            raise ValueError("Reranker passage overlap must be smaller than the passage size")
        if self.llm_provider not in {"groq", "gemini", "ollama"}:
            raise ValueError("Unsupported LLM provider")
        if self.embedding_provider not in {"local", "gemini"}:
            raise ValueError("Unsupported embedding provider")
        if self.reranker_provider != "local":
            raise ValueError("Unsupported reranker provider")
        if self.llm_fallback_provider and (
            self.llm_fallback_provider not in {"groq", "gemini", "ollama"}
            or not self.llm_fallback_model or not self.llm_fallback_base_url
        ):
            raise ValueError("Fallback provider requires model and endpoint")
        if self.observability_enabled and not (
            self.langfuse_host and self.langfuse_public_key.get_secret_value()
            and self.langfuse_secret_key.get_secret_value()
        ):
            raise ValueError("Observability credentials and host are required")
        return self

    def fingerprint(self, *, embedding: bool = False) -> str:
        keys = ("embedding_provider", "embedding_model", "embedding_dimension",
                "embedding_base_url") if embedding else tuple(type(self).model_fields)
        values = {key: str(getattr(self, key)) for key in keys
                  if not isinstance(getattr(self, key), SecretStr)}
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


@lru_cache
def get_settings() -> Settings:
    return Settings()
