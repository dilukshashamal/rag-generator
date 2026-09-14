from redis import Redis

from src.config import Settings
from src.providers.base import EmbeddingProvider, LLMProvider, ObservabilityProvider
from src.providers.embeddings import GeminiEmbeddingProvider, LocalEmbeddingProvider
from src.providers.llm import GeminiLLMProvider, GroqLLMProvider, OllamaLLMProvider
from src.providers.reliability import CircuitBreaker, ReliableLLM

LLM_REGISTRY = {"groq": GroqLLMProvider, "gemini": GeminiLLMProvider, "ollama": OllamaLLMProvider}


def build_llm(settings: Settings, redis: Redis, telemetry: ObservabilityProvider) -> ReliableLLM:
    primary = (settings.llm_provider, settings.llm_model, settings.llm_base_url,
               settings.llm_api_key.get_secret_value())
    configurations = [primary]
    if settings.llm_same_provider_fallback_model:
        configurations.append((primary[0], settings.llm_same_provider_fallback_model, primary[2], primary[3]))
    if settings.llm_fallback_provider:
        configurations.append((settings.llm_fallback_provider, settings.llm_fallback_model,
                               settings.llm_fallback_base_url, settings.llm_fallback_api_key.get_secret_value()))
    providers: list[LLMProvider] = [LLM_REGISTRY[name](name, model, endpoint, key,
                                                   settings.llm_timeout_seconds, settings.max_output_tokens)
                                    for name, model, endpoint, key in configurations]
    return ReliableLLM(providers, settings, telemetry, CircuitBreaker(redis, settings))


def build_embeddings(settings: Settings) -> EmbeddingProvider:
    if settings.embedding_provider == "local":
        return LocalEmbeddingProvider(settings)
    return GeminiEmbeddingProvider(settings.embedding_provider, settings.embedding_model,
                                   settings.embedding_base_url, settings.embedding_api_key.get_secret_value(),
                                   settings.llm_timeout_seconds, settings.max_output_tokens,
                                   settings.embedding_dimension)
