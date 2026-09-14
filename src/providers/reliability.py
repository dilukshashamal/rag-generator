import hashlib
import random
import time
from collections.abc import Callable

from redis import Redis

from src.config import Settings
from src.models.schemas import RequestMetrics
from src.providers.base import LLMProvider, ObservabilityProvider
from src.providers.llm import ProviderError


class CircuitBreaker:
    def __init__(self, redis: Redis, settings: Settings) -> None:
        self.redis, self.settings = redis, settings

    @staticmethod
    def key(provider: LLMProvider) -> str:
        identity = f"{provider.name}:{provider.model}:{getattr(provider, 'endpoint', '')}"
        return "circuit:" + hashlib.sha256(identity.encode()).hexdigest()

    def available(self, provider: LLMProvider) -> bool:
        return int(self.redis.get(self.key(provider)) or 0) < self.settings.circuit_failure_threshold

    def failure(self, provider: LLMProvider) -> None:
        self.redis.eval(
            "local n=redis.call('INCR',KEYS[1]); if n==1 then redis.call('EXPIRE',KEYS[1],ARGV[1]) end; return n",
            1, self.key(provider), self.settings.circuit_cooldown_seconds,
        )

    def success(self, provider: LLMProvider) -> None:
        self.redis.delete(self.key(provider))


class ReliableLLM:
    """Bounded transient retries followed by configured ordered fallbacks."""

    def __init__(self, providers: list[LLMProvider], settings: Settings,
                 telemetry: ObservabilityProvider, circuit: CircuitBreaker,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.providers, self.settings = providers, settings
        self.telemetry, self.circuit, self.sleep = telemetry, circuit, sleep

    def generate(self, system: str, payload: str, metrics: RequestMetrics) -> str:
        for index, provider in enumerate(self.providers):
            if not self.circuit.available(provider):
                continue
            metrics.provider, metrics.model = provider.name, provider.model
            if index:
                metrics.fallback_used = True
                self.telemetry.event("fallback", metrics.model_dump(mode="json"))
            for attempt in range(1, self.settings.llm_max_attempts + 1):
                metrics.llm_calls += 1
                metrics.estimated_tokens += (len(system) + len(payload)) // 4
                try:
                    response = provider.generate(system, payload)
                    metrics.estimated_tokens += len(response) // 4
                    self.circuit.success(provider)
                    return response
                except ProviderError as error:
                    self.telemetry.event("provider_failure", {
                        **metrics.model_dump(mode="json"), "error_type": error.code,
                        "attempt": attempt,
                    })
                    if error.safety:
                        raise
                    if not error.transient:
                        break
                    if attempt == self.settings.llm_max_attempts:
                        self.circuit.failure(provider)
                        break
                    metrics.retries += 1
                    delay = min(2 ** (attempt - 1), 8) + random.uniform(0, 0.5)
                    self.telemetry.event("retry", {**metrics.model_dump(mode="json"),
                                                    "attempt": attempt, "delay_seconds": delay})
                    self.sleep(delay)
        raise ProviderError("AllProvidersUnavailable")
