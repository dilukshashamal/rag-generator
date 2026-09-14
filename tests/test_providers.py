from typing import Any
from uuid import uuid4

import httpx
import pytest

from src.config import Settings
from src.observability.events import Telemetry
from src.providers.llm import GeminiLLMProvider, GroqLLMProvider, OllamaLLMProvider, ProviderError
from src.security.questions import ValidationError
from src.security.rate_limit import RateLimiter
from tests.conftest import MemoryRedis


@pytest.mark.parametrize("status,transient", [(429, True), (500, True), (502, True), (503, True), (504, True),
                                               (400, False), (401, False), (403, False), (422, False)])
def test_http_status_classification(status: int, transient: bool) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(status, json={"secret": "do-not-log"})))
    provider = GroqLLMProvider("groq", "test", "https://example.invalid", "fake", 1, 100, client)
    with pytest.raises(ProviderError) as error:
        provider.generate("system", "payload")
    assert error.value.transient == transient
    assert "do-not-log" not in str(error.value)


@pytest.mark.parametrize("kind,body", [
    (GroqLLMProvider, {"choices": [{"message": {"content": '{"claims":[]}'}}]}),
    (GeminiLLMProvider, {"candidates": [{"content": {"parts": [{"text": '{"claims":[]}' }]}}]}),
    (OllamaLLMProvider, {"message": {"content": '{"claims":[]}'}}),
])
def test_provider_adapter_success(kind: Any, body: dict[str, Any]) -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=body)))
    provider = kind("test", "test", "https://example.invalid", "fake", 1, 100, client)
    assert provider.generate("system", "payload") == '{"claims":[]}'


def test_safety_response_is_distinct() -> None:
    client = httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})))
    provider = GeminiLLMProvider("test", "test", "https://example.invalid", "fake", 1, 100, client)
    with pytest.raises(ProviderError) as error:
        provider.generate("system", "payload")
    assert error.value.safety


def test_telemetry_drops_sensitive_fields(settings: Settings, capsys: Any) -> None:
    telemetry = Telemetry(settings)
    telemetry.event("request", {"collection_id": str(uuid4()), "api_key": "secret-key",
                                "question": "sensitive user question", "document": "private document"})
    captured = capsys.readouterr().err
    assert "secret-key" not in captured and "sensitive user question" not in captured
    assert "collection_id" in captured


def test_rate_limit_is_scoped() -> None:
    limiter = RateLimiter(MemoryRedis(), 60)
    collection_id = uuid4()
    limiter.check(collection_id, "user", "chat", 1)
    with pytest.raises(ValidationError):
        limiter.check(collection_id, "user", "chat", 1)
    limiter.check(uuid4(), "user", "chat", 1)
