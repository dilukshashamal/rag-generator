"""Small HTTP adapters that normalize failures without leaking response bodies."""

from typing import Any

import httpx


class ProviderError(Exception):
    def __init__(self, code: str, *, transient: bool = False, safety: bool = False) -> None:
        super().__init__(code)
        self.code = code
        self.transient = transient
        self.safety = safety


class HTTPProvider:
    def __init__(self, name: str, model: str, endpoint: str, key: str,
                 timeout: float, max_tokens: int, client: httpx.Client | None = None) -> None:
        self.name, self.model, self.endpoint = name, model, endpoint.rstrip("/")
        self.key, self.max_tokens = key, max_tokens
        self.client = client or httpx.Client(timeout=timeout, follow_redirects=False)

    def post(self, path: str, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        try:
            response = self.client.post(f"{self.endpoint}{path}", json=body, headers=headers)
        except (httpx.TimeoutException, httpx.NetworkError) as error:
            raise ProviderError(type(error).__name__, transient=True) from None
        if response.status_code >= 400:
            raise ProviderError(f"HTTP{response.status_code}",
                                transient=response.status_code == 429 or response.status_code >= 500)
        try:
            return response.json()
        except ValueError:
            raise ProviderError("MalformedResponse") from None


class GroqLLMProvider(HTTPProvider):
    def generate(self, system: str, payload: str) -> str:
        if not self.key:
            raise ProviderError("MissingCredentials")
        data = self.post("/chat/completions", {
            "model": self.model, "temperature": 0, "max_tokens": self.max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": payload}],
        }, {"Authorization": f"Bearer {self.key}"})
        try:
            choice = data["choices"][0]
            if choice["message"].get("refusal") or choice.get("finish_reason") == "content_filter":
                raise ProviderError("SafetyRefusal", safety=True)
            content = choice["message"]["content"]
            if not isinstance(content, str):
                raise ValueError
            return content
        except (KeyError, IndexError, TypeError, ValueError):
            raise ProviderError("MalformedResponse") from None


class GeminiLLMProvider(HTTPProvider):
    def generate(self, system: str, payload: str) -> str:
        if not self.key:
            raise ProviderError("MissingCredentials")
        data = self.post(f"/models/{self.model}:generateContent", {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": payload}]}],
            "generationConfig": {"temperature": 0, "maxOutputTokens": self.max_tokens,
                                 "responseMimeType": "application/json"},
        }, {"x-goog-api-key": self.key})
        if data.get("promptFeedback", {}).get("blockReason"):
            raise ProviderError("SafetyRefusal", safety=True)
        try:
            candidate = data["candidates"][0]
            if candidate.get("finishReason") in {"SAFETY", "RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT"}:
                raise ProviderError("SafetyRefusal", safety=True)
            return "".join(part.get("text", "") for part in candidate["content"]["parts"])
        except (KeyError, IndexError, TypeError):
            raise ProviderError("MalformedResponse") from None


class OllamaLLMProvider(HTTPProvider):
    def generate(self, system: str, payload: str) -> str:
        data = self.post("/api/chat", {
            "model": self.model, "stream": False, "format": "json",
            "options": {"temperature": 0, "num_predict": self.max_tokens},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": payload}],
        }, {"Authorization": f"Bearer {self.key}"} if self.key else {})
        try:
            content = data["message"]["content"]
            if not isinstance(content, str):
                raise ValueError
            return content
        except (KeyError, TypeError, ValueError):
            raise ProviderError("MalformedResponse") from None
