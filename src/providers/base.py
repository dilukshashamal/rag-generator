from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

from src.models.schemas import Evidence


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, system: str, payload: str) -> str: ...


class EmbeddingProvider(Protocol):
    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]: ...


class RerankerProvider(Protocol):
    def rerank(self, question: str, evidence: list[Evidence]) -> list[Evidence]: ...


class StorageProvider(Protocol):
    def put(self, collection_id: UUID, document_id: UUID, content: bytes) -> str: ...
    def read(self, key: str) -> bytes: ...
    def delete(self, key: str) -> None: ...
    def path(self, key: str) -> Path: ...


class ObservabilityProvider(Protocol):
    def event(self, event: str, metadata: dict[str, Any]) -> None: ...
