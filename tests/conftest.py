from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from src.cache.service import Cache
from src.config import Settings
from src.models.schemas import Evidence
from src.providers.embeddings import CachedEmbeddings
from src.providers.reliability import CircuitBreaker, ReliableLLM
from src.rag.pipeline import RAGPipeline
from src.security.rate_limit import RateLimiter


@pytest.fixture
def settings(tmp_path: Any) -> Settings:
    return Settings(_env_file=None, database_url="postgresql+psycopg://test:test@localhost/test",
                    redis_url="redis://localhost/0", llm_provider="groq", llm_model="test-model",
                    llm_base_url="https://provider.invalid", embedding_provider="local",
                    embedding_model="test-embedding", embedding_dimension=3, reranker_provider="local",
                    reranker_model="test-reranker", tokenizer_encoding="cl100k_base",
                    storage_root=tmp_path / "uploads", results_root=tmp_path / "evals")


class MemoryRedis:
    def __init__(self) -> None:
        self.values: dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self.values.get(key)

    def set(self, key: str, value: Any, ex: int | None = None) -> None:
        self.values[key] = value

    def delete(self, key: str) -> None:
        self.values.pop(key, None)

    def eval(self, script: str, count: int, key: str, window: int) -> int:
        self.values[key] = int(self.values.get(key, 0)) + 1
        return self.values[key]

    def scan_iter(self, match: str, count: int) -> list[str]:
        return [key for key in self.values if key.startswith(match.rstrip("*"))]


class Events:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict[str, Any]]] = []

    def event(self, event: str, metadata: dict[str, Any]) -> None:
        self.events.append((event, metadata))


class FakeEmbeddings:
    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        self.calls += len(texts)
        return [[1.0, 0.1, 0.1] for _ in texts]


class FakeLLM:
    name = "test"
    model = "test-model"

    def __init__(self, responses: list[Any]) -> None:
        self.responses = iter(responses)
        self.calls = 0

    def generate(self, system: str, payload: str) -> str:
        self.calls += 1
        result = next(self.responses)
        if isinstance(result, Exception):
            raise result
        return result


class MemoryRepository:
    def __init__(self, settings: Settings) -> None:
        self.cols: dict[UUID, Any] = {}
        self.docs: dict[UUID, Any] = {}
        self.chunks: list[Evidence] = []
        self.settings = settings

    def create_collection(self, name: str, fingerprint: str) -> Any:
        item = SimpleNamespace(collection_id=uuid4(), name=name, revision=0, embedding_fingerprint=fingerprint)
        self.cols[item.collection_id] = item
        return item

    def collection(self, collection_id: UUID) -> Any:
        import copy
        return copy.copy(self.cols[collection_id])

    def documents(self, collection_id: UUID) -> list[Any]:
        return [doc for doc in self.docs.values() if doc.collection_id == collection_id]

    def add_document(self, document: Any) -> None:
        document.attempts = 0
        self.docs[document.document_id] = document
        self.cols[document.collection_id].revision += 1

    @contextmanager
    def indexing_lock(self, document_id: UUID) -> Any:
        yield True

    def start_indexing(self, collection_id: UUID, document_id: UUID) -> Any:
        doc = self.docs.get(document_id)
        if not doc or doc.collection_id != collection_id or doc.status == "indexed":
            return None
        doc.status = "processing"
        doc.attempts += 1
        return doc

    def finish_indexing(self, collection_id: UUID, document_id: UUID, chunks: list[Any], vectors: list[Any]) -> bool:
        doc = self.docs[document_id]
        self.chunks = [item for item in self.chunks if item.document_id != document_id]
        self.chunks.extend(Evidence(chunk_id=uuid4(), collection_id=collection_id, document_id=document_id,
                                    filename=doc.filename, page=chunk.page, heading=chunk.heading,
                                    content=chunk.text, score=0.95) for chunk in chunks)
        doc.status = "indexed"
        self.cols[collection_id].revision += 1
        return True

    def fail_indexing(self, collection_id: UUID, document_id: UUID, code: str, *, permanent: bool = False) -> None:
        self.docs[document_id].status = "failed"


class FakeRetriever:
    def __init__(self, evidence: list[Evidence]) -> None:
        self.evidence, self.calls = evidence, 0

    def retrieve(self, collection_id: UUID, question: str, vector: list[float]) -> list[Evidence]:
        self.calls += 1
        return self.evidence


class FakeReranker:
    def rerank(self, question: str, evidence: list[Evidence]) -> list[Evidence]:
        return evidence


@pytest.fixture
def harness(settings: Settings) -> Any:
    redis = MemoryRedis()
    cache = Cache(redis)
    repository = MemoryRepository(settings)
    collection = repository.create_collection("Policies", settings.fingerprint(embedding=True))
    collection_id = collection.collection_id
    evidence = Evidence(chunk_id=uuid4(), collection_id=collection_id, document_id=uuid4(),
                        filename="policy.md", heading="Annual Leave", score=0.95,
                        content="Employees receive 20 days of annual leave per year.")
    events = Events()
    embeddings = CachedEmbeddings(FakeEmbeddings(), cache, settings)
    def pipeline(responses: list[Any], sources: list[Evidence] | None = None) -> RAGPipeline:
        llm = ReliableLLM([FakeLLM(responses)], settings, events, CircuitBreaker(redis, settings), sleep=lambda _: None)
        return RAGPipeline(repository, FakeRetriever(sources if sources is not None else [evidence]),
                           FakeReranker(), embeddings, llm, cache,
                           RateLimiter(redis, settings.rate_window_seconds), events, settings)
    return SimpleNamespace(redis=redis, cache=cache, repo=repository, collection_id=collection_id,
                           evidence=evidence, pipeline=pipeline, settings=settings, events=events)
