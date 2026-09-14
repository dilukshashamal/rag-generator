"""Real PostgreSQL/pgvector + Redis integration with deterministic model doubles."""

import json
import os
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import pytest
from redis import Redis
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from src.cache.service import Cache
from src.config import Settings, get_settings
from src.database import Database
from src.ingestion.service import IngestionService
from src.models.entities import Chunk
from src.providers.embeddings import CachedEmbeddings
from src.providers.reliability import CircuitBreaker, ReliableLLM
from src.providers.storage import LocalStorageProvider
from src.rag.pipeline import RAGPipeline
from src.rag.retrieval import HybridRetriever
from src.repository import Repository
from src.security.rate_limit import RateLimiter
from tests.conftest import Events, FakeEmbeddings, FakeLLM
from tests.test_pdf_regression import PDF, QUESTIONS

pytestmark = pytest.mark.integration


@pytest.fixture
def live(settings: Settings) -> Iterator[Any]:
    if os.getenv("RUN_INTEGRATION") != "1":
        pytest.skip("Set RUN_INTEGRATION=1 with migrated PostgreSQL and Redis running")
    from types import SimpleNamespace
    actual = get_settings()
    config = settings.model_copy(update={"database_url": actual.database_url, "redis_url": actual.redis_url})
    database = Database(config)
    redis = Redis.from_url(config.redis_url.get_secret_value(), decode_responses=True)
    cache, repository = Cache(redis), Repository(database)
    collections = [repository.create_collection(f"integration-{uuid4()}", config.fingerprint(embedding=True)) for _ in range(2)]
    telemetry = Events()
    embeddings = CachedEmbeddings(FakeEmbeddings(), cache, config)
    limiter = RateLimiter(redis, 60)
    service = IngestionService(repository, LocalStorageProvider(config.storage_root), embeddings, limiter, cache, telemetry, config)
    try:
        yield SimpleNamespace(config=config, database=database, redis=redis, repository=repository,
                              collections=collections, embeddings=embeddings, cache=cache,
                              service=service, telemetry=telemetry, limiter=limiter)
    finally:
        for collection in collections:
            service.delete_collection(collection.collection_id)
        database.engine.dispose()
        redis.close()


class HighReranker:
    def rerank(self, question: str, evidence: list[Any]) -> list[Any]:
        return [item.model_copy(update={"score": 0.95}) for item in evidence]


@pytest.mark.parametrize("question,page,quote", QUESTIONS)
def test_real_pdf_upload_index_search_and_citation(live: Any, question: str, page: int, quote: str) -> None:
    collection_id = live.collections[0].collection_id
    document_id = live.service.upload(collection_id, PDF.name, "application/pdf", PDF.read_bytes(), "integration")
    live.service.index(collection_id, document_id)
    retriever = HybridRetriever(live.database, live.config)
    evidence = retriever.retrieve(collection_id, question, [1.0, 0.1, 0.1])
    assert len(evidence) == 4
    assert not retriever.retrieve(live.collections[1].collection_id, question, [1.0, 0.1, 0.1])
    source = next(item for item in evidence if item.page == page)
    response = json.dumps({"claims": [{"text": quote, "quote": quote, "chunk_id": str(source.chunk_id)}]})
    llm = ReliableLLM([FakeLLM([response])], live.config, live.telemetry, CircuitBreaker(live.redis, live.config))
    pipeline = RAGPipeline(live.repository, retriever, HighReranker(), live.embeddings, llm,
                           live.cache, live.limiter, live.telemetry, live.config)
    answer = pipeline.ask(collection_id, question, "integration")
    assert answer.grounded and answer.citations[0].page == page
    assert answer.citations[0].document_id == document_id


def test_real_upload_index_search_citation_and_isolation(live: Any) -> None:
    first, second = [item.collection_id for item in live.collections]
    document_id = live.service.upload(first, "policy.md", "text/markdown", b"# Leave\nEmployees receive 20 days of annual leave per year.", "integration")
    live.service.index(first, document_id)
    assert live.repository.document(first, document_id).status == "indexed"
    assert live.repository.document(second, document_id) is None
    retriever = HybridRetriever(live.database, live.config)
    evidence = retriever.retrieve(first, "annual leave", [1.0, 0.1, 0.1])
    assert len(evidence) == 1 and not retriever.retrieve(second, "annual leave", [1.0, 0.1, 0.1])
    original_chunk_id = evidence[0].chunk_id
    calls = live.embeddings.provider.calls
    live.service.index(first, document_id)
    assert live.embeddings.provider.calls == calls
    assert retriever.retrieve(first, "leave", [1.0, 0.1, 0.1])[0].chunk_id == original_chunk_id
    quote = "Employees receive 20 days of annual leave per year."
    response = json.dumps({"claims": [{"text": quote, "quote": quote, "chunk_id": str(original_chunk_id)}]})
    llm = ReliableLLM([FakeLLM([response])], live.config, live.telemetry, CircuitBreaker(live.redis, live.config))
    pipeline = RAGPipeline(live.repository, retriever, HighReranker(), live.embeddings, llm,
                           live.cache, live.limiter, live.telemetry, live.config)
    answer = pipeline.ask(first, "How many days of annual leave?", "integration")
    assert answer.grounded and answer.citations[0].document_id == document_id
    assert pipeline.ask(first, "How many days of annual leave?", "integration").metrics.cache_hit
    live.service.delete_document(second, document_id)
    assert live.repository.document(first, document_id) is not None
    live.service.delete_document(first, document_id)
    assert not retriever.retrieve(first, "leave", [1.0, 0.1, 0.1])


def test_database_rejects_cross_collection_chunk(live: Any) -> None:
    first, second = [item.collection_id for item in live.collections]
    document_id = live.service.upload(first, "policy.txt", "text/plain", b"Sample document with useful text.", "integration")
    with pytest.raises(IntegrityError), live.database.session() as session:
        session.add(Chunk(collection_id=second, document_id=document_id, filename="policy.txt", chunk_index=0,
                          content="wrong collection", content_hash="a" * 64, embedding=[1.0, 0.1, 0.1]))
        session.flush()


def test_pending_job_recovery_and_delete_during_index(live: Any) -> None:
    collection_id = live.collections[0].collection_id
    document_id = live.service.upload(collection_id, "policy.txt", "text/plain", b"Sample document with useful text.", "integration")
    assert (collection_id, document_id) in live.repository.pending_documents()
    assert live.repository.start_indexing(collection_id, document_id) is not None
    live.service.delete_document(collection_id, document_id)
    assert not live.repository.finish_indexing(collection_id, document_id, [], [])
    with live.database.session() as session:
        assert not session.scalars(select(Chunk).where(Chunk.collection_id == collection_id)).all()
