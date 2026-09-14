import json
from typing import Any

import pytest
import tiktoken

from src.ingestion.chunking import chunk_sections
from src.ingestion.service import IngestionService
from src.models.schemas import Section
from src.providers.storage import LocalStorageProvider
from src.security.questions import ValidationError


def test_chunk_token_bounds_and_source_locations(harness: Any) -> None:
    section = Section(text="A paragraph about employee leave entitlement. " * 400, page=7, heading="Leave")
    chunks = chunk_sections([section], harness.settings)
    encoding = tiktoken.get_encoding(harness.settings.tokenizer_encoding)
    assert len(chunks) > 1
    assert all(len(encoding.encode(chunk.text)) <= 700 for chunk in chunks)
    assert all(chunk.page == 7 and chunk.heading == "Leave" for chunk in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_upload_index_query_citation_offline(harness: Any) -> None:
    pipeline = harness.pipeline([])
    service = IngestionService(harness.repo, LocalStorageProvider(harness.settings.storage_root),
                               pipeline.embeddings, pipeline.limiter, harness.cache, harness.events, harness.settings)
    data = b"# Leave\nEmployees receive 20 days of annual leave per year."
    document_id = service.upload(harness.collection_id, "policy.md", "text/markdown", data, "user")
    assert harness.repo.docs[document_id].status == "pending"
    service.index(harness.collection_id, document_id)
    assert harness.repo.docs[document_id].status == "indexed"
    calls = pipeline.embeddings.provider.calls
    service.index(harness.collection_id, document_id)
    assert pipeline.embeddings.provider.calls == calls
    with pytest.raises(ValidationError, match="already exists"):
        service.upload(harness.collection_id, "another-name.md", "text/markdown", data, "user")
    evidence = harness.repo.chunks[0]
    quote = "Employees receive 20 days of annual leave per year."
    pipeline = harness.pipeline([json.dumps({"claims": [{"text": quote, "quote": quote,
                                                        "chunk_id": str(evidence.chunk_id)}]})], [evidence])
    result = pipeline.ask(harness.collection_id, "How many annual leave days?", "user")
    assert result.grounded and result.citations[0].document_id == document_id
    assert result.citations[0].heading == "Leave"
