"""The reported four-page PDF exercises real extraction; model calls are doubles."""

import json
from pathlib import Path
from typing import Any

import pytest

from src.config import Settings
from src.ingestion.chunking import chunk_sections
from src.ingestion.extract import extract
from src.ingestion.service import IngestionService
from src.providers.storage import LocalStorageProvider

PDF = Path(__file__).parent / "fixtures/microgrid_operations_and_response_guide.pdf"
QUESTIONS = [
    ("What is the normal battery state-of-charge range?", 1,
     "Normal state of charge: 35% to 90%"),
    ("What should an operator do if battery state of charge falls below 35%?", 2,
     "20% to 34%\nLow reserve\nNotify on-call technician and verify generator\nreadiness."),
]


def test_pdf_extracts_all_pages_and_battery_evidence(settings: Settings) -> None:
    sections = extract(PDF.name, "application/pdf", PDF.read_bytes(), settings)
    chunks = chunk_sections(sections, settings)
    assert [chunk.page for chunk in chunks] == [1, 2, 3, 4]
    for _, page, quote in QUESTIONS:
        assert quote in chunks[page - 1].text
    assert "Notify on-call technician\nwithin 30 minutes." in chunks[2].text
    assert "telemetry older than 15 minutes" in chunks[3].text


@pytest.mark.parametrize("question,page,quote", QUESTIONS)
def test_pdf_upload_index_retrieve_answer_and_citation(
    harness: Any, question: str, page: int, quote: str,
) -> None:
    pipeline = harness.pipeline([])
    service = IngestionService(harness.repo, LocalStorageProvider(harness.settings.storage_root),
                               pipeline.embeddings, pipeline.limiter, harness.cache,
                               harness.events, harness.settings)
    document_id = service.upload(harness.collection_id, PDF.name, "application/pdf", PDF.read_bytes(), "pdf-test")
    service.index(harness.collection_id, document_id)
    sources = harness.repo.chunks
    assert len(sources) == 4
    source = next(item for item in sources if item.page == page)
    raw = json.dumps({"claims": [{"text": quote, "quote": quote, "chunk_id": str(source.chunk_id)}]})
    result = harness.pipeline([raw], sources).ask(harness.collection_id, question, "pdf-test")
    assert result.status == "answered" and result.grounded
    assert result.citations[0].page == page
    assert result.citations[0].document_id == document_id
    assert result.citations[0].excerpt == quote
