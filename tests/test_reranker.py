from typing import Any
from uuid import uuid4

import pytest

from src.config import Settings
from src.providers.reranker import LocalCrossEncoderReranker


class WordTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool) -> list[str]:
        return text.split()

    def decode(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class PassageModel:
    """Only the late answer passage is relevant, mimicking a long PDF page."""

    tokenizer = WordTokenizer()

    def __init__(self) -> None:
        self.pairs: list[tuple[str, str]] = []

    def predict(self, pairs: list[tuple[str, str]], **kwargs: Any) -> list[float]:
        self.pairs = pairs
        return [4.0 if "35% to 90%" in passage else -8.0 for _, passage in pairs]


def test_reranks_late_pdf_passages_and_preserves_citation_source(harness: Any) -> None:
    reranker = LocalCrossEncoderReranker(harness.settings)
    reranker.model = PassageModel()
    relevant = harness.evidence.model_copy(update={
        "content": "Introduction " * 550 + "\nNormal state of charge: 35% to 90%",
        "page": 2, "filename": "manual.pdf",
    })
    irrelevant = harness.evidence.model_copy(update={"chunk_id": uuid4()})
    ranked = reranker.rerank("Normal battery range?", [irrelevant, relevant])
    assert ranked[0].chunk_id == relevant.chunk_id
    assert ranked[0].score > harness.settings.min_rerank_score
    assert ranked[0].content == relevant.content
    assert ranked[0].page == 2 and ranked[0].filename == "manual.pdf"
    assert ranked[1].score < harness.settings.min_rerank_score
    assert all(len(passage.split()) <= harness.settings.reranker_passage_tokens
               for _, passage in reranker.model.pairs)


def test_window_budget_covers_tail_and_bounds_candidates(harness: Any) -> None:
    config = harness.settings.model_copy(update={"reranker_max_passages": 3, "rerank_top_k": 2})
    reranker = LocalCrossEncoderReranker(config)
    reranker.model = PassageModel()
    source = harness.evidence.model_copy(update={"content": "Filler " * 3000 + "35% to 90%"})
    assert len(reranker.rerank("Range?", [source] * 10)) == 2
    assert len(reranker.model.pairs) == 6
    assert "35% to 90%" in reranker.model.pairs[-1][1]


def test_empty_evidence_does_not_load_model(settings: Settings) -> None:
    reranker = LocalCrossEncoderReranker(settings)
    assert reranker.rerank("Range?", []) == []
    assert reranker.model is None


def test_passage_overlap_is_validated(settings: Settings) -> None:
    with pytest.raises(ValueError, match="overlap"):
        Settings(**{**settings.model_dump(), "reranker_passage_tokens": 64,
                    "reranker_passage_overlap": 64})
