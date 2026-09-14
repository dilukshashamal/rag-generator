from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

NO_ANSWER = "I could not find enough reliable information in the uploaded documents to answer this question."
UNAVAILABLE = "The model is temporarily unavailable. Please try again later."


class Section(BaseModel):
    text: str
    page: int | None = None
    heading: str | None = None


class ChunkData(Section):
    chunk_index: int
    content_hash: str


class Evidence(BaseModel):
    chunk_id: UUID
    collection_id: UUID
    document_id: UUID
    filename: str
    page: int | None = None
    heading: str | None = None
    content: str
    score: float = 0


class Claim(BaseModel):
    text: str = Field(min_length=1, max_length=3000)
    chunk_id: UUID
    quote: str = Field(min_length=1, max_length=3000)


class GeneratedAnswer(BaseModel):
    claims: list[Claim] = Field(default_factory=list, max_length=8)


class Citation(BaseModel):
    chunk_id: UUID
    document_id: UUID
    filename: str
    page: int | None
    heading: str | None
    excerpt: str


class RequestMetrics(BaseModel):
    request_id: UUID
    collection_id: UUID
    retrieval_attempts: int = 0
    rewrite_used: bool = False
    generation_attempts: int = 0
    llm_calls: int = 0
    retries: int = 0
    fallback_used: bool = False
    cache_hit: bool = False
    embedding_cache_hit: bool = False
    provider: str = ""
    model: str = ""
    retrieval_ms: float = 0
    latency_ms: float = 0
    estimated_tokens: int = 0
    selected_chunk_ids: list[str] = Field(default_factory=list)
    retrieval_scores: list[float] = Field(default_factory=list)
    reranker_scores: list[float] = Field(default_factory=list)
    error_type: str | None = None
    validation_result: str = "accepted"
    abstention_reason: str | None = None


class Answer(BaseModel):
    collection_id: UUID
    answer: str = NO_ANSWER
    grounded: bool = False
    confidence: Literal["high", "medium", "low"] = "low"
    status: Literal["answered", "abstained", "unavailable", "rejected"] = "abstained"
    citations: list[Citation] = Field(default_factory=list)
    metrics: RequestMetrics


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=4000)
