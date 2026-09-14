import json
import logging
from collections import deque
from datetime import UTC, datetime
from typing import Any

from src.config import Settings

ALLOWED = frozenset({
    "request_id", "collection_id", "validation_result", "retrieval_attempts", "rewrite_used",
    "generation_attempts", "llm_calls", "retries", "fallback_used", "cache_hit",
    "embedding_cache_hit", "provider", "model", "retrieval_ms", "latency_ms",
    "estimated_tokens", "selected_chunk_ids", "retrieval_scores", "reranker_scores",
    "error_type", "status", "attempt", "delay_seconds", "document_id", "run_id", "abstention_reason",
})
EVENTS = frozenset({"request", "retry", "provider_failure", "fallback", "ingestion", "evaluation", "failure"})


class Telemetry:
    """Drop unknown fields at the sink; never serialize exception messages."""

    def __init__(self, settings: Settings) -> None:
        self.logger = logging.getLogger("rag.metadata")
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            self.logger.addHandler(logging.StreamHandler())
        self.logger.propagate = False
        self.recent_errors: deque[dict[str, Any]] = deque(maxlen=20)
        self.client = None
        if settings.observability_enabled:
            from langfuse import Langfuse
            self.client = Langfuse(host=settings.langfuse_host,
                                  public_key=settings.langfuse_public_key.get_secret_value(),
                                  secret_key=settings.langfuse_secret_key.get_secret_value())

    def event(self, event: str, metadata: dict[str, Any]) -> None:
        safe = {key: value for key, value in metadata.items() if key in ALLOWED}
        record = {"event": event if event in EVENTS else "failure",
                  "time": datetime.now(UTC).isoformat(), **safe}
        self.logger.info(json.dumps(record, default=str))
        if safe.get("error_type"):
            self.recent_errors.append(record)
        if self.client is not None:
            try:
                with self.client.start_as_current_observation(name=record["event"], metadata=safe):
                    pass
            except Exception:
                self.logger.warning('{"event":"failure","error_type":"TelemetryUnavailable"}')
