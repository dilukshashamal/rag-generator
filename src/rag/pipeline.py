import json
import time
from uuid import UUID, uuid4

import tiktoken

from src.cache.service import Cache
from src.config import Settings
from src.models.schemas import UNAVAILABLE, Answer, ChatMessage, Evidence, RequestMetrics
from src.providers.base import ObservabilityProvider, RerankerProvider
from src.providers.embeddings import CachedEmbeddings
from src.providers.llm import ProviderError
from src.providers.reliability import ReliableLLM
from src.rag.grounding import SYSTEM_PROMPT, validate_grounding
from src.rag.retrieval import HybridRetriever
from src.repository import Repository
from src.security.questions import ValidationError, injection_flag, validate_question
from src.security.rate_limit import RateLimiter


class RAGPipeline:
    """At most two retrievals, one rewrite and two answer generations per request."""

    def __init__(self, repository: Repository, retriever: HybridRetriever, reranker: RerankerProvider,
                 embeddings: CachedEmbeddings, llm: ReliableLLM, cache: Cache, limiter: RateLimiter,
                 telemetry: ObservabilityProvider, settings: Settings) -> None:
        self.repository, self.retriever, self.reranker, self.embeddings = repository, retriever, reranker, embeddings
        self.llm, self.cache, self.limiter, self.telemetry, self.settings = llm, cache, limiter, telemetry, settings

    def ask(self, collection_id: UUID, question: str, principal: str,
            history: list[ChatMessage] | None = None, *, use_cache: bool = True) -> Answer:
        started = time.perf_counter()
        metrics = RequestMetrics(request_id=uuid4(), collection_id=collection_id)
        result = Answer(collection_id=collection_id, metrics=metrics)
        try:
            self.limiter.check(collection_id, principal, "chat", self.settings.chat_rate_limit)
            question = validate_question(question)
            collection = self.repository.collection(collection_id)
            if collection.embedding_fingerprint != self.settings.fingerprint(embedding=True):
                raise ValidationError("The embedding configuration changed. Select a collection indexed with the current model.")
            bounded_history = (history or [])[-self.settings.chat_history_messages:] if self.settings.chat_history_messages else []
            history_data = [message.model_dump() for message in bounded_history]
            key = self.cache.key(collection_id, "answer", collection.revision, question.casefold(),
                                 self.settings.fingerprint(), history_data)
            cached = self.cache.get(key) if use_cache else None
            if cached:
                if self.repository.collection(collection_id).revision != collection.revision:
                    return result
                result = Answer.model_validate(cached)
                result.metrics = metrics
                metrics.cache_hit = True
                return result
            evidence: list[Evidence] = []
            retrieval_question = question
            for attempt in range(2):
                tick = time.perf_counter()
                metrics.retrieval_attempts += 1
                vectors, hit = self.embeddings.embed(collection_id, [retrieval_question], query=True)
                metrics.embedding_cache_hit = hit
                candidates = self.retriever.retrieve(collection_id, retrieval_question, vectors[0])
                if any(item.collection_id != collection_id for item in candidates):
                    raise ValidationError("Collection isolation check failed.")
                metrics.retrieval_scores = [item.score for item in candidates]
                ranked = self.reranker.rerank(question, candidates[:self.settings.rerank_top_k])
                metrics.reranker_scores = [item.score for item in ranked]
                metrics.retrieval_ms += (time.perf_counter() - tick) * 1000
                evidence = self._context([item for item in ranked if item.score >= self.settings.min_rerank_score
                                          and not injection_flag(item.content)])
                if evidence:
                    break
                if attempt == 0:
                    raw = self.llm.generate(
                        SYSTEM_PROMPT + '\nFor this call only, return {"query":"short search query"}. '
                        'Rephrase the question for document search. Add no new facts.',
                        json.dumps({"question": question}), metrics)
                    try:
                        retrieval_question = validate_question(json.loads(raw)["query"])
                    except (ValueError, KeyError, TypeError):
                        break
                    metrics.rewrite_used = True
            if not evidence:
                metrics.abstention_reason = "no_relevant_evidence"
                return result
            metrics.selected_chunk_ids = [str(item.chunk_id) for item in evidence]
            payload = json.dumps({"question": question, "history": history_data,
                                  "evidence": [{"chunk_id": str(item.chunk_id), "content": item.content}
                                               for item in evidence]})
            for generation in range(2):
                metrics.generation_attempts += 1
                strict = "\nPrevious output failed validation. Copy complete source sentences exactly or return empty claims." if generation else ""
                raw = self.llm.generate(SYSTEM_PROMPT + strict, payload, metrics)
                validated = validate_grounding(raw, evidence, collection_id)
                if validated:
                    text, citations = validated
                    result = Answer(collection_id=collection_id, answer=text, grounded=True, status="answered",
                                    confidence="high" if min(item.score for item in evidence) >= 0.8 else "medium",
                                    citations=citations, metrics=metrics)
                    break
            # A concurrent delete/index must not serve an answer assembled from an old revision.
            if self.repository.collection(collection_id).revision != collection.revision:
                metrics.abstention_reason = "collection_changed"
                result = Answer(collection_id=collection_id, metrics=metrics)
            elif result.grounded and use_cache:
                self.cache.set(key, result.model_dump(mode="json"), self.settings.answer_cache_ttl)
            if result.status == "abstained" and metrics.abstention_reason is None:
                metrics.abstention_reason = "no_supported_claims"
            return result
        except ValidationError as error:
            metrics.validation_result, metrics.error_type = "rejected", "ValidationError"
            result = Answer(collection_id=collection_id, answer=str(error), status="rejected", metrics=metrics)
            return result
        except ProviderError as error:
            metrics.error_type = error.code
            if error.safety:
                result = Answer(collection_id=collection_id, metrics=metrics)
            else:
                result = Answer(collection_id=collection_id, answer=UNAVAILABLE, status="unavailable", metrics=metrics)
            return result
        except Exception as error:
            metrics.error_type = type(error).__name__
            result = Answer(collection_id=collection_id, answer="A required service is temporarily unavailable. Please try again later.",
                            status="unavailable", metrics=metrics)
            return result
        finally:
            metrics.latency_ms = (time.perf_counter() - started) * 1000
            self.telemetry.event("request", {**metrics.model_dump(mode="json"), "status": result.status})

    def _context(self, ranked: list[Evidence]) -> list[Evidence]:
        encoding = tiktoken.get_encoding(self.settings.tokenizer_encoding)
        selected: list[Evidence] = []
        count = 0
        for item in ranked[:self.settings.context_top_k]:
            tokens = len(encoding.encode(item.content, disallowed_special=()))
            if count + tokens <= self.settings.max_context_tokens:
                selected.append(item)
                count += tokens
        return selected
