import math
from uuid import UUID

import httpx

from src.cache.service import Cache
from src.config import Settings
from src.providers.base import EmbeddingProvider
from src.providers.llm import HTTPProvider, ProviderError


def validate_vectors(vectors: list[list[float]], count: int, dimension: int) -> list[list[float]]:
    if len(vectors) != count or any(
        len(v) != dimension or not all(math.isfinite(x) for x in v) or not any(v)
        for v in vectors
    ):
        raise ProviderError("InvalidEmbeddingShape")
    return vectors


class LocalEmbeddingProvider:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.model = None

    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.settings.embedding_model,
                                             device=self.settings.model_device, trust_remote_code=False)
        import numpy as np
        output: list[list[float]] = []
        tokenizer = self.model.tokenizer
        window = self.model.max_seq_length - 8
        for text in texts:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            parts = [tokenizer.decode(tokens[start:start + window]) for start in range(0, len(tokens), window)]
            vectors = self.model.encode(parts, normalize_embeddings=True, show_progress_bar=False, batch_size=16)
            vector = np.mean(vectors, axis=0)
            vector /= max(float(np.linalg.norm(vector)), 1e-12)
            output.append(vector.tolist())
        return output


class GeminiEmbeddingProvider(HTTPProvider):
    def __init__(self, name: str, model: str, endpoint: str, key: str,
                 timeout: float, max_tokens: int, dimension: int | None = None,
                 client: httpx.Client | None = None) -> None:
        super().__init__(name, model, endpoint, key, timeout, max_tokens, client=client)
        self.dimension = dimension

    def embed(self, texts: list[str], *, query: bool = False) -> list[list[float]]:
        if not self.key:
            raise ProviderError("MissingCredentials")
        reqs = []
        for value in texts:
            item = {
                "model": f"models/{self.model}",
                "content": {"parts": [{"text": value}]},
                "taskType": "RETRIEVAL_QUERY" if query else "RETRIEVAL_DOCUMENT"
            }
            if self.dimension:
                item["outputDimensionality"] = self.dimension
            reqs.append(item)
        data = self.post(f"/models/{self.model}:batchEmbedContents", {
            "requests": reqs,
        }, {"x-goog-api-key": self.key})
        try:
            return [embedding["values"] for embedding in data["embeddings"]]
        except (KeyError, TypeError):
            raise ProviderError("MalformedEmbeddingResponse") from None


class CachedEmbeddings:
    def __init__(self, provider: EmbeddingProvider, cache: Cache, settings: Settings) -> None:
        self.provider, self.cache, self.settings = provider, cache, settings

    def embed(self, collection_id: UUID, texts: list[str], *, query: bool = False) -> tuple[list[list[float]], bool]:
        vectors: list[list[float]] = []
        all_cached = True
        for content in texts:
            key = self.cache.key(collection_id, "embedding", self.settings.fingerprint(embedding=True), query, content)
            value = self.cache.get(key)
            if value is None:
                all_cached = False
                value = validate_vectors(self.provider.embed([content], query=query), 1,
                                         self.settings.embedding_dimension)[0]
                self.cache.set(key, value, self.settings.embedding_cache_ttl)
            vectors.append(value)
        return validate_vectors(vectors, len(texts), self.settings.embedding_dimension), all_cached
