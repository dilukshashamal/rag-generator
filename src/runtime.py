from functools import lru_cache
from typing import Any

from redis import Redis
from sqlalchemy import text

from src.cache.service import Cache
from src.config import Settings, get_settings
from src.database import Database
from src.ingestion.service import IngestionService
from src.observability.events import Telemetry
from src.providers.embeddings import CachedEmbeddings
from src.providers.factory import build_embeddings, build_llm
from src.providers.reranker import LocalCrossEncoderReranker
from src.providers.storage import LocalStorageProvider
from src.rag.pipeline import RAGPipeline
from src.rag.retrieval import HybridRetriever
from src.repository import Repository
from src.security.rate_limit import RateLimiter


class Runtime:
    """Compose services lazily; do not download models just to render the UI."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.database = Database(settings)
        self.redis = Redis.from_url(settings.redis_url.get_secret_value(), decode_responses=True,
                                   socket_connect_timeout=3, socket_timeout=5)
        self.repository = Repository(self.database)
        self.cache = Cache(self.redis)
        self.telemetry = Telemetry(settings)
        self.limiter = RateLimiter(self.redis, settings.rate_window_seconds)
        self.storage = LocalStorageProvider(settings.storage_root)
        self.embeddings = CachedEmbeddings(build_embeddings(settings), self.cache, settings)
        self.reranker = LocalCrossEncoderReranker(settings)
        self.llm = build_llm(settings, self.redis, self.telemetry)
        self.ingestion = IngestionService(self.repository, self.storage, self.embeddings, self.limiter,
                                          self.cache, self.telemetry, settings)
        self.pipeline = RAGPipeline(self.repository, HybridRetriever(self.database, settings), self.reranker,
                                    self.embeddings, self.llm, self.cache, self.limiter, self.telemetry, settings)

    def health(self) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for name, probe in [("Database", lambda: self._database_ready()),
                            ("Redis", lambda: self.redis.ping())]:
            try:
                probe()
                rows.append({"Service": name, "Status": "Ready"})
            except Exception as error:
                self.telemetry.event("failure", {"error_type": type(error).__name__})
                rows.append({"Service": name, "Status": "Unavailable"})
        for provider in self.llm.providers:
            try:
                state = "Configured (not probed)" if self.llm.circuit.available(provider) else "Circuit open"
                if provider.name != "ollama" and not getattr(provider, "key", None):
                    state = "Credentials missing"
            except Exception:
                state = "Circuit status unavailable"
            rows.append({"Service": f"LLM: {provider.name}", "Status": state})
        rows.append({"Service": "Embeddings", "Status": "Loaded" if getattr(self.embeddings.provider, "model", None) is not None else "Configured (lazy load)"})
        rows.append({"Service": "Reranker", "Status": "Loaded" if self.reranker.model is not None else "Configured (lazy load)"})
        return rows

    def _database_ready(self) -> Any:
        with self.database.session() as session:
            return session.execute(text("SELECT version FROM schema_versions WHERE version = 1")).scalar_one()


@lru_cache
def get_runtime() -> Runtime:
    return Runtime(get_settings())
