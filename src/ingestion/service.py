import hashlib
from uuid import UUID, uuid4

from src.cache.service import Cache
from src.config import Settings
from src.ingestion.chunking import chunk_sections
from src.ingestion.extract import extract, safe_filename
from src.models.entities import Document
from src.providers.base import ObservabilityProvider, StorageProvider
from src.providers.embeddings import CachedEmbeddings
from src.providers.llm import ProviderError
from src.repository import Repository
from src.security.questions import ValidationError
from src.security.rate_limit import RateLimiter


class IngestionService:
    def __init__(self, repository: Repository, storage: StorageProvider, embeddings: CachedEmbeddings,
                 limiter: RateLimiter, cache: Cache, telemetry: ObservabilityProvider, settings: Settings) -> None:
        self.repository, self.storage, self.embeddings = repository, storage, embeddings
        self.limiter, self.cache, self.telemetry, self.settings = limiter, cache, telemetry, settings

    def upload(self, collection_id: UUID, filename: str, mime: str, data: bytes, principal: str) -> UUID:
        self.limiter.check(collection_id, principal, "upload", self.settings.upload_rate_limit)
        collection = self.repository.collection(collection_id)
        if collection.embedding_fingerprint != self.settings.fingerprint(embedding=True):
            raise ValidationError("The embedding configuration changed. Create a new collection.")
        filename = safe_filename(filename)
        extract(filename, mime, data, self.settings)
        content_hash = hashlib.sha256(data).hexdigest()
        if any(doc.content_hash == content_hash for doc in self.repository.documents(collection_id)):
            raise ValidationError("This file already exists in the selected collection.")
        document_id = uuid4()
        key = self.storage.put(collection_id, document_id, data)
        try:
            self.repository.add_document(Document(document_id=document_id, collection_id=collection_id,
                                                   filename=filename, mime_type=mime, content_hash=content_hash,
                                                   storage_key=key, status="pending"))
        except Exception:
            self.storage.delete(key)
            raise
        return document_id

    def index(self, collection_id: UUID, document_id: UUID) -> None:
        with self.repository.indexing_lock(document_id) as acquired:
            if not acquired:
                return
            document = self.repository.start_indexing(collection_id, document_id)
            if document is None:
                return
            try:
                collection = self.repository.collection(collection_id)
                if collection.embedding_fingerprint != self.settings.fingerprint(embedding=True):
                    raise ValidationError("Embedding configuration changed")
                sections = extract(document.filename, document.mime_type,
                                   self.storage.read(document.storage_key), self.settings)
                chunks = chunk_sections(sections, self.settings)
                vectors, hit = self.embeddings.embed(collection_id, [chunk.text for chunk in chunks])
                committed = self.repository.finish_indexing(collection_id, document_id, chunks, vectors)
                self.telemetry.event("ingestion", {"collection_id": str(collection_id), "document_id": str(document_id),
                                                    "status": "indexed" if committed else "deleted",
                                                    "embedding_cache_hit": hit})
            except Exception as error:
                permanent = isinstance(error, ValidationError) or (isinstance(error, ProviderError) and not error.transient)
                self.repository.fail_indexing(collection_id, document_id, type(error).__name__, permanent=permanent)
                self.telemetry.event("ingestion", {"collection_id": str(collection_id), "document_id": str(document_id),
                                                    "status": "failed", "error_type": type(error).__name__})

    def delete_document(self, collection_id: UUID, document_id: UUID) -> None:
        key = self.repository.delete_document(collection_id, document_id)
        if key:
            self.storage.delete(key)
        self.cache.clear_collection(collection_id)

    def delete_collection(self, collection_id: UUID) -> None:
        for key in self.repository.delete_collection(collection_id):
            self.storage.delete(key)
        self.cache.clear_collection(collection_id)
