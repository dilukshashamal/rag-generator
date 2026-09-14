"""All data access is scoped; collection locks serialize content mutations."""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy import delete, or_, select, text
from sqlalchemy.exc import IntegrityError

from src.database import Database
from src.models.entities import Chunk, Collection, Document, EvaluationRun
from src.models.schemas import ChunkData
from src.security.questions import ValidationError


class Repository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def collections(self) -> list[Collection]:
        with self.database.session() as session:
            return list(session.scalars(select(Collection).order_by(Collection.created_at.desc())))

    def collection(self, collection_id: UUID) -> Collection:
        with self.database.session() as session:
            value = session.get(Collection, collection_id)
            if value is None:
                raise ValidationError("This collection no longer exists.")
            return value

    def create_collection(self, name: str, fingerprint: str) -> Collection:
        name = " ".join(name.split())
        if not name or len(name) > 120:
            raise ValidationError("Collection names must contain 1 to 120 characters.")
        with self.database.session() as session:
            value = Collection(name=name, embedding_fingerprint=fingerprint)
            session.add(value)
            session.flush()
            return value

    def documents(self, collection_id: UUID) -> list[Document]:
        with self.database.session() as session:
            return list(session.scalars(select(Document).where(Document.collection_id == collection_id)
                                         .order_by(Document.created_at.desc())))

    def document(self, collection_id: UUID, document_id: UUID) -> Document | None:
        with self.database.session() as session:
            return session.scalar(select(Document).where(Document.collection_id == collection_id,
                                                         Document.document_id == document_id))

    def add_document(self, document: Document) -> None:
        try:
            with self.database.session() as session:
                collection = session.scalar(select(Collection).where(
                    Collection.collection_id == document.collection_id).with_for_update())
                if collection is None:
                    raise ValidationError("This collection no longer exists.")
                session.add(document)
                collection.revision += 1
                session.flush()
        except IntegrityError:
            raise ValidationError("This file already exists in the selected collection.") from None

    def delete_document(self, collection_id: UUID, document_id: UUID) -> str | None:
        with self.database.session() as session:
            collection = session.scalar(select(Collection).where(Collection.collection_id == collection_id)
                                         .with_for_update())
            document = session.scalar(select(Document).where(Document.collection_id == collection_id,
                                                            Document.document_id == document_id))
            if document is None or collection is None:
                return None
            key = document.storage_key
            session.delete(document)
            collection.revision += 1
            return key

    def delete_collection(self, collection_id: UUID) -> list[str]:
        with self.database.session() as session:
            collection = session.scalar(select(Collection).where(Collection.collection_id == collection_id)
                                         .with_for_update())
            if collection is None:
                return []
            keys = list(session.scalars(select(Document.storage_key).where(Document.collection_id == collection_id)))
            session.delete(collection)
            return keys

    @contextmanager
    def indexing_lock(self, document_id: UUID) -> Iterator[bool]:
        key = int.from_bytes(document_id.bytes[:8], "big", signed=True)
        with self.database.engine.connect() as connection:
            locked = bool(connection.scalar(text("SELECT pg_try_advisory_lock(:key)"), {"key": key}))
            connection.commit()
            try:
                yield locked
            finally:
                if locked:
                    connection.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})
                    connection.commit()

    def start_indexing(self, collection_id: UUID, document_id: UUID) -> Document | None:
        with self.database.session() as session:
            document = session.scalar(select(Document).where(Document.collection_id == collection_id,
                                                            Document.document_id == document_id).with_for_update())
            if document is None or document.status == "indexed":
                return None
            if document.attempts >= 3:
                if document.status == "processing":
                    document.status, document.error_code = "failed", "WorkerInterrupted"
                return None
            document.status, document.error_code = "processing", None
            document.attempts += 1
            session.flush()
            return document

    def finish_indexing(self, collection_id: UUID, document_id: UUID,
                        chunks: list[ChunkData], vectors: list[list[float]]) -> bool:
        with self.database.session() as session:
            collection = session.scalar(select(Collection).where(Collection.collection_id == collection_id)
                                         .with_for_update())
            document = session.scalar(select(Document).where(Document.collection_id == collection_id,
                                                            Document.document_id == document_id).with_for_update())
            if collection is None or document is None:
                return False
            session.execute(delete(Chunk).where(Chunk.collection_id == collection_id, Chunk.document_id == document_id))
            for chunk, vector in zip(chunks, vectors, strict=True):
                session.add(Chunk(collection_id=collection_id, document_id=document_id,
                                  filename=document.filename, page=chunk.page, heading=chunk.heading,
                                  chunk_index=chunk.chunk_index, content=chunk.text,
                                  content_hash=chunk.content_hash, embedding=vector))
            document.status, document.error_code, document.chunk_count = "indexed", None, len(chunks)
            collection.revision += 1
            return True

    def fail_indexing(self, collection_id: UUID, document_id: UUID, code: str, *, permanent: bool = False) -> None:
        with self.database.session() as session:
            document = session.scalar(select(Document).where(Document.collection_id == collection_id,
                                                            Document.document_id == document_id).with_for_update())
            if document is not None:
                document.status, document.error_code = "failed", code
                if permanent:
                    document.attempts = 3

    def retry_document(self, collection_id: UUID, document_id: UUID) -> None:
        with self.database.session() as session:
            document = session.scalar(select(Document).where(Document.collection_id == collection_id,
                                                            Document.document_id == document_id).with_for_update())
            if document is not None and document.status == "failed":
                document.status, document.attempts, document.error_code = "pending", 0, None

    def pending_documents(self) -> list[tuple[UUID, UUID]]:
        with self.database.session() as session:
            return list(session.execute(select(Document.collection_id, Document.document_id).where(
                or_((Document.status == "pending") & (Document.attempts < 3),
                    (Document.status == "failed") & (Document.attempts < 3) & (Document.updated_at < datetime.now(UTC) - timedelta(seconds=60)),
                    (Document.status == "processing") & (Document.updated_at < datetime.now(UTC) - timedelta(minutes=16))),
            ).limit(100)).all())

    def chunks(self, collection_id: UUID, chunk_ids: list[UUID]) -> list[Chunk]:
        with self.database.session() as session:
            return list(session.scalars(select(Chunk).where(Chunk.collection_id == collection_id,
                                                            Chunk.chunk_id.in_(chunk_ids))))

    def create_evaluation(self, collection_id: UUID) -> EvaluationRun:
        with self.database.session() as session:
            collection = session.scalar(select(Collection).where(Collection.collection_id == collection_id).with_for_update())
            if collection is None:
                raise ValidationError("This collection no longer exists.")
            if session.scalar(select(EvaluationRun).where(EvaluationRun.collection_id == collection_id,
                                                          EvaluationRun.status.in_(["pending", "processing"]))):
                raise ValidationError("An evaluation is already queued or running for this collection.")
            run = EvaluationRun(collection_id=collection_id)
            session.add(run)
            session.flush()
            return run

    def evaluations(self, collection_id: UUID) -> list[EvaluationRun]:
        with self.database.session() as session:
            return list(session.scalars(select(EvaluationRun).where(EvaluationRun.collection_id == collection_id)
                                         .order_by(EvaluationRun.created_at.desc()).limit(20)))
