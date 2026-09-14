from uuid import UUID

from sqlalchemy import func, select

from src.config import Settings
from src.database import Database
from src.models.entities import Chunk, Document
from src.models.schemas import Evidence


def reciprocal_rank_fusion(rankings: list[list[Evidence]], limit: int = 10) -> list[Evidence]:
    scores: dict[UUID, float] = {}
    chunks: dict[UUID, Evidence] = {}
    for ranking in rankings:
        seen: set[UUID] = set()
        for rank, item in enumerate(ranking, 1):
            if item.chunk_id in seen:
                continue
            seen.add(item.chunk_id)
            chunks[item.chunk_id] = item
            scores[item.chunk_id] = scores.get(item.chunk_id, 0) + 1 / (60 + rank)
    return [chunks[key].model_copy(update={"score": score})
            for key, score in sorted(scores.items(), key=lambda pair: (-pair[1], str(pair[0])))[:limit]]


class HybridRetriever:
    def __init__(self, database: Database, settings: Settings) -> None:
        self.database, self.settings = database, settings

    def retrieve(self, collection_id: UUID, question: str, vector: list[float]) -> list[Evidence]:
        with self.database.session() as session:
            base = select(Chunk).join(Document, (Document.document_id == Chunk.document_id)
                                      & (Document.collection_id == Chunk.collection_id)).where(
                Chunk.collection_id == collection_id, Document.status == "indexed")
            vector_rows = session.scalars(base.order_by(Chunk.embedding.cosine_distance(vector))
                                           .limit(self.settings.vector_top_k)).all()
            document = func.to_tsvector("english", Chunk.content)
            query = func.plainto_tsquery("english", question)
            keyword_rows = session.scalars(base.where(document.op("@@")(query))
                                            .order_by(func.ts_rank_cd(document, query).desc())
                                            .limit(self.settings.keyword_top_k)).all()
            def convert(rows: list[Chunk]) -> list[Evidence]:
                return [Evidence(chunk_id=row.chunk_id, collection_id=row.collection_id,
                                 document_id=row.document_id, filename=row.filename,
                                 page=row.page, heading=row.heading, content=row.content) for row in rows]
            return reciprocal_rank_fusion([convert(vector_rows), convert(keyword_rows)], self.settings.rerank_top_k)
