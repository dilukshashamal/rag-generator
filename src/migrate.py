"""Versioned bootstrap migration, serialized across concurrent starts."""

from sqlalchemy import text

from src.config import get_settings
from src.database import Database
from src.models.entities import Base


def migrate() -> None:
    database = Database(get_settings())
    with database.engine.begin() as connection:
        connection.execute(text("SELECT pg_advisory_xact_lock(81002001)"))
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.execute(text("CREATE TABLE IF NOT EXISTS schema_versions (version integer PRIMARY KEY)"))
        if not connection.execute(text("SELECT 1 FROM schema_versions WHERE version = 1")).scalar():
            Base.metadata.create_all(connection)
            connection.execute(text("CREATE INDEX ix_chunks_fts ON chunks USING gin (to_tsvector('english', content))"))
            connection.execute(text("INSERT INTO schema_versions VALUES (1)"))
    database.engine.dispose()


if __name__ == "__main__":
    migrate()
