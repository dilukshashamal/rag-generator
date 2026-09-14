from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine
from sqlalchemy.orm import Session, sessionmaker

from src.config import Settings


class Database:
    """Own connection pooling and short transaction boundaries."""

    def __init__(self, settings: Settings) -> None:
        self.engine: Engine = create_engine(settings.database_url.get_secret_value(),
                                            pool_pre_ping=True, pool_size=5, max_overflow=5,
                                            connect_args={"connect_timeout": 5})
        self.sessions = sessionmaker(self.engine, expire_on_commit=False)

    @contextmanager
    def session(self) -> Iterator[Session]:
        with self.sessions.begin() as session:
            yield session
