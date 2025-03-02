from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from typing import Optional, Callable
import os
from dotenv import load_dotenv

load_dotenv()

Base = declarative_base()

DEFAULT_DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/crypto_trader"
)


class Database:
    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
        self.engine = None
        self.SessionLocal = None
        self._setup_engine()

    def _setup_engine(self) -> None:
        self.engine = create_engine(self.database_url, pool_pre_ping=True)

        self.SessionLocal = scoped_session(
            sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        )

    def create_database(self) -> None:
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> scoped_session:
        return self.SessionLocal()

    def session_scope(self) -> Callable:
        from contextlib import contextmanager

        @contextmanager
        def session_scope_cm():
            session = self.get_session()
            try:
                yield session
                session.commit()
            except Exception:
                session.rollback()
                raise
            finally:
                session.close()

        return session_scope_cm


db = Database()


def get_db():
    session = db.get_session()
    try:
        yield session
    finally:
        session.close()
