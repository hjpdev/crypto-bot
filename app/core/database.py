from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker, scoped_session
from sqlalchemy.exc import SQLAlchemyError, DBAPIError, DisconnectionError
from sqlalchemy.pool import QueuePool
from typing import Optional, Callable, Generator, Any
import os
import time
import random
from contextlib import contextmanager
from dotenv import load_dotenv

from app.core.exceptions import DatabaseError, DatabaseConnectionError
from app.utils.logger import logger

load_dotenv()

Base = declarative_base()

DEFAULT_DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/crypto_trader"
)

MAX_RETRIES = int(os.environ.get("DB_MAX_RETRIES", "5"))
RETRY_DELAY = float(os.environ.get("DB_RETRY_DELAY", "0.5"))
MAX_POOL_SIZE = int(os.environ.get("DB_MAX_POOL_SIZE", "20"))
POOL_OVERFLOW = int(os.environ.get("DB_POOL_OVERFLOW", "10"))
POOL_TIMEOUT = int(os.environ.get("DB_POOL_TIMEOUT", "30"))
POOL_RECYCLE = int(os.environ.get("DB_POOL_RECYCLE", "1800"))  # 30 minutes


class Database:
    """
    Attributes:
        database_url (str): The connection URL for the database
        engine: SQLAlchemy engine instance
        SessionLocal: SQLAlchemy sessionmaker instance
    """

    def __init__(self, database_url: Optional[str] = None):
        self.database_url = database_url or os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)
        self.engine = None
        self.SessionLocal = None
        self._setup_engine()

    def _setup_engine(self) -> None:
        try:
            logger.info("Setting up database engine with connection pooling")
            self.engine = create_engine(
                self.database_url,
                poolclass=QueuePool,
                pool_size=MAX_POOL_SIZE,
                max_overflow=POOL_OVERFLOW,
                pool_timeout=POOL_TIMEOUT,
                pool_recycle=POOL_RECYCLE,
                pool_pre_ping=True,  # Verify connection before usage
            )

            event.listen(self.engine, "connect", self._on_connect)
            event.listen(self.engine, "checkout", self._on_checkout)
            event.listen(self.engine, "checkin", self._on_checkin)
            event.listen(self.engine, "engine_connect", self._on_engine_connect)

            self.SessionLocal = scoped_session(
                sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
            )

            logger.info("Database engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to set up database engine: {str(e)}")
            raise DatabaseConnectionError(f"Failed to set up database engine: {str(e)}") from e

    def _on_connect(self, dbapi_connection, connection_record):
        """Event listener called when a connection is created."""
        logger.debug(f"Database connection established: {connection_record}")

    def _on_checkout(self, dbapi_connection, connection_record, connection_proxy):
        """Event listener called when a connection is checked out from the pool."""
        logger.debug(f"Database connection checked out: {connection_record}")

    def _on_checkin(self, dbapi_connection, connection_record):
        """Event listener called when a connection is checked back into the pool."""
        logger.debug(f"Database connection checked in: {connection_record}")

    def _on_engine_connect(self, connection):
        """Event listener called when engine connects."""
        logger.debug("Database engine connection event triggered")

    def create_all_tables(self) -> None:
        """
        Create all tables defined in SQLAlchemy models if they don't exist.
        """
        try:
            logger.info("Creating database tables if they don't exist")
            Base.metadata.create_all(bind=self.engine)
            logger.info("Database tables created successfully")
        except SQLAlchemyError as e:
            logger.error(f"Failed to create database tables: {str(e)}")
            raise DatabaseError(f"Failed to create database tables: {str(e)}") from e

    def get_engine(self):
        return self.engine

    def get_session(self):
        return self.SessionLocal()

    @contextmanager
    def session_scope(self):
        session = self.get_session()

        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Database operation failed: {str(e)}")
            raise
        finally:
            session.close()

    def execute_with_retry(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute a database operation with retry logic for temporary connection failures."""
        retries = 0
        last_error = None

        while retries <= MAX_RETRIES:
            try:
                if retries > 0:
                    logger.info(f"Retry attempt {retries} for database operation")
                return operation(*args, **kwargs)
            except (SQLAlchemyError, DBAPIError, DisconnectionError) as e:
                last_error = e
                retries += 1

                if retries <= MAX_RETRIES:
                    # Exponential backoff with jitter
                    delay = RETRY_DELAY * (2 ** (retries - 1)) + random.uniform(0, 0.1)
                    logger.warning(
                        f"Database connection error: {str(e)}. Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)

                    # Try to reconnect if necessary
                    if isinstance(e, DisconnectionError) or "connection" in str(e).lower():
                        try:
                            logger.info("Attempting to refresh database engine")
                            self._setup_engine()
                        except Exception as refresh_err:
                            logger.error(f"Failed to refresh database engine: {str(refresh_err)}")
                else:
                    logger.error(f"Database operation failed after {MAX_RETRIES} retry attempts")
                    raise DatabaseConnectionError(
                        f"Database operation failed after {MAX_RETRIES} attempts: {str(last_error)}"
                    ) from last_error
            except Exception as e:
                # For non-connection related exceptions, don't retry
                logger.error(f"Database operation error: {str(e)}")
                raise

    def dispose(self):
        """
        Close all connection pool connections and dispose of the engine.
        """
        if self.engine:
            logger.info("Disposing database engine and connection pool")
            self.engine.dispose()
            logger.info("Database engine disposed")

    def create_database(self) -> None:
        """
        Create all tables defined in SQLAlchemy models if they don't exist.
        Alias for create_all_tables for backward compatibility.
        """
        return self.create_all_tables()


db = Database()


def get_engine():
    """Get the global database engine."""
    return db.get_engine()


def get_db() -> Generator:
    """Get a database session. Intended for use as a dependency in FastAPI endpoints."""
    session = db.get_session()
    try:
        yield session
    finally:
        session.close()


def create_all_tables():
    db.create_all_tables()
