import os
import pytest
from unittest.mock import patch, MagicMock

from sqlalchemy import Column, Integer, String, inspect
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import Session

from app.core.database import Database, Base
from app.core.exceptions import DatabaseConnectionError, DatabaseError


class TestModel(Base):
    """Test model for database operations."""
    __tablename__ = "test_model"

    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)


class TestDatabase:
    """Tests for the Database class."""

    @pytest.fixture
    def sqlite_db_url(self):
        """Fixture to provide an in-memory SQLite database URL."""
        return "sqlite:///:memory:"

    @pytest.fixture
    def test_db(self, sqlite_db_url):
        """Fixture to create a test Database instance with in-memory SQLite."""
        db = Database(sqlite_db_url)
        # Create the test table
        TestModel.__table__.create(db.engine)
        yield db
        db.dispose()

    def test_init_creates_engine(self, sqlite_db_url):
        """Test that initializing Database creates an engine."""
        db = Database(sqlite_db_url)
        assert db.engine is not None
        assert db.SessionLocal is not None
        db.dispose()

    def test_get_engine(self, test_db):
        """Test get_engine returns the correct engine."""
        engine = test_db.get_engine()
        assert engine is test_db.engine

    def test_get_session(self, test_db):
        """Test get_session returns a valid session."""
        session = test_db.get_session()
        assert isinstance(session, Session)
        session.close()

    def test_create_all_tables(self, sqlite_db_url):
        """Test create_all_tables creates tables."""
        # Create a new database with no tables
        db = Database(sqlite_db_url)

        # Create a new model dynamically
        class DynamicModel(Base):
            __tablename__ = "dynamic_model"
            id = Column(Integer, primary_key=True)
            value = Column(String(50))

        # Create all tables including our dynamic model
        db.create_all_tables()

        # Verify table exists
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        assert "dynamic_model" in tables

        db.dispose()

    def test_session_scope(self, test_db):
        """Test session_scope properly manages sessions."""
        # Insert test data
        with test_db.session_scope() as session:
            test_item = TestModel(name="test_item")
            session.add(test_item)

        # Verify data was committed
        with test_db.session_scope() as session:
            result = session.query(TestModel).filter_by(name="test_item").first()
            assert result is not None
            assert result.name == "test_item"

    def test_session_scope_rolls_back_on_error(self, test_db):
        """Test session_scope rolls back on error."""
        # Count existing items
        with test_db.session_scope() as session:
            initial_count = session.query(TestModel).count()

        # Try to insert data but raise an exception
        try:
            with test_db.session_scope() as session:
                test_item = TestModel(name="rollback_test")
                session.add(test_item)
                raise ValueError("Test exception to trigger rollback")
        except ValueError:
            pass

        # Verify data was rolled back
        with test_db.session_scope() as session:
            new_count = session.query(TestModel).count()
            assert new_count == initial_count

    @patch('time.sleep', return_value=None)  # Don't actually sleep in tests
    def test_execute_with_retry_success(self, mock_sleep, test_db):
        """Test execute_with_retry succeeds with a valid operation."""
        def operation():
            with test_db.session_scope() as session:
                return session.query(TestModel).count()

        result = test_db.execute_with_retry(operation)
        assert isinstance(result, int)
        assert not mock_sleep.called  # Sleep should not be called for successful operations

    @patch('time.sleep', return_value=None)
    def test_execute_with_retry_temporary_failure(self, mock_sleep, test_db):
        """Test execute_with_retry handles temporary failures."""

        # Define a mock function that fails twice and then succeeds
        mock_operation = MagicMock(side_effect=[
            OperationalError("statement", "params", "orig"),
            OperationalError("statement", "params", "orig"),
            5  # Success on third attempt, returning 5
        ])

        result = test_db.execute_with_retry(mock_operation)

        assert result == 5
        assert mock_operation.call_count == 3
        assert mock_sleep.call_count == 2  # Should sleep twice (after first and second failures)

    @patch('time.sleep', return_value=None)
    def test_execute_with_retry_permanent_failure(self, mock_sleep, test_db):
        """Test execute_with_retry raises DatabaseConnectionError after max retries."""

        # Configure mock to always raise an OperationalError
        mock_operation = MagicMock(side_effect=OperationalError("statement", "params", "orig"))

        # Override MAX_RETRIES for faster test execution
        with patch('app.core.database.MAX_RETRIES', 3):
            with pytest.raises(DatabaseConnectionError):
                test_db.execute_with_retry(mock_operation)

            assert mock_operation.call_count == 4  # Initial + 3 retries
            assert mock_sleep.call_count == 3  # Sleep after each retry

    def test_create_all_tables_error_handling(self, sqlite_db_url):
        """Test error handling in create_all_tables."""
        db = Database(sqlite_db_url)

        # Mock the metadata.create_all to raise an exception
        with patch('sqlalchemy.MetaData.create_all', side_effect=SQLAlchemyError("Test error")):
            with pytest.raises(DatabaseError):
                db.create_all_tables()

        db.dispose()

    @patch('app.core.database.create_engine')
    def test_setup_engine_error_handling(self, mock_create_engine):
        """Test error handling in _setup_engine."""
        mock_create_engine.side_effect = Exception("Test engine creation error")

        with pytest.raises(DatabaseConnectionError):
            Database("sqlite:///:memory:")

    def test_connection_pool_settings(self, sqlite_db_url):
        """Test that connection pool settings are applied."""
        with patch('app.core.database.MAX_POOL_SIZE', 5), \
             patch('app.core.database.POOL_OVERFLOW', 3), \
             patch('app.core.database.POOL_RECYCLE', 60):

            db = Database(sqlite_db_url)

            # For SQLite, pool settings might not apply the same way as PostgreSQL,
            # so we just verify the connection was successful
            assert db.engine is not None
            assert db.SessionLocal is not None

            db.dispose()


@pytest.mark.integration
class TestDatabaseIntegration:
    """Integration tests for the Database class with a real PostgreSQL database."""

    @pytest.fixture
    def postgres_db_url(self):
        """
        Fixture to provide a real PostgreSQL database URL.

        This should be run only in environments where a PostgreSQL database is available.
        Typically used in CI/CD pipelines with a dedicated test database.
        """
        db_url = os.environ.get("TEST_DATABASE_URL")
        if not db_url:
            pytest.skip("TEST_DATABASE_URL environment variable not set")
        return db_url

    @pytest.fixture
    def postgres_db(self, postgres_db_url):
        """Fixture to create a test Database instance with PostgreSQL."""
        db = Database(postgres_db_url)

        # Create the test table
        Base.metadata.create_all(db.engine)

        yield db

        # Clean up the test table
        TestModel.__table__.drop(db.engine)
        db.dispose()

    @pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"),
                        reason="TEST_DATABASE_URL environment variable not set")
    def test_postgres_connection(self, postgres_db):
        """Test connection to a real PostgreSQL database."""
        with postgres_db.session_scope() as session:
            # Simple database operation to verify connection
            session.execute("SELECT 1")

    @pytest.mark.skipif(not os.environ.get("TEST_DATABASE_URL"),
                        reason="TEST_DATABASE_URL environment variable not set")
    def test_postgres_crud_operations(self, postgres_db):
        """Test CRUD operations on a real PostgreSQL database."""
        # Create
        with postgres_db.session_scope() as session:
            test_item = TestModel(name="integration_test")
            session.add(test_item)

        # Read
        with postgres_db.session_scope() as session:
            result = session.query(TestModel).filter_by(name="integration_test").first()
            assert result is not None
            assert result.name == "integration_test"

            # Update
            result.name = "updated_integration_test"

        # Verify Update
        with postgres_db.session_scope() as session:
            updated = session.query(TestModel).filter_by(name="updated_integration_test").first()
            assert updated is not None

            # Delete
            session.delete(updated)

        # Verify Delete
        with postgres_db.session_scope() as session:
            deleted = session.query(TestModel).filter_by(name="updated_integration_test").first()
            assert deleted is None
