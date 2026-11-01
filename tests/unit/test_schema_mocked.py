"""Mocked unit tests for database configuration and schema initialization/inspection (F-08 Part 2).

Verifies:
- Schema initialization behavior on empty/existing database.
- Schema inspection returns expected table sets and flags.
- Missing database configuration handling.
- Invalid database connection and error redaction without leaking credentials.
- No live MySQL server required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
from sqlalchemy.exc import OperationalError

from learnpath_ai.database import Database, DatabaseError
from learnpath_ai.schema import (
    EXPECTED_TABLES,
    SCHEMA_VERSION,
    InitResult,
    SchemaError,
    SchemaStatus,
    initialize_schema,
    inspect_schema,
)
from learnpath_ai.settings import SettingsError, from_environ


def _mock_settings():
    return from_environ(
        {
            "MYSQL_HOST": "127.0.0.1",
            "MYSQL_PORT": "3306",
            "MYSQL_USER": "learnpath_app",
            "MYSQL_PASSWORD": "super-secret-password-12345",
            "MYSQL_DATABASE": "learnpath_ai",
        }
    )


def test_missing_database_configuration_raises_settings_error(monkeypatch):
    """Missing required MySQL environment variables raises SettingsError with variable names."""
    monkeypatch.delenv("MYSQL_HOST", raising=False)
    monkeypatch.delenv("MYSQL_PORT", raising=False)
    monkeypatch.delenv("MYSQL_USER", raising=False)
    monkeypatch.delenv("MYSQL_PASSWORD", raising=False)
    monkeypatch.delenv("MYSQL_DATABASE", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)

    with pytest.raises(SettingsError) as exc_info:
        Database.from_env()
    assert "MYSQL_HOST" in str(exc_info.value)


def test_invalid_database_connection_redacts_credentials():
    """OperationalError during connection redacts the secret password from error messages."""
    db = Database(_mock_settings())

    class FailingEngine:
        def connect(self):
            raise OperationalError(
                "SELECT 1",
                {},
                Exception("Access denied for user 'learnpath_app'@'127.0.0.1' (using password: YES) super-secret-password-12345"),
            )

    db._engine = FailingEngine()  # type: ignore[assignment]
    db._session_factory = object()  # type: ignore[assignment]

    with pytest.raises(DatabaseError) as exc_info:
        db.ping()

    err_msg = str(exc_info.value)
    assert "super-secret-password-12345" not in err_msg
    assert "learnpath_app" in err_msg


def test_mocked_schema_inspection_detects_tables():
    """inspect_schema queries database existence and reports present tables."""
    db = Database(_mock_settings())

    fake_conn = MagicMock()
    fake_conn.execute.return_value.first.return_value = (SCHEMA_VERSION,)
    fake_engine = MagicMock()
    fake_engine.connect.return_value.__enter__.return_value = fake_conn

    with patch.object(db, "connect_without_database", return_value=MagicMock()), \
         patch.object(db, "engine", return_value=fake_engine), \
         patch("learnpath_ai.schema.database_exists", return_value=True), \
         patch("learnpath_ai.schema.existing_tables", return_value=list(EXPECTED_TABLES)):

        status = inspect_schema(db)
        assert status.database_exists is True
        assert status.missing_tables == []
        assert status.schema_version == SCHEMA_VERSION
        assert status.compatible is True


def test_mocked_schema_initialization_when_tables_already_present():
    """initialize_schema reports 'already_present' when compatible tables already exist."""
    db = Database(_mock_settings())

    fake_status = SchemaStatus(
        database="learnpath_ai",
        database_exists=True,
        existing_tables=list(EXPECTED_TABLES),
        missing_tables=[],
        unexpected_tables=[],
        schema_version=SCHEMA_VERSION,
        compatible=True,
    )

    with patch("learnpath_ai.schema.inspect_schema", return_value=fake_status):
        result = initialize_schema(db)
        assert result.action == "already_present"
        assert result.created_tables == []


def test_mocked_schema_initialization_fails_if_database_missing():
    """initialize_schema raises SchemaError if target database does not exist."""
    db = Database(_mock_settings())

    fake_status = SchemaStatus(
        database="learnpath_ai",
        database_exists=False,
        messages=["Database 'learnpath_ai' does not exist."],
    )

    with patch("learnpath_ai.schema.inspect_schema", return_value=fake_status):
        with pytest.raises(SchemaError, match="does not exist"):
            initialize_schema(db)
