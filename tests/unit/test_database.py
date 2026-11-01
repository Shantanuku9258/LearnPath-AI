from __future__ import annotations

import pytest
from sqlalchemy.exc import OperationalError

from learnpath_ai.database import Database, DatabaseError
from learnpath_ai.settings import from_environ


def _settings():
    return from_environ(
        {
            "MYSQL_HOST": "127.0.0.1",
            "MYSQL_PORT": "3306",
            "MYSQL_USER": "learnpath_app",
            "MYSQL_PASSWORD": "unit-test-secret",
            "MYSQL_DATABASE": "learnpath_ai",
        }
    )


def test_database_does_not_connect_on_init() -> None:
    db = Database(_settings())
    assert db._engine is None


def test_ping_wraps_operational_error_without_password() -> None:
    db = Database(_settings())

    class FakeEngine:
        def connect(self):
            raise OperationalError(
                "SELECT VERSION()",
                {},
                Exception(
                    "Can't connect to MySQL server on '127.0.0.1' "
                    "(using password: YES) unit-test-secret"
                ),
            )

    db._engine = FakeEngine()  # type: ignore[assignment]
    db._session_factory = object()  # type: ignore[assignment]

    with pytest.raises(DatabaseError) as exc_info:
        db.ping()
    message = str(exc_info.value)
    assert "unit-test-secret" not in message
    assert "127.0.0.1" in message
    assert "learnpath_app" in message
    assert "learnpath_ai" in message


def test_transaction_commits_on_success() -> None:
    db = Database(_settings())
    session = _FakeSession()
    db._engine = object()  # type: ignore[assignment]
    db._session_factory = lambda: session  # type: ignore[assignment]

    with db.transaction() as txn:
        assert txn is session
    assert session.commits == 1
    assert session.rollbacks == 0
    assert session.closed == 1


def test_transaction_rolls_back_on_error() -> None:
    db = Database(_settings())
    session = _FakeSession()
    db._engine = object()  # type: ignore[assignment]
    db._session_factory = lambda: session  # type: ignore[assignment]

    with pytest.raises(RuntimeError, match="boom"):
        with db.transaction():
            raise RuntimeError("boom")
    assert session.commits == 0
    assert session.rollbacks == 1
    assert session.closed == 1


class _FakeSession:
    def __init__(self) -> None:
        self.commits = 0
        self.rollbacks = 0
        self.closed = 0

    def commit(self) -> None:
        self.commits += 1

    def rollback(self) -> None:
        self.rollbacks += 1

    def close(self) -> None:
        self.closed += 1
