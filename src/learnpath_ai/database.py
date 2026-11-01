"""SQLAlchemy 2.x + PyMySQL connection helpers.

No engine is created at import time. Passwords are never printed.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session, sessionmaker

from learnpath_ai.settings import (
    DatabaseSettings,
    SettingsError,
    build_sqlalchemy_url,
    load_database_settings,
    redact_secret,
)

__all__ = [
    "Database",
    "DatabaseError",
    "DatabaseSettings",
    "SettingsError",
    "build_sqlalchemy_url",
    "load_database_settings",
    "redact_url",
]


def redact_url(url: str) -> str:
    from learnpath_ai.settings import redact_url as _redact_url

    return _redact_url(url)


class DatabaseError(RuntimeError):
    """Connection or transaction failure with secrets stripped from the message."""


@dataclass(frozen=True)
class PingResult:
    server_version: str
    current_database: str | None
    user: str
    host: str
    port: int


class Database:
    """Lazy engine and session factory bound to one settings object."""

    def __init__(self, settings: DatabaseSettings) -> None:
        self.settings = settings
        self._engine: Engine | None = None
        self._session_factory: sessionmaker[Session] | None = None

    @classmethod
    def from_env(cls, **kwargs) -> Database:
        return cls(load_database_settings(**kwargs))

    def engine(self) -> Engine:
        if self._engine is None:
            url = build_sqlalchemy_url(self.settings, include_database=True)
            try:
                self._engine = create_engine(
                    url,
                    pool_pre_ping=True,
                    pool_recycle=1800,
                    future=True,
                )
            except SQLAlchemyError as exc:
                raise self._wrap(exc, "Failed to create the database engine.") from None
            self._session_factory = sessionmaker(
                bind=self._engine,
                autoflush=False,
                autocommit=False,
                expire_on_commit=False,
                class_=Session,
            )
        return self._engine

    def session(self) -> Session:
        if self._session_factory is None:
            self.engine()
        assert self._session_factory is not None
        return self._session_factory()

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        """Commit on success, rollback on exception, always close the session."""
        session = self.session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def ping(self) -> PingResult:
        try:
            with self.engine().connect() as conn:
                version = conn.execute(text("SELECT VERSION()")).scalar_one()
                current_db = conn.execute(text("SELECT DATABASE()")).scalar_one()
        except SQLAlchemyError as exc:
            raise self._wrap(
                exc,
                "Failed to connect to MySQL at "
                f"{self.settings.host}:{self.settings.port} as {self.settings.user} "
                f"(database={self.settings.database}).",
            ) from None
        return PingResult(
            server_version=str(version),
            current_database=str(current_db) if current_db is not None else None,
            user=self.settings.user,
            host=self.settings.host,
            port=self.settings.port,
        )

    def connect_without_database(self) -> Engine:
        """Engine bound to the server only, used to verify the target database exists."""
        url = build_sqlalchemy_url(self.settings, include_database=False)
        try:
            return create_engine(url, pool_pre_ping=True, future=True)
        except SQLAlchemyError as exc:
            raise self._wrap(exc, "Failed to create a server-level MySQL engine.") from None

    def dispose(self) -> None:
        if self._engine is not None:
            self._engine.dispose()
            self._engine = None
            self._session_factory = None

    def _wrap(self, exc: BaseException, prefix: str) -> DatabaseError:
        detail = redact_secret(str(exc), self.settings)
        return DatabaseError(f"{prefix} {detail}".strip())
