from __future__ import annotations

import pytest

from learnpath_ai.settings import (
    SettingsError,
    build_sqlalchemy_url,
    from_environ,
    redact_secret,
    redact_url,
)


def test_missing_required_variables() -> None:
    with pytest.raises(SettingsError, match="MYSQL_HOST"):
        from_environ(
            {
                "MYSQL_PORT": "3306",
                "MYSQL_USER": "learnpath_app",
                "MYSQL_PASSWORD": "secret",
                "MYSQL_DATABASE": "learnpath_ai",
            }
        )


def test_missing_password_key() -> None:
    with pytest.raises(SettingsError, match="MYSQL_PASSWORD"):
        from_environ(
            {
                "MYSQL_HOST": "127.0.0.1",
                "MYSQL_PORT": "3306",
                "MYSQL_USER": "learnpath_app",
                "MYSQL_DATABASE": "learnpath_ai",
            }
        )


def test_invalid_port() -> None:
    with pytest.raises(SettingsError, match="MYSQL_PORT"):
        from_environ(
            {
                "MYSQL_HOST": "127.0.0.1",
                "MYSQL_PORT": "abc",
                "MYSQL_USER": "learnpath_app",
                "MYSQL_PASSWORD": "secret",
                "MYSQL_DATABASE": "learnpath_ai",
            }
        )


def test_sqlite_database_url_rejected() -> None:
    with pytest.raises(SettingsError, match="SQLite"):
        from_environ({"DATABASE_URL": "sqlite:///learnpath.db"})


def test_non_pymysql_mysql_url_rejected() -> None:
    with pytest.raises(SettingsError, match="pymysql"):
        from_environ({"DATABASE_URL": "mysql://user:secret@127.0.0.1:3306/learnpath_ai"})


def test_build_url_encodes_special_password() -> None:
    settings = from_environ(
        {
            "MYSQL_HOST": "127.0.0.1",
            "MYSQL_PORT": "3306",
            "MYSQL_USER": "learnpath_app",
            "MYSQL_PASSWORD": "p@ss/w:rd",
            "MYSQL_DATABASE": "learnpath_ai",
        }
    )
    url = build_sqlalchemy_url(settings)
    assert url.startswith("mysql+pymysql://")
    assert "p@ss/w:rd" not in url
    assert "p%40ss%2Fw%3Ard" in url
    assert "learnpath_ai" in url
    assert "charset=utf8mb4" in url


def test_redact_url_hides_password() -> None:
    url = "mysql+pymysql://learnpath_app:super-secret@127.0.0.1:3306/learnpath_ai"
    redacted = redact_url(url)
    assert "super-secret" not in redacted
    assert "learnpath_app:***@" in redacted


def test_redact_secret_strips_raw_and_encoded_password() -> None:
    settings = from_environ(
        {
            "MYSQL_HOST": "127.0.0.1",
            "MYSQL_PORT": "3306",
            "MYSQL_USER": "learnpath_app",
            "MYSQL_PASSWORD": "p@ss",
            "MYSQL_DATABASE": "learnpath_ai",
        }
    )
    url = build_sqlalchemy_url(settings)
    message = f"failed connecting with {url} password=p@ss"
    redacted = redact_secret(message, settings)
    assert "p@ss" not in redacted
    assert "p%40ss" not in redacted
