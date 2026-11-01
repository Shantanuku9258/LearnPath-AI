"""Environment-based settings. Does not open database connections."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote_plus, urlparse, unquote

from dotenv import load_dotenv

REQUIRED_MYSQL_VARS = (
    "MYSQL_HOST",
    "MYSQL_PORT",
    "MYSQL_USER",
    "MYSQL_PASSWORD",
    "MYSQL_DATABASE",
)

FORBIDDEN_URL_SCHEMES = ("sqlite", "sqlite+pysqlite", "sqlite+aiosqlite")


class SettingsError(ValueError):
    """Invalid or incomplete database configuration."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_env_file(dotenv_path: Path | None = None, *, environ: dict[str, str] | None = None) -> None:
    """Load `.env` into os.environ. Safe to call multiple times. Does not connect."""
    path = dotenv_path or (project_root() / ".env")
    if environ is None:
        load_dotenv(path, override=False)
        return
    if path.is_file():
        load_dotenv(path, override=False)


@dataclass(frozen=True)
class DatabaseSettings:
    host: str
    port: int
    user: str
    password: str
    database: str
    database_url: str | None = None

    @property
    def source(self) -> str:
        return "DATABASE_URL" if self.database_url else "MYSQL_*"


def from_environ(environ: dict[str, str] | os._Environ[str] | None = None) -> DatabaseSettings:
    """Build settings from the given mapping. Does not read files or connect."""
    env = dict(os.environ if environ is None else environ)
    raw_url = (env.get("DATABASE_URL") or "").strip()
    if raw_url:
        return _from_url(raw_url)

    missing: list[str] = []
    for name in REQUIRED_MYSQL_VARS:
        if name == "MYSQL_PASSWORD":
            if name not in env:
                missing.append(name)
            continue
        if not str(env.get(name, "")).strip():
            missing.append(name)
    if missing:
        raise SettingsError(
            "Missing required database environment variables: "
            + ", ".join(missing)
            + ". Copy .env.example to .env and fill in local values."
        )

    port_raw = str(env["MYSQL_PORT"]).strip()
    try:
        port = int(port_raw)
    except ValueError as exc:
        raise SettingsError(f"MYSQL_PORT must be an integer, got {port_raw!r}.") from exc
    if not (1 <= port <= 65535):
        raise SettingsError(f"MYSQL_PORT must be between 1 and 65535, got {port}.")

    database = str(env["MYSQL_DATABASE"]).strip()
    _reject_sqlite_name(database)

    return DatabaseSettings(
        host=str(env["MYSQL_HOST"]).strip(),
        port=port,
        user=str(env["MYSQL_USER"]).strip(),
        password=str(env.get("MYSQL_PASSWORD", "")),
        database=database,
        database_url=None,
    )


def load_database_settings(
    *,
    dotenv_path: Path | None = None,
    environ: dict[str, str] | None = None,
    load_dotenv_file: bool = True,
) -> DatabaseSettings:
    if load_dotenv_file and environ is None:
        load_env_file(dotenv_path)
    return from_environ(environ)


def _from_url(raw_url: str) -> DatabaseSettings:
    parsed = urlparse(raw_url)
    scheme = (parsed.scheme or "").lower()
    if scheme in FORBIDDEN_URL_SCHEMES or scheme.startswith("sqlite"):
        raise SettingsError(
            "SQLite is not the primary database. Use mysql+pymysql:// credentials instead."
        )
    if not scheme.startswith("mysql"):
        raise SettingsError(
            f"Unsupported DATABASE_URL scheme {scheme!r}. Expected mysql+pymysql."
        )
    if "pymysql" not in scheme:
        raise SettingsError("DATABASE_URL must use the mysql+pymysql driver.")
    if not parsed.hostname:
        raise SettingsError("DATABASE_URL is missing a host.")
    database = (parsed.path or "").lstrip("/").split("?")[0].strip()
    if not database:
        raise SettingsError("DATABASE_URL is missing a database name.")
    _reject_sqlite_name(database)
    port = parsed.port or 3306
    user = unquote(parsed.username or "")
    password = unquote(parsed.password or "")
    if not user:
        raise SettingsError("DATABASE_URL is missing a user.")
    return DatabaseSettings(
        host=parsed.hostname,
        port=port,
        user=user,
        password=password,
        database=database,
        database_url=raw_url,
    )


def _reject_sqlite_name(database: str) -> None:
    lowered = database.lower()
    if lowered.endswith(".db") or lowered.endswith(".sqlite") or lowered.endswith(".sqlite3"):
        raise SettingsError(
            "Database name looks like a SQLite file. MySQL 8.4 is required."
        )


def build_sqlalchemy_url(settings: DatabaseSettings, *, include_database: bool = True) -> str:
    """Construct a SQLAlchemy URL. Never log the return value without redaction."""
    if settings.database_url and include_database:
        return settings.database_url
    user = quote_plus(settings.user)
    password = quote_plus(settings.password)
    host = settings.host
    port = settings.port
    path = quote_plus(settings.database) if include_database else ""
    return f"mysql+pymysql://{user}:{password}@{host}:{port}/{path}?charset=utf8mb4"


def redact_secret(text: str, settings: DatabaseSettings | None = None) -> str:
    """Remove passwords and credential URLs from a string."""
    redacted = text
    redacted = _redact_url_passwords(redacted)
    if settings and settings.password:
        if settings.password:
            redacted = redacted.replace(settings.password, "***")
        encoded = quote_plus(settings.password)
        if encoded and encoded != "***":
            redacted = redacted.replace(encoded, "***")
    return redacted


def redact_url(url: str) -> str:
    return _redact_url_passwords(url)


def _redact_url_passwords(text: str) -> str:
    # mysql+pymysql://user:password@host
    import re

    return re.sub(
        r"(mysql(?:\+[a-z0-9]+)?://[^:/?#\s]+):([^@/\s]+)@",
        r"\1:***@",
        text,
        flags=re.IGNORECASE,
    )
