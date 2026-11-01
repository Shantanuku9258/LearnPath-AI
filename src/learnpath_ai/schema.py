"""Schema file parsing, expected-object inventory, and safe initialization.

Never CREATE DATABASE, DROP DATABASE, or DROP TABLE.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import inspect, text
from sqlalchemy.engine import Connection, Engine
from sqlalchemy.exc import SQLAlchemyError

from learnpath_ai.database import Database, DatabaseError
from learnpath_ai.settings import project_root, redact_secret

SCHEMA_VERSION = "1.1.0"
SCHEMA_COMPONENT = "learnpath_ai"
FORBIDDEN_DDL = re.compile(
    r"^\s*(DROP\s+DATABASE|DROP\s+TABLE|CREATE\s+DATABASE|TRUNCATE)\b",
    re.IGNORECASE,
)

# Creation order matches schemas/schema.sql
EXPECTED_TABLES: tuple[str, ...] = (
    "schema_meta",
    "course",
    "learner",
    "skill",
    "skill_prerequisite",
    "content_item",
    "content_skill",
    "learning_path",
    "recommendation",
    "learning_path_item",
    "attempt",
    "assessment_result",
    "mastery_snapshot",
    "recommendation_outcome",
    "event_history",
    "data_quality_issue",
)

MANDATORY_ENTITY_TABLES: tuple[str, ...] = (
    "learner",
    "course",
    "skill",
    "skill_prerequisite",
    "content_item",
    "content_skill",
    "attempt",
    "assessment_result",
    "learning_path",
    "recommendation",
)


class SchemaError(DatabaseError):
    """Schema is missing, incomplete, or incompatible. No destructive repair is attempted."""


@dataclass
class SchemaStatus:
    database: str
    database_exists: bool
    existing_tables: list[str] = field(default_factory=list)
    missing_tables: list[str] = field(default_factory=list)
    unexpected_tables: list[str] = field(default_factory=list)
    schema_version: str | None = None
    compatible: bool = False
    messages: list[str] = field(default_factory=list)


@dataclass
class InitResult:
    action: str
    status: SchemaStatus
    created_tables: list[str] = field(default_factory=list)


def default_schema_path() -> Path:
    return project_root() / "schemas" / "schema.sql"


def schema_checksum(sql_text: str) -> str:
    canonical = "\n".join(line.rstrip() for line in sql_text.replace("\r\n", "\n").split("\n"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def load_schema_sql(path: Path | None = None) -> str:
    schema_path = path or default_schema_path()
    return schema_path.read_text(encoding="utf-8")


def parse_sql_statements(sql_text: str) -> list[str]:
    """Split a MySQL script into executable statements. Rejects destructive DDL."""
    without_line_comments = []
    for line in sql_text.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("--"):
            continue
        without_line_comments.append(line)
    body = "\n".join(without_line_comments)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.DOTALL)

    statements: list[str] = []
    for chunk in body.split(";"):
        stmt = chunk.strip()
        if not stmt:
            continue
        if FORBIDDEN_DDL.search(stmt):
            raise SchemaError(
                "Refusing to execute destructive or database-creating SQL: "
                f"{stmt.splitlines()[0][:80]}"
            )
        if re.match(r"^USE\s+", stmt, re.IGNORECASE):
            continue
        statements.append(stmt)
    return statements


def list_create_table_names(sql_text: str) -> list[str]:
    names: list[str] = []
    for stmt in parse_sql_statements(sql_text):
        match = re.match(r"CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+`?([A-Za-z0-9_]+)`?", stmt, re.I)
        if match:
            names.append(match.group(1).lower())
    return names


def database_exists(engine: Engine, database: str) -> bool:
    with engine.connect() as conn:
        row = conn.execute(
            text(
                "SELECT SCHEMA_NAME FROM INFORMATION_SCHEMA.SCHEMATA "
                "WHERE SCHEMA_NAME = :name"
            ),
            {"name": database},
        ).first()
    return row is not None


def existing_tables(conn: Connection, database: str) -> list[str]:
    inspector = inspect(conn)
    names = inspector.get_table_names(schema=database)
    return sorted(name.lower() for name in names)


def inspect_schema(db: Database) -> SchemaStatus:
    settings = db.settings
    status = SchemaStatus(database=settings.database, database_exists=False)
    server_engine = db.connect_without_database()
    try:
        try:
            status.database_exists = database_exists(server_engine, settings.database)
        except SQLAlchemyError as exc:
            raise DatabaseError(
                redact_secret(
                    "Failed to query INFORMATION_SCHEMA for database existence. " + str(exc),
                    settings,
                )
            ) from None
        if not status.database_exists:
            status.messages.append(
                f"Database {settings.database!r} does not exist. "
                "Create it manually, then rerun init. This tool will not CREATE DATABASE."
            )
            return status

        with db.engine().connect() as conn:
            tables = existing_tables(conn, settings.database)
            status.existing_tables = tables
            expected = set(EXPECTED_TABLES)
            present = set(tables)
            status.missing_tables = [name for name in EXPECTED_TABLES if name not in present]
            status.unexpected_tables = sorted(present - expected)
            if "schema_meta" in present:
                row = conn.execute(
                    text(
                        "SELECT version FROM schema_meta WHERE component = :component"
                    ),
                    {"component": SCHEMA_COMPONENT},
                ).first()
                if row:
                    status.schema_version = str(row[0])

        if not status.missing_tables:
            if status.schema_version and status.schema_version != SCHEMA_VERSION:
                status.compatible = False
                status.messages.append(
                    f"Existing schema_meta.version={status.schema_version!r} "
                    f"does not match {SCHEMA_VERSION!r}. Refusing to alter or drop objects."
                )
            else:
                status.compatible = True
                if status.schema_version is None:
                    status.messages.append(
                        "All expected tables exist but schema_meta has no version row. "
                        "Treat as compatible inventory; version will be recorded on a later explicit repair."
                    )
                else:
                    status.messages.append("Existing schema matches expected table inventory.")
        elif status.existing_tables:
            status.compatible = False
            status.messages.append(
                "Partial schema detected. Missing: "
                + ", ".join(status.missing_tables)
                + ". Refusing to drop or rebuild existing tables."
            )
        else:
            status.messages.append("Database exists and contains none of the expected tables.")
        return status
    finally:
        server_engine.dispose()


def initialize_schema(db: Database, *, schema_path: Path | None = None) -> InitResult:
    """Apply DDL only when the target database is empty of expected tables."""
    sql_text = load_schema_sql(schema_path)
    statements = parse_sql_statements(sql_text)
    checksum = schema_checksum(sql_text)
    status = inspect_schema(db)

    if not status.database_exists:
        raise SchemaError(status.messages[0] if status.messages else "Target database is missing.")

    if not status.missing_tables:
        if status.compatible:
            return InitResult(action="already_present", status=status, created_tables=[])
        raise SchemaError(
            "Existing schema is incompatible and will not be dropped or altered. "
            + " ".join(status.messages)
        )

    if status.existing_tables and status.missing_tables:
        raise SchemaError(
            "Refusing to initialize a partial schema. "
            f"Present: {', '.join(status.existing_tables) or '(none)'}. "
            f"Missing: {', '.join(status.missing_tables)}. "
            "No DROP TABLE was executed."
        )

    created: list[str] = []
    try:
        with db.engine().begin() as conn:
            for stmt in statements:
                conn.execute(text(stmt))
                match = re.match(
                    r"CREATE\s+TABLE(?:\s+IF\s+NOT\s+EXISTS)?\s+`?([A-Za-z0-9_]+)`?",
                    stmt,
                    re.I,
                )
                if match:
                    created.append(match.group(1).lower())
            conn.execute(
                text(
                    "INSERT INTO schema_meta (component, version, checksum, applied_at) "
                    "VALUES (:component, :version, :checksum, :applied_at) "
                    "ON DUPLICATE KEY UPDATE version = VALUES(version), "
                    "checksum = VALUES(checksum), applied_at = VALUES(applied_at)"
                ),
                {
                    "component": SCHEMA_COMPONENT,
                    "version": SCHEMA_VERSION,
                    "checksum": checksum,
                    "applied_at": datetime.now(timezone.utc).replace(tzinfo=None),
                },
            )
    except SQLAlchemyError as exc:
        raise SchemaError(
            redact_secret("Schema application failed. No DROP was issued. " + str(exc), db.settings)
        ) from None

    refreshed = inspect_schema(db)
    if refreshed.missing_tables:
        raise SchemaError(
            "Schema application finished but tables are still missing: "
            + ", ".join(refreshed.missing_tables)
        )
    refreshed.messages.append(f"Applied schema version {SCHEMA_VERSION}.")
    return InitResult(action="created", status=refreshed, created_tables=created)
