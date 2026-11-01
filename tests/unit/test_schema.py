from __future__ import annotations

import pytest

from learnpath_ai.schema import (
    EXPECTED_TABLES,
    MANDATORY_ENTITY_TABLES,
    SchemaError,
    default_schema_path,
    list_create_table_names,
    load_schema_sql,
    parse_sql_statements,
)


def test_schema_file_declares_all_mandatory_entities() -> None:
    sql = load_schema_sql()
    names = set(list_create_table_names(sql))
    missing = [table for table in MANDATORY_ENTITY_TABLES if table not in names]
    assert missing == []
    for table in EXPECTED_TABLES:
        assert table in names


def test_schema_has_no_drop_or_create_database() -> None:
    statements = parse_sql_statements(load_schema_sql())
    joined = "\n".join(statements).upper()
    assert "DROP DATABASE" not in joined
    assert "DROP TABLE" not in joined
    assert "CREATE DATABASE" not in joined
    assert "TRUNCATE" not in joined
    assert statements
    assert all(stmt.upper().startswith("CREATE TABLE") for stmt in statements)


def test_one_active_path_unique_constraint_present() -> None:
    sql = load_schema_sql()
    assert "uq_learning_path_one_active" in sql
    assert "GENERATED ALWAYS AS" in sql
    assert "status = 'active'" in sql


def test_parse_rejects_drop_table() -> None:
    with pytest.raises(SchemaError, match="destructive"):
        parse_sql_statements("DROP TABLE learner;")


def test_schema_path_points_at_repo_file() -> None:
    path = default_schema_path()
    assert path.is_file()
    assert path.name == "schema.sql"
