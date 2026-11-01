# Database design and operations — LearnPath AI

Phase 1 document. Live MySQL 8.4 initialization has not been executed on this machine.

## 1. Role of MySQL

MySQL 8.4 LTS is the system of record. CSV/JSON files are interchange formats. SQLite is rejected by configuration code.

## 2. One active learning path per learner per course

`learning_path.active_learner_course` is a stored generated column:

- When `status = 'active'`, the value is `CONCAT(learner_id, '#', course_id)`.
- Otherwise the value is `NULL`.

`UNIQUE KEY uq_learning_path_one_active` allows many NULLs (draft/completed/abandoned) and at most one active pair. Activating a second path for the same learner and course is a constraint violation. Application code must abandon or complete the previous active path first.

## 3. Prerequisite DAG

Self-edges are rejected in SQL (`ck_prereq_not_self`). Cycles cannot be expressed as CHECK constraints. `learnpath_ai.prerequisites.require_dag` must run before persisting edges.

## 4. Eligibility vs ranking

`learnpath_ai.eligibility.content_passes_required_prerequisites` is a hard gate. `assert_rankable` raises if a ranker is given ineligible content. Ranking scores, path continuity, and future ML probabilities cannot override a failed required-prerequisite check. `relationship_type = recommended` is a ranking signal only.

## 5. Transactions

`Database.transaction()`:

1. Opens a SQLAlchemy 2.x `Session`.
2. Yields it to the caller.
3. `commit()` on a clean exit.
4. `rollback()` on any exception.
5. `close()` in `finally`.

DDL application uses `engine.begin()`. MySQL may autocommit some DDL; init still never issues DROP DATABASE, DROP TABLE, CREATE DATABASE, or TRUNCATE.

## 6. Initialization policy

The application user is **not** used to create databases.

Operator steps:

1. Install MySQL Server **8.4 LTS** locally on Windows.
2. Create database and user in a privileged session (not this app).
3. Put credentials in `.env` (never committed).
4. Run `python -m learnpath_ai db ping`.
5. Run `python -m learnpath_ai db init` only if ping succeeds.

`db init` behavior:

| Database state | Action |
|---|---|
| Database missing | Fail. Do not create it. |
| No expected tables | Apply `schemas/schema.sql`. |
| All expected tables, compatible version | Report `already_present`. Do not drop. |
| Partial table set | Fail. Do not drop or rebuild. |
| Version mismatch | Fail. Do not alter in place. |

## 7. Secrets

Engine URLs are built with `quote_plus`. Logs and exceptions pass through `redact_secret`. Connection is not opened at import time.

## 8. Path IDs

`path_id` matches `PATH-YYYY-000001` where `YYYY` is the UTC calendar year of `created_at`. Enforced by application generators (not yet implemented) plus SQL regex.
