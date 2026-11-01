# AGENTS.md — LearnPath AI

Instructions for coding agents working in this repository.

## Mission

Build an explainable learning-activity recommendation pipeline: synthetic data → validate/quarantine → mastery → recommend → evaluate. Challenge context: YottaFlex.ai Associate AI FDE Round 2.

## Source of truth

1. The official challenge problem statement (held by the human).
2. `docs/` in this repo.
3. `schemas/schema.sql` and `config/default.json`.

If those disagree, stop and ask. Do not invent extra product scope.

## Stack (do not replace)

Python 3.12, Pandas, NumPy, MySQL 8.4, SQLAlchemy 2.x, PyMySQL, Scikit-learn, Pytest. CLI-first. No SQLite as primary DB. No extra frameworks without a written reason in `docs/design_decisions.md`.

## Hard rules

- Never hard-code database passwords. Use `.env` (gitignored) and `.env.example` placeholders.
- Never commit secrets or personal information.
- Never print passwords or full database URLs.
- Never `CREATE DATABASE`, `DROP DATABASE`, `DROP TABLE`, or `TRUNCATE` automatically.
- Never expose `data/ground_truth` labels as model features.
- Never use future attempts in mastery or ranking features (`as_of_timestamp` cutoff).
- Never let ranking override a failed required-prerequisite eligibility check.
- Never silently drop anomalies; quarantine and attach issue types.
- Never claim metrics or tests that were not run.
- Do not copy from the `skewbench` project.
- Do not add a license file or SPDX license without human approval.

## Phased work

| Phase | Allowed | Not allowed until approved |
|---|---|---|
| 0 | Docs, schema, config, empty package | Data gen, logic, ML, UI, pip install, git remote |
| 1 (current) | MySQL connection, schema init, unit tests | Datasets, mastery, ranking, ML, frontend, git push |
| 2+ | Implementation the human requests | Scope creep, undeclared dependencies |

## Data and IDs

Deterministic IDs: `LRN-`, `CRS-`, `SKL-`, `CNT-`, `ATT-<learner-id>-`, `PATH-YYYY-` (YYYY = UTC year of `created_at`). Default seed `20261022`. Masters before events. Configurable `dev` (8 courses) / `full` (40 courses) profiles. One active learning path per learner per course.

## Tests

Place tests under `tests/unit`, `tests/integration`, `tests/acceptance`. Integration tests that need MySQL must skip cleanly when credentials are absent.

## Frontend

Do not add Streamlit or any UI until the CLI pipeline and evaluation reports work and the human asks for it.

## Git

Do not `git init`, commit, or push unless the human explicitly asks.
