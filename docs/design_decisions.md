# Design decisions — LearnPath AI

Phase 0 decisions plus Phase 1 database approvals.

## 1. Why this stack

| Choice | Reason |
|---|---|
| Python 3.12 | Challenge language; typing and performance adequate for 500k attempts |
| Pandas / NumPy | Tabular generation, joins, and evaluation without extra frameworks |
| MySQL 8.4 LTS | Required primary store; FKs, CHECK, JSON, transactions |
| SQLAlchemy 2.x | Explicit mappings, connection lifecycle, bulk load later |
| PyMySQL | Pure-Python MySQL driver specified by the stack |
| Scikit-learn | Standard, reproducible ML comparison vs rules |
| Pytest | Unit / integration / acceptance layout |
| JSON + env | Config vs secrets split |
| No Streamlit in Phase 0 | Frontend is optional; CLI must stand alone |

**Rejected:** SQLite as primary DB (disallowed). Extra web frameworks (no requirement). Docker as a hard dependency (not in the mandated stack; local MySQL 8.4 is the assumption).

**Justified extra libraries:**

- `python-dotenv` — load `.env` without hard-coding secrets.
- `cryptography` — often required by PyMySQL for `caching_sha2_password` on MySQL 8.

## 2. Entity and ID design

Deterministic public IDs are the primary keys (not auto-increment integers) so CSV, MySQL, and reports share the same keys.

| Entity | Pattern | Example |
|---|---|---|
| Learner | `LRN-000001` | `LRN-000001` |
| Course | `CRS-000001` | `CRS-000001` |
| Skill | `SKL-000001` | `SKL-000001` |
| ContentItem | `CNT-000001` | `CNT-000001` |
| Attempt | `ATT-<learner-id>-000001` | `ATT-LRN-000001-000001` |
| LearningPath | `PATH-YYYY-000001` | `PATH-2026-000001` |
| Recommendation | `REC-000001` | `REC-000001` |
| AssessmentResult | `RES-000001` | `RES-000001` |
| LearningPathItem | `LPI-000001` | `LPI-000001` |
| MasterySnapshot | `MST-000001` | `MST-000001` |
| RecommendationOutcome | `ROC-000001` | `ROC-000001` |
| EventHistory | `EVT-000001` | `EVT-000001` |
| DataQualityIssue | `DQI-000001` | `DQI-000001` |

Recommendation / result / support IDs are not in the challenge’s six examples; they follow the same zero-padded style so loaders stay uniform.

## 3. Cardinality

- Course 1—N ContentItem (content belongs to one course).
- ContentItem N—N Skill via `content_skill`.
- Skill N—N Skill via `skill_prerequisite` (directed; no self-edges; application `require_dag` rejects cycles).
- Learner 1—N Attempt, LearningPath, Recommendation, MasterySnapshot.
- At most one **active** LearningPath per (learner, course).
- Attempt 1—0..1 AssessmentResult.
- Recommendation 1—0..1 RecommendationOutcome.
- LearningPath 1—N LearningPathItem.

## 4. Mastery

Mastery is a derived snapshot, not a column on `learner`. That preserves history and enables as-of queries.

Planned estimator (to implement later): recency-weighted average of normalized scores on content that maps to the skill, using only evidence with timestamp `<= as_of`. Default pass threshold 0.70; prerequisite gate 0.60 (`config/default.json`).

## 5. Prerequisites & Recommendation (Phase 6, `rule-v1`)

`relationship_type = required` is an immutable hard eligibility filter implemented in `learnpath_ai.eligibility`. Ranking, path continuity, and ML scores are strictly forbidden from overriding a failed gate. Only content satisfying all required prerequisites at the temporal cutoff `as_of_timestamp` is submitted to the rule-based ranker.

Ranker score formula (`rule-v1`):
Score = 0.40 * SkillGap + 0.25 * DiffMatch + 0.20 * w_cs + 0.15 * PerfFactor
- SkillGap: (1.0 - mastery) * w_cs
- DiffMatch: 1.0 - abs(diff - target_diff) / 4.0
- PerfFactor: 0.90 for retrying failed content, 0.70 for fresh unattempted content, 0.20 for already passed content.
- Deterministic tie-breaking by score desc, skill gap desc, difficulty match desc, content ID asc.

## 6. Synthetic data

- Default seed `20261022`.
- Profiles `dev` and `full` in `config/default.json`.
- Same seed + profile ⇒ same logical records.
- Masters before events.
- Anomalies injected **after** a valid baseline, with labels only in `data/ground_truth/`.

Generator algorithm (planned, not coded):

1. RNG from seed.
2. Courses in domains; skills per domain; DAG prerequisites with bounded in-degree.
3. Content items per course with difficulty 1.00–5.00 and 1–4 skills.
4. Learners with cohort and starting level; enroll in a published course.
5. Chronological attempts; scores from a difficulty-vs-mastery function plus noise.
6. Results when status is `scored`.
7. Paths as ordered unpublished then activated item lists.
8. Baseline recommendations consistent with eligibility.
9. Separate anomaly injector mutates copies and writes ground truth.

## 7. Database initialization

1. Operator installs MySQL **8.4 LTS** locally and creates the empty database plus app user (privileged session, not this app).
2. Credentials live in `.env` only.
3. `python -m learnpath_ai db ping` verifies connectivity without printing secrets.
4. `python -m learnpath_ai db init` applies `schemas/schema.sql` only to a database with no expected tables. It never creates or drops databases/tables automatically.
5. Later phases will load CSV in FK order inside transactions and reconcile counts.

Live init has not been executed here because MySQL 8.4 is not installed.

## 8. Evaluation philosophy

Metrics are computed from **this synthetic world**. They measure pipeline correctness and ranking quality on simulated outcomes. They are **not** evidence of real classroom effectiveness. No numbers are claimed in Phase 0.

## 9. Frontend

Deferred. CLI remains the operator interface.

## 10. Version control

GitHub submission is required at the end of the challenge. Do not `git init`, commit, or push until explicitly approved.

## 11. License

No license is declared. Do not assume MIT.

## 12. Evaluation relevance

Primary label: skill improvement (`improved` / positive `score_delta`). Secondary: successful completion (`completed`).
