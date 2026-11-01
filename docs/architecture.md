# Architecture — LearnPath AI

Design document. Phase 1 implemented database settings, SQLAlchemy access, schema inspection/init, CLI `db` commands, DAG validation, and the prerequisite eligibility gate. Generation, mastery, ranking, and evaluation remain unimplemented.

## 1. Objective

LearnPath AI recommends the next learning activities for a learner using:

- current skill mastery
- skill prerequisite edges
- historical attempts and scores
- content difficulty
- prior recommendations and subsequent outcomes

Every recommendation must carry reason codes, respect prerequisites, avoid temporal leakage, and be evaluable.

## 2. Runtime shape

The product is **CLI-first**. Operators run generation, validation, recommendation, and evaluation as sequenced commands. Streamlit is an optional later presentation layer only. The core must work with no UI.

```text
config JSON + .env
        │
        ▼
┌─────────────────────────────────────────────────────────────┐
│  CLI / pipeline orchestrator (planned)                      │
└─────────────────────────────────────────────────────────────┘
        │
        ├── generate master data
        ├── generate dependent events
        ├── inject labelled anomalies (separate from baseline)
        ├── validate + quarantine
        ├── load MySQL (transactional)
        ├── compute mastery as-of T
        ├── filter eligible content
        ├── rank (rule-v1, later ml-v1)
        ├── persist recommendations, paths, events
        └── evaluate + write reports
```

## 3. Pipeline stages

| # | Stage | Input | Output | Notes |
|---|---|---|---|---|
| 1 | Master data generation | seed, profile | learners, courses, skills, content | Parents first |
| 2 | Dependent events | masters + seed | attempts, results, paths | Chronological timestamps |
| 3 | Schema and relationship validation | generated tables | pass/fail + issue rows | PK/FK/uniqueness |
| 4 | Anomaly detection and quarantine | valid + injected | `data/quarantine`, `data_quality_issue` | Labels stay in ground truth |
| 5 | Mastery estimation | attempts with `ts <= as_of` | `mastery_snapshot` | No future attempts |
| 6 | Eligibility filter | mastery, prereqs, content | candidate set | Drop prereq violations |
| 7 | Ranking | candidates + difficulty + history | top-K | rule-v1 first |
| 8 | Reason codes and history | ranked items | `recommendation`, `event_history` | Explainable |
| 9 | Path and outcomes | recs + later attempts | `learning_path*`, `recommendation_outcome` | Simulated outcomes |
| 10 | Evaluation | recs vs later activity | `reports/` | Chronological split |

## 4. Recommendation engines

### 4.1 Rule-based baseline (`rule-v1`) — first to implement

Planned scoring (weights to be calibrated on the **dev** profile, not invented here):

1. Skill-gap: lower mastery on primary skills increases priority.
2. Prerequisite gate (hard, runs **before** ranking): required prerequisites must meet `min_mastery_required`. Failures never enter the candidate set and cannot be overridden by rank scores.
3. Difficulty match: prefer content near current mastery band.
4. Retry: failed recent assessments may be re-offered after a cooldown.
5. Path continuity: prefer the next incomplete `learning_path_item`.
6. Dedup: do not re-recommend completed-and-mastered content unless mastery decayed.

Output: ranked list of size `K` (default 5) plus reason codes from `config/default.json`.

### 4.2 Machine-learning extension (`ml-v1`) — later

Scikit-learn classifier or calibrated probability model predicting **successful completion or mastery improvement** in a fixed outcome window.

- Features built only from events at or before `as_of_timestamp`.
- Labels from **future** attempts after `as_of_timestamp` (training labels are not input features).
- Compare Precision@K / Recall@K and coverage against `rule-v1`. Do not assume ML wins.

Candidate families (to be chosen in a later phase after feature audits): logistic regression, gradient boosting, or a simple random forest. Keep the model small and explainable.

## 5. Temporal contract (anti-leakage)

For a recommendation issued at `as_of_timestamp T`:

- Mastery uses attempts with `submitted_at <= T` (or `started_at` if not submitted).
- Outcomes, later scores, and ground-truth anomaly labels are **not** features.
- Evaluation splits are chronological on `recommended_at` / `as_of_timestamp`, not random row shuffles.

## 6. Data stores

| Store | Role |
|---|---|
| CSV / JSON under `data/` | Generation, interchange, quarantine, ground truth |
| MySQL 8.4 | System of record: FKs, transactions, indexes |
| SQLAlchemy 2.x + PyMySQL | Access layer (`src/learnpath_ai/database.py`) |

SQLite is not the primary database.

## 7. Package layout (planned, not yet filled)

```text
src/learnpath_ai/
  __init__.py          # package version
  __main__.py          # python -m learnpath_ai
  cli.py               # db ping | status | init
  settings.py          # env + URL construction
  database.py          # lazy engine, sessions, transactions
  schema.py            # inspect / apply DDL safely
  prerequisites.py     # DAG validation
  eligibility.py       # hard prerequisite gate (not a ranker)
  ids.py               # planned
  generate/            # planned
  mastery/             # implemented (Phase 5, calculator, validate, writer)
  recommend/           # implemented (Phase 6, rule-v1 ranker, validate, writer)
  persist/             # planned ORM mappings
  evaluate/            # planned
```

## 8. Anomaly handling (Phase 4 - implemented)

Valid baseline data is generated first using the Phase 2 and Phase 3 generators. Five labelled anomaly types are then injected deterministically using `numpy.random.default_rng(seed)` and stored in `data/quarantine/<profile>/`. Ground-truth labels are written separately to `data/ground_truth/ground_truth_anomalies.csv`. Recommendation features never include those labels (`never_use_labels_as_features = true` in config).

**Five anomaly types (all implemented):**

| Type | Table | Detection rule |
|---|---|---|
| `REC_PREREQ_VIOLATION` | `recommendation` | Learner lacks required prerequisite for recommended content primary skill |
| `ATTEMPT_TIMESTAMP_ORDER` | `attempt` | `submitted_at` < `started_at` |
| `CONTENT_MISSING_SKILL` | `content_skill` | `skill_id` not in `skill` table (FK break) |
| `MASTERY_FUTURE_ATTEMPT` | `attempt` | Attempt `started_at` > mastery `as_of_timestamp` |
| `PATH_LOOP_NO_PROGRESS` | `learning_path` | Same content repeated with no score improvement |

**CLI commands:**

```
python -m learnpath_ai generate anomalies --profile dev
python -m learnpath_ai validate quality --profile dev
```

**New modules:** `src/learnpath_ai/quality/anomalies.py`, `detectors.py`, `quarantine.py`, `report.py`.

See `docs/evaluation_methodology.md` for ground-truth label usage policy.

## 9. Frontend

Not in scope until the CLI pipeline produces real evaluation artifacts. If added, Streamlit would read MySQL or report files only; it would not own business logic.

## 10. Submission shape

A dedicated public GitHub repository with working code, setup/run docs, artifacts, and `docs/ai_usage.md`. Secrets stay out of git.
