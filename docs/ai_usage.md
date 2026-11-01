# AI Usage Disclosure & Human Engineering Contribution

> **Challenge Context:** YottaFlex.ai Associate AI Forward Deployed Engineer Round 2, 72-Hour Code Challenge.  
> **Repository:** `LearnPath AI`  
> **Status:** Submission-Ready Verification Log

---

## A. Development Approach

This project was developed through an AI-assisted engineering workflow. AI tools assisted with implementation, debugging, technical exploration, and documentation drafting. The human engineer remained responsible for understanding the challenge requirements, reviewing proposed solutions, evaluating technical decisions, checking implementation against expected behavior, and validating the delivered system.

AI coding assistants were utilized as productivity accelerators—generating initial boilerplate, drafting test scaffolding, and constructing vectorized routines. However, code was not committed without local testing, schema validation, and checksum verification. The developer reviewed each component to ensure it complied with the challenge specification and maintained architectural consistency.

---

## B. AI Tools Used

| Tool | Primary Environment | Actual Documented Role |
|---|---|---|
| **Cursor (with Grok 4.6 Agent)** | IDE Workspace | Initial Phase 0 repository structure setup, Phase 1 database connection helper stubs, and initial unit test skeletons. |
| **ChatGPT (OpenAI o1 / GPT-4o)** | Browser / Architectural Advisory | Exploratory technical discussions regarding psychometric mastery estimation heuristics, temporal recency decay formulations, and NDCG metric calculations. |
| **Google DeepMind Antigravity** | Agentic IDE & Terminal Sandbox | Code implementation across Phases 2 through 10: vectorized synthetic data generators, prerequisite DAG topological verification, rule-based and ML ranking engines, anomaly detection and quarantine, CLI command wiring, test suite expansion (180 tests), and Docker packaging. |

*Note on tool and model names:* Tool names reflect the actual software environments used. Specific model versions (such as Grok 4.6 in Cursor and OpenAI o1 / GPT-4o in browser sessions) represent the documented interfaces used during development sessions.

---

## C. Human Engineering Contribution

The developer maintained technical oversight across the implementation lifecycle, ensuring that challenge requirements were correctly understood, technical trade-offs were evaluated, and all outputs were verified.

### 1. Challenge Requirement Analysis & Decomposition
- **System Function:** Translates the challenge statement (*"Recommend next learning activities based on skills, prerequisites, assessment results, and content difficulty"*) into a structured 10-phase execution plan.
- **Relevant Files:** [`AGENTS.md`](../AGENTS.md), [`docs/architecture.md`](architecture.md).
- **Documented AI Assistance:** AI assisted in drafting initial phase checklists and file scaffolds.
- **Developer Review & Decisions:** Established the phased delivery boundaries (Phase 0 through Phase 9). Disallowed premature ML training, unauthorized cloud dependencies, or frontend bloat. Mandated deterministic random seeding (`20261022`) and immutable ID conventions (`LRN-`, `CRS-`, `SKL-`, `CNT-`, `ATT-`, `PATH-`).
- **Verification Evidence:** Phase gates documented in [`AGENTS.md`](../AGENTS.md).

### 2. Relational Database Design & Entity Relationships
- **System Function:** Implements a 16-table relational schema for **MySQL 8.4 LTS InnoDB** enforcing referential integrity, status enums, and temporal constraints.
- **Relevant Files:** [`schemas/schema.sql`](../schemas/schema.sql), [`docs/database.md`](database.md), [`docs/data_dictionary.md`](data_dictionary.md).
- **Documented AI Assistance:** AI generated initial SQL DDL table declarations.
- **Developer Review & Decisions:** Reviewed primary and surrogate keys, resolved N:M relationships with dedicated bridge tables (`content_skill`, `skill_prerequisite`), configured 1:1 foreign keys (`assessment_result` $\to$ `attempt`), added generated columns for single active path constraints, and added MySQL `CHECK` constraints (`ck_mastery_no_future`) preventing temporal leakage at the database layer.
- **Verification Evidence:** `validate master` checks foreign keys; database unit tests in [`tests/unit/test_schema_mocked.py`](../tests/unit/test_schema_mocked.py).

### 3. Skill Prerequisites & Dependency Handling
- **System Function:** Models curriculum dependencies as a Directed Acyclic Graph (DAG) and enforces prerequisite mastery.
- **Relevant Files:** [`src/learnpath_ai/prerequisites.py`](../src/learnpath_ai/prerequisites.py), [`src/learnpath_ai/eligibility.py`](../src/learnpath_ai/eligibility.py).
- **Documented AI Assistance:** AI drafted graph traversal routines and cycle detection functions.
- **Developer Review & Decisions:** Evaluated and accepted topological sort validation (`require_dag`) to reject cycles and self-loops. Configured the **0.60 minimum mastery hard gate**: ranking algorithms cannot recommend content whose required prerequisite skills fall below 0.60.
- **Verification Evidence:** [`tests/unit/test_prerequisites.py`](../tests/unit/test_prerequisites.py); 0 prerequisite gate violations across all 21,046 generated recommendations.

### 4. Mastery Estimation & Assessment History
- **System Function:** Computes reproducible, time-decayed skill mastery from historical assessment attempts subject to timestamp cutoffs (`as_of_timestamp`).
- **Relevant Files:** [`src/learnpath_ai/mastery/calculator.py`](../src/learnpath_ai/mastery/calculator.py).
- **Documented AI Assistance:** AI generated vectorized exponential decay routines and pandas aggregation code.
- **Developer Review & Decisions:** Adopted an exponential recency decay model with a configured 90-day half-life ($7,776,000$ seconds), primary/secondary skill mapping weights ($w_{cs}$), credibility regularization ($k=1.0$) to avoid premature 1.0 scores on single attempts, and an unattempted baseline of $0.0000$.
- **Verification Evidence:** [`tests/unit/test_mastery.py`](../tests/unit/test_mastery.py).

### 5. Recommendation Eligibility & Rule-v1 Scoring
- **System Function:** Deterministic multi-objective heuristic recommendation engine selecting eligible next learning activities.
- **Relevant Files:** [`src/learnpath_ai/recommend/engine.py`](../src/learnpath_ai/recommend/engine.py).
- **Documented AI Assistance:** AI implemented candidate filtering loops and output record serialization.
- **Developer Review & Decisions:** Evaluated and configured the 4-part scoring heuristic:
  $$\text{Score} = 0.40 \cdot \text{SkillGap} + 0.25 \cdot \text{DiffMatch} + 0.20 \cdot w_{cs} + 0.15 \cdot \text{PerfFactor}$$
  Reviewed explainability reason codes (`GAP_PRIMARY_SKILL`, `LOW_MASTERY_PRIORITY`, `PREREQUISITE_SATISFIED`, `DIFFICULTY_MATCH`, `RETRY_AFTER_FAIL`) and deterministic tie-breaking logic.
- **Verification Evidence:** [`tests/unit/test_recommendations.py`](../tests/unit/test_recommendations.py).

### 6. Cold-Start Learner Handling
- **System Function:** Guarantees valid recommendations for brand-new learners with zero assessment history without bypassing prerequisite safety.
- **Relevant Files:** [`src/learnpath_ai/recommend/engine.py`](../src/learnpath_ai/recommend/engine.py), [`src/learnpath_ai/eligibility.py`](../src/learnpath_ai/eligibility.py).
- **Documented AI Assistance:** AI wrote catalog filtering logic.
- **Developer Review & Decisions:** Identified and resolved an edge case where cold-start learners failed prerequisite checks on courses without introductory content. Mandated the **foundational content guarantee**: every course must publish at least one zero-required-prerequisite item. Verified across all 8 DEV courses and 40 FULL courses.
- **Verification Evidence:** [`tests/unit/test_recommendation_edge_cases.py`](../tests/unit/test_recommendation_edge_cases.py); 100% cold-start recommendation coverage in evaluation reports.

### 7. Anomaly Detection & Quarantine
- **System Function:** Detects and quarantines corrupted records across 5 specific anomaly scenarios without silently dropping data.
- **Relevant Files:** [`src/learnpath_ai/quality/`](../src/learnpath_ai/quality/).
- **Documented AI Assistance:** AI generated synthetic anomaly injection fixtures and detector helper functions.
- **Developer Review & Decisions:** Reviewed the quarantine directory layout (`data/quarantine/dev/`) and ground-truth benchmark separation (`data/ground_truth/ground_truth_anomalies.csv`). Formulated detector criteria for all 5 scenarios (prerequisite bypass, inverted timestamps, missing skills, future attempt leakage, and loop stalls).
- **Verification Evidence:** [`tests/unit/test_anomaly_detection.py`](../tests/unit/test_anomaly_detection.py); 25 injected dev anomalies detected with 100% precision and recall.

### 8. Temporal Leakage Prevention
- **System Function:** Guarantees zero look-ahead bias across mastery estimation, feature extraction, and evaluation.
- **Relevant Files:** [`src/learnpath_ai/ml/dataset.py`](../src/learnpath_ai/ml/dataset.py), [`src/learnpath_ai/mastery/calculator.py`](../src/learnpath_ai/mastery/calculator.py).
- **Documented AI Assistance:** AI wrote chronological filtering helpers.
- **Developer Review & Decisions:** Inspected and corrected potential leakage issues where attempts occurring after `as_of_timestamp` could affect mastery or features. Mandated strict temporal boundaries: `scored_at <= as_of_timestamp` for features and evaluation slices.
- **Verification Evidence:** [`tests/unit/test_ml.py`](../tests/unit/test_ml.py) (`test_strict_scored_at_leakage_prevention`).

### 9. ML-v1 Implementation & Evaluation
- **System Function:** Logistic Regression ranking model predicting learner success probability to re-rank candidate activities.
- **Relevant Files:** [`src/learnpath_ai/ml/`](../src/learnpath_ai/ml/), [`models/ml-v1_meta.json`](../models/ml-v1_meta.json).
- **Documented AI Assistance:** AI drafted Scikit-learn training and inference pipelines.
- **Developer Review & Decisions:** Reviewed feature definitions and removed collinear variables (`skill_gap` which was linear with mastery). Enforced feature parity: `ml-v1` must respect the identical 0.60 prerequisite gate as `rule-v1`.
- **Verification Evidence:** Validation ROC-AUC of **0.8630** documented in [`models/ml-v1_meta.json`](../models/ml-v1_meta.json).

### 10. DEV Training & FULL Zero-Shot Transfer Evaluation
- **System Function:** Evaluates how well a model trained strictly on DEV sample interactions generalizes to the 500,000-attempt FULL dataset.
- **Relevant Files:** [`reports/evaluation_f03_dev.txt`](../reports/evaluation_f03_dev.txt), [`reports/evaluation_f03_full.txt`](../reports/evaluation_f03_full.txt).
- **Documented AI Assistance:** AI executed evaluation scripts across DEV and FULL profiles.
- **Developer Review & Decisions:** Decided to retain the DEV-trained model artifact rather than retraining on FULL, establishing an authentic out-of-domain transfer benchmark. Confirmed that evaluation runs do not regenerate datasets or modify saved model weights.
- **Verification Evidence:** Evaluation reports demonstrating superior Precision@1 on FULL (0.1117 for ML vs. 0.1050 for Rule).

### 11. Automated & MySQL Integration Testing
- **System Function:** Multi-tiered test suite ensuring pipeline correctness.
- **Relevant Files:** [`tests/`](../tests/), [`pyproject.toml`](../pyproject.toml).
- **Documented AI Assistance:** AI generated parameterized unit test cases and CLI invocation tests.
- **Developer Review & Decisions:** Structured testing taxonomy (`unit`, `integration`, `acceptance`). Preserved strict test execution separation: **180 passed, 1 skipped** in standard isolated runs, and **1 passed separately** for live MySQL 8.4 integration. Kept credentials out of source code via `.env` and `.env.example`.
- **Verification Evidence:** Pytest test suite execution.

### 12. Docker Packaging & Production Readiness
- **System Function:** Containerized CLI pipeline.
- **Relevant Files:** [`Dockerfile`](../Dockerfile), [`docker-compose.yml`](../docker-compose.yml), [`.dockerignore`](../.dockerignore).
- **Documented AI Assistance:** AI drafted initial container specifications.
- **Developer Review & Decisions:** Refined the Dockerfile into a single-stage production image based on `python:3.12-slim`. Kept MySQL optional for CSV-based evaluation. Configured `.dockerignore` to prevent leaking local `.env` or `.venv` artifacts.
- **Verification Evidence:** Docker image build and CLI execution in project development records.

### 13. Final Technical & Documentation Audits
- **System Function:** Verification of project against challenge guidelines.
- **Relevant Files:** [`README.md`](../README.md), [`docs/`](./).
- **Documented AI Assistance:** AI generated draft text and assisted in editing documentation.
- **Developer Review & Decisions:** Audited and corrected discrepancies regarding Docker architecture, MySQL service ports, and prompt history availability.
- **Verification Evidence:** Manifest and checksum verification across all datasets.

---

## D. Technical Decisions and Their Rationale

The table below summarizes key architectural and algorithmic choices implemented in the project, their rationale, and how they were verified:

| Decision | Implementation | Purpose / Rationale | Verification Evidence |
|---|---|---|---|
| **0.60 Prerequisite Mastery Threshold** | `content_passes_required_prerequisites` in `eligibility.py` | Configured threshold to ensure learners master foundational skills before receiving advanced content, avoiding frustration. | [`tests/unit/test_prerequisites.py`](../tests/unit/test_prerequisites.py); 0 violations in 21k generated recs. |
| **90-Day Mastery Recency Half-Life** | Exponential decay in `calculator.py` ($\lambda = \ln(2) / 7776000$) | Balances recent performance against past track record; older attempts gradually lose influence. | [`tests/unit/test_mastery.py`](../tests/unit/test_mastery.py); verified recency weighting. |
| **Credibility Regularization ($k=1.0$)** | Credibility factor $\frac{n}{n + 1.0}$ in `calculator.py` | Prevents a single lucky attempt from granting 1.0 perfect mastery; requires multiple observations for high confidence. | Unit tests in `test_mastery.py` checking low-evidence mastery dampening. |
| **Cold-Start Baseline (0.0000)** | Default mastery score in `calculator.py` | Explicit numerical assignment representing absence of evidence; avoids NULL propagation issues in ranking formulas. | Cold-start unit tests in `test_recommendations.py`. |
| **Strict Chronological Evaluation Split** | Out-of-sample test split partitioned by cutoff timestamp $T$ | Prevents temporal look-ahead bias; simulates true forward-in-time recommendation deployment. | `evaluate compare` command; zero test attempts dated prior to cutoff. |
| **DEV-Only Model Training** | Serialized model artifact trained on `data/ml/dev/train.csv` | Keeps model training lightweight and inspectable; tests generalization capability. | `models/ml-v1_meta.json` records 2,358 training rows from DEV. |
| **FULL Zero-Shot Transfer Evaluation** | Applied DEV-trained model to score and rank candidates on FULL dataset | Serves as an authentic out-of-domain transfer test on 10,000 learners without overfitting to full-scale data. | [`reports/evaluation_f03_full.txt`](../reports/evaluation_f03_full.txt). |
| **Rule-v1 vs. ML-v1 Dual Strategy** | Side-by-side evaluation in `evaluate compare` | Benchmarks an explainable heuristic baseline against a machine learning model under identical conditions. | Consistent metrics comparison in `reports/evaluation_f03_dev.txt` and `full.txt`. |
| **Five Explicit Anomaly Categories** | Five typed detectors in `src/learnpath_ai/quality/` | Isolates known failure modes (prereq bypass, timestamp order, missing skills, future leakage, loops) without silently dropping data. | [`tests/unit/test_anomaly_detection.py`](../tests/unit/test_anomaly_detection.py); 25/25 detected in quarantine. |

---

## E. AI Assistance vs. Human Responsibility Matrix

| Development Activity | AI Assistance Provided | Human Responsibility & Verification | Project Evidence |
|---|---|---|---|
| **Pipeline Architecture** | Proposed directory layouts and stub files. | Decomposed challenge into 10 phases; enforced safety rules and deterministic seeding. | [`AGENTS.md`](../AGENTS.md), [`docs/architecture.md`](architecture.md) |
| **Relational Schema** | Drafted initial SQL DDL statements. | Reviewed keys, bridge tables, constraints, status enums, and generated columns. | [`schemas/schema.sql`](../schemas/schema.sql), [`docs/data_dictionary.md`](data_dictionary.md) |
| **Master Data Generation** | Drafted generation loops and ID formatters. | Evaluated course-domain bounds, DAG acyclicity via `require_dag`, and manifest checksums. | [`src/learnpath_ai/generate/master.py`](../src/learnpath_ai/generate/master.py), [`tests/unit/test_master_generator.py`](../tests/unit/test_master_generator.py) |
| **Activity Simulation** | Vectorized attempt generation code. | Evaluated cold-start distributions (10%), score reconciliation, and timestamp validation. | [`src/learnpath_ai/generate/activity.py`](../src/learnpath_ai/generate/activity.py), [`tests/unit/test_activity_generator.py`](../tests/unit/test_activity_generator.py) |
| **Mastery Estimation** | Drafted exponential math routines. | Configured recency half-life (90 days), credibility weighting ($k=1.0$), and cutoff gating. | [`src/learnpath_ai/mastery/calculator.py`](../src/learnpath_ai/mastery/calculator.py), [`tests/unit/test_mastery.py`](../tests/unit/test_mastery.py) |
| **Recommendation Engine** | Scaffolding candidate sorting and top-K. | Configured 0.60 hard prerequisite gate, multi-objective formula, and reason codes. | [`src/learnpath_ai/recommend/engine.py`](../src/learnpath_ai/recommend/engine.py), [`tests/unit/test_recommendations.py`](../tests/unit/test_recommendations.py) |
| **Cold-Start Policy** | Identified candidate filtering logic. | Established foundational content guarantee across all 48 courses (8 DEV, 40 FULL). | [`tests/unit/test_recommendation_edge_cases.py`](../tests/unit/test_recommendation_edge_cases.py) |
| **Anomaly Detection** | Generated injection fixtures and helper detectors. | Defined 5 anomaly scenarios, verified quarantine separation, checked 100% precision/recall. | [`src/learnpath_ai/quality/`](../src/learnpath_ai/quality/), [`tests/unit/test_anomaly_detection.py`](../tests/unit/test_anomaly_detection.py) |
| **ML Ranking Model** | Scripted Logistic Regression training pipeline. | Selected 8 non-collinear features, enforced temporal cutoff, verified DEV training scope. | [`models/ml-v1_meta.json`](../models/ml-v1_meta.json), [`src/learnpath_ai/ml/`](../src/learnpath_ai/ml/) |
| **Model Evaluation** | Implemented P@K, R@K, and NDCG@K formulas. | Evaluated chronological test slice strategy; verified zero dataset or model overwrites. | [`reports/evaluation_f03_dev.txt`](../reports/evaluation_f03_dev.txt), [`docs/evaluation_methodology.md`](evaluation_methodology.md) |
| **Automated Testing** | Generated parameterized test cases. | Structured unit/integration suites; verified 180 passing tests; kept MySQL tests optional. | [`tests/`](../tests/) (180 passed, 1 skipped) |
| **Containerization** | Generated initial Dockerfile. | Refined into single-stage Python 3.12 image; secured `.dockerignore`; kept DB optional. | [`Dockerfile`](../Dockerfile), [`docker-compose.yml`](../docker-compose.yml) |

---

## F. Prompt History & Session Records

### Prompt History Transparency Statement
Some early prompts could not be recovered verbatim. The available development intent is documented through clearly labelled summaries below. These summaries are not represented as exact original prompts. Later sessions (Sessions 7–12) have complete structured task prompts preserved in conversation transcripts. In accordance with challenge integrity rules, missing verbatim prompts have not been retroactively invented or reconstructed.

### Session Logs

#### Session 1 — Phase 0 Foundation (2026-10-30)
- **Tool:** Cursor (Grok 4.6 agent).
- **Documented Intent (Summary):** Establish Phase 0 repository foundation. Enforce workspace root `LearnPath AI`. Create directory tree, schema stubs, config, and documentation. No datasets, ML, or Git operations.
- **Prompts (Condensed Summary):**
  1. Inspect `Documents` directory, establish workspace at `LearnPath AI`, prohibit modifying unrelated projects (`skewbench`).
  2. Create standard folder structure, 10 core entities, MySQL schema skeleton, evaluation plan, and stop for human approval.
- **Outcome:** Phase 0 stubs approved.

#### Session 2 — Phase 1 MySQL Foundation (2026-10-30)
- **Tool:** Cursor (Grok 4.6 agent).
- **Documented Intent (Summary):** Build database connection utilities, schema apply logic, CLI `db` commands, and unit tests without modifying existing host databases.
- **Prompts (Condensed Summary):**
  1. Inspect local MySQL environment. Implement `settings.py`, `database.py`, and `schema.py`.
  2. Implement CLI subcommands `db ping`, `db status`, and `db init` with strict safety rules (no `DROP TABLE` or `CREATE DATABASE`).
  3. Create unit tests and mock integration tests.
- **Outcome:** Connection layer established; live MySQL tests cleanly skipped when unconfigured.

#### Session 3 — Phase 2 Master Data Generator (2026-10-30)
- **Tool:** Google DeepMind Antigravity.
- **Documented Intent (Summary):** Build reproducible master data generator for Learner, Course, Skill, SkillPrerequisite, ContentItem, and ContentSkill entities across DEV and FULL profiles.
- **Prompts (Condensed Summary):**
  1. Implement `src/learnpath_ai/generate/master.py` with fixed seed `20261022` and deterministic IDs.
  2. Implement `validate.py` verifying PKs, FKs, DAG acyclicity (`require_dag`), and SHA-256 manifests.
  3. Create CLI commands `generate master` and `validate master`.
- **Outcome:** Generated and validated `data/sample/` master dataset.

#### Session 4 — Phase 3 Activity & Assessment Generator (2026-10-30)
- **Tool:** Google DeepMind Antigravity.
- **Documented Intent (Summary):** Generate realistic, reproducible attempt logs and scored assessment results linked to Phase 2 master data.
- **Prompts (Condensed Summary):**
  1. Implement `src/learnpath_ai/generate/activity.py` supporting DEV (~5,000 attempts) and FULL (500,000 attempts) profiles.
  2. Model cold-start learners (10%), Pareto activity distributions, repeated attempts, and score reconciliation.
  3. Add CLI commands `generate activity` and `validate activity` with manifest verification.
- **Outcome:** Generated and validated DEV and FULL activity datasets without altering master data baselines.

#### Session 5 — Phase 4 Data Quality & Anomaly Quarantine (2026-10-30)
- **Tool:** Google DeepMind Antigravity.
- **Documented Intent (Summary):** Build anomaly injection, detection, and quarantine pipeline demonstrating all 5 required challenge scenarios without corrupting baseline data.
- **Prompts (Condensed Summary):**
  1. Create `src/learnpath_ai/quality/` containing `anomalies.py`, `detectors.py`, `quarantine.py`, and `report.py`.
  2. Inject 5 anomaly types: prerequisite violations, timestamp inversions, missing skills, future leakage, and loop stalls.
  3. Quarantine corrupt records to `data/quarantine/dev/` and write ground-truth labels to `data/ground_truth/ground_truth_anomalies.csv`.
- **Outcome:** 25 injected fixtures detected with 100% precision and recall; baseline data preserved.

#### Session 6 — Phase 5 Mastery Calculation Engine (2026-10-30)
- **Tool:** Google DeepMind Antigravity.
- **Documented Intent (Summary):** Implement temporal mastery estimation with recency decay, credibility regularization, and strict leakage prevention.
- **Prompts (Condensed Summary):**
  1. Implement `src/learnpath_ai/mastery/calculator.py` with exponential decay (90-day half-life) and credibility weighting.
  2. Enforce point-in-time cutoffs (`started_at <= as_of_timestamp` and `scored_at <= as_of_timestamp`).
  3. Add CLI subcommands `generate mastery` and `validate mastery`.
- **Outcome:** Generated and validated 920 DEV snapshots and 638,203 FULL snapshots.

#### Session 7 — Phase 6 Rule-Based Recommendation Engine (2026-10-30)
- **Tool:** Google DeepMind Antigravity.
- **Structured Task Intent:** Implement deterministic, explainable `rule-v1` recommendation engine enforcing hard prerequisite gating, multi-objective scoring, and machine-readable reason codes.
- **Outcome:** Verified top-K recommendations with zero prerequisite violations.

#### Session 8 — Phase 7 Learning Path Generation (2026-10-30)
- **Tool:** Google DeepMind Antigravity.
- **Structured Task Intent:** Implement deterministic learning path generation and sequencing, cycle detection, and single active path constraint per learner/course.
- **Outcome:** Generated 97 DEV paths and 9,502 FULL paths.

#### Session 9 — Phase 8 Machine Learning Extension (2026-10-30 to 2026-10-31)
- **Tool:** Google DeepMind Antigravity.
- **Structured Task Intent:** Build point-in-time training dataset, train explainable Logistic Regression model (`ml-v1`), generate recommendations, and implement chronological strategy comparison.
- **Outcome:** Trained model reaching ROC-AUC 0.8630 on 8 point-in-time features.

#### Session 10 — Phase 9 Evaluation & Final Verification (2026-10-31)
- **Tool:** Google DeepMind Antigravity.
- **Structured Task Intent:** End-to-end verification across profiles, run full pytest suite, compile evaluation reports, and document benchmarks.
- **Outcome:** All 10 validators passed; evaluation reports compiled.

#### Session 11 — F-01 through F-10 Stabilization & FULL Regeneration (2026-10-31)
- **Tool:** Google DeepMind Antigravity.
- **Structured Task Intent:** Execute stabilization tickets: verify cold-start foundational content across all 40 courses, enforce feature parity between ML training and inference, regenerate FULL reference data cleanly, and expand test suite to 180 tests.
- **Outcome:** All 40 FULL courses verified with foundational content; 180 tests passing.

#### Session 12 — Final Master Audit & Submission Packaging (2026-11-01)
- **Tool:** Google DeepMind Antigravity.
- **Structured Task Intent:** Audit repository against official submission guidelines: verify single-stage Dockerfile, single-service Compose, MySQL 8.4 on port 3307, verify dataset manifests, ensure `.env` is excluded, and prepare complete installation instructions.
- **Outcome:** Final audit completed; submission packaging verified.

---

## G. Testing & Verification Evidence

All claims in this repository are supported by verifiable execution evidence:

1. **Automated Pytest Suite:**
   - Command: `.venv\Scripts\python.exe -m pytest -ra --tb=short`
   - Result: **180 passed, 1 skipped in 15.67s (0 failures, 0 warnings)**.
   - The single skipped test is `tests/integration/test_mysql.py`, which is designed to skip cleanly when live database credentials are not set in the environment.
2. **Separate MySQL Integration Test:**
   - Status: **1 passed separately** when executed in a privileged environment with live MySQL 8.4.13 credentials and `LEARNPATH_MYSQL_IT=1`.
   - Result: `tests/integration/test_mysql.py::test_live_ping PASSED`.
   - *Note on test totals:* The automated test suite result (180 passed, 1 skipped) and the MySQL integration test result (1 passed) represent separate runs under different environment configurations. They are not combined into a single fabricated total.
3. **Docker Verification Status:**
   - During the earlier development workflow, the Docker image was successfully built and the application CLI and DEV evaluation were executed inside the container.
   - During the later documentation-only audit, Docker commands were not re-executed because the Docker binary was unavailable in that specific execution shell. The Dockerfile and compose syntax remain intact.
4. **Dataset Manifest & Checksum Verification:**
   - 10 CLI validation commands (`validate master`, `validate activity`, `validate mastery`, `validate recommendations`, `validate paths` for both `dev` and `full`) executed with `status=valid`.
   - Every file matches its cryptographic SHA-256 hash in `manifest.json`.
5. **Data Quality & Quarantine Benchmark:**
   - 25 injected fixtures across 5 anomaly types detected with **100% Precision and 100% Recall** against `data/ground_truth/ground_truth_anomalies.csv`.
6. **Zero-Shot Transfer Evaluation on FULL Profile:**
   - Model trained strictly on DEV sample interactions achieved superior ranking metrics over the heuristic baseline when transferred to the 500,000-attempt FULL dataset (P@1: 0.1117 vs. 0.1050; NDCG@5: 0.0919 vs. 0.0894).
   - Both strategies achieved **100.0% recommendation coverage** and **100.0% cold-start coverage**.

---

## H. Limitations & Transparency

- **Early Prompt History:** Verbatim prompt records for Sessions 1–6 are partially unavailable as historical artifacts; their scopes and instructions are accurately reflected via intent summaries.
- **Synthetic Data:** Student interactions, attempt durations, and assessment scores are synthetically simulated to test architectural scalability, data validation, and prerequisite routing. They do not claim psychometric validation against human behavioral profiles.
- **ML Training Scope:** The serialized model artifact (`models/ml-v1.pkl`) was deliberately trained on DEV sample interactions to serve as an honest generalization benchmark against the FULL reference dataset.
- **Local MySQL Port:** Host-level MySQL 8.4.13 binds to port 3307. When configuring `.env`, `MYSQL_PORT=3307` must be specified.

---

## I. Final Statement

AI coding assistants were utilized in this project as implementation accelerators. However, responsibility for understanding requirements, evaluating proposed technical solutions, reviewing code, making architectural decisions, and validating the delivered system remained with the human engineer. Every component was tested, reviewed, and verified against the challenge specification.
