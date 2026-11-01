# Data dictionary — LearnPath AI

All timestamps are UTC `DATETIME(6)`. Status values are closed enumerations. IDs are deterministic strings.

Legend: **R** = required, **O** = optional.

## Shared lifecycle

Slowly changing masters (`learner`, `course`, `skill`, `content_item`, `skill_prerequisite`) carry:

- `effective_from` (R)
- `effective_to` (O, null = currently effective)
- Rule: `effective_to IS NULL OR effective_to >= effective_from`

---

## 1. Learner (`learner`)

| Field | Type | R/O | Notes |
|---|---|---|---|
| learner_id | VARCHAR(16) | R | PK. `LRN-000001`. Unique. |
| display_name | VARCHAR(120) | R | Synthetic display name only. |
| cohort_code | VARCHAR(32) | O | Batch / intake code. |
| starting_level | ENUM | R | `beginner`, `intermediate`, `advanced`. |
| status | ENUM | R | `active`, `paused`, `completed`, `withdrawn`. |
| current_course_id | VARCHAR(16) | O | FK → `course.course_id`. |
| timezone | VARCHAR(64) | R | Default `UTC`. |
| created_at / updated_at | DATETIME(6) | R | Audit. |
| effective_from / effective_to | DATETIME(6) | R / O | Lifecycle. |

**Cardinality:** 1 learner → N attempts, paths, recommendations, mastery snapshots.

**Validation:** ID regex `^LRN-[0-9]{6}$`. Withdrawn learners are not recommended.

---

## 2. Course (`course`)

| Field | Type | R/O | Notes |
|---|---|---|---|
| course_id | VARCHAR(16) | R | PK. `CRS-000001`. |
| title | VARCHAR(200) | R | Unique with `domain`. |
| domain | VARCHAR(80) | R | Subject area. |
| difficulty_band | ENUM | R | `intro`, `core`, `advanced`. |
| status | ENUM | R | `draft`, `published`, `archived`. |
| published_at | DATETIME(6) | O | Required if status = published. |
| archived_at | DATETIME(6) | O | Set when archived. |
| created_at / updated_at | DATETIME(6) | R | |
| effective_from / effective_to | DATETIME(6) | R / O | |

**Cardinality:** 1 course → N content items, N learning paths.

---

## 3. Skill (`skill`)

| Field | Type | R/O | Notes |
|---|---|---|---|
| skill_id | VARCHAR(16) | R | PK. `SKL-000001`. |
| skill_name | VARCHAR(160) | R | Unique with `domain`. |
| domain | VARCHAR(80) | R | |
| bloom_level | ENUM | R | `remember` … `create`. |
| status | ENUM | R | `active`, `deprecated`. |
| created_at / updated_at | DATETIME(6) | R | |
| effective_from / effective_to | DATETIME(6) | R / O | |

Deprecated skills remain for history but are not newly recommended as primary targets.

---

## 4. SkillPrerequisite (`skill_prerequisite`)

Bridge: directed edge **skill requires prerequisite_skill**.

| Field | Type | R/O | Notes |
|---|---|---|---|
| skill_id | VARCHAR(16) | R | PK part. FK → skill. |
| prerequisite_skill_id | VARCHAR(16) | R | PK part. FK → skill. |
| relationship_type | ENUM | R | `required` (hard gate), `recommended` (soft). |
| min_mastery_required | DECIMAL(6,4) | R | 0–1, default 0.6000. |
| created_at | DATETIME(6) | R | |
| effective_from / effective_to | DATETIME(6) | R / O | |

**Uniqueness:** `(skill_id, prerequisite_skill_id)`.  
**Validation:** no self-edge. Graph must be a DAG (application check; not a DB CHECK).

**Cardinality:** Skill N—N Skill.

---

## 5. ContentItem (`content_item`)

| Field | Type | R/O | Notes |
|---|---|---|---|
| content_id | VARCHAR(16) | R | PK. `CNT-000001`. |
| course_id | VARCHAR(16) | R | FK → course. |
| title | VARCHAR(200) | R | |
| content_type | ENUM | R | `lesson`, `practice`, `assessment`, `project`. |
| difficulty | DECIMAL(3,2) | R | 1.00–5.00. |
| estimated_minutes | SMALLINT UNSIGNED | R | 1–600. |
| status | ENUM | R | `draft`, `published`, `retired`. |
| created_at / updated_at | DATETIME(6) | R | |
| effective_from / effective_to | DATETIME(6) | R / O | |

Only `published` content is eligible for new recommendations.

---

## 6. ContentSkill (`content_skill`)

Many-to-many bridge.

| Field | Type | R/O | Notes |
|---|---|---|---|
| content_id | VARCHAR(16) | R | PK part. FK → content_item. |
| skill_id | VARCHAR(16) | R | PK part. FK → skill. |
| skill_role | ENUM | R | `primary`, `supporting`. |
| weight | DECIMAL(6,4) | R | (0, 1]. |
| created_at | DATETIME(6) | R | |

**Validation:** published content should have at least one `primary` skill (application rule). Anomaly type `CONTENT_MISSING_SKILL` covers missing/nonexistent skill references before FK load.

---

## 7. Attempt (`attempt`)

| Field | Type | R/O | Notes |
|---|---|---|---|
| attempt_id | VARCHAR(32) | R | PK. `ATT-LRN-000001-000001`. |
| learner_id | VARCHAR(16) | R | FK → learner. |
| content_id | VARCHAR(16) | R | FK → content_item. |
| attempt_number | SMALLINT UNSIGNED | R | ≥ 1. Unique with learner+content. |
| status | ENUM | R | `in_progress`, `submitted`, `scored`, `abandoned`, `invalid`. |
| started_at | DATETIME(6) | R | |
| submitted_at | DATETIME(6) | O | Must be ≥ started_at. |
| score | DECIMAL(5,2) | O | 0..max_score; required when scored. |
| max_score | DECIMAL(5,2) | R | Default 100.00. |
| source_recommendation_id | VARCHAR(16) | O | FK → recommendation. |
| created_at | DATETIME(6) | R | |

**Anomaly:** `ATTEMPT_TIMESTAMP_ORDER` when `submitted_at < started_at` or attempt_number order disagrees with timestamps.

---

## 8. AssessmentResult (`assessment_result`)

| Field | Type | R/O | Notes |
|---|---|---|---|
| result_id | VARCHAR(16) | R | PK. `RES-000001`. |
| attempt_id | VARCHAR(32) | R | Unique FK → attempt. |
| outcome | ENUM | R | `pass`, `fail`, `incomplete`. |
| score | DECIMAL(5,2) | R | |
| passed | TINYINT(1) | R | 1 iff outcome = pass. |
| mastery_delta | DECIMAL(7,4) | O | Change vs prior snapshot. |
| scored_at | DATETIME(6) | R | ≥ attempt.submitted_at. |
| created_at | DATETIME(6) | R | |

Pass threshold is application config (`pass_score_threshold`, default 70.0), not a DB CHECK on the numeric cut.

---

## 9. LearningPath (`learning_path`)

| Field | Type | R/O | Notes |
|---|---|---|---|
| path_id | VARCHAR(24) | R | PK. `PATH-YYYY-000001`. YYYY is the UTC calendar year of `created_at`. |
| learner_id | VARCHAR(16) | R | FK → learner. |
| course_id | VARCHAR(16) | R | FK → course. |
| status | ENUM | R | `draft`, `active`, `completed`, `abandoned`. |
| generated_at | DATETIME(6) | R | |
| activated_at / completed_at | DATETIME(6) | O | `activated_at` required when status is `active`. |
| generator_version | VARCHAR(32) | R | `rule-v1` or `ml-v1`. |
| created_at / updated_at | DATETIME(6) | R | |
| active_learner_course | VARCHAR(40) | generated | `learner_id#course_id` when status is `active`, else NULL. Unique. |

**Uniqueness:** at most one active path per (`learner_id`, `course_id`). Draft/completed/abandoned paths may coexist.

**Cardinality:** 1 path → N `learning_path_item`.

---

## 10. Recommendation (`recommendation`)

| Field | Type | R/O | Notes |
|---|---|---|---|
| recommendation_id | VARCHAR(16) | R | PK. `REC-000001`. |
| learner_id | VARCHAR(16) | R | FK → learner. |
| content_id | VARCHAR(16) | R | FK → content_item. |
| path_id | VARCHAR(24) | O | FK → learning_path. |
| rank_position | TINYINT UNSIGNED | R | 1–20. |
| engine_version | VARCHAR(32) | R | `rule-v1`, `ml-v1`. |
| reason_codes | JSON | R | Array of code strings. |
| score | DECIMAL(8,6) | O | Ranker score / probability. |
| status | ENUM | R | `pending`, `accepted`, `dismissed`, `expired`, `completed`, `quarantined`. |
| recommended_at | DATETIME(6) | R | Persist time. |
| as_of_timestamp | DATETIME(6) | R | Feature cutoff; ≤ recommended_at. |
| expires_at | DATETIME(6) | O | |
| created_at | DATETIME(6) | R | |

**Uniqueness:** `(learner_id, content_id, as_of_timestamp, engine_version)`.  
**Anomaly:** `REC_PREREQ_VIOLATION` if required prerequisites are unmet at `as_of_timestamp`.

---

## Supporting tables

### LearningPathItem (`learning_path_item`)

| Field | Type | R/O | Notes |
|---|---|---|---|
| path_item_id | VARCHAR(16) | R | PK. `LPI-000001`. |
| path_id | VARCHAR(24) | R | FK. Unique with `sequence_no`. |
| sequence_no | SMALLINT UNSIGNED | R | ≥ 1. |
| content_id | VARCHAR(16) | R | FK. |
| recommendation_id | VARCHAR(16) | O | FK. |
| status | ENUM | R | `planned`, `in_progress`, `completed`, `skipped`. |
| planned_at / completed_at | DATETIME(6) | R / O | |

**Anomaly:** `PATH_LOOP_NO_PROGRESS` — repeating content with no mastery/score improvement.

### MasterySnapshot (`mastery_snapshot`)

| Field | Type | R/O | Notes |
|---|---|---|---|
| snapshot_id | VARCHAR(16) | R | PK. `MST-000001`. |
| learner_id / skill_id | VARCHAR(16) | R | FKs. Unique with `as_of_timestamp`. |
| mastery_score | DECIMAL(6,4) | R | 0–1. |
| evidence_attempt_count | INT UNSIGNED | R | |
| as_of_timestamp | DATETIME(6) | R | Decision time. |
| evidence_cutoff_ts | DATETIME(6) | R | Must be ≤ as_of. |
| calculator_version | VARCHAR(32) | R | `mastery-v1`. |
| created_at | DATETIME(6) | R | |

**Anomaly:** `MASTERY_FUTURE_ATTEMPT` if evidence includes attempts after `as_of_timestamp`.

### RecommendationOutcome (`recommendation_outcome`)

| Field | Type | R/O | Notes |
|---|---|---|---|
| outcome_id | VARCHAR(16) | R | PK. `ROC-000001`. |
| recommendation_id | VARCHAR(16) | R | Unique FK. |
| subsequent_attempt_id | VARCHAR(32) | O | FK → attempt. |
| window_hours | INT UNSIGNED | R | Default 168. |
| completed / improved | TINYINT(1) | R | Simulated flags. |
| score_delta | DECIMAL(6,2) | O | |
| observed_at / created_at | DATETIME(6) | R | |

These outcomes are **simulated** labels for ranking evaluation, not real educational studies.

### EventHistory (`event_history`)

Append-only audit: `event_id` `EVT-000001`, `event_type`, `entity_type`, `entity_id`, JSON `payload`, `occurred_at`, `recorded_at`, `actor`.

### DataQualityIssue (`data_quality_issue`)

| Field | Type | R/O | Notes |
|---|---|---|---|
| issue_id | VARCHAR(16) | R | PK. `DQI-000001`. |
| issue_type | ENUM | R | The five required anomaly types. |
| source_table / source_pk | VARCHAR | R | Pointer to original record. |
| severity | ENUM | R | `low` … `critical`. |
| status | ENUM | R | `open`, `quarantined`, `accepted_exception`, `resolved`. |
| expected_decision | VARCHAR(64) | R | From ground truth. |
| details | JSON | R | Audit payload including original values. |
| detected_at / created_at | DATETIME(6) | R | |

Original invalid records are preserved in quarantine files; they are not silently repaired.

---

## File interchange (planned CSV names)

| File | Folder | When |
|---|---|---|
| learners.csv, courses.csv, skills.csv, skill_prerequisites.csv, content_items.csv, content_skills.csv | `data/input` or `data/sample` | After generation |
| attempts.csv, assessment_results.csv, learning_paths.csv, recommendations.csv | `data/input` | After generation |
| ground_truth_anomalies.csv | `data/ground_truth` | After injection |
| quarantined_*.csv | `data/quarantine` | After validation |
| reports | `reports/` | After evaluation |

No data files are generated in Phase 0.### DataQualityIssue (data_quality_issue)

| Field | Type | R/O | Notes |
|---|---|---|---|
| issue_id | VARCHAR(16) | R | PK. \DQI-000001\. |
| issue_type | ENUM | R | One of five anomaly types (see below). |
| source_table / source_pk | VARCHAR | R | Pointer to original record. |
| severity | ENUM | R | \low\, \medium\, \high\, \critical\. |
| status | ENUM | R | \open\, \quarantined\, \ccepted_exception\, esolved\. |
| expected_decision | VARCHAR(64) | R | Ground-truth label (eject\, \xclude\, eview\). |
| details | JSON | R | Audit payload including original field values and reason. |
| detected_at / created_at | DATETIME(6) | R | |

**Issue type ENUM values (Phase 4 implemented):**

| Value | Anomaly | Severity | Description |
|---|---|---|---|
| \REC_PREREQ_VIOLATION\ | A | critical | Recommendation issued for content whose primary skill has an unsatisfied prerequisite. |
| \ATTEMPT_TIMESTAMP_ORDER\ | B | high | \submitted_at\ < \started_at\ or \scored_at\ < \submitted_at\. |
| \CONTENT_MISSING_SKILL\ | C | high | \content_skill\ mapping references a \skill_id\ that does not exist in the \skill\ table. |
| \MASTERY_FUTURE_ATTEMPT\ | D | critical | Mastery calculation includes an attempt whose \started_at\ is after the \s_of_timestamp\ (temporal leakage). |
| \PATH_LOOP_NO_PROGRESS\ | E | medium | Learning path contains repeated content without score improvement. |

**Quarantine files (Phase 4, per profile):**

| File | Location | Contents |
|---|---|---|
| ec_prereq_violations.csv\ | \data/quarantine/<profile>/\ | Anomaly A records |
| \ttempt_timestamp_orders.csv\ | \data/quarantine/<profile>/\ | Anomaly B records |
| \content_missing_skills.csv\ | \data/quarantine/<profile>/\ | Anomaly C records |
| \mastery_future_attempts.csv\ | \data/quarantine/<profile>/\ | Anomaly D records |
| \path_loop_no_progress.csv\ | \data/quarantine/<profile>/\ | Anomaly E records |
| \data_quality_issues.csv\ | \data/quarantine/<profile>/\ | Combined all-type CSV |
| \quality_manifest.json\ | \data/quarantine/<profile>/\ | Seed, profile, row counts, SHA256 |
| \ground_truth_anomalies.csv\ | \data/ground_truth/\ | Labels only — never used as features |

Original invalid records are preserved in quarantine files; they are not silently repaired.
