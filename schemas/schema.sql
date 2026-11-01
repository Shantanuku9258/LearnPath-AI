-- LearnPath AI — MySQL 8.4 schema
-- Character set: utf8mb4
-- Engine: InnoDB for FK, unique, and transaction support
-- Time convention: all DATETIME(6) values are UTC
--
-- This file contains TABLE DDL only.
-- It does NOT create or drop databases.
-- It does NOT drop tables.
-- The operator must create an empty database and application user first:
--
--   CREATE DATABASE learnpath_ai CHARACTER SET utf8mb4 COLLATE utf8mb4_0900_ai_ci;
--   CREATE USER 'learnpath_app'@'localhost' IDENTIFIED BY '...';
--   GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, ALTER, INDEX,
--         REFERENCES, CREATE VIEW ON learnpath_ai.* TO 'learnpath_app'@'localhost';
--
-- Then: python -m learnpath_ai db init
--
-- Load order (parent/master before dependents):
--   schema_meta, course, skill, learner (FK to course),
--   skill_prerequisite, content_item, content_skill,
--   learning_path, recommendation, learning_path_item,
--   attempt, assessment_result, mastery_snapshot,
--   recommendation_outcome, event_history, data_quality_issue

-- ---------------------------------------------------------------------------
-- schema_meta — applied DDL version (compatibility checks; never drop)
-- ---------------------------------------------------------------------------
CREATE TABLE schema_meta (
  component             VARCHAR(64)  NOT NULL,
  version               VARCHAR(32)  NOT NULL,
  checksum              CHAR(64)     NOT NULL,
  applied_at            DATETIME(6)  NOT NULL,
  PRIMARY KEY (component)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ---------------------------------------------------------------------------
-- 2. course
-- PK: course_id  (CRS-000001)
-- Created before learner so current_course_id can reference it inline.
-- ---------------------------------------------------------------------------
CREATE TABLE course (
  course_id             VARCHAR(16)  NOT NULL,
  title                 VARCHAR(200) NOT NULL,
  domain                VARCHAR(80)  NOT NULL,
  difficulty_band       ENUM('intro', 'core', 'advanced') NOT NULL DEFAULT 'core',
  status                ENUM('draft', 'published', 'archived') NOT NULL DEFAULT 'draft',
  published_at          DATETIME(6)  NULL,
  archived_at           DATETIME(6)  NULL,
  created_at            DATETIME(6)  NOT NULL,
  updated_at            DATETIME(6)  NOT NULL,
  effective_from        DATETIME(6)  NOT NULL,
  effective_to          DATETIME(6)  NULL,
  PRIMARY KEY (course_id),
  UNIQUE KEY uq_course_title_domain (title, domain),
  KEY ix_course_status (status),
  KEY ix_course_domain (domain),
  CONSTRAINT ck_course_id CHECK (course_id REGEXP '^CRS-[0-9]{6}$'),
  CONSTRAINT ck_course_publish CHECK (
    (status <> 'published') OR published_at IS NOT NULL
  ),
  CONSTRAINT ck_course_effective CHECK (effective_to IS NULL OR effective_to >= effective_from)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ---------------------------------------------------------------------------
-- 1. learner
-- PK: learner_id  (LRN-000001)
-- ---------------------------------------------------------------------------
CREATE TABLE learner (
  learner_id            VARCHAR(16)  NOT NULL,
  display_name          VARCHAR(120) NOT NULL,
  cohort_code           VARCHAR(32)  NULL,
  starting_level        ENUM('beginner', 'intermediate', 'advanced') NOT NULL DEFAULT 'beginner',
  status                ENUM('active', 'paused', 'completed', 'withdrawn') NOT NULL DEFAULT 'active',
  current_course_id     VARCHAR(16)  NULL,
  timezone              VARCHAR(64)  NOT NULL DEFAULT 'UTC',
  created_at            DATETIME(6)  NOT NULL,
  updated_at            DATETIME(6)  NOT NULL,
  effective_from        DATETIME(6)  NOT NULL,
  effective_to          DATETIME(6)  NULL,
  PRIMARY KEY (learner_id),
  KEY ix_learner_status (status),
  KEY ix_learner_cohort (cohort_code),
  KEY ix_learner_current_course (current_course_id),
  CONSTRAINT fk_learner_current_course
    FOREIGN KEY (current_course_id) REFERENCES course (course_id)
    ON UPDATE RESTRICT ON DELETE SET NULL,
  CONSTRAINT ck_learner_id CHECK (learner_id REGEXP '^LRN-[0-9]{6}$'),
  CONSTRAINT ck_learner_effective CHECK (effective_to IS NULL OR effective_to >= effective_from)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ---------------------------------------------------------------------------
-- 3. skill
-- PK: skill_id  (SKL-000001)
-- ---------------------------------------------------------------------------
CREATE TABLE skill (
  skill_id              VARCHAR(16)  NOT NULL,
  skill_name            VARCHAR(160) NOT NULL,
  domain                VARCHAR(80)  NOT NULL,
  bloom_level           ENUM('remember', 'understand', 'apply', 'analyze', 'evaluate', 'create')
                        NOT NULL DEFAULT 'understand',
  status                ENUM('active', 'deprecated') NOT NULL DEFAULT 'active',
  created_at            DATETIME(6)  NOT NULL,
  updated_at            DATETIME(6)  NOT NULL,
  effective_from        DATETIME(6)  NOT NULL,
  effective_to          DATETIME(6)  NULL,
  PRIMARY KEY (skill_id),
  UNIQUE KEY uq_skill_name_domain (skill_name, domain),
  KEY ix_skill_domain (domain),
  KEY ix_skill_status (status),
  CONSTRAINT ck_skill_id CHECK (skill_id REGEXP '^SKL-[0-9]{6}$'),
  CONSTRAINT ck_skill_effective CHECK (effective_to IS NULL OR effective_to >= effective_from)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ---------------------------------------------------------------------------
-- 4. skill_prerequisite  (directed edges; DAG enforced in application logic)
-- PK: (skill_id, prerequisite_skill_id)
-- ---------------------------------------------------------------------------
CREATE TABLE skill_prerequisite (
  skill_id                  VARCHAR(16)    NOT NULL,
  prerequisite_skill_id     VARCHAR(16)    NOT NULL,
  relationship_type         ENUM('required', 'recommended') NOT NULL DEFAULT 'required',
  min_mastery_required      DECIMAL(6,4)   NOT NULL DEFAULT 0.6000,
  created_at                DATETIME(6)    NOT NULL,
  effective_from            DATETIME(6)    NOT NULL,
  effective_to              DATETIME(6)    NULL,
  PRIMARY KEY (skill_id, prerequisite_skill_id),
  KEY ix_prereq_prerequisite (prerequisite_skill_id),
  CONSTRAINT fk_prereq_skill
    FOREIGN KEY (skill_id) REFERENCES skill (skill_id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT fk_prereq_required_skill
    FOREIGN KEY (prerequisite_skill_id) REFERENCES skill (skill_id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT ck_prereq_not_self CHECK (skill_id <> prerequisite_skill_id),
  CONSTRAINT ck_prereq_mastery CHECK (
    min_mastery_required >= 0.0000 AND min_mastery_required <= 1.0000
  ),
  CONSTRAINT ck_prereq_effective CHECK (effective_to IS NULL OR effective_to >= effective_from)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ---------------------------------------------------------------------------
-- 5. content_item
-- PK: content_id  (CNT-000001)
-- ---------------------------------------------------------------------------
CREATE TABLE content_item (
  content_id            VARCHAR(16)    NOT NULL,
  course_id             VARCHAR(16)    NOT NULL,
  title                 VARCHAR(200)   NOT NULL,
  content_type          ENUM('video', 'article', 'quiz', 'exercise', 'project') NOT NULL,
  difficulty            DECIMAL(3,2)   NOT NULL,
  estimated_minutes     SMALLINT UNSIGNED NOT NULL,
  status                ENUM('draft', 'published', 'retired') NOT NULL DEFAULT 'draft',
  created_at            DATETIME(6)    NOT NULL,
  updated_at            DATETIME(6)    NOT NULL,
  effective_from        DATETIME(6)    NOT NULL,
  effective_to          DATETIME(6)    NULL,
  PRIMARY KEY (content_id),
  KEY ix_content_course (course_id),
  KEY ix_content_type_status (content_type, status),
  KEY ix_content_difficulty (difficulty),
  CONSTRAINT fk_content_course
    FOREIGN KEY (course_id) REFERENCES course (course_id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT ck_content_id CHECK (content_id REGEXP '^CNT-[0-9]{6}$'),
  CONSTRAINT ck_content_difficulty CHECK (difficulty >= 1.00 AND difficulty <= 5.00),
  CONSTRAINT ck_content_minutes CHECK (estimated_minutes BETWEEN 1 AND 600),
  CONSTRAINT ck_content_effective CHECK (effective_to IS NULL OR effective_to >= effective_from)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ---------------------------------------------------------------------------
-- 6. content_skill  (many-to-many bridge)
-- PK: (content_id, skill_id)
-- ---------------------------------------------------------------------------
CREATE TABLE content_skill (
  content_id            VARCHAR(16)    NOT NULL,
  skill_id              VARCHAR(16)    NOT NULL,
  skill_role            ENUM('primary', 'supporting') NOT NULL DEFAULT 'primary',
  weight                DECIMAL(6,4)   NOT NULL DEFAULT 1.0000,
  created_at            DATETIME(6)    NOT NULL,
  PRIMARY KEY (content_id, skill_id),
  KEY ix_content_skill_skill (skill_id),
  KEY ix_content_skill_role (skill_role),
  CONSTRAINT fk_content_skill_content
    FOREIGN KEY (content_id) REFERENCES content_item (content_id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT fk_content_skill_skill
    FOREIGN KEY (skill_id) REFERENCES skill (skill_id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT ck_content_skill_weight CHECK (weight > 0.0000 AND weight <= 1.0000)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ---------------------------------------------------------------------------
-- 9. learning_path
-- PK: path_id  (PATH-YYYY-000001) where YYYY is the UTC calendar year of created_at
-- At most one ACTIVE path per (learner_id, course_id) via generated unique column.
-- Multiple draft/completed/abandoned paths are allowed.
-- ---------------------------------------------------------------------------
CREATE TABLE learning_path (
  path_id               VARCHAR(24)  NOT NULL,
  learner_id            VARCHAR(16)  NOT NULL,
  course_id             VARCHAR(16)  NOT NULL,
  status                ENUM('draft', 'active', 'completed', 'abandoned') NOT NULL DEFAULT 'draft',
  generated_at          DATETIME(6)  NOT NULL,
  activated_at          DATETIME(6)  NULL,
  completed_at          DATETIME(6)  NULL,
  generator_version     VARCHAR(32)  NOT NULL DEFAULT 'rule-v1',
  created_at            DATETIME(6)  NOT NULL,
  updated_at            DATETIME(6)  NOT NULL,
  active_learner_course VARCHAR(40)
    GENERATED ALWAYS AS (
      IF(status = 'active', CONCAT(learner_id, '#', course_id), NULL)
    ) STORED,
  PRIMARY KEY (path_id),
  UNIQUE KEY uq_learning_path_one_active (active_learner_course),
  KEY ix_path_learner_status (learner_id, status),
  KEY ix_path_learner_course (learner_id, course_id),
  KEY ix_path_course (course_id),
  KEY ix_path_generated_at (generated_at),
  CONSTRAINT fk_path_learner
    FOREIGN KEY (learner_id) REFERENCES learner (learner_id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT fk_path_course
    FOREIGN KEY (course_id) REFERENCES course (course_id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT ck_path_id CHECK (path_id REGEXP '^PATH-[0-9]{4}-[0-9]{6}$'),
  CONSTRAINT ck_path_active_ts CHECK (
    (status <> 'active') OR activated_at IS NOT NULL
  )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ---------------------------------------------------------------------------
-- 10. recommendation
-- PK: recommendation_id  (REC-000001)
-- as_of_timestamp is the decision time; features must use data <= this instant.
-- Ranking never receives content that failed the prerequisite eligibility gate.
-- ---------------------------------------------------------------------------
CREATE TABLE recommendation (
  recommendation_id     VARCHAR(16)  NOT NULL,
  learner_id            VARCHAR(16)  NOT NULL,
  content_id            VARCHAR(16)  NOT NULL,
  path_id               VARCHAR(24)  NULL,
  rank_position         TINYINT UNSIGNED NOT NULL,
  engine_version        VARCHAR(32)  NOT NULL DEFAULT 'rule-v1',
  reason_codes          JSON         NOT NULL,
  score                 DECIMAL(8,6) NULL,
  status                ENUM('pending', 'accepted', 'dismissed', 'expired', 'completed', 'quarantined')
                        NOT NULL DEFAULT 'pending',
  recommended_at        DATETIME(6)  NOT NULL,
  as_of_timestamp       DATETIME(6)  NOT NULL,
  expires_at            DATETIME(6)  NULL,
  created_at            DATETIME(6)  NOT NULL,
  PRIMARY KEY (recommendation_id),
  UNIQUE KEY uq_rec_learner_content_asof_engine (learner_id, content_id, as_of_timestamp, engine_version),
  UNIQUE KEY uq_rec_learner_asof_engine_rank (learner_id, as_of_timestamp, engine_version, rank_position),
  KEY ix_rec_learner_time (learner_id, recommended_at),
  KEY ix_rec_content (content_id),
  KEY ix_rec_path (path_id),
  KEY ix_rec_status (status),
  CONSTRAINT fk_rec_learner
    FOREIGN KEY (learner_id) REFERENCES learner (learner_id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT fk_rec_content
    FOREIGN KEY (content_id) REFERENCES content_item (content_id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT fk_rec_path
    FOREIGN KEY (path_id) REFERENCES learning_path (path_id)
    ON UPDATE RESTRICT ON DELETE SET NULL,
  CONSTRAINT ck_rec_id CHECK (recommendation_id REGEXP '^REC-[0-9]{6}$'),
  CONSTRAINT ck_rec_rank CHECK (rank_position BETWEEN 1 AND 20),
  CONSTRAINT ck_rec_as_of CHECK (as_of_timestamp <= recommended_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ---------------------------------------------------------------------------
-- Support: learning_path_item
-- PK: path_item_id  (LPI-000001)
-- Ordered items on a path. Unique (path_id, sequence_no).
-- Duplicate content on a path is allowed so loop anomalies can be stored.
-- ---------------------------------------------------------------------------
CREATE TABLE learning_path_item (
  path_item_id          VARCHAR(16)  NOT NULL,
  path_id               VARCHAR(24)  NOT NULL,
  sequence_no           SMALLINT UNSIGNED NOT NULL,
  content_id            VARCHAR(16)  NOT NULL,
  recommendation_id     VARCHAR(16)  NULL,
  status                ENUM('planned', 'in_progress', 'completed', 'skipped') NOT NULL DEFAULT 'planned',
  planned_at            DATETIME(6)  NOT NULL,
  completed_at          DATETIME(6)  NULL,
  PRIMARY KEY (path_item_id),
  UNIQUE KEY uq_path_sequence (path_id, sequence_no),
  KEY ix_path_item_content (content_id),
  KEY ix_path_item_recommendation (recommendation_id),
  CONSTRAINT fk_path_item_path
    FOREIGN KEY (path_id) REFERENCES learning_path (path_id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT fk_path_item_content
    FOREIGN KEY (content_id) REFERENCES content_item (content_id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT fk_path_item_recommendation
    FOREIGN KEY (recommendation_id) REFERENCES recommendation (recommendation_id)
    ON UPDATE RESTRICT ON DELETE SET NULL,
  CONSTRAINT ck_path_item_id CHECK (path_item_id REGEXP '^LPI-[0-9]{6}$'),
  CONSTRAINT ck_path_item_seq CHECK (sequence_no >= 1)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ---------------------------------------------------------------------------
-- 7. attempt
-- PK: attempt_id  (ATT-LRN-000001-000001)
-- submitted_at must be >= started_at when present.
-- ---------------------------------------------------------------------------
CREATE TABLE attempt (
  attempt_id            VARCHAR(32)    NOT NULL,
  learner_id            VARCHAR(16)    NOT NULL,
  content_id            VARCHAR(16)    NOT NULL,
  attempt_number        SMALLINT UNSIGNED NOT NULL,
  status                ENUM('in_progress', 'submitted', 'scored', 'abandoned', 'invalid')
                        NOT NULL DEFAULT 'in_progress',
  started_at            DATETIME(6)    NOT NULL,
  submitted_at          DATETIME(6)    NULL,
  score                 DECIMAL(5,2)   NULL,
  max_score             DECIMAL(5,2)   NOT NULL DEFAULT 100.00,
  source_recommendation_id VARCHAR(16) NULL,
  created_at            DATETIME(6)    NOT NULL,
  PRIMARY KEY (attempt_id),
  UNIQUE KEY uq_attempt_learner_content_number (learner_id, content_id, attempt_number),
  KEY ix_attempt_learner_started (learner_id, started_at),
  KEY ix_attempt_content (content_id),
  KEY ix_attempt_status (status),
  KEY ix_attempt_recommendation (source_recommendation_id),
  CONSTRAINT fk_attempt_learner
    FOREIGN KEY (learner_id) REFERENCES learner (learner_id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT fk_attempt_content
    FOREIGN KEY (content_id) REFERENCES content_item (content_id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT fk_attempt_recommendation
    FOREIGN KEY (source_recommendation_id) REFERENCES recommendation (recommendation_id)
    ON UPDATE RESTRICT ON DELETE SET NULL,
  CONSTRAINT ck_attempt_id CHECK (attempt_id REGEXP '^ATT-LRN-[0-9]{6}-[0-9]{6}$'),
  CONSTRAINT ck_attempt_number CHECK (attempt_number >= 1),
  CONSTRAINT ck_attempt_times CHECK (submitted_at IS NULL OR submitted_at >= started_at),
  CONSTRAINT ck_attempt_score CHECK (score IS NULL OR (score >= 0.00 AND score <= max_score)),
  CONSTRAINT ck_attempt_max_score CHECK (max_score > 0.00)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ---------------------------------------------------------------------------
-- 8. assessment_result
-- PK: result_id  (RES-000001)
-- 1:1 with a scored attempt.
-- ---------------------------------------------------------------------------
CREATE TABLE assessment_result (
  result_id             VARCHAR(16)    NOT NULL,
  attempt_id            VARCHAR(32)    NOT NULL,
  outcome               ENUM('pass', 'fail', 'incomplete') NOT NULL,
  score                 DECIMAL(5,2)   NOT NULL,
  passed                TINYINT(1)     NOT NULL,
  mastery_delta         DECIMAL(7,4)   NULL,
  scored_at             DATETIME(6)    NOT NULL,
  created_at            DATETIME(6)    NOT NULL,
  PRIMARY KEY (result_id),
  UNIQUE KEY uq_result_attempt (attempt_id),
  KEY ix_result_outcome (outcome),
  KEY ix_result_scored_at (scored_at),
  CONSTRAINT fk_result_attempt
    FOREIGN KEY (attempt_id) REFERENCES attempt (attempt_id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT ck_result_id CHECK (result_id REGEXP '^RES-[0-9]{6}$'),
  CONSTRAINT ck_result_score CHECK (score >= 0.00),
  CONSTRAINT ck_result_passed CHECK (passed IN (0, 1)),
  CONSTRAINT ck_result_pass_consistency CHECK (
    (outcome = 'pass' AND passed = 1) OR (outcome <> 'pass' AND passed = 0)
  )
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ---------------------------------------------------------------------------
-- Support: mastery_snapshot
-- Point-in-time mastery. Unique (learner_id, skill_id, as_of_timestamp).
-- evidence_cutoff_ts must be <= as_of_timestamp (no future attempts).
-- ---------------------------------------------------------------------------
CREATE TABLE mastery_snapshot (
  snapshot_id           VARCHAR(16)    NOT NULL,
  learner_id            VARCHAR(16)    NOT NULL,
  skill_id              VARCHAR(16)    NOT NULL,
  mastery_score         DECIMAL(6,4)   NOT NULL,
  evidence_attempt_count INT UNSIGNED  NOT NULL DEFAULT 0,
  as_of_timestamp       DATETIME(6)    NOT NULL,
  evidence_cutoff_ts    DATETIME(6)    NOT NULL,
  calculator_version    VARCHAR(32)    NOT NULL DEFAULT 'mastery-v1',
  created_at            DATETIME(6)    NOT NULL,
  PRIMARY KEY (snapshot_id),
  UNIQUE KEY uq_mastery_learner_skill_asof (learner_id, skill_id, as_of_timestamp),
  KEY ix_mastery_skill (skill_id),
  KEY ix_mastery_as_of (as_of_timestamp),
  CONSTRAINT fk_mastery_learner
    FOREIGN KEY (learner_id) REFERENCES learner (learner_id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT fk_mastery_skill
    FOREIGN KEY (skill_id) REFERENCES skill (skill_id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT ck_mastery_id CHECK (snapshot_id REGEXP '^MST-[0-9]{6}$'),
  CONSTRAINT ck_mastery_score CHECK (mastery_score >= 0.0000 AND mastery_score <= 1.0000),
  CONSTRAINT ck_mastery_no_future CHECK (evidence_cutoff_ts <= as_of_timestamp)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ---------------------------------------------------------------------------
-- Support: recommendation_outcome
-- Subsequent observed result of a recommendation. Simulated in this challenge.
-- Relevance: skill improvement is primary; completion is secondary.
-- ---------------------------------------------------------------------------
CREATE TABLE recommendation_outcome (
  outcome_id            VARCHAR(16)    NOT NULL,
  recommendation_id     VARCHAR(16)    NOT NULL,
  subsequent_attempt_id VARCHAR(32)    NULL,
  window_hours          INT UNSIGNED   NOT NULL DEFAULT 168,
  completed             TINYINT(1)     NOT NULL DEFAULT 0,
  improved              TINYINT(1)     NOT NULL DEFAULT 0,
  score_delta           DECIMAL(6,2)   NULL,
  observed_at           DATETIME(6)    NOT NULL,
  created_at            DATETIME(6)    NOT NULL,
  PRIMARY KEY (outcome_id),
  UNIQUE KEY uq_outcome_recommendation (recommendation_id),
  KEY ix_outcome_attempt (subsequent_attempt_id),
  KEY ix_outcome_improved (improved),
  CONSTRAINT fk_outcome_recommendation
    FOREIGN KEY (recommendation_id) REFERENCES recommendation (recommendation_id)
    ON UPDATE RESTRICT ON DELETE RESTRICT,
  CONSTRAINT fk_outcome_attempt
    FOREIGN KEY (subsequent_attempt_id) REFERENCES attempt (attempt_id)
    ON UPDATE RESTRICT ON DELETE SET NULL,
  CONSTRAINT ck_outcome_id CHECK (outcome_id REGEXP '^ROC-[0-9]{6}$'),
  CONSTRAINT ck_outcome_flags CHECK (completed IN (0, 1) AND improved IN (0, 1))
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ---------------------------------------------------------------------------
-- Support: event_history
-- Append-only audit of decisions and state changes.
-- ---------------------------------------------------------------------------
CREATE TABLE event_history (
  event_id              VARCHAR(16)  NOT NULL,
  event_type            VARCHAR(64)  NOT NULL,
  entity_type           VARCHAR(64)  NOT NULL,
  entity_id             VARCHAR(32)  NOT NULL,
  payload               JSON         NOT NULL,
  occurred_at           DATETIME(6)  NOT NULL,
  recorded_at           DATETIME(6)  NOT NULL,
  actor                 VARCHAR(64)  NOT NULL DEFAULT 'system',
  PRIMARY KEY (event_id),
  KEY ix_event_entity (entity_type, entity_id),
  KEY ix_event_type_time (event_type, occurred_at),
  CONSTRAINT ck_event_id CHECK (event_id REGEXP '^EVT-[0-9]{6}$')
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;

-- ---------------------------------------------------------------------------
-- Support: data_quality_issue
-- Quarantine and audit of detected anomalies. Ground-truth labels stay off-model.
-- ---------------------------------------------------------------------------
CREATE TABLE data_quality_issue (
  issue_id              VARCHAR(16)  NOT NULL,
  issue_type            ENUM(
                          'REC_PREREQ_VIOLATION',
                          'ATTEMPT_TIMESTAMP_ORDER',
                          'CONTENT_MISSING_SKILL',
                          'MASTERY_FUTURE_ATTEMPT',
                          'PATH_LOOP_NO_PROGRESS'
                        ) NOT NULL,
  source_table          VARCHAR(64)  NOT NULL,
  source_pk             VARCHAR(32)  NOT NULL,
  severity              ENUM('low', 'medium', 'high', 'critical') NOT NULL DEFAULT 'high',
  status                ENUM('open', 'quarantined', 'accepted_exception', 'resolved')
                        NOT NULL DEFAULT 'quarantined',
  expected_decision     VARCHAR(64)  NOT NULL,
  details               JSON         NOT NULL,
  detected_at           DATETIME(6)  NOT NULL,
  created_at            DATETIME(6)  NOT NULL,
  PRIMARY KEY (issue_id),
  KEY ix_dqi_type_status (issue_type, status),
  KEY ix_dqi_source (source_table, source_pk),
  CONSTRAINT ck_dqi_id CHECK (issue_id REGEXP '^DQI-[0-9]{6}$')
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_0900_ai_ci;
