"""CSV column order matching `schemas/schema.sql` / the data dictionary."""

from __future__ import annotations

COURSE_COLUMNS = (
    "course_id",
    "title",
    "domain",
    "difficulty_band",
    "status",
    "published_at",
    "archived_at",
    "created_at",
    "updated_at",
    "effective_from",
    "effective_to",
)

LEARNER_COLUMNS = (
    "learner_id",
    "display_name",
    "cohort_code",
    "starting_level",
    "status",
    "current_course_id",
    "timezone",
    "created_at",
    "updated_at",
    "effective_from",
    "effective_to",
)

SKILL_COLUMNS = (
    "skill_id",
    "skill_name",
    "domain",
    "bloom_level",
    "status",
    "created_at",
    "updated_at",
    "effective_from",
    "effective_to",
)

SKILL_PREREQUISITE_COLUMNS = (
    "skill_id",
    "prerequisite_skill_id",
    "relationship_type",
    "min_mastery_required",
    "created_at",
    "effective_from",
    "effective_to",
)

CONTENT_ITEM_COLUMNS = (
    "content_id",
    "course_id",
    "title",
    "content_type",
    "difficulty",
    "estimated_minutes",
    "status",
    "created_at",
    "updated_at",
    "effective_from",
    "effective_to",
)

CONTENT_SKILL_COLUMNS = (
    "content_id",
    "skill_id",
    "skill_role",
    "weight",
    "created_at",
)

ATTEMPT_COLUMNS = (
    "attempt_id",
    "learner_id",
    "content_id",
    "attempt_number",
    "status",
    "started_at",
    "submitted_at",
    "score",
    "max_score",
    "source_recommendation_id",
    "created_at",
)

ASSESSMENT_RESULT_COLUMNS = (
    "result_id",
    "attempt_id",
    "outcome",
    "score",
    "passed",
    "mastery_delta",
    "scored_at",
    "created_at",
)

FILE_COLUMNS = {
    "courses.csv": COURSE_COLUMNS,
    "learners.csv": LEARNER_COLUMNS,
    "skills.csv": SKILL_COLUMNS,
    "skill_prerequisites.csv": SKILL_PREREQUISITE_COLUMNS,
    "content_items.csv": CONTENT_ITEM_COLUMNS,
    "content_skills.csv": CONTENT_SKILL_COLUMNS,
    "attempts.csv": ATTEMPT_COLUMNS,
    "assessment_results.csv": ASSESSMENT_RESULT_COLUMNS,
}

MASTER_FILES = (
    "courses.csv",
    "learners.csv",
    "skills.csv",
    "skill_prerequisites.csv",
    "content_items.csv",
    "content_skills.csv",
)

ACTIVITY_FILES = (
    "attempts.csv",
    "assessment_results.csv",
)

MANIFEST_NAME = "master_manifest.json"
ACTIVITY_MANIFEST_NAME = "activity_manifest.json"

DATA_QUALITY_ISSUE_COLUMNS = (
    "issue_id",
    "issue_type",
    "source_table",
    "source_pk",
    "severity",
    "status",
    "expected_decision",
    "details",
    "detected_at",
    "created_at",
)

QUARANTINE_FILES = ("data_quality_issues.csv",)
QUALITY_MANIFEST_NAME = "quality_manifest.json"

MASTERY_SNAPSHOT_COLUMNS = (
    "snapshot_id",
    "learner_id",
    "skill_id",
    "mastery_score",
    "evidence_attempt_count",
    "as_of_timestamp",
    "evidence_cutoff_ts",
    "calculator_version",
    "created_at",
)

MASTERY_FILES = ("mastery_snapshots.csv",)
MASTERY_MANIFEST_NAME = "mastery_manifest.json"

RECOMMENDATION_COLUMNS = (
    "recommendation_id",
    "learner_id",
    "content_id",
    "path_id",
    "rank_position",
    "engine_version",
    "reason_codes",
    "score",
    "status",
    "recommended_at",
    "as_of_timestamp",
    "expires_at",
    "created_at",
)

RECOMMENDATION_FILES = ("recommendations.csv",)
RECOMMENDATION_MANIFEST_NAME = "recommendation_manifest.json"
