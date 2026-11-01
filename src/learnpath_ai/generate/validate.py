"""Master-data and activity validation for LearnPath AI."""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from learnpath_ai.config_loader import load_app_config, profile_counts
from learnpath_ai.generate.catalog import (
    ASSESSMENT_OUTCOMES,
    ATTEMPT_STATUSES,
    BLOOM_LEVELS,
    CONTENT_STATUSES,
    CONTENT_TYPES,
    COURSE_STATUSES,
    DIFFICULTY_BANDS,
    LEARNER_LEVELS,
    LEARNER_STATUSES,
    RELATIONSHIP_TYPES,
    SKILL_ROLES,
    SKILL_STATUSES,
)
from learnpath_ai.generate.columns import (
    ACTIVITY_FILES,
    ACTIVITY_MANIFEST_NAME,
    FILE_COLUMNS,
    MANIFEST_NAME,
    MASTER_FILES,
)
from learnpath_ai.ids import is_valid_id
from learnpath_ai.prerequisites import PrerequisiteEdge, PrerequisiteGraphError, require_dag


class MasterDataError(ValueError):
    """Raised when master data integrity, schema, or consistency checks fail."""


class ActivityDataError(ValueError):
    """Raised when activity / assessment data integrity, schema, or consistency checks fail."""


def _parse_ts(val: str, field_name: str, table_name: str, row_id: str) -> datetime:
    text = str(val).strip()
    if not text:
        raise MasterDataError(
            f"Table {table_name!r} record {row_id!r} has empty timestamp for field {field_name!r}"
        )
    if "T" in text or "+" in text or text.endswith("Z"):
        clean = text.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(clean)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except Exception as exc:
            raise MasterDataError(
                f"Table {table_name!r} record {row_id!r} has invalid timestamp {val!r} for {field_name!r}"
            ) from exc
    try:
        if "." in text:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S.%f")
        else:
            parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        return parsed.replace(tzinfo=timezone.utc)
    except Exception as exc:
        raise MasterDataError(
            f"Table {table_name!r} record {row_id!r} has invalid timestamp {val!r} for {field_name!r}"
        ) from exc


def _check_effective_dates(
    row: dict[str, Any], table_name: str, row_id: str
) -> None:
    ef_str = str(row.get("effective_from") or "").strip()
    if not ef_str:
        raise MasterDataError(
            f"Table {table_name!r} record {row_id!r} missing required 'effective_from'."
        )
    ef = _parse_ts(ef_str, "effective_from", table_name, row_id)

    et_str = str(row.get("effective_to") or "").strip()
    if et_str:
        et = _parse_ts(et_str, "effective_to", table_name, row_id)
        if et < ef:
            raise MasterDataError(
                f"Table {table_name!r} record {row_id!r} has effective_to ({et_str}) < effective_from ({ef_str})."
            )

    created_str = str(row.get("created_at") or "").strip()
    updated_str = str(row.get("updated_at") or "").strip()
    if created_str and updated_str:
        c_ts = _parse_ts(created_str, "created_at", table_name, row_id)
        u_ts = _parse_ts(updated_str, "updated_at", table_name, row_id)
        if u_ts < c_ts:
            raise MasterDataError(
                f"Table {table_name!r} record {row_id!r} has updated_at ({updated_str}) < created_at ({created_str})."
            )


def validate_master_tables(
    tables: Any,
    *,
    expected_counts: dict[str, int] | None = None,
    profile: str | None = None,
) -> None:
    """Validate in-memory master tables against schema rules, constraints, and DAG acyclicity."""
    if hasattr(tables, "courses"):
        courses = tables.courses
        learners = tables.learners
        skills = tables.skills
        skill_prerequisites = tables.skill_prerequisites
        content_items = tables.content_items
        content_skills = tables.content_skills
        dataset_profile = profile or getattr(tables, "profile", None)
    elif isinstance(tables, dict):
        courses = tables.get("courses") or []
        learners = tables.get("learners") or []
        skills = tables.get("skills") or []
        skill_prerequisites = tables.get("skill_prerequisites") or []
        content_items = tables.get("content_items") or []
        content_skills = tables.get("content_skills") or []
        dataset_profile = profile or tables.get("profile")
    else:
        raise MasterDataError(f"Unsupported tables type: {type(tables)}")

    # 1. Expected dataset counts
    target_counts = expected_counts
    if target_counts is None and dataset_profile:
        try:
            cfg = load_app_config()
            target_counts = profile_counts(cfg, dataset_profile)
        except Exception:
            target_counts = None

    if target_counts:
        actual_counts = {
            "courses": len(courses),
            "learners": len(learners),
            "skills": len(skills),
            "content_items": len(content_items),
        }
        for key in ("courses", "learners", "skills", "content_items"):
            if key in target_counts and actual_counts[key] != target_counts[key]:
                raise MasterDataError(
                    f"Row count mismatch for {key!r}: expected {target_counts[key]}, got {actual_counts[key]}"
                )

    # 2. Courses validation
    course_ids: set[str] = set()
    course_titles_domain: set[tuple[str, str]] = set()
    for row in courses:
        cid = str(row.get("course_id") or "").strip()
        if not cid:
            raise MasterDataError("Course record missing 'course_id'.")
        if not is_valid_id(cid, "CRS"):
            raise MasterDataError(f"Invalid course_id format: {cid!r}")
        if cid in course_ids:
            raise MasterDataError(f"Duplicate course_id: {cid!r}")
        course_ids.add(cid)

        title = str(row.get("title") or "").strip()
        domain = str(row.get("domain") or "").strip()
        if not title:
            raise MasterDataError(f"Course {cid} missing required 'title'.")
        if not domain:
            raise MasterDataError(f"Course {cid} missing required 'domain'.")
        key = (title, domain)
        if key in course_titles_domain:
            raise MasterDataError(f"Duplicate course (title, domain) combination: {key}")
        course_titles_domain.add(key)

        diff = str(row.get("difficulty_band") or "").strip()
        if diff not in DIFFICULTY_BANDS:
            raise MasterDataError(
                f"Course {cid} invalid difficulty_band {diff!r}. Allowed: {DIFFICULTY_BANDS}"
            )

        status = str(row.get("status") or "").strip()
        if status not in COURSE_STATUSES:
            raise MasterDataError(
                f"Course {cid} invalid status {status!r}. Allowed: {COURSE_STATUSES}"
            )

        pub_at = str(row.get("published_at") or "").strip()
        if status == "published" and not pub_at:
            raise MasterDataError(f"Course {cid} has status 'published' but published_at is missing.")
        if pub_at:
            _parse_ts(pub_at, "published_at", "course", cid)

        arch_at = str(row.get("archived_at") or "").strip()
        if status == "archived" and not arch_at:
            raise MasterDataError(f"Course {cid} has status 'archived' but archived_at is missing.")
        if arch_at:
            _parse_ts(arch_at, "archived_at", "course", cid)

        _check_effective_dates(row, "course", cid)

    # 3. Skills validation
    skill_ids: set[str] = set()
    skill_names_domain: set[tuple[str, str]] = set()
    for row in skills:
        skid = str(row.get("skill_id") or "").strip()
        if not skid:
            raise MasterDataError("Skill record missing 'skill_id'.")
        if not is_valid_id(skid, "SKL"):
            raise MasterDataError(f"Invalid skill_id format: {skid!r}")
        if skid in skill_ids:
            raise MasterDataError(f"Duplicate skill_id: {skid!r}")
        skill_ids.add(skid)

        sname = str(row.get("skill_name") or "").strip()
        domain = str(row.get("domain") or "").strip()
        if not sname:
            raise MasterDataError(f"Skill {skid} missing required 'skill_name'.")
        if not domain:
            raise MasterDataError(f"Skill {skid} missing required 'domain'.")
        key = (sname, domain)
        if key in skill_names_domain:
            raise MasterDataError(f"Duplicate skill (skill_name, domain) combination: {key}")
        skill_names_domain.add(key)

        bloom = str(row.get("bloom_level") or "").strip()
        if bloom not in BLOOM_LEVELS:
            raise MasterDataError(
                f"Skill {skid} invalid bloom_level {bloom!r}. Allowed: {BLOOM_LEVELS}"
            )

        status = str(row.get("status") or "").strip()
        if status not in SKILL_STATUSES:
            raise MasterDataError(
                f"Skill {skid} invalid status {status!r}. Allowed: {SKILL_STATUSES}"
            )

        _check_effective_dates(row, "skill", skid)

    # 4. SkillPrerequisites validation (DAG + constraints)
    seen_prereq_edges: set[tuple[str, str]] = set()
    dag_edges: list[PrerequisiteEdge] = []
    for row in skill_prerequisites:
        skid = str(row.get("skill_id") or "").strip()
        prereq_id = str(row.get("prerequisite_skill_id") or "").strip()
        pair_id = f"({skid}, {prereq_id})"

        if not skid:
            raise MasterDataError(f"SkillPrerequisite record {pair_id} missing 'skill_id'.")
        if not prereq_id:
            raise MasterDataError(f"SkillPrerequisite record {pair_id} missing 'prerequisite_skill_id'.")

        if skid not in skill_ids:
            raise MasterDataError(
                f"Foreign key violation: skill_id {skid!r} does not exist in skills."
            )
        if prereq_id not in skill_ids:
            raise MasterDataError(
                f"Foreign key violation: prerequisite_skill_id {prereq_id!r} does not exist in skills."
            )

        if skid == prereq_id:
            raise MasterDataError(
                f"Skill prerequisite self-reference is not allowed: {skid} -> {prereq_id}"
            )

        edge_key = (skid, prereq_id)
        if edge_key in seen_prereq_edges:
            raise MasterDataError(f"Duplicate skill_prerequisite edge: {edge_key}")
        seen_prereq_edges.add(edge_key)

        rel = str(row.get("relationship_type") or "").strip()
        if rel not in RELATIONSHIP_TYPES:
            raise MasterDataError(
                f"SkillPrerequisite {pair_id} invalid relationship_type {rel!r}. Allowed: {RELATIONSHIP_TYPES}"
            )

        raw_mastery = str(row.get("min_mastery_required") or "").strip()
        try:
            mastery = float(raw_mastery)
            if mastery < 0.0 or mastery > 1.0:
                raise ValueError()
        except ValueError as exc:
            raise MasterDataError(
                f"SkillPrerequisite {pair_id} min_mastery_required ({raw_mastery}) must be between 0.0 and 1.0"
            ) from exc

        _check_effective_dates(row, "skill_prerequisite", pair_id)
        dag_edges.append(
            PrerequisiteEdge(
                skill_id=skid,
                prerequisite_skill_id=prereq_id,
                relationship_type=rel,
                min_mastery_required=mastery,
            )
        )

    # DAG cycle check
    try:
        require_dag(dag_edges)
    except PrerequisiteGraphError as exc:
        raise MasterDataError(f"Prerequisite graph cycle detected: {exc}") from exc

    # 5. ContentItems validation
    content_ids: set[str] = set()
    for row in content_items:
        cnt_id = str(row.get("content_id") or "").strip()
        if not cnt_id:
            raise MasterDataError("ContentItem record missing 'content_id'.")
        if not is_valid_id(cnt_id, "CNT"):
            raise MasterDataError(f"Invalid content_id format: {cnt_id!r}")
        if cnt_id in content_ids:
            raise MasterDataError(f"Duplicate content_id: {cnt_id!r}")
        content_ids.add(cnt_id)

        course_id = str(row.get("course_id") or "").strip()
        if not course_id:
            raise MasterDataError(f"ContentItem {cnt_id} missing 'course_id'.")
        if course_id not in course_ids:
            raise MasterDataError(
                f"Foreign key violation: ContentItem {cnt_id} references non-existent course_id {course_id!r}."
            )

        title = str(row.get("title") or "").strip()
        if not title:
            raise MasterDataError(f"ContentItem {cnt_id} missing required 'title'.")

        ctype = str(row.get("content_type") or "").strip()
        if ctype not in CONTENT_TYPES:
            raise MasterDataError(
                f"ContentItem {cnt_id} invalid content_type {ctype!r}. Allowed: {CONTENT_TYPES}"
            )

        status = str(row.get("status") or "").strip()
        if status not in CONTENT_STATUSES:
            raise MasterDataError(
                f"ContentItem {cnt_id} invalid status {status!r}. Allowed: {CONTENT_STATUSES}"
            )

        raw_diff = str(row.get("difficulty") or "").strip()
        try:
            diff = float(raw_diff)
            if diff < 1.00 or diff > 5.00:
                raise ValueError()
        except ValueError as exc:
            raise MasterDataError(
                f"ContentItem {cnt_id} difficulty ({raw_diff}) must be between 1.00 and 5.00"
            ) from exc

        raw_minutes = str(row.get("estimated_minutes") or "").strip()
        try:
            minutes = int(raw_minutes)
            if minutes < 1 or minutes > 600:
                raise ValueError()
        except ValueError as exc:
            raise MasterDataError(
                f"ContentItem {cnt_id} estimated_minutes ({raw_minutes}) must be between 1 and 600"
            ) from exc

        _check_effective_dates(row, "content_item", cnt_id)

    # 6. ContentSkills validation
    content_skill_pairs: set[tuple[str, str]] = set()
    content_roles: dict[str, list[str]] = {cid: [] for cid in content_ids}
    for row in content_skills:
        cnt_id = str(row.get("content_id") or "").strip()
        skid = str(row.get("skill_id") or "").strip()
        pair_id = f"({cnt_id}, {skid})"

        if not cnt_id:
            raise MasterDataError("ContentSkill missing 'content_id'.")
        if not skid:
            raise MasterDataError("ContentSkill missing 'skill_id'.")

        if cnt_id not in content_ids:
            raise MasterDataError(
                f"Foreign key violation: ContentSkill references non-existent content_id {cnt_id!r}."
            )
        if skid not in skill_ids:
            raise MasterDataError(
                f"Foreign key violation: ContentSkill references non-existent skill_id {skid!r}."
            )

        pair_key = (cnt_id, skid)
        if pair_key in content_skill_pairs:
            raise MasterDataError(f"Duplicate content_skill mapping: {pair_key}")
        content_skill_pairs.add(pair_key)

        role = str(row.get("skill_role") or "").strip()
        if role not in SKILL_ROLES:
            raise MasterDataError(
                f"ContentSkill {pair_id} invalid skill_role {role!r}. Allowed: {SKILL_ROLES}"
            )
        content_roles[cnt_id].append(role)

        raw_weight = str(row.get("weight") or "").strip()
        try:
            weight = float(raw_weight)
            if weight <= 0.0 or weight > 1.0:
                raise ValueError()
        except ValueError as exc:
            raise MasterDataError(
                f"ContentSkill {pair_id} weight ({raw_weight}) must be > 0.0000 and <= 1.0000"
            ) from exc

        c_str = str(row.get("created_at") or "").strip()
        if not c_str:
            raise MasterDataError(f"ContentSkill {pair_id} missing required 'created_at'.")
        _parse_ts(c_str, "created_at", "content_skill", pair_id)

    # Check that published content items have at least one primary skill
    content_status_map = {row["content_id"]: row.get("status") for row in content_items}
    for cid, roles in content_roles.items():
        if not roles:
            raise MasterDataError(f"ContentItem {cid} has no associated skills in content_skill.")
        if content_status_map.get(cid) == "published" and "primary" not in roles:
            raise MasterDataError(
                f"Published ContentItem {cid} must have at least one 'primary' skill mapped."
            )

    # 7. Learners validation
    learner_ids: set[str] = set()
    for row in learners:
        lid = str(row.get("learner_id") or "").strip()
        if not lid:
            raise MasterDataError("Learner record missing 'learner_id'.")
        if not is_valid_id(lid, "LRN"):
            raise MasterDataError(f"Invalid learner_id format: {lid!r}")
        if lid in learner_ids:
            raise MasterDataError(f"Duplicate learner_id: {lid!r}")
        learner_ids.add(lid)

        dname = str(row.get("display_name") or "").strip()
        if not dname:
            raise MasterDataError(f"Learner {lid} missing required 'display_name'.")

        level = str(row.get("starting_level") or "").strip()
        if level not in LEARNER_LEVELS:
            raise MasterDataError(
                f"Learner {lid} invalid starting_level {level!r}. Allowed: {LEARNER_LEVELS}"
            )

        status = str(row.get("status") or "").strip()
        if status not in LEARNER_STATUSES:
            raise MasterDataError(
                f"Learner {lid} invalid status {status!r}. Allowed: {LEARNER_STATUSES}"
            )

        current_course = str(row.get("current_course_id") or "").strip()
        if current_course:
            if current_course not in course_ids:
                raise MasterDataError(
                    f"Foreign key violation: Learner {lid} references non-existent course_id {current_course!r}."
                )

        tz = str(row.get("timezone") or "").strip()
        if not tz:
            raise MasterDataError(f"Learner {lid} missing required 'timezone'.")

        _check_effective_dates(row, "learner", lid)


def validate_activity_tables(
    activity_tables: Any,
    master_tables: Any,
    *,
    expected_attempts: int | None = None,
    profile: str | None = None,
) -> None:
    """Validate attempts and assessment_results against schema and master tables."""
    if hasattr(activity_tables, "attempts"):
        attempts = activity_tables.attempts
        results = activity_tables.assessment_results
        dataset_profile = profile or getattr(activity_tables, "profile", None)
    elif isinstance(activity_tables, dict):
        attempts = activity_tables.get("attempts", [])
        results = activity_tables.get("assessment_results", [])
        dataset_profile = profile or activity_tables.get("profile")
    else:
        raise ActivityDataError(f"Unsupported activity_tables type: {type(activity_tables)}")

    # Extract master sets
    if hasattr(master_tables, "learners"):
        m_learners = master_tables.learners
        m_courses = master_tables.courses
        m_contents = master_tables.content_items
    elif isinstance(master_tables, dict):
        m_learners = master_tables.get("learners", [])
        m_courses = master_tables.get("courses", [])
        m_contents = master_tables.get("content_items", [])
    else:
        raise ActivityDataError(f"Unsupported master_tables type: {type(master_tables)}")

    learner_ids = {l["learner_id"] for l in m_learners}
    course_ids = {c["course_id"] for c in m_courses}
    content_map = {c["content_id"]: c for c in m_contents}

    # 1. Expected attempt count check
    if expected_attempts is not None and len(attempts) != expected_attempts:
        raise ActivityDataError(
            f"Expected {expected_attempts} attempts, got {len(attempts)}"
        )

    # 2. Attempts validation
    attempt_ids: set[str] = set()
    learner_attempt_numbers: dict[str, set[int]] = {}
    learner_content_numbers: set[tuple[str, str, int]] = set()
    attempt_map: dict[str, dict[str, Any]] = {}
    learners_with_attempts: set[str] = set()

    for row in attempts:
        att_id = str(row.get("attempt_id") or "").strip()
        if not att_id:
            raise ActivityDataError("Attempt missing attempt_id")
        if not is_valid_id(att_id, "ATT"):
            raise ActivityDataError(f"Invalid attempt_id format: {att_id!r}")
        if att_id in attempt_ids:
            raise ActivityDataError(f"Duplicate attempt_id: {att_id}")
        attempt_ids.add(att_id)
        attempt_map[att_id] = row

        lid = str(row.get("learner_id") or "").strip()
        if not lid or lid not in learner_ids:
            raise ActivityDataError(f"Attempt {att_id} foreign key violation: invalid learner_id {lid!r}")
        learners_with_attempts.add(lid)

        # Check attempt_id format matches learner_id: ATT-<learner_id>-<seq>
        expected_prefix = f"ATT-{lid}-"
        if not att_id.startswith(expected_prefix):
            raise ActivityDataError(
                f"Attempt {att_id} ID prefix does not match learner_id {lid}"
            )
        seq_num = int(att_id[len(expected_prefix):])
        if seq_num < 1:
            raise ActivityDataError(f"Attempt {att_id} sequence must be >= 1")

        cnt_id = str(row.get("content_id") or "").strip()
        if not cnt_id or cnt_id not in content_map:
            raise ActivityDataError(f"Attempt {att_id} foreign key violation: invalid content_id {cnt_id!r}")

        # Course relationship check
        c_item = content_map[cnt_id]
        c_course_id = c_item.get("course_id")
        if not c_course_id or c_course_id not in course_ids:
            raise ActivityDataError(f"Attempt {att_id} content {cnt_id} belongs to invalid course {c_course_id!r}")

        # Attempt number
        raw_num = str(row.get("attempt_number") or "").strip()
        try:
            att_num = int(raw_num)
            if att_num < 1:
                raise ValueError()
        except ValueError as exc:
            raise ActivityDataError(
                f"Attempt {att_id} invalid attempt_number: {raw_num!r}"
            ) from exc

        # Unique key (learner_id, content_id, attempt_number)
        lc_key = (lid, cnt_id, att_num)
        if lc_key in learner_content_numbers:
            raise ActivityDataError(
                f"Duplicate attempt for learner+content+attempt_number: {lc_key}"
            )
        learner_content_numbers.add(lc_key)

        # Status
        status = str(row.get("status") or "").strip()
        if status not in ATTEMPT_STATUSES:
            raise ActivityDataError(f"Attempt {att_id} invalid status: {status!r}. Allowed: {ATTEMPT_STATUSES}")

        # Timestamps
        started_str = str(row.get("started_at") or "").strip()
        if not started_str:
            raise ActivityDataError(f"Attempt {att_id} missing started_at")
        started_dt = _parse_ts(started_str, "started_at", "attempt", att_id)

        submitted_str = str(row.get("submitted_at") or "").strip()
        if status in ("submitted", "scored") and not submitted_str:
            raise ActivityDataError(f"Attempt {att_id} status {status} requires non-empty submitted_at")
        if submitted_str:
            submitted_dt = _parse_ts(submitted_str, "submitted_at", "attempt", att_id)
            if submitted_dt < started_dt:
                raise ActivityDataError(
                    f"Attempt {att_id} submitted_at ({submitted_str}) < started_at ({started_str})"
                )

        # Scores
        raw_score = str(row.get("score") or "").strip()
        raw_max = str(row.get("max_score") or "").strip()
        try:
            max_s = float(raw_max)
            if max_s <= 0.0:
                raise ValueError()
        except ValueError as exc:
            raise ActivityDataError(f"Attempt {att_id} invalid max_score: {raw_max!r}") from exc

        if status == "scored":
            if not raw_score:
                raise ActivityDataError(f"Attempt {att_id} is scored but score is empty")
            try:
                s_val = float(raw_score)
                if s_val < 0.0 or s_val > max_s:
                    raise ValueError()
            except ValueError as exc:
                raise ActivityDataError(
                    f"Attempt {att_id} score ({raw_score}) must be between 0.00 and max_score ({max_s})"
                ) from exc

    # 3. Assessment results validation
    result_ids: set[str] = set()
    result_attempt_ids: set[str] = set()

    for row in results:
        res_id = str(row.get("result_id") or "").strip()
        if not res_id:
            raise ActivityDataError("AssessmentResult missing result_id")
        if not is_valid_id(res_id, "RES"):
            raise ActivityDataError(f"Invalid result_id format: {res_id!r}")
        if res_id in result_ids:
            raise ActivityDataError(f"Duplicate result_id: {res_id}")
        result_ids.add(res_id)

        att_id = str(row.get("attempt_id") or "").strip()
        if not att_id or att_id not in attempt_map:
            raise ActivityDataError(f"AssessmentResult {res_id} references nonexistent attempt: {att_id!r}")
        if att_id in result_attempt_ids:
            raise ActivityDataError(f"AssessmentResult {res_id} duplicate 1:1 attempt_id: {att_id}")
        result_attempt_ids.add(att_id)

        parent_att = attempt_map[att_id]
        if parent_att.get("status") != "scored":
            raise ActivityDataError(
                f"AssessmentResult {res_id} attached to non-scored attempt {att_id} (status: {parent_att.get('status')})"
            )

        outcome = str(row.get("outcome") or "").strip()
        if outcome not in ASSESSMENT_OUTCOMES:
            raise ActivityDataError(
                f"AssessmentResult {res_id} invalid outcome: {outcome!r}. Allowed: {ASSESSMENT_OUTCOMES}"
            )

        passed_str = str(row.get("passed") or "").strip()
        if passed_str not in ("0", "1"):
            raise ActivityDataError(f"AssessmentResult {res_id} passed flag must be 0 or 1, got {passed_str!r}")
        passed_int = int(passed_str)

        # Consistency: outcome == 'pass' <=> passed == 1
        if (outcome == "pass" and passed_int != 1) or (outcome != "pass" and passed_int == 1):
            raise ActivityDataError(
                f"AssessmentResult {res_id} pass consistency failure: outcome={outcome}, passed={passed_int}"
            )

        raw_score = str(row.get("score") or "").strip()
        try:
            res_score = float(raw_score)
            if res_score < 0.0:
                raise ValueError()
        except ValueError as exc:
            raise ActivityDataError(f"AssessmentResult {res_id} invalid score: {raw_score!r}") from exc

        # Score reconciles with parent attempt score
        parent_score = float(parent_att.get("score"))
        if abs(res_score - parent_score) > 0.001:
            raise ActivityDataError(
                f"AssessmentResult {res_id} score ({res_score}) does not match attempt score ({parent_score})"
            )

        # Scored_at >= attempt submitted_at
        scored_str = str(row.get("scored_at") or "").strip()
        if not scored_str:
            raise ActivityDataError(f"AssessmentResult {res_id} missing scored_at")
        scored_dt = _parse_ts(scored_str, "scored_at", "assessment_result", res_id)

        submitted_dt = _parse_ts(parent_att["submitted_at"], "submitted_at", "attempt", att_id)
        if scored_dt < submitted_dt:
            raise ActivityDataError(
                f"AssessmentResult {res_id} scored_at ({scored_str}) < attempt submitted_at ({parent_att['submitted_at']})"
            )

    # 4. 1:1 check: Every scored attempt must have exactly one assessment result
    scored_attempt_ids = {att["attempt_id"] for att in attempts if att.get("status") == "scored"}
    if scored_attempt_ids != result_attempt_ids:
        missing_results = scored_attempt_ids - result_attempt_ids
        if missing_results:
            sample_missing = list(missing_results)[:3]
            raise ActivityDataError(
                f"{len(missing_results)} scored attempts lack assessment results (e.g. {sample_missing})"
            )


def validate_master_dataset_dir(
    dir_path: Path | str,
    *,
    expected_profile: str | None = None,
    expected_counts: dict[str, int] | None = None,
    verify_manifest: bool = True,
) -> dict[str, Any]:
    """Validate all generated master CSV files in a directory along with the manifest."""
    directory = Path(dir_path)
    if not directory.exists() or not directory.is_dir():
        raise MasterDataError(f"Directory not found: {directory}")

    missing_files = [fname for fname in MASTER_FILES if not (directory / fname).is_file()]
    if missing_files:
        raise MasterDataError(
            f"Missing required CSV files in {directory}: {', '.join(missing_files)}"
        )

    loaded_tables: dict[str, list[dict[str, Any]]] = {}
    row_counts: dict[str, int] = {}
    actual_checksums: dict[str, str] = {}

    table_key_map = {
        "courses.csv": "courses",
        "learners.csv": "learners",
        "skills.csv": "skills",
        "skill_prerequisites.csv": "skill_prerequisites",
        "content_items.csv": "content_items",
        "content_skills.csv": "content_skills",
    }

    for fname in MASTER_FILES:
        fpath = directory / fname
        digest = hashlib.sha256(fpath.read_bytes()).hexdigest()
        actual_checksums[fname] = digest

        with fpath.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            expected_cols = list(FILE_COLUMNS[fname])
            if reader.fieldnames != expected_cols:
                raise MasterDataError(
                    f"CSV column mismatch in {fname}. Expected: {expected_cols}, got: {reader.fieldnames}"
                )
            rows = list(reader)

        table_key = table_key_map[fname]
        loaded_tables[table_key] = rows
        row_counts[fname] = len(rows)

    manifest_data: dict[str, Any] = {}
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.is_file():
        alt_path = directory / "manifest.json"
        if alt_path.is_file():
            manifest_path = alt_path

    if manifest_path.is_file():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise MasterDataError(f"Failed to parse manifest {manifest_path}: {exc}") from exc

        if verify_manifest:
            manifest_checksums = manifest_data.get("checksums_sha256") or manifest_data.get("checksums") or {}
            for fname in MASTER_FILES:
                if fname in manifest_checksums:
                    if manifest_checksums[fname] != actual_checksums[fname]:
                        raise MasterDataError(
                            f"Checksum mismatch for {fname}: manifest={manifest_checksums[fname]}, actual={actual_checksums[fname]}"
                        )
            manifest_counts = manifest_data.get("row_counts") or {}
            for fname in MASTER_FILES:
                if fname in manifest_counts and manifest_counts[fname] != row_counts[fname]:
                    raise MasterDataError(
                        f"Manifest row count mismatch for {fname}: manifest={manifest_counts[fname]}, actual={row_counts[fname]}"
                    )
    elif verify_manifest:
        raise MasterDataError(f"Required manifest file missing in {directory}")

    profile = expected_profile or manifest_data.get("dataset_profile")
    validate_master_tables(
        loaded_tables,
        expected_counts=expected_counts,
        profile=profile,
    )

    return {
        "status": "valid",
        "directory": str(directory),
        "profile": profile,
        "row_counts": row_counts,
        "checksums": actual_checksums,
        "manifest": manifest_data,
    }


def validate_activity_dataset_dir(
    dir_path: Path | str,
    *,
    expected_profile: str | None = None,
    verify_manifest: bool = True,
) -> dict[str, Any]:
    """Validate attempts.csv and assessment_results.csv in dir against master CSVs and manifest."""
    directory = Path(dir_path)
    if not directory.exists() or not directory.is_dir():
        raise ActivityDataError(f"Directory not found: {directory}")

    # Check activity files
    missing_files = [fname for fname in ACTIVITY_FILES if not (directory / fname).is_file()]
    if missing_files:
        raise ActivityDataError(
            f"Missing required activity CSV files in {directory}: {', '.join(missing_files)}"
        )

    # Validate master dataset first to load parent records
    master_res = validate_master_dataset_dir(directory, expected_profile=expected_profile, verify_manifest=verify_manifest)
    
    # Load master tables
    master_tables: dict[str, list[dict[str, Any]]] = {}
    table_key_map = {
        "courses.csv": "courses",
        "learners.csv": "learners",
        "skills.csv": "skills",
        "skill_prerequisites.csv": "skill_prerequisites",
        "content_items.csv": "content_items",
        "content_skills.csv": "content_skills",
    }
    for fname, k in table_key_map.items():
        with (directory / fname).open("r", encoding="utf-8") as handle:
            master_tables[k] = list(csv.DictReader(handle))

    # Load activity tables
    actual_checksums: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    activity_tables: dict[str, list[dict[str, Any]]] = {}

    for fname, k in (("attempts.csv", "attempts"), ("assessment_results.csv", "assessment_results")):
        fpath = directory / fname
        digest = hashlib.sha256(fpath.read_bytes()).hexdigest()
        actual_checksums[fname] = digest
        with fpath.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            expected_cols = list(FILE_COLUMNS[fname])
            if reader.fieldnames != expected_cols:
                raise ActivityDataError(
                    f"CSV column mismatch in {fname}. Expected: {expected_cols}, got: {reader.fieldnames}"
                )
            rows = list(reader)
        activity_tables[k] = rows
        row_counts[fname] = len(rows)

    # Verify activity manifest
    manifest_data: dict[str, Any] = {}
    manifest_path = directory / ACTIVITY_MANIFEST_NAME
    if manifest_path.is_file():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ActivityDataError(f"Failed to parse activity manifest {manifest_path}: {exc}") from exc

        if verify_manifest:
            m_checksums = manifest_data.get("checksums_sha256") or manifest_data.get("checksums") or {}
            for fname in ACTIVITY_FILES:
                if fname in m_checksums:
                    if m_checksums[fname] != actual_checksums[fname]:
                        raise ActivityDataError(
                            f"Checksum mismatch for {fname}: manifest={m_checksums[fname]}, actual={actual_checksums[fname]}"
                        )
            m_counts = manifest_data.get("row_counts") or {}
            for fname in ACTIVITY_FILES:
                if fname in m_counts and m_counts[fname] != row_counts[fname]:
                    raise ActivityDataError(
                        f"Manifest row count mismatch for {fname}: manifest={m_counts[fname]}, actual={row_counts[fname]}"
                    )
    elif verify_manifest:
        raise ActivityDataError(f"Required activity manifest missing in {directory}: {ACTIVITY_MANIFEST_NAME}")

    profile = expected_profile or manifest_data.get("dataset_profile")
    validate_activity_tables(activity_tables, master_tables, profile=profile)

    return {
        "status": "valid",
        "directory": str(directory),
        "profile": profile,
        "row_counts": row_counts,
        "checksums": actual_checksums,
        "manifest": manifest_data,
    }
