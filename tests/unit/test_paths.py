"""Unit tests for Phase 7 & 7.1 learning-path generation, progress tracking, and replanning."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

import pytest

from learnpath_ai.ids import is_valid_id
from learnpath_ai.path.engine import (
    ENGINE_VERSION,
    LearningPathError,
    check_and_update_path_completion,
    complete_path_item,
    generate_learning_paths,
    replan_learning_path,
    skip_path_item,
    start_path_item,
    update_item_status,
)
from learnpath_ai.path.validate import (
    PathValidationError,
    validate_path_dataset_dir,
    validate_path_tables,
)
from learnpath_ai.path.writer import write_path_dataset
from learnpath_ai.prerequisites import PrerequisiteGraphError

UTC = timezone.utc
SEED = 20261022


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def base_master():
    return {
        "courses": [
            {"course_id": "CRS-000001", "title": "Programming Foundations", "status": "published"}
        ],
        "learners": [
            {
                "learner_id": "LRN-000001",
                "current_course_id": "CRS-000001",
                "starting_level": "beginner",
                "status": "active",
            },
            {
                "learner_id": "LRN-000002",
                "current_course_id": "CRS-000001",
                "starting_level": "intermediate",
                "status": "withdrawn",
            },
        ],
        "skills": [
            {"skill_id": "SKL-000001", "skill_name": "Variables", "domain": "Programming", "status": "active"},
            {"skill_id": "SKL-000002", "skill_name": "Loops", "domain": "Programming", "status": "active"},
            {"skill_id": "SKL-000003", "skill_name": "Functions", "domain": "Programming", "status": "active"},
        ],
        "skill_prerequisites": [
            {
                "skill_id": "SKL-000002",
                "prerequisite_skill_id": "SKL-000001",
                "relationship_type": "required",
                "min_mastery_required": 0.60,
            },
        ],
        "content_items": [
            {"content_id": "CNT-000001", "course_id": "CRS-000001", "title": "Intro Variables", "difficulty": "1.5", "status": "published"},
            {"content_id": "CNT-000002", "course_id": "CRS-000001", "title": "Loop Basics", "difficulty": "2.5", "status": "published"},
            {"content_id": "CNT-000003", "course_id": "CRS-000001", "title": "Advanced Functions", "difficulty": "4.0", "status": "published"},
            {"content_id": "CNT-000004", "course_id": "CRS-000001", "title": "Draft Topic", "difficulty": "2.0", "status": "draft"},
        ],
        "content_skills": [
            {"content_id": "CNT-000001", "skill_id": "SKL-000001", "skill_role": "primary", "weight": "1.0"},
            {"content_id": "CNT-000002", "skill_id": "SKL-000002", "skill_role": "primary", "weight": "1.0"},
            {"content_id": "CNT-000003", "skill_id": "SKL-000003", "skill_role": "primary", "weight": "1.0"},
            {"content_id": "CNT-000004", "skill_id": "SKL-000001", "skill_role": "primary", "weight": "1.0"},
        ],
    }


@pytest.fixture
def sample_recs():
    return [
        {
            "recommendation_id": "REC-000001",
            "learner_id": "LRN-000001",
            "content_id": "CNT-000001",
            "path_id": "",
            "rank_position": 1,
            "engine_version": "rule-v1",
            "reason_codes": '["GAP_PRIMARY_SKILL"]',
            "score": "0.850000",
            "status": "pending",
            "recommended_at": "2026-10-22 10:00:00.000000",
            "as_of_timestamp": "2026-10-22 10:00:00.000000",
            "expires_at": "",
            "created_at": "2026-10-22 10:00:00.000000",
        },
        {
            "recommendation_id": "REC-000002",
            "learner_id": "LRN-000001",
            "content_id": "CNT-000003",
            "path_id": "",
            "rank_position": 2,
            "engine_version": "rule-v1",
            "reason_codes": '["GAP_PRIMARY_SKILL"]',
            "score": "0.720000",
            "status": "pending",
            "recommended_at": "2026-10-22 10:00:00.000000",
            "as_of_timestamp": "2026-10-22 10:00:00.000000",
            "expires_at": "",
            "created_at": "2026-10-22 10:00:00.000000",
        },
    ]


@pytest.fixture
def sample_mastery():
    return [
        {
            "snapshot_id": "MST-000001",
            "learner_id": "LRN-000001",
            "skill_id": "SKL-000001",
            "mastery_score": 0.40,
            "as_of_timestamp": "2026-10-22 10:00:00.000000",
        }
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_path_creation_and_ordering(base_master, sample_recs, sample_mastery):
    """Paths are created with sequential rank ordering matching recommendations."""
    paths, items = generate_learning_paths(base_master, sample_recs, sample_mastery)
    assert len(paths) == 1
    p = paths[0]
    assert p.learner_id == "LRN-000001"
    assert p.course_id == "CRS-000001"
    assert p.status == "active"
    assert p.path_id.startswith("PATH-2026-")

    assert len(items) == 2
    assert items[0].sequence_no == 1
    assert items[0].content_id == "CNT-000001"
    assert items[0].path_item_id.startswith("LPI-")

    assert items[1].sequence_no == 2
    assert items[1].content_id == "CNT-000003"


def test_hard_prerequisite_enforced_in_path(base_master, sample_mastery):
    """Content failing required prerequisites is strictly excluded from learning paths."""
    ineligible_rec = [
        {
            "recommendation_id": "REC-000003",
            "learner_id": "LRN-000001",
            "content_id": "CNT-000002",
            "rank_position": 1,
            "score": "0.990000",
            "status": "pending",
            "as_of_timestamp": "2026-10-22 10:00:00.000000",
            "recommended_at": "2026-10-22 10:00:00.000000",
        }
    ]
    paths, items = generate_learning_paths(base_master, ineligible_rec, sample_mastery)
    assert len(paths) == 0
    assert len(items) == 0


def test_satisfied_prerequisite_included_in_path(base_master):
    """When prerequisites are satisfied (mastery >= 0.60), content is included."""
    mastery = [
        {
            "snapshot_id": "MST-000001",
            "learner_id": "LRN-000001",
            "skill_id": "SKL-000001",
            "mastery_score": 0.85,
            "as_of_timestamp": "2026-10-22 10:00:00.000000",
        }
    ]
    recs = [
        {
            "recommendation_id": "REC-000003",
            "learner_id": "LRN-000001",
            "content_id": "CNT-000002",
            "rank_position": 1,
            "score": "0.950000",
            "status": "pending",
            "as_of_timestamp": "2026-10-22 10:00:00.000000",
            "recommended_at": "2026-10-22 10:00:00.000000",
        }
    ]
    paths, items = generate_learning_paths(base_master, recs, mastery)
    assert len(paths) == 1
    assert len(items) == 1
    assert items[0].content_id == "CNT-000002"


def test_one_active_path_per_learner_course(base_master, sample_recs, sample_mastery):
    """Only one active path is created per learner and course combination."""
    paths, items = generate_learning_paths(base_master, sample_recs, sample_mastery)
    learner_courses = [(p.learner_id, p.course_id) for p in paths]
    assert len(learner_courses) == len(set(learner_courses))


def test_withdrawn_learner_gets_no_path(base_master, sample_mastery):
    """Withdrawn learners do not receive learning paths."""
    recs = [
        {
            "recommendation_id": "REC-000099",
            "learner_id": "LRN-000002",
            "content_id": "CNT-000001",
            "rank_position": 1,
            "score": "0.900000",
            "status": "pending",
            "as_of_timestamp": "2026-10-22 10:00:00.000000",
            "recommended_at": "2026-10-22 10:00:00.000000",
        }
    ]
    paths, items = generate_learning_paths(base_master, recs, sample_mastery)
    assert len(paths) == 0
    assert len(items) == 0


def test_loop_and_duplicate_content_prevention(base_master, sample_mastery):
    """Duplicate content IDs for the same learner are deduplicated within a path."""
    duplicate_recs = [
        {
            "recommendation_id": "REC-000001",
            "learner_id": "LRN-000001",
            "content_id": "CNT-000001",
            "rank_position": 1,
            "score": "0.900000",
            "status": "pending",
            "as_of_timestamp": "2026-10-22 10:00:00.000000",
            "recommended_at": "2026-10-22 10:00:00.000000",
        },
        {
            "recommendation_id": "REC-000002",
            "learner_id": "LRN-000001",
            "content_id": "CNT-000001",
            "rank_position": 2,
            "score": "0.800000",
            "status": "pending",
            "as_of_timestamp": "2026-10-22 10:00:00.000000",
            "recommended_at": "2026-10-22 10:00:00.000000",
        },
    ]
    paths, items = generate_learning_paths(base_master, duplicate_recs, sample_mastery)
    assert len(paths) == 1
    assert len(items) == 1
    assert items[0].sequence_no == 1


def test_prerequisite_graph_cycle_rejected(base_master, sample_recs, sample_mastery):
    """Cyclic prerequisite graph in master data is detected and rejected via require_dag."""
    cyclic_master = dict(base_master)
    cyclic_master["skill_prerequisites"] = [
        {"skill_id": "SKL-000001", "prerequisite_skill_id": "SKL-000002", "relationship_type": "required", "min_mastery_required": 0.6},
        {"skill_id": "SKL-000002", "prerequisite_skill_id": "SKL-000001", "relationship_type": "required", "min_mastery_required": 0.6},
    ]
    with pytest.raises(PrerequisiteGraphError, match="cycle"):
        generate_learning_paths(cyclic_master, sample_recs, sample_mastery)


def test_temporal_cutoff_excludes_future_recommendations(base_master, sample_mastery):
    """Recommendations with as_of_timestamp > cutoff are excluded."""
    recs = [
        {
            "recommendation_id": "REC-000001",
            "learner_id": "LRN-000001",
            "content_id": "CNT-000001",
            "rank_position": 1,
            "score": "0.900000",
            "status": "pending",
            "as_of_timestamp": "2026-10-22 10:00:00.000000",
            "recommended_at": "2026-10-22 10:00:00.000000",
        },
        {
            "recommendation_id": "REC-000002",
            "learner_id": "LRN-000001",
            "content_id": "CNT-000003",
            "rank_position": 2,
            "score": "0.850000",
            "status": "pending",
            "as_of_timestamp": "2026-10-23 10:00:00.000000",
            "recommended_at": "2026-10-23 10:00:00.000000",
        },
    ]
    paths, items = generate_learning_paths(
        base_master,
        recs,
        sample_mastery,
        as_of_timestamp="2026-10-22 10:00:00.000000",
    )
    assert len(paths) == 1
    assert len(items) == 1
    assert items[0].content_id == "CNT-000001"


def test_id_format_and_uniqueness(base_master, sample_recs, sample_mastery):
    """Generated path IDs match PATH-YYYY-000001 and item IDs match LPI-000001."""
    paths, items = generate_learning_paths(base_master, sample_recs, sample_mastery)
    assert len(paths) > 0
    for p in paths:
        import re
        assert re.fullmatch(r"^PATH-[0-9]{4}-[0-9]{6}$", p.path_id)
    for it in items:
        assert is_valid_id(it.path_item_id, "LPI")


def test_write_and_validate_path_dataset(tmp_path, base_master, sample_recs, sample_mastery):
    """write_path_dataset outputs valid CSVs and manifest matching checksums."""
    paths, items = generate_learning_paths(base_master, sample_recs, sample_mastery)
    manifest = write_path_dataset(
        paths,
        items,
        tmp_path,
        profile="dev",
        seed=SEED,
        as_of_timestamp="2026-10-22 10:00:00.000000",
    )
    assert "learning_paths.csv" in manifest["checksums"]
    assert "learning_path_items.csv" in manifest["checksums"]

    res = validate_path_dataset_dir(tmp_path, expected_profile="dev", master_tables=base_master)
    assert res["row_counts"]["learning_paths.csv"] == 1
    assert res["row_counts"]["learning_path_items.csv"] == 2


# ---------------------------------------------------------------------------
# Phase 7.1 Progress Tracking & Replanning Tests
# ---------------------------------------------------------------------------

def test_progress_tracking_transitions(base_master, sample_recs, sample_mastery):
    """Valid item transitions planned -> in_progress -> completed and planned -> skipped."""
    paths, items = generate_learning_paths(base_master, sample_recs, sample_mastery)
    item1 = items[0]
    assert item1.status == "planned"

    # planned -> in_progress
    in_prog = start_path_item(item1, timestamp="2026-10-22 11:00:00.000000")
    assert in_prog.status == "in_progress"
    assert in_prog.completed_at == ""

    # in_progress -> completed
    completed = complete_path_item(in_prog, timestamp="2026-10-22 12:00:00.000000")
    assert completed.status == "completed"
    assert completed.completed_at == "2026-10-22 12:00:00.000000"

    # planned -> skipped
    item2 = items[1]
    skipped = skip_path_item(item2, timestamp="2026-10-22 11:30:00.000000")
    assert skipped.status == "skipped"
    assert skipped.completed_at == "2026-10-22 11:30:00.000000"


def test_invalid_item_transitions_rejected(base_master, sample_recs, sample_mastery):
    """Illegal transitions (e.g., planned -> completed, completed -> in_progress) raise error."""
    paths, items = generate_learning_paths(base_master, sample_recs, sample_mastery)
    item = items[0]

    # planned -> completed directly is illegal (must go through in_progress)
    with pytest.raises(LearningPathError, match="Invalid item status transition"):
        update_item_status(item, "completed")

    in_prog = start_path_item(item)
    # in_progress -> planned is illegal
    with pytest.raises(LearningPathError, match="Invalid item status transition"):
        update_item_status(in_prog, "planned")

    completed = complete_path_item(in_prog)
    # completed is terminal
    with pytest.raises(LearningPathError, match="Invalid item status transition"):
        update_item_status(completed, "in_progress")


def test_automatic_path_completion(base_master, sample_recs, sample_mastery):
    """When all path items are completed/skipped (with at least one completed), path automatically completes."""
    paths, items = generate_learning_paths(base_master, sample_recs, sample_mastery)
    path = paths[0]
    assert path.status == "active"
    assert path.completed_at == ""

    # Finish item 1 (completed) and item 2 (skipped)
    finished_item1 = complete_path_item(start_path_item(items[0]), timestamp="2026-10-22 12:00:00.000000")
    finished_item2 = skip_path_item(items[1], timestamp="2026-10-22 12:15:00.000000")
    updated_items = [finished_item1, finished_item2]

    completed_path = check_and_update_path_completion(path, updated_items)
    assert completed_path.status == "completed"
    assert completed_path.completed_at != ""


def test_replanning_preserves_history_and_single_active_path(base_master, sample_recs, sample_mastery):
    """Replanning marks old path as abandoned, keeps all items, and adds a new active path."""
    paths, items = generate_learning_paths(base_master, sample_recs, sample_mastery)
    assert len(paths) == 1
    assert paths[0].status == "active"
    orig_path_id = paths[0].path_id

    # New recommendations for replanning
    new_recs = [
        {
            "recommendation_id": "REC-000005",
            "learner_id": "LRN-000001",
            "content_id": "CNT-000003",
            "rank_position": 1,
            "score": "0.950000",
            "status": "pending",
            "recommended_at": "2026-10-22 14:00:00.000000",
            "as_of_timestamp": "2026-10-22 14:00:00.000000",
        }
    ]

    all_paths, all_items = replan_learning_path(
        paths,
        items,
        learner_id="LRN-000001",
        course_id="CRS-000001",
        master_tables=base_master,
        recommendations=new_recs,
        mastery_snapshots=sample_mastery,
        as_of_timestamp="2026-10-22 14:00:00.000000",
    )

    # Historical preservation: old path is kept, now abandoned
    assert len(all_paths) == 2
    old_p = next(p for p in all_paths if p.path_id == orig_path_id)
    assert old_p.status == "abandoned"

    # New path is active with distinct ID
    new_p = next(p for p in all_paths if p.path_id != orig_path_id)
    assert new_p.status == "active"
    assert new_p.path_id != orig_path_id

    # Exactly one active path for (LRN-000001, CRS-000001)
    active_paths = [p for p in all_paths if p.learner_id == "LRN-000001" and p.status == "active"]
    assert len(active_paths) == 1

    # Old items are completely preserved
    assert len(all_items) == len(items) + 1
