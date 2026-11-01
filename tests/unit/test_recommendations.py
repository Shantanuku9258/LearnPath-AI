"""Unit tests for Phase 6 explainable rule-based recommendation engine."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from learnpath_ai.ids import is_valid_id
from learnpath_ai.recommend.engine import (
    ENGINE_VERSION,
    RecommendationError,
    RecommendationRecord,
    generate_recommendations,
)
from learnpath_ai.recommend.validate import (
    RecommendationValidationError,
    validate_recommendation_dataset_dir,
    validate_recommendations,
)
from learnpath_ai.recommend.writer import (
    write_recommendation_dataset,
    write_recommendation_manifest,
)

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
                "starting_level": "advanced",
                "status": "active",
            },
            {
                "learner_id": "LRN-000003",
                "current_course_id": "CRS-000001",
                "starting_level": "beginner",
                "status": "withdrawn",  # Withdrawn learners must not be recommended to
            },
        ],
        "skills": [
            {"skill_id": "SKL-000001", "skill_name": "Variables", "domain": "Programming Fundamentals"},
            {"skill_id": "SKL-000002", "skill_name": "Conditionals", "domain": "Programming Fundamentals"},
            {"skill_id": "SKL-000003", "skill_name": "Loops", "domain": "Programming Fundamentals"},
        ],
        "skill_prerequisites": [
            # Loops (SKL-000003) requires Conditionals (SKL-000002) with min mastery 0.60
            {
                "skill_id": "SKL-000003",
                "prerequisite_skill_id": "SKL-000002",
                "relationship_type": "required",
                "min_mastery_required": 0.60,
            }
        ],
        "content_items": [
            # Content 1: Variables (difficulty 1.5, published)
            {
                "content_id": "CNT-000001",
                "course_id": "CRS-000001",
                "title": "Intro to Variables",
                "difficulty": 1.5,
                "status": "published",
            },
            # Content 2: Conditionals (difficulty 2.5, published)
            {
                "content_id": "CNT-000002",
                "course_id": "CRS-000001",
                "title": "Working with Conditionals",
                "difficulty": 2.5,
                "status": "published",
            },
            # Content 3: Loops (difficulty 3.5, published) - requires Conditionals
            {
                "content_id": "CNT-000003",
                "course_id": "CRS-000001",
                "title": "Deep dive Loops",
                "difficulty": 3.5,
                "status": "published",
            },
            # Content 4: Draft content - must be excluded
            {
                "content_id": "CNT-000004",
                "course_id": "CRS-000001",
                "title": "Unpublished Draft",
                "difficulty": 2.0,
                "status": "draft",
            },
        ],
        "content_skills": [
            {"content_id": "CNT-000001", "skill_id": "SKL-000001", "skill_role": "primary", "weight": 1.0},
            {"content_id": "CNT-000002", "skill_id": "SKL-000002", "skill_role": "primary", "weight": 1.0},
            {"content_id": "CNT-000003", "skill_id": "SKL-000003", "skill_role": "primary", "weight": 1.0},
            {"content_id": "CNT-000004", "skill_id": "SKL-000001", "skill_role": "primary", "weight": 1.0},
        ],
    }


@pytest.fixture
def base_activity():
    return {
        "attempts": [
            {
                "attempt_id": "ATT-LRN-000001-000001",
                "learner_id": "LRN-000001",
                "content_id": "CNT-000001",
                "started_at": "2025-01-10 10:00:00.000000",
                "submitted_at": "2025-01-10 10:30:00.000000",
                "status": "scored",
                "max_score": 100.0,
            }
        ],
        "assessment_results": [
            {
                "result_id": "RES-000001",
                "attempt_id": "ATT-LRN-000001-000001",
                "score": 50.0,
                "outcome": "fail",
                "scored_at": "2025-01-10 10:35:00.000000",
            }
        ],
    }


# ---------------------------------------------------------------------------
# Test 1: Hard prerequisite gate blocks ineligible content
# ---------------------------------------------------------------------------

def test_hard_prerequisite_blocks_content(base_master, base_activity):
    # LRN-000001 has 0.0 mastery in SKL-000002 (Conditionals).
    # Therefore CNT-000003 (Loops) MUST NOT be recommended.
    mastery = [
        {"learner_id": "LRN-000001", "skill_id": "SKL-000001", "mastery_score": 0.20, "as_of_timestamp": "2025-02-01 00:00:00.000000"},
        {"learner_id": "LRN-000001", "skill_id": "SKL-000002", "mastery_score": 0.10, "as_of_timestamp": "2025-02-01 00:00:00.000000"},
    ]
    recs = generate_recommendations(
        base_master,
        base_activity,
        mastery,
        as_of_timestamp="2025-02-01 00:00:00.000000",
        k=5,
    )
    lrn1_recs = [r for r in recs if r.learner_id == "LRN-000001"]
    recommended_content_ids = {r.content_id for r in lrn1_recs}

    assert "CNT-000003" not in recommended_content_ids, "Prerequisite failure: CNT-000003 was recommended despite unmet prereq!"
    assert "CNT-000001" in recommended_content_ids
    assert "CNT-000002" in recommended_content_ids


# ---------------------------------------------------------------------------
# Test 2: Satisfied prerequisite allows recommendation
# ---------------------------------------------------------------------------

def test_satisfied_prerequisite_allows_content(base_master, base_activity):
    # LRN-000001 now has 0.85 mastery in SKL-000002 (>= 0.60 required).
    # Now CNT-000003 (Loops) CAN be recommended.
    mastery = [
        {"learner_id": "LRN-000001", "skill_id": "SKL-000001", "mastery_score": 0.90, "as_of_timestamp": "2025-02-01 00:00:00.000000"},
        {"learner_id": "LRN-000001", "skill_id": "SKL-000002", "mastery_score": 0.85, "as_of_timestamp": "2025-02-01 00:00:00.000000"},
        {"learner_id": "LRN-000001", "skill_id": "SKL-000003", "mastery_score": 0.10, "as_of_timestamp": "2025-02-01 00:00:00.000000"},
    ]
    recs = generate_recommendations(
        base_master,
        base_activity,
        mastery,
        as_of_timestamp="2025-02-01 00:00:00.000000",
        k=5,
    )
    lrn1_recs = [r for r in recs if r.learner_id == "LRN-000001"]
    content_ids = {r.content_id for r in lrn1_recs}
    assert "CNT-000003" in content_ids, "CNT-000003 should be eligible when prereq is satisfied"


# ---------------------------------------------------------------------------
# Test 3: Draft/unpublished content is excluded
# ---------------------------------------------------------------------------

def test_draft_content_is_excluded(base_master, base_activity):
    recs = generate_recommendations(
        base_master,
        base_activity,
        [],
        as_of_timestamp="2025-02-01 00:00:00.000000",
    )
    content_ids = {r.content_id for r in recs}
    assert "CNT-000004" not in content_ids, "Draft content must be excluded"


# ---------------------------------------------------------------------------
# Test 4: Withdrawn learners are not recommended to
# ---------------------------------------------------------------------------

def test_withdrawn_learners_excluded(base_master, base_activity):
    recs = generate_recommendations(
        base_master,
        base_activity,
        [],
        as_of_timestamp="2025-02-01 00:00:00.000000",
    )
    learner_ids = {r.learner_id for r in recs}
    assert "LRN-000003" not in learner_ids, "Withdrawn learners must receive no recommendations"


# ---------------------------------------------------------------------------
# Test 5: Skill gap priority
# ---------------------------------------------------------------------------

def test_skill_gap_ranking_priority(base_master, base_activity):
    # LRN-000002 has high mastery in Variables (0.95), low in Conditionals (0.10)
    mastery = [
        {"learner_id": "LRN-000002", "skill_id": "SKL-000001", "mastery_score": 0.95, "as_of_timestamp": "2025-02-01 00:00:00.000000"},
        {"learner_id": "LRN-000002", "skill_id": "SKL-000002", "mastery_score": 0.10, "as_of_timestamp": "2025-02-01 00:00:00.000000"},
    ]
    recs = generate_recommendations(
        base_master,
        base_activity,
        mastery,
        as_of_timestamp="2025-02-01 00:00:00.000000",
    )
    lrn2_recs = [r for r in recs if r.learner_id == "LRN-000002"]
    assert len(lrn2_recs) >= 2
    # Conditionals (CNT-000002) should rank above Variables (CNT-000001) because of the mastery gap
    r1 = lrn2_recs[0]
    assert r1.content_id == "CNT-000002"
    assert r1.rank_position == 1


# ---------------------------------------------------------------------------
# Test 6: Difficulty suitability matching
# ---------------------------------------------------------------------------

def test_difficulty_suitability_influences_score(base_master):
    # Beginner (target diff 1.5) vs Advanced (target diff 4.5)
    recs = generate_recommendations(
        base_master,
        {"attempts": [], "assessment_results": []},
        [],
        as_of_timestamp="2025-02-01 00:00:00.000000",
    )
    lrn1_recs = [r for r in recs if r.learner_id == "LRN-000001"]  # beginner
    lrn2_recs = [r for r in recs if r.learner_id == "LRN-000002"]  # advanced

    # For beginner, CNT-000001 (diff 1.5) should score higher on difficulty match than CNT-000002 (diff 2.5)
    r1_b = [r for r in lrn1_recs if r.content_id == "CNT-000001"][0]
    r2_b = [r for r in lrn1_recs if r.content_id == "CNT-000002"][0]
    assert r1_b.score > r2_b.score


# ---------------------------------------------------------------------------
# Test 7: Retry after fail reason code and factor
# ---------------------------------------------------------------------------

def test_retry_after_fail_reason_code(base_master, base_activity):
    # LRN-000001 failed CNT-000001 in base_activity
    recs = generate_recommendations(
        base_master,
        base_activity,
        [],
        as_of_timestamp="2025-02-01 00:00:00.000000",
    )
    r = [r for r in recs if r.learner_id == "LRN-000001" and r.content_id == "CNT-000001"][0]
    assert "RETRY_AFTER_FAIL" in r.reason_codes


# ---------------------------------------------------------------------------
# Test 8: Cold-start learners handled gracefully
# ---------------------------------------------------------------------------

def test_cold_start_learners_handled(base_master):
    # No activity and no mastery
    recs = generate_recommendations(
        base_master,
        {"attempts": [], "assessment_results": []},
        [],
        as_of_timestamp="2025-02-01 00:00:00.000000",
    )
    assert len(recs) > 0
    for r in recs:
        assert 0.0 <= r.score <= 1.0
        assert r.rank_position >= 1
        assert "LOW_MASTERY_PRIORITY" in r.reason_codes or "GAP_PRIMARY_SKILL" in r.reason_codes


# ---------------------------------------------------------------------------
# Test 9: Top-K limits strictly respected
# ---------------------------------------------------------------------------

def test_top_k_limits_respected(base_master):
    recs = generate_recommendations(
        base_master,
        {"attempts": [], "assessment_results": []},
        [],
        as_of_timestamp="2025-02-01 00:00:00.000000",
        k=1,
    )
    lrn1_recs = [r for r in recs if r.learner_id == "LRN-000001"]
    assert len(lrn1_recs) == 1
    assert lrn1_recs[0].rank_position == 1


# ---------------------------------------------------------------------------
# Test 10: Deterministic ranking and tie-breaking
# ---------------------------------------------------------------------------

def test_deterministic_ranking(base_master, base_activity):
    recs1 = generate_recommendations(
        base_master,
        base_activity,
        [],
        as_of_timestamp="2025-02-01 00:00:00.000000",
    )
    recs2 = generate_recommendations(
        base_master,
        base_activity,
        [],
        as_of_timestamp="2025-02-01 00:00:00.000000",
    )
    assert [r.to_dict() for r in recs1] == [r.to_dict() for r in recs2]


# ---------------------------------------------------------------------------
# Test 11: Future-data exclusion (Temporal Leakage Guard)
# ---------------------------------------------------------------------------

def test_future_attempts_excluded_from_recommendation_features(base_master):
    # Attempt in the future (May 2025)
    future_activity = {
        "attempts": [
            {
                "attempt_id": "ATT-LRN-000001-000001",
                "learner_id": "LRN-000001",
                "content_id": "CNT-000001",
                "started_at": "2025-05-10 10:00:00.000000",
                "submitted_at": "2025-05-10 10:30:00.000000",
                "status": "scored",
                "max_score": 100.0,
            }
        ],
        "assessment_results": [
            {
                "result_id": "RES-000001",
                "attempt_id": "ATT-LRN-000001-000001",
                "score": 50.0,
                "outcome": "fail",
                "scored_at": "2025-05-10 10:35:00.000000",
            }
        ],
    }
    # Evaluated at February 2025 (before attempt)
    recs = generate_recommendations(
        base_master,
        future_activity,
        [],
        as_of_timestamp="2025-02-01 00:00:00.000000",
    )
    r = [r for r in recs if r.learner_id == "LRN-000001" and r.content_id == "CNT-000001"][0]
    # Since attempt occurred in May, at February it is unseen, so RETRY_AFTER_FAIL must NOT be present
    assert "RETRY_AFTER_FAIL" not in r.reason_codes


# ---------------------------------------------------------------------------
# Test 12: Recommendation ID format and uniqueness
# ---------------------------------------------------------------------------

def test_recommendation_id_format_and_uniqueness(base_master):
    recs = generate_recommendations(
        base_master,
        {"attempts": [], "assessment_results": []},
        [],
        as_of_timestamp="2025-02-01 00:00:00.000000",
    )
    seen_ids = set()
    for r in recs:
        assert is_valid_id(r.recommendation_id, "REC")
        assert r.recommendation_id not in seen_ids
        seen_ids.add(r.recommendation_id)


# ---------------------------------------------------------------------------
# Test 13: Schema and Constraint validation function
# ---------------------------------------------------------------------------

def test_validate_recommendations_schema_and_constraints(base_master):
    recs = generate_recommendations(
        base_master,
        {"attempts": [], "assessment_results": []},
        [],
        as_of_timestamp="2025-02-01 00:00:00.000000",
    )
    rows = [r.to_dict() for r in recs]
    validate_recommendations(rows, master_tables=base_master)

    # Corrupt rank
    bad_rank_rows = [dict(r) for r in rows]
    bad_rank_rows[0]["rank_position"] = 25
    with pytest.raises(RecommendationValidationError, match="out of bounds"):
        validate_recommendations(bad_rank_rows, master_tables=base_master)

    # Corrupt timestamp order (as_of > recommended_at)
    bad_ts_rows = [dict(r) for r in rows]
    bad_ts_rows[0]["as_of_timestamp"] = "2025-03-01 00:00:00.000000"
    bad_ts_rows[0]["recommended_at"] = "2025-02-01 00:00:00.000000"
    with pytest.raises(RecommendationValidationError, match="Temporal violation"):
        validate_recommendations(bad_ts_rows, master_tables=base_master)


# ---------------------------------------------------------------------------
# Test 14: Manifest and checksum verification
# ---------------------------------------------------------------------------

def test_write_and_validate_recommendation_dataset(tmp_path, base_master):
    recs = generate_recommendations(
        base_master,
        {"attempts": [], "assessment_results": []},
        [],
        as_of_timestamp="2025-02-01 00:00:00.000000",
    )
    manifest = write_recommendation_dataset(
        recs,
        tmp_path,
        profile="dev",
        seed=SEED,
        overwrite=True,
    )
    assert (tmp_path / "recommendations.csv").is_file()
    assert (tmp_path / "recommendation_manifest.json").is_file()
    assert "recommendations.csv" in manifest["checksums"]

    # Validate directory
    result = validate_recommendation_dataset_dir(
        tmp_path,
        expected_profile="dev",
        master_tables=base_master,
    )
    assert result["row_counts"]["recommendations.csv"] == len(recs)

