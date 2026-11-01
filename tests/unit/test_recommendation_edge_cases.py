"""Targeted unit tests for specific recommendation behavior edge cases (F-08 Part 2).

Covers:
- Already-passed content receives lower ranking priority.
- Course-local candidate selection (candidates strictly match learner current_course_id).
- Unknown skill mappings handling (safe fallback when primary skill is not explicitly marked).
- Hard prerequisite enforcement blocks content when prerequisites are unmet.
- PATH_NEXT_ITEM behavior (documented inactive in current heuristic rules).
"""

from __future__ import annotations

import pytest

from learnpath_ai.recommend.engine import generate_recommendations


@pytest.fixture
def multi_course_master():
    return {
        "courses": [
            {"course_id": "CRS-001", "status": "published"},
            {"course_id": "CRS-002", "status": "published"},
        ],
        "learners": [
            {"learner_id": "LRN-C1", "current_course_id": "CRS-001", "status": "active"},
            {"learner_id": "LRN-C2", "current_course_id": "CRS-002", "status": "active"},
        ],
        "skills": [
            {"skill_id": "SKL-001", "status": "active"},
            {"skill_id": "SKL-002", "status": "active"},
        ],
        "skill_prerequisites": [
            {"skill_id": "SKL-002", "prerequisite_skill_id": "SKL-001", "relationship_type": "required", "min_mastery_required": 0.60}
        ],
        "content_items": [
            {"content_id": "CNT-C1-A", "course_id": "CRS-001", "status": "published", "difficulty": "1.00"},
            {"content_id": "CNT-C1-B", "course_id": "CRS-001", "status": "published", "difficulty": "2.00"},
            {"content_id": "CNT-C2-A", "course_id": "CRS-002", "status": "published", "difficulty": "1.00"},
            {"content_id": "CNT-C1-DRAFT", "course_id": "CRS-001", "status": "draft", "difficulty": "1.00"},
        ],
        "content_skills": [
            {"content_id": "CNT-C1-A", "skill_id": "SKL-001", "skill_role": "primary", "weight": "1.0"},
            {"content_id": "CNT-C1-B", "skill_id": "SKL-002", "skill_role": "primary", "weight": "1.0"},
            {"content_id": "CNT-C2-A", "skill_id": "SKL-001", "skill_role": "primary", "weight": "1.0"},
            {"content_id": "CNT-C1-DRAFT", "skill_id": "SKL-001", "skill_role": "primary", "weight": "1.0"},
        ],
    }


def test_course_local_candidate_selection_strictly_enforced(multi_course_master):
    """Learners only receive recommendations from their own current_course_id."""
    recs = generate_recommendations(
        multi_course_master,
        {"attempts": [], "assessment_results": []},
        [],
        as_of_timestamp="2025-01-01 00:00:00.000000",
    )

    c1_recs = [r for r in recs if r.learner_id == "LRN-C1"]
    c2_recs = [r for r in recs if r.learner_id == "LRN-C2"]

    assert all(r.content_id.startswith("CNT-C1-") for r in c1_recs)
    assert all(r.content_id.startswith("CNT-C2-") for r in c2_recs)
    assert "CNT-C2-A" not in {r.content_id for r in c1_recs}
    assert "CNT-C1-A" not in {r.content_id for r in c2_recs}


def test_already_passed_content_deprioritized(multi_course_master):
    """When a learner has already passed an item, its priority is lowered relative to unpassed items."""
    # LRN-C1 passed CNT-C1-A
    activity = {
        "attempts": [
            {
                "attempt_id": "ATT-001",
                "learner_id": "LRN-C1",
                "content_id": "CNT-C1-A",
                "status": "scored",
                "started_at": "2025-01-01 10:00:00.000000",
            }
        ],
        "assessment_results": [
            {
                "result_id": "RES-001",
                "attempt_id": "ATT-001",
                "outcome": "pass",
                "passed": 1,
                "score": 90.0,
                "scored_at": "2025-01-01 10:30:00.000000",
            }
        ],
    }
    # Meet prerequisite for CNT-C1-B so both items are eligible
    mastery = [
        {"learner_id": "LRN-C1", "skill_id": "SKL-001", "mastery_score": 0.80, "as_of_timestamp": "2025-01-02 00:00:00.000000"}
    ]

    recs = generate_recommendations(
        multi_course_master,
        activity,
        mastery,
        as_of_timestamp="2025-01-02 00:00:00.000000",
        k=5,
    )
    c1_recs = [r for r in recs if r.learner_id == "LRN-C1"]
    ranks = {r.content_id: r.rank_position for r in c1_recs}

    # CNT-C1-B (unpassed) should rank above CNT-C1-A (already passed)
    assert ranks["CNT-C1-B"] < ranks["CNT-C1-A"]


def test_unknown_or_secondary_only_skill_mappings_handled_safely():
    """If content has no explicitly marked 'primary' role, the engine falls back to first mapped skill."""
    master = {
        "courses": [{"course_id": "CRS-001", "status": "published"}],
        "learners": [{"learner_id": "LRN-001", "current_course_id": "CRS-001", "status": "active"}],
        "skills": [{"skill_id": "SKL-001", "status": "active"}],
        "skill_prerequisites": [],
        "content_items": [
            {"content_id": "CNT-SEC", "course_id": "CRS-001", "difficulty": "1.00", "status": "published"}
        ],
        "content_skills": [
            # Only secondary role, no primary role
            {"content_id": "CNT-SEC", "skill_id": "SKL-001", "skill_role": "secondary", "weight": "0.5"}
        ],
    }

    recs = generate_recommendations(
        master,
        {"attempts": [], "assessment_results": []},
        [],
        as_of_timestamp="2025-01-01 00:00:00.000000",
    )
    assert len(recs) == 1
    assert recs[0].content_id == "CNT-SEC"


def test_hard_prerequisite_blocks_candidate_without_sufficient_mastery(multi_course_master):
    """Candidate CNT-C1-B requires SKL-001 >= 0.60; with 0.40 mastery it must be strictly excluded."""
    mastery = [
        {"learner_id": "LRN-C1", "skill_id": "SKL-001", "mastery_score": 0.40, "as_of_timestamp": "2025-01-01 00:00:00.000000"}
    ]
    recs = generate_recommendations(
        multi_course_master,
        {"attempts": [], "assessment_results": []},
        mastery,
        as_of_timestamp="2025-01-01 00:00:00.000000",
    )
    c1_cids = {r.content_id for r in recs if r.learner_id == "LRN-C1"}
    assert "CNT-C1-A" in c1_cids
    assert "CNT-C1-B" not in c1_cids, "CNT-C1-B must be blocked by failed prerequisite check"
