"""Deterministic end-to-end acceptance tests for the core LearnPath AI workflow.

Verifies:
- Active eligible learners receive recommendations.
- Required prerequisites cannot be bypassed.
- Cold-start learners receive foundational (zero-prerequisite) content.
- Withdrawn learners are strictly excluded.
- The prepared deterministic fixture achieves 100% recommendation coverage.
- Future activity evidence cannot leak into point-in-time mastery or recommendations.
"""

from __future__ import annotations

import pytest

from learnpath_ai.mastery.calculator import calculate_mastery_snapshots
from learnpath_ai.recommend.engine import generate_recommendations


@pytest.fixture
def acceptance_fixture():
    """Deterministic, fast multi-learner workflow fixture."""
    master = {
        "courses": [
            {"course_id": "CRS-001", "status": "published"}
        ],
        "learners": [
            {"learner_id": "LRN-ACTIVE-COLD", "current_course_id": "CRS-001", "status": "active"},
            {"learner_id": "LRN-ACTIVE-INTER", "current_course_id": "CRS-001", "status": "active"},
            {"learner_id": "LRN-WITHDRAWN", "current_course_id": "CRS-001", "status": "withdrawn"},
        ],
        "skills": [
            {"skill_id": "SKL-ROOT", "status": "active"},
            {"skill_id": "SKL-ADV", "status": "active"},
        ],
        "skill_prerequisites": [
            {
                "skill_id": "SKL-ADV",
                "prerequisite_skill_id": "SKL-ROOT",
                "relationship_type": "required",
                "min_mastery_required": 0.60,
            }
        ],
        "content_items": [
            {"content_id": "CNT-FOUNDATIONAL", "course_id": "CRS-001", "difficulty": "1.20", "status": "published"},
            {"content_id": "CNT-ADVANCED", "course_id": "CRS-001", "difficulty": "2.80", "status": "published"},
        ],
        "content_skills": [
            {"content_id": "CNT-FOUNDATIONAL", "skill_id": "SKL-ROOT", "skill_role": "primary", "weight": "1.0"},
            {"content_id": "CNT-ADVANCED", "skill_id": "SKL-ADV", "skill_role": "primary", "weight": "1.0"},
        ],
    }

    # Activity history up to T1=2025-01-02 00:00:00:
    # LRN-ACTIVE-INTER has 1 pass on CNT-FOUNDATIONAL
    # Future attempt at T2=2025-01-10 00:00:00:
    # LRN-ACTIVE-COLD has an attempt on CNT-FOUNDATIONAL in the future
    activity = {
        "attempts": [
            {
                "attempt_id": "ATT-001",
                "learner_id": "LRN-ACTIVE-INTER",
                "content_id": "CNT-FOUNDATIONAL",
                "status": "scored",
                "started_at": "2025-01-01 10:00:00.000000",
            },
            {
                "attempt_id": "ATT-FUTURE",
                "learner_id": "LRN-ACTIVE-COLD",
                "content_id": "CNT-FOUNDATIONAL",
                "status": "scored",
                "started_at": "2025-01-10 10:00:00.000000",  # Future!
            },
        ],
        "assessment_results": [
            {
                "result_id": "RES-001",
                "attempt_id": "ATT-001",
                "outcome": "pass",
                "passed": 1,
                "score": 100.0,
                "scored_at": "2025-01-01 10:30:00.000000",
            },
            {
                "result_id": "RES-FUTURE",
                "attempt_id": "ATT-FUTURE",
                "outcome": "pass",
                "passed": 1,
                "score": 100.0,
                "scored_at": "2025-01-10 10:30:00.000000",
            },
        ],
    }
    return {"master": master, "activity": activity}


def test_acceptance_core_recommendation_pipeline(acceptance_fixture):
    """End-to-end verification of eligibility, exclusion, prerequisite gates, and temporal integrity."""
    master = acceptance_fixture["master"]
    activity = acceptance_fixture["activity"]
    as_of = "2025-01-02 00:00:00.000000"

    # 1. Point-in-time mastery calculation
    mastery_snapshots = calculate_mastery_snapshots(
        master,
        activity,
        as_of_timestamp=as_of,
        include_cold_start=True,
    )

    # Verify future evidence is NOT in point-in-time mastery
    cold_snap = next(
        s for s in mastery_snapshots
        if s.learner_id == "LRN-ACTIVE-COLD" and s.skill_id == "SKL-ROOT"
    )
    assert cold_snap.evidence_attempt_count == 0
    assert cold_snap.mastery_score == 0.0

    inter_snap = next(
        s for s in mastery_snapshots
        if s.learner_id == "LRN-ACTIVE-INTER" and s.skill_id == "SKL-ROOT"
    )
    assert inter_snap.evidence_attempt_count == 1
    # Credibility shrinkage: 100 * (1 / (1 + 1)) = 50.0% = 0.5000
    assert inter_snap.mastery_score == 0.5000

    # 2. Recommendation generation
    recs = generate_recommendations(
        master,
        activity,
        mastery_snapshots,
        as_of_timestamp=as_of,
        k=5,
    )

    # Verification: Withdrawn learners are excluded
    rec_learners = {r.learner_id for r in recs}
    assert "LRN-WITHDRAWN" not in rec_learners

    # Verification: 100% coverage among active eligible learners
    active_learners = {"LRN-ACTIVE-COLD", "LRN-ACTIVE-INTER"}
    assert rec_learners == active_learners

    # Verification: Cold-start learner receives foundational content
    cold_recs = [r for r in recs if r.learner_id == "LRN-ACTIVE-COLD"]
    assert len(cold_recs) == 1
    assert cold_recs[0].content_id == "CNT-FOUNDATIONAL"

    # Verification: Required prerequisite cannot be bypassed
    # For LRN-ACTIVE-INTER, mastery on SKL-ROOT is 0.5000 < min_mastery 0.60
    # Therefore CNT-ADVANCED is strictly blocked
    inter_recs = [r for r in recs if r.learner_id == "LRN-ACTIVE-INTER"]
    inter_cids = {r.content_id for r in inter_recs}
    assert "CNT-ADVANCED" not in inter_cids
    assert "CNT-FOUNDATIONAL" in inter_cids

