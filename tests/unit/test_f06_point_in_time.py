"""F-06 point-in-time evaluation regression tests.

These tests lock the shared cutoff, mastery recalculation, future-snapshot rejection,
no-leakage, and cold-start gate behavior used by rule-v1 and ml-v1. They do not
change production ranking logic.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from learnpath_ai.eligibility import ContentSkillLink, content_passes_required_prerequisites
from learnpath_ai.mastery.calculator import calculate_mastery_snapshots
from learnpath_ai.ml.engine import generate_ml_recommendations
from learnpath_ai.ml.features import FEATURE_NAMES
from learnpath_ai.ml.model import ModelArtifact
from learnpath_ai.prerequisites import PrerequisiteEdge
from learnpath_ai.recommend.engine import generate_recommendations

CUTOFF = "2025-01-03 00:00:00.000000"
FUTURE = "2025-01-10 00:00:00.000000"

MASTER = {
    "courses": [{"course_id": "CRS-000001", "title": "CS 101", "status": "published"}],
    "learners": [
        {"learner_id": "LRN-000001", "current_course_id": "CRS-000001", "starting_level": "beginner", "status": "active"},
        {"learner_id": "LRN-000002", "current_course_id": "CRS-000001", "starting_level": "beginner", "status": "active"},
    ],
    "skills": [
        {"skill_id": "SKL-000001", "skill_name": "Variables", "domain": "CS", "status": "active"},
        {"skill_id": "SKL-000002", "skill_name": "Conditionals", "domain": "CS", "status": "active"},
    ],
    "skill_prerequisites": [
        {
            "skill_id": "SKL-000002",
            "prerequisite_skill_id": "SKL-000001",
            "relationship_type": "required",
            "min_mastery_required": 0.60,
        }
    ],
    "content_items": [
        {"content_id": "CNT-000001", "course_id": "CRS-000001", "title": "Vars", "difficulty": "1.5", "status": "published"},
        {"content_id": "CNT-000002", "course_id": "CRS-000001", "title": "Branches", "difficulty": "2.5", "status": "published"},
    ],
    "content_skills": [
        {"content_id": "CNT-000001", "skill_id": "SKL-000001", "skill_role": "primary", "weight": "1.0"},
        {"content_id": "CNT-000002", "skill_id": "SKL-000002", "skill_role": "primary", "weight": "1.0"},
    ],
}

ACTIVITY = {
    "attempts": [
        {
            "attempt_id": "ATT-BEFORE",
            "learner_id": "LRN-000001",
            "content_id": "CNT-000001",
            "status": "scored",
            "started_at": "2025-01-01 10:00:00.000000",
            "max_score": 100.0,
        },
        {
            "attempt_id": "ATT-AFTER",
            "learner_id": "LRN-000001",
            "content_id": "CNT-000001",
            "status": "scored",
            "started_at": "2025-01-08 10:00:00.000000",
            "max_score": 100.0,
        },
    ],
    "assessment_results": [
        {
            "result_id": "RES-BEFORE",
            "attempt_id": "ATT-BEFORE",
            "outcome": "pass",
            "passed": 1,
            "score": 80.0,
            "scored_at": "2025-01-01 10:30:00.000000",
        },
        {
            "result_id": "RES-AFTER",
            "attempt_id": "ATT-AFTER",
            "outcome": "fail",
            "passed": 0,
            "score": 20.0,
            "scored_at": "2025-01-08 10:30:00.000000",
        },
    ],
}


def _dummy_artifact() -> ModelArtifact:
    model = LogisticRegression()
    model.fit(np.zeros((2, 8)), [0, 1])
    return ModelArtifact(
        model=model,
        feature_names=list(FEATURE_NAMES),
        model_version="ml-v1",
        trained_at="2025-01-01",
        metrics={},
        coefficients={name: 0.0 for name in FEATURE_NAMES},
        intercept=0.0,
    )


def test_f06_rule_and_ml_use_the_same_evaluation_cutoff():
    snaps = calculate_mastery_snapshots(
        MASTER, ACTIVITY, as_of_timestamp=CUTOFF, include_cold_start=True
    )
    artifact = _dummy_artifact()
    rule_recs = generate_recommendations(
        MASTER, ACTIVITY, snaps, as_of_timestamp=CUTOFF, k=5
    )
    ml_recs = generate_ml_recommendations(
        MASTER, ACTIVITY, snaps, artifact, as_of_timestamp=CUTOFF, k=5
    )
    assert rule_recs
    assert ml_recs
    assert {r.as_of_timestamp for r in rule_recs} == {CUTOFF}
    assert {r.as_of_timestamp for r in ml_recs} == {CUTOFF}
    assert {r.as_of_timestamp for r in rule_recs} == {r.as_of_timestamp for r in ml_recs}


def test_f06_mastery_is_recalculated_at_the_cutoff():
    snaps_cutoff = calculate_mastery_snapshots(
        MASTER, ACTIVITY, as_of_timestamp=CUTOFF, include_cold_start=True
    )
    snaps_future = calculate_mastery_snapshots(
        MASTER, ACTIVITY, as_of_timestamp=FUTURE, include_cold_start=True
    )
    cutoff_snap = next(
        s for s in snaps_cutoff
        if s.learner_id == "LRN-000001" and s.skill_id == "SKL-000001"
    )
    future_snap = next(
        s for s in snaps_future
        if s.learner_id == "LRN-000001" and s.skill_id == "SKL-000001"
    )
    assert cutoff_snap.evidence_attempt_count == 1
    assert future_snap.evidence_attempt_count == 2
    assert cutoff_snap.mastery_score != future_snap.mastery_score
    assert cutoff_snap.evidence_cutoff_ts <= CUTOFF
    assert cutoff_snap.as_of_timestamp <= CUTOFF


def test_f06_future_mastery_snapshots_are_rejected_by_both_engines():
    future_snaps = calculate_mastery_snapshots(
        MASTER, ACTIVITY, as_of_timestamp=FUTURE, include_cold_start=True
    )
    artifact = _dummy_artifact()
    rule_recs = generate_recommendations(
        MASTER, ACTIVITY, future_snaps, as_of_timestamp=CUTOFF, k=5
    )
    ml_recs = generate_ml_recommendations(
        MASTER, ACTIVITY, future_snaps, artifact, as_of_timestamp=CUTOFF, k=5
    )
    # Future snapshots have as_of > CUTOFF, so engines treat mastery as missing (0.0)
    # and the required SKL-000001 >= 0.60 gate blocks CNT-000002.
    assert all(r.content_id != "CNT-000002" for r in rule_recs)
    assert all(r.content_id != "CNT-000002" for r in ml_recs)


def test_f06_attempts_after_cutoff_do_not_leak_into_evaluation():
    snaps = calculate_mastery_snapshots(
        MASTER, ACTIVITY, as_of_timestamp=CUTOFF, include_cold_start=True
    )
    artifact = _dummy_artifact()
    rule_recs = generate_recommendations(
        MASTER, ACTIVITY, snaps, as_of_timestamp=CUTOFF, k=5
    )
    ml_recs = generate_ml_recommendations(
        MASTER, ACTIVITY, snaps, artifact, as_of_timestamp=CUTOFF, k=5
    )
    rule_item = next(
        r for r in rule_recs
        if r.learner_id == "LRN-000001" and r.content_id == "CNT-000001"
    )
    ml_item = next(
        r for r in ml_recs
        if r.learner_id == "LRN-000001" and r.content_id == "CNT-000001"
    )
    # ATT-AFTER (fail on Jan 8) must not create RETRY_AFTER_FAIL at the Jan 3 cutoff.
    assert "RETRY_AFTER_FAIL" not in rule_item.reason_codes
    assert "RETRY_AFTER_FAIL" not in ml_item.reason_codes
    snap = next(
        s for s in snaps
        if s.learner_id == "LRN-000001" and s.skill_id == "SKL-000001"
    )
    assert snap.evidence_attempt_count == 1


def test_f06_cold_start_zero_baselines_cannot_pass_positive_prerequisite_thresholds():
    links = [ContentSkillLink(content_id="CNT-000002", skill_id="SKL-000002", skill_role="primary")]
    edges = [
        PrerequisiteEdge(
            skill_id="SKL-000002",
            prerequisite_skill_id="SKL-000001",
            relationship_type="required",
            min_mastery_required=0.60,
        )
    ]
    cold_mastery = {"SKL-000001": 0.0}
    assert not content_passes_required_prerequisites(
        "CNT-000002",
        links,
        edges,
        cold_mastery,
        missing_mastery=0.0,
    )
    assert not content_passes_required_prerequisites(
        "CNT-000002",
        links,
        edges,
        {},
        missing_mastery=0.0,
    )
