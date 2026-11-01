"""Explicit tests documenting recommendation history timestamp boundaries and reason codes.

Documents:
- Rule history boundary: started_at <= as_of_timestamp
- ML prior history boundary: started_at < as_of_timestamp
- RETRY_AFTER_FAIL reason-code behavior in both engines.
"""

from __future__ import annotations

import unittest.mock
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from learnpath_ai.ml.features import FEATURE_NAMES
from learnpath_ai.ml.engine import generate_ml_recommendations
from learnpath_ai.ml.model import ModelArtifact
from learnpath_ai.recommend.engine import generate_recommendations


@pytest.fixture
def boundary_master():
    return {
        "courses": [{"course_id": "CRS-001", "status": "published"}],
        "learners": [
            {"learner_id": "LRN-001", "current_course_id": "CRS-001", "status": "active"}
        ],
        "skills": [{"skill_id": "SKL-001", "status": "active"}],
        "skill_prerequisites": [],
        "content_items": [
            {"content_id": "CNT-001", "course_id": "CRS-001", "difficulty": "1.00", "status": "published"}
        ],
        "content_skills": [
            {"content_id": "CNT-001", "skill_id": "SKL-001", "skill_role": "primary", "weight": "1.0"}
        ],
    }


def _dummy_ml_artifact() -> ModelArtifact:
    mock_lr = LogisticRegression()
    mock_lr.fit(np.zeros((2, len(FEATURE_NAMES))), [0, 1])
    return ModelArtifact(
        model=mock_lr,
        feature_names=FEATURE_NAMES,
        model_version="test-v1",
        trained_at="2025-01-01",
        metrics={},
        coefficients={f: 1.0 for f in FEATURE_NAMES},
        intercept=0.0,
    )


def test_rule_history_boundary_includes_attempt_at_exact_as_of(boundary_master):
    """Rule engine boundary test: an attempt starting at exact as_of is counted (started_at <= as_of).

    If an attempt started exactly at T with a fail result scored by T:
    - Rule-v1 indexes it as a prior attempt (<= T) and flags RETRY_AFTER_FAIL.
    """
    cutoff = "2025-01-01 10:00:00.000000"
    activity = {
        "attempts": [
            {
                "attempt_id": "ATT-001",
                "learner_id": "LRN-001",
                "content_id": "CNT-001",
                "status": "scored",
                "started_at": cutoff,  # Exactly at cutoff!
            }
        ],
        "assessment_results": [
            {
                "result_id": "RES-001",
                "attempt_id": "ATT-001",
                "outcome": "fail",
                "passed": 0,
                "score": 40.0,
                "scored_at": cutoff,
            }
        ],
    }

    recs = generate_recommendations(
        boundary_master,
        activity,
        mastery_snapshots=[],
        as_of_timestamp=cutoff,
    )
    rec = next(r for r in recs if r.learner_id == "LRN-001" and r.content_id == "CNT-001")
    # In rule engine: started_at <= as_of -> included -> outcome='fail' -> RETRY_AFTER_FAIL
    assert "RETRY_AFTER_FAIL" in rec.reason_codes, (
        "Rule engine must include attempt at exact as_of_timestamp (<= boundary)"
    )


def test_ml_history_boundary_excludes_attempt_at_exact_as_of(boundary_master):
    """ML engine boundary test: an attempt starting at exact as_of is excluded from prior counts (started_at < as_of).

    For the ML engine, an attempt starting at T represents a current/candidate attempt at T,
    so prior evidence only includes events strictly before T (att_dt < as_of_dt).
    Therefore, prior_count is 0 and RETRY_AFTER_FAIL is NOT attached.
    """
    cutoff = "2025-01-01 10:00:00.000000"
    activity = {
        "attempts": [
            {
                "attempt_id": "ATT-001",
                "learner_id": "LRN-001",
                "content_id": "CNT-001",
                "status": "scored",
                "started_at": cutoff,  # Exactly at cutoff!
            }
        ],
        "assessment_results": [
            {
                "result_id": "RES-001",
                "attempt_id": "ATT-001",
                "outcome": "fail",
                "passed": 0,
                "score": 40.0,
                "scored_at": cutoff,
            }
        ],
    }

    artifact = _dummy_ml_artifact()
    recs = generate_ml_recommendations(
        boundary_master,
        activity,
        mastery_snapshots=[],
        model_artifact=artifact,
        as_of_timestamp=cutoff,
    )
    rec = next(r for r in recs if r.learner_id == "LRN-001" and r.content_id == "CNT-001")
    # In ML engine: started_at < as_of -> excluded (0 prior attempts) -> no RETRY_AFTER_FAIL
    assert "RETRY_AFTER_FAIL" not in rec.reason_codes, (
        "ML engine must strictly exclude attempts starting at exact as_of_timestamp (< boundary)"
    )


def test_retry_after_fail_reason_code_in_both_engines_when_prior_to_cutoff(boundary_master):
    """When a failed attempt started strictly prior to as_of, both engines trigger RETRY_AFTER_FAIL."""
    prior_time = "2025-01-01 09:00:00.000000"
    cutoff = "2025-01-01 10:00:00.000000"
    activity = {
        "attempts": [
            {
                "attempt_id": "ATT-001",
                "learner_id": "LRN-001",
                "content_id": "CNT-001",
                "status": "scored",
                "started_at": prior_time,
            }
        ],
        "assessment_results": [
            {
                "result_id": "RES-001",
                "attempt_id": "ATT-001",
                "outcome": "fail",
                "passed": 0,
                "score": 40.0,
                "scored_at": prior_time,
            }
        ],
    }

    # 1. Rule Engine
    rule_recs = generate_recommendations(
        boundary_master,
        activity,
        mastery_snapshots=[],
        as_of_timestamp=cutoff,
    )
    r_rec = next(r for r in rule_recs if r.learner_id == "LRN-001" and r.content_id == "CNT-001")
    assert "RETRY_AFTER_FAIL" in r_rec.reason_codes

    # 2. ML Engine
    artifact = _dummy_ml_artifact()
    ml_recs = generate_ml_recommendations(
        boundary_master,
        activity,
        mastery_snapshots=[],
        model_artifact=artifact,
        as_of_timestamp=cutoff,
    )
    m_rec = next(r for r in ml_recs if r.learner_id == "LRN-001" and r.content_id == "CNT-001")
    assert "RETRY_AFTER_FAIL" in m_rec.reason_codes

