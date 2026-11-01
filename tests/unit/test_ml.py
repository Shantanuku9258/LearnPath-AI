"""Unit tests for Phase 8 ML recommendation extension."""

from __future__ import annotations

import math
import unittest.mock
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from learnpath_ai.ml.dataset import (
    FEATURE_NAMES,
    MLTrainingExample,
    build_ml_dataset,
    chronological_split,
)
from learnpath_ai.ml.engine import generate_ml_recommendations
from learnpath_ai.ml.evaluate import (
    evaluate_recommendation_strategies,
    evaluate_recommendations_for_test_slice,
)
from learnpath_ai.ml.model import ModelArtifact, train_model
from learnpath_ai.recommend.engine import RecommendationRecord, generate_recommendations


@pytest.fixture
def mock_master():
    return {
        "courses": [{"course_id": "CRS-000001", "title": "CS 101", "status": "published"}],
        "learners": [
            {"learner_id": "LRN-000001", "current_course_id": "CRS-000001", "starting_level": "beginner", "status": "active"},
            {"learner_id": "LRN-000002", "current_course_id": "CRS-000001", "starting_level": "intermediate", "status": "active"},
            {"learner_id": "LRN-000003", "current_course_id": "CRS-000001", "starting_level": "advanced", "status": "withdrawn"},
        ],
        "skills": [
            {"skill_id": "SKL-000001", "skill_name": "Variables", "domain": "CS", "status": "active"},
            {"skill_id": "SKL-000002", "skill_name": "Conditionals", "domain": "CS", "status": "active"},
        ],
        "skill_prerequisites": [
            {"skill_id": "SKL-000002", "prerequisite_skill_id": "SKL-000001", "relationship_type": "required", "min_mastery_required": 0.60}
        ],
        "content_items": [
            {"content_id": "CNT-000001", "course_id": "CRS-000001", "title": "Vars", "difficulty": "1.5", "status": "published"},
            {"content_id": "CNT-000002", "course_id": "CRS-000001", "title": "Branches", "difficulty": "2.5", "status": "published"},
            {"content_id": "CNT-000003", "course_id": "CRS-000001", "title": "Draft Topic", "difficulty": "2.0", "status": "draft"},
        ],
        "content_skills": [
            {"content_id": "CNT-000001", "skill_id": "SKL-000001", "skill_role": "primary", "weight": "1.0"},
            {"content_id": "CNT-000002", "skill_id": "SKL-000002", "skill_role": "primary", "weight": "1.0"},
            {"content_id": "CNT-000003", "skill_id": "SKL-000001", "skill_role": "primary", "weight": "1.0"},
        ],
    }


@pytest.fixture
def mock_activity():
    attempts = [
        {"attempt_id": "ATT-001", "learner_id": "LRN-000001", "content_id": "CNT-000001", "status": "scored", "started_at": "2025-01-01 10:00:00.000000"},
        {"attempt_id": "ATT-002", "learner_id": "LRN-000001", "content_id": "CNT-000001", "status": "scored", "started_at": "2025-01-02 10:00:00.000000"},
        {"attempt_id": "ATT-003", "learner_id": "LRN-000002", "content_id": "CNT-000001", "status": "scored", "started_at": "2025-01-03 10:00:00.000000"},
        {"attempt_id": "ATT-004", "learner_id": "LRN-000002", "content_id": "CNT-000002", "status": "scored", "started_at": "2025-01-04 10:00:00.000000"},
    ]
    results = [
        {"result_id": "RES-001", "attempt_id": "ATT-001", "outcome": "pass", "passed": 1, "score": 90.0, "scored_at": "2025-01-01 10:30:00.000000"},
        {"result_id": "RES-002", "attempt_id": "ATT-002", "outcome": "fail", "passed": 0, "score": 45.0, "scored_at": "2025-01-02 10:30:00.000000"},
        {"result_id": "RES-003", "attempt_id": "ATT-003", "outcome": "fail", "passed": 0, "score": 50.0, "scored_at": "2025-01-03 10:30:00.000000"},
        {"result_id": "RES-004", "attempt_id": "ATT-004", "outcome": "pass", "passed": 1, "score": 85.0, "scored_at": "2025-01-04 10:30:00.000000"},
    ]
    return {"attempts": attempts, "assessment_results": results}


@pytest.fixture
def balanced_ml_activity():
    """Eight chronological scored attempts so a 50/25/25 split has both classes in val.

    Order: pass, fail, fail, pass | fail, pass | fail, pass
    """
    attempts = [
        {"attempt_id": "ATT-001", "learner_id": "LRN-000001", "content_id": "CNT-000001", "status": "scored", "started_at": "2025-01-01 10:00:00.000000"},
        {"attempt_id": "ATT-002", "learner_id": "LRN-000001", "content_id": "CNT-000001", "status": "scored", "started_at": "2025-01-02 10:00:00.000000"},
        {"attempt_id": "ATT-003", "learner_id": "LRN-000002", "content_id": "CNT-000001", "status": "scored", "started_at": "2025-01-03 10:00:00.000000"},
        {"attempt_id": "ATT-004", "learner_id": "LRN-000002", "content_id": "CNT-000002", "status": "scored", "started_at": "2025-01-04 10:00:00.000000"},
        {"attempt_id": "ATT-005", "learner_id": "LRN-000001", "content_id": "CNT-000001", "status": "scored", "started_at": "2025-01-05 10:00:00.000000"},
        {"attempt_id": "ATT-006", "learner_id": "LRN-000002", "content_id": "CNT-000001", "status": "scored", "started_at": "2025-01-06 10:00:00.000000"},
        {"attempt_id": "ATT-007", "learner_id": "LRN-000001", "content_id": "CNT-000001", "status": "scored", "started_at": "2025-01-07 10:00:00.000000"},
        {"attempt_id": "ATT-008", "learner_id": "LRN-000002", "content_id": "CNT-000001", "status": "scored", "started_at": "2025-01-08 10:00:00.000000"},
    ]
    results = [
        {"result_id": "RES-001", "attempt_id": "ATT-001", "outcome": "pass", "passed": 1, "score": 90.0, "scored_at": "2025-01-01 10:30:00.000000"},
        {"result_id": "RES-002", "attempt_id": "ATT-002", "outcome": "fail", "passed": 0, "score": 45.0, "scored_at": "2025-01-02 10:30:00.000000"},
        {"result_id": "RES-003", "attempt_id": "ATT-003", "outcome": "fail", "passed": 0, "score": 50.0, "scored_at": "2025-01-03 10:30:00.000000"},
        {"result_id": "RES-004", "attempt_id": "ATT-004", "outcome": "pass", "passed": 1, "score": 85.0, "scored_at": "2025-01-04 10:30:00.000000"},
        {"result_id": "RES-005", "attempt_id": "ATT-005", "outcome": "fail", "passed": 0, "score": 40.0, "scored_at": "2025-01-05 10:30:00.000000"},
        {"result_id": "RES-006", "attempt_id": "ATT-006", "outcome": "pass", "passed": 1, "score": 80.0, "scored_at": "2025-01-06 10:30:00.000000"},
        {"result_id": "RES-007", "attempt_id": "ATT-007", "outcome": "fail", "passed": 0, "score": 35.0, "scored_at": "2025-01-07 10:30:00.000000"},
        {"result_id": "RES-008", "attempt_id": "ATT-008", "outcome": "pass", "passed": 1, "score": 88.0, "scored_at": "2025-01-08 10:30:00.000000"},
    ]
    return {"attempts": attempts, "assessment_results": results}


def _train_two_class_model(master, activity):
    """Train ml-v1 on a chronological split whose validation labels include both classes."""
    examples = build_ml_dataset(master, activity)
    split = chronological_split(examples, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    assert split.train, "training split must not be empty"
    assert split.val, "validation split must not be empty"
    assert {ex.label for ex in split.train} == {0, 1}
    val_labels = {ex.label for ex in split.val}
    assert val_labels == {0, 1}, f"validation labels must include both classes, got {sorted(val_labels)}"
    artifact = train_model(split, seed=20261022)
    auc = artifact.metrics["roc_auc"]
    assert math.isfinite(auc), f"ROC-AUC must be finite, got {auc!r}"
    assert 0.0 <= auc <= 1.0, f"ROC-AUC out of range: {auc}"
    return artifact


def test_build_ml_dataset_and_no_leakage(mock_master, mock_activity):
    """Features at time T only use information strictly prior to T."""
    examples = build_ml_dataset(mock_master, mock_activity)
    assert len(examples) == 4

    # ATT-001 was first attempt for LRN-000001: prior attempts must be 0
    ex1 = next(e for e in examples if e.attempt_id == "ATT-001")
    assert ex1.learner_prior_attempts_on_content == 0
    assert ex1.learner_prior_passes_on_content == 0
    assert ex1.label == 1

    # ATT-002 was second attempt for LRN-000001: prior attempts must be 1, passes 1
    ex2 = next(e for e in examples if e.attempt_id == "ATT-002")
    assert ex2.learner_prior_attempts_on_content == 1
    assert ex2.learner_prior_passes_on_content == 1
    assert ex2.label == 0


def test_strict_scored_at_leakage_prevention(mock_master):
    """Assessment outcome scored after a later attempt's start must not leak into that attempt's features.

    Scenario:
      ATT-A starts at T1=2025-01-01 10:00, scored at T3=2025-01-03 10:00.
      ATT-B starts at T2=2025-01-02 10:00 — BEFORE ATT-A is scored.

    F-10 policy:
      - Behavioral counts (learner_prior_attempts_on_content): count all non-invalid attempts
        started strictly before T. ATT-A started before ATT-B, so count = 1.
      - Pass/scored evidence (learner_prior_passes_on_content): only scored results available
        by the cutoff T (scored_at <= T). ATT-A's result scored at Jan 3 > Jan 2, so passes = 0.
    """
    activity = {
        "attempts": [
            {
                "attempt_id": "ATT-A",
                "learner_id": "LRN-000001",
                "content_id": "CNT-000001",
                "status": "scored",
                "started_at": "2025-01-01 10:00:00.000000",
            },
            {
                "attempt_id": "ATT-B",
                "learner_id": "LRN-000001",
                "content_id": "CNT-000001",
                "status": "scored",
                "started_at": "2025-01-02 10:00:00.000000",
            },
        ],
        "assessment_results": [
            {
                "result_id": "RES-A",
                "attempt_id": "ATT-A",
                "outcome": "pass",
                "passed": 1,
                "score": 90.0,
                # scored AFTER ATT-B starts — pass must NOT appear in ATT-B's evidence
                "scored_at": "2025-01-03 10:00:00.000000",
            },
            {
                "result_id": "RES-B",
                "attempt_id": "ATT-B",
                "outcome": "fail",
                "passed": 0,
                "score": 40.0,
                "scored_at": "2025-01-02 12:00:00.000000",
            },
        ],
    }
    examples = build_ml_dataset(mock_master, activity)
    ex_b = next(e for e in examples if e.attempt_id == "ATT-B")
    # ATT-A started before ATT-B → counts as 1 behavioral prior attempt on content (F-10 policy)
    assert ex_b.learner_prior_attempts_on_content == 1, (
        f"Expected 1 prior attempt for ATT-B (ATT-A started before it); got {ex_b.learner_prior_attempts_on_content}."
    )
    # ATT-A's result scored at Jan 3 > ATT-B start Jan 2 → 0 prior passes (score not yet available)
    assert ex_b.learner_prior_passes_on_content == 0, (
        f"Expected 0 prior passes for ATT-B; got {ex_b.learner_prior_passes_on_content}. "
        "ATT-A outcome was not yet available at ATT-B start time."
    )


def test_chronological_split(mock_master, mock_activity):
    """Split orders by timestamp strictly without overlap."""
    examples = build_ml_dataset(mock_master, mock_activity)
    split = chronological_split(examples, train_ratio=0.5, val_ratio=0.25, test_ratio=0.25)
    assert len(split.train) == 2
    assert len(split.val) == 1
    assert len(split.test) == 1
    assert split.train[-1].timestamp <= split.val[0].timestamp
    assert split.val[-1].timestamp <= split.test[0].timestamp


def test_model_training_and_serialization(tmp_path, mock_master, balanced_ml_activity):
    """Model trains reproducibly and saves/loads artifacts."""
    artifact = _train_two_class_model(mock_master, balanced_ml_activity)

    assert artifact.model_version == "ml-v1"
    assert len(artifact.feature_names) == len(FEATURE_NAMES)
    assert artifact.intercept is not None

    artifact.save(tmp_path)
    loaded = ModelArtifact.load(tmp_path, "ml-v1")
    assert loaded.model_version == artifact.model_version
    assert loaded.intercept == artifact.intercept


def test_ml_recommendations_enforce_hard_prerequisite_gate(mock_master, balanced_ml_activity):
    """ML model cannot recommend content failing required prerequisites."""
    artifact = _train_two_class_model(mock_master, balanced_ml_activity)

    # Learner 1 has mastery on SKL-000001 = 0.0 (fails requirement for CNT-000002)
    mastery = [
        {"snapshot_id": "MST-01", "learner_id": "LRN-000001", "skill_id": "SKL-000001", "mastery_score": 0.20, "as_of_timestamp": "2025-01-05 00:00:00.000000"}
    ]
    recs = generate_ml_recommendations(
        mock_master,
        balanced_ml_activity,
        mastery,
        artifact,
        as_of_timestamp="2025-01-05 00:00:00.000000",
        k=5,
    )

    recs_l1 = [r for r in recs if r.learner_id == "LRN-000001"]
    assert len(recs_l1) == 1
    # CNT-000002 requires SKL-000001 >= 0.60; it MUST be excluded
    assert recs_l1[0].content_id == "CNT-000001"


def test_cold_start_learners_handled_safely(mock_master, balanced_ml_activity):
    """Cold-start learners with 0 history evaluate gracefully and receive recommendations."""
    artifact = _train_two_class_model(mock_master, balanced_ml_activity)

    recs = generate_ml_recommendations(
        mock_master,
        balanced_ml_activity,
        [],  # No mastery snapshots
        artifact,
        as_of_timestamp="2025-01-05 00:00:00.000000",
        k=5,
    )
    assert len(recs) > 0
    # Withdrawn learner LRN-000003 must be excluded
    assert all(r.learner_id != "LRN-000003" for r in recs)


def test_evaluation_metrics_comparison():
    """Precision, Recall, and NDCG calculate accurately."""
    from learnpath_ai.recommend.engine import RecommendationRecord
    recs_rule = [
        RecommendationRecord(
            recommendation_id="REC-1", learner_id="LRN-1", content_id="CNT-1", path_id="",
            rank_position=1, engine_version="rule-v1", reason_codes=[], score=0.9, status="pending",
            recommended_at="2026-01-01", as_of_timestamp="2026-01-01", expires_at="", created_at="2026-01-01"
        )
    ]
    recs_ml = [
        RecommendationRecord(
            recommendation_id="REC-2", learner_id="LRN-1", content_id="CNT-2", path_id="",
            rank_position=1, engine_version="ml-v1", reason_codes=[], score=0.8, status="pending",
            recommended_at="2026-01-01", as_of_timestamp="2026-01-01", expires_at="", created_at="2026-01-01"
        )
    ]
    ground_truth = [{"learner_id": "LRN-1", "content_id": "CNT-1", "passed": 1, "outcome": "pass"}]

    res = evaluate_recommendation_strategies(
        recs_rule,
        recs_ml,
        ground_truth,
        eligible_learners=["LRN-1"],
        cold_start_learners=[],
        k_list=[1],
    )
    assert res["rule-v1"].precision_at_k[1] == 1.0
    assert res["ml-v1"].precision_at_k[1] == 0.0


def _metric_rec(learner_id: str, content_id: str, rank: int, engine: str = "rule-v1") -> RecommendationRecord:
    return RecommendationRecord(
        recommendation_id=f"REC-{learner_id}-{rank}",
        learner_id=learner_id,
        content_id=content_id,
        path_id="",
        rank_position=rank,
        engine_version=engine,
        reason_codes=[],
        score=1.0 - (rank * 0.05),
        status="pending",
        recommended_at="2026-01-01",
        as_of_timestamp="2026-01-01",
        expires_at="",
        created_at="2026-01-01",
    )


def test_evaluation_coverage_and_cold_start_coverage():
    recs = [
        _metric_rec("LRN-A", "CNT-1", 1),
        _metric_rec("LRN-C", "CNT-9", 1),
    ]
    ground_truth = [
        {"learner_id": "LRN-A", "content_id": "CNT-1", "passed": 1, "outcome": "pass"},
    ]
    metrics = evaluate_recommendations_for_test_slice(
        recs,
        ground_truth,
        eligible_learners=["LRN-A", "LRN-B", "LRN-C"],
        cold_start_learners=["LRN-B", "LRN-C"],
        k_list=[1, 3, 5],
        strategy_name="rule-v1",
    )
    assert metrics.total_learners == 3
    assert metrics.covered_learners == 2
    assert metrics.coverage == 0.6667
    assert metrics.cold_start_coverage == 0.5


def test_evaluation_precision_recall_ndcg_at_k():
    """Manually verifiable P/R/NDCG for one learner with two relevant items."""
    recs = [
        _metric_rec("LRN-A", "CNT-1", 1),
        _metric_rec("LRN-A", "CNT-2", 2),
        _metric_rec("LRN-A", "CNT-3", 3),
        _metric_rec("LRN-A", "CNT-4", 4),
        _metric_rec("LRN-A", "CNT-5", 5),
    ]
    ground_truth = [
        {"learner_id": "LRN-A", "content_id": "CNT-1", "passed": 1, "outcome": "pass"},
        {"learner_id": "LRN-A", "content_id": "CNT-3", "passed": 1, "outcome": "pass"},
        {"learner_id": "LRN-A", "content_id": "CNT-9", "passed": 0, "outcome": "fail"},
    ]
    metrics = evaluate_recommendations_for_test_slice(
        recs,
        ground_truth,
        eligible_learners=["LRN-A"],
        cold_start_learners=[],
        k_list=[1, 3, 5],
    )
    # rel = {CNT-1, CNT-3}; ranked = CNT-1, CNT-2, CNT-3, CNT-4, CNT-5
    assert metrics.precision_at_k[1] == pytest.approx(1.0)
    assert metrics.precision_at_k[3] == pytest.approx(2 / 3)
    assert metrics.precision_at_k[5] == pytest.approx(2 / 5)
    assert metrics.recall_at_k[1] == pytest.approx(0.5)
    assert metrics.recall_at_k[3] == pytest.approx(1.0)
    assert metrics.recall_at_k[5] == pytest.approx(1.0)
    # NDCG@1 = 1. NDCG@3 DCG=1 + 1/log2(4)=1.5; IDCG=1 + 1/log2(3)
    assert metrics.ndcg_at_k[1] == pytest.approx(1.0)
    idcg3 = 1.0 + 1.0 / math.log2(3)
    assert metrics.ndcg_at_k[3] == pytest.approx(1.5 / idcg3)
    idcg5 = idcg3
    dcg5 = 1.0 + 1.0 / math.log2(4)
    assert metrics.ndcg_at_k[5] == pytest.approx(dcg5 / idcg5)


def test_evaluation_ignores_learners_with_no_relevant_test_items():
    recs = [
        _metric_rec("LRN-HIT", "CNT-1", 1),
        _metric_rec("LRN-MISS", "CNT-2", 1),
    ]
    ground_truth = [
        {"learner_id": "LRN-HIT", "content_id": "CNT-1", "passed": 1, "outcome": "pass"},
        {"learner_id": "LRN-MISS", "content_id": "CNT-9", "passed": 0, "outcome": "fail"},
    ]
    metrics = evaluate_recommendations_for_test_slice(
        recs,
        ground_truth,
        eligible_learners=["LRN-HIT", "LRN-MISS"],
        cold_start_learners=[],
        k_list=[1, 3, 5],
    )
    assert metrics.coverage == 1.0
    # Only LRN-HIT contributes to ranking metrics
    assert metrics.precision_at_k[1] == pytest.approx(1.0)
    assert metrics.recall_at_k[1] == pytest.approx(1.0)
    assert metrics.ndcg_at_k[1] == pytest.approx(1.0)


def test_evaluation_empty_recommendation_list():
    ground_truth = [
        {"learner_id": "LRN-A", "content_id": "CNT-1", "passed": 1, "outcome": "pass"},
    ]
    metrics = evaluate_recommendations_for_test_slice(
        [],
        ground_truth,
        eligible_learners=["LRN-A", "LRN-B"],
        cold_start_learners=["LRN-B"],
        k_list=[1, 3, 5],
    )
    assert metrics.coverage == 0.0
    assert metrics.covered_learners == 0
    assert metrics.cold_start_coverage == 0.0
    assert metrics.precision_at_k[1] == 0.0
    assert metrics.precision_at_k[5] == 0.0
    assert metrics.recall_at_k[3] == 0.0
    assert metrics.ndcg_at_k[5] == 0.0


def test_evaluation_fewer_candidates_than_k():
    recs = [_metric_rec("LRN-A", "CNT-1", 1)]
    ground_truth = [
        {"learner_id": "LRN-A", "content_id": "CNT-1", "passed": 1, "outcome": "pass"},
        {"learner_id": "LRN-A", "content_id": "CNT-2", "passed": 1, "outcome": "pass"},
    ]
    metrics = evaluate_recommendations_for_test_slice(
        recs,
        ground_truth,
        eligible_learners=["LRN-A"],
        cold_start_learners=[],
        k_list=[1, 3, 5],
    )
    # hits@1=1, hits@3=1, hits@5=1; denominator for precision is K not list length
    assert metrics.precision_at_k[1] == pytest.approx(1.0)
    assert metrics.precision_at_k[3] == pytest.approx(1 / 3)
    assert metrics.precision_at_k[5] == pytest.approx(1 / 5)
    assert metrics.recall_at_k[1] == pytest.approx(0.5)
    assert metrics.recall_at_k[3] == pytest.approx(0.5)
    assert metrics.recall_at_k[5] == pytest.approx(0.5)
    assert metrics.ndcg_at_k[1] == pytest.approx(1.0)


def test_point_in_time_mastery_evaluation_no_future_leakage(mock_master):
    """Evidence after evaluation cutoff does not affect point-in-time mastery,
    and evidence before cutoff is properly aggregated.
    """
    from learnpath_ai.mastery.calculator import calculate_mastery_snapshots

    # Mock activity with attempt 1 before cutoff and attempt 2 after cutoff
    cutoff = "2025-01-02 00:00:00.000000"
    activity = {
        "attempts": [
            {
                "attempt_id": "ATT-001", "learner_id": "LRN-000001", "content_id": "CNT-000001",
                "status": "scored", "started_at": "2025-01-01 10:00:00.000000"
            },
            {
                "attempt_id": "ATT-002", "learner_id": "LRN-000001", "content_id": "CNT-000001",
                "status": "scored", "started_at": "2025-01-03 10:00:00.000000"  # After cutoff
            },
        ],
        "assessment_results": [
            {"result_id": "RES-001", "attempt_id": "ATT-001", "score": 80.0, "scored_at": "2025-01-01 10:30:00.000000"},
            {"result_id": "RES-002", "attempt_id": "ATT-002", "score": 20.0, "scored_at": "2025-01-03 10:30:00.000000"},
        ]
    }

    # Point-in-time calculation at cutoff
    pit_snaps = calculate_mastery_snapshots(
        mock_master,
        activity,
        as_of_timestamp=cutoff,
        include_cold_start=True,
    )

    # Learner 1 SKL-000001 snapshot should ONLY reflect ATT-001 (score 80)
    snap = next(s for s in pit_snaps if s.learner_id == "LRN-000001" and s.skill_id == "SKL-000001")
    assert snap.evidence_attempt_count == 1
    # Credibility for n=1: 80% * (1 / (1 + 1)) = 40.0% = 0.4000
    assert snap.mastery_score == 0.4000
    assert snap.as_of_timestamp <= cutoff
    assert snap.evidence_cutoff_ts <= cutoff

    # Calculation without cutoff includes ATT-002 which lowers score
    full_snaps = calculate_mastery_snapshots(
        mock_master,
        activity,
        as_of_timestamp="2025-01-10 00:00:00.000000",
        include_cold_start=True,
    )
    full_snap = next(s for s in full_snaps if s.learner_id == "LRN-000001" and s.skill_id == "SKL-000001")
    assert full_snap.evidence_attempt_count == 2
    assert full_snap.mastery_score != snap.mastery_score


def test_engines_receive_same_point_in_time_mastery(mock_master, balanced_ml_activity):
    """Rule-based and ML engines evaluate identical point-in-time mastery snapshots."""
    from learnpath_ai.mastery.calculator import calculate_mastery_snapshots

    cutoff = "2025-01-03 00:00:00.000000"
    pit_snaps = calculate_mastery_snapshots(
        mock_master,
        balanced_ml_activity,
        as_of_timestamp=cutoff,
        include_cold_start=True,
    )

    rule_recs = generate_recommendations(
        mock_master,
        balanced_ml_activity,
        pit_snaps,
        as_of_timestamp=cutoff,
        k=5,
    )

    artifact = _train_two_class_model(mock_master, balanced_ml_activity)

    ml_recs = generate_ml_recommendations(
        mock_master,
        balanced_ml_activity,
        pit_snaps,
        artifact,
        as_of_timestamp=cutoff,
        k=5,
    )

    # Both engines receive the exact same snapshot inputs and check the same prerequisite gates
    # Verify neither engine failed with unhandled exceptions
    assert len(rule_recs) > 0
    assert len(ml_recs) > 0

    # Prerequisite gating is strictly identical across both engines
    rule_content_ids = {r.content_id for r in rule_recs}
    ml_content_ids = {r.content_id for r in ml_recs}
    assert rule_content_ids == ml_content_ids
    # For LRN-000001, SKL-000001 mastery is below 0.60 at 2025-01-03, so CNT-000002 must be excluded in both
    assert "CNT-000002" not in rule_content_ids
    assert "CNT-000002" not in ml_content_ids


def test_future_dated_snapshots_rejected_by_engines(mock_master, balanced_ml_activity):
    """Snapshots with as_of_timestamp after recommendation as_of_timestamp are rejected by engines."""
    from learnpath_ai.mastery.calculator import calculate_mastery_snapshots

    # Generate snapshots with future timestamp relative to recommendation issuance
    future_snaps = calculate_mastery_snapshots(
        mock_master,
        balanced_ml_activity,
        as_of_timestamp="2025-01-10 00:00:00.000000",
        include_cold_start=True,
    )

    past_cutoff = "2025-01-02 00:00:00.000000"

    # Rule recommendations at past_cutoff should reject future snapshots
    recs_rule = generate_recommendations(
        mock_master,
        balanced_ml_activity,
        future_snaps,
        as_of_timestamp=past_cutoff,
        k=5,
    )

    artifact = _train_two_class_model(mock_master, balanced_ml_activity)

    recs_ml = generate_ml_recommendations(
        mock_master,
        balanced_ml_activity,
        future_snaps,
        artifact,
        as_of_timestamp=past_cutoff,
        k=5,
    )

    # Since all future_snaps have as_of (2025-01-10) > past_cutoff (2025-01-02),
    # mastery_by_learner in both engines becomes empty ({})
    # As a result, prerequisite checking defaults to 0.0, strictly filtering out CNT-000002
    assert all(r.content_id != "CNT-000002" for r in recs_rule)
    assert all(r.content_id != "CNT-000002" for r in recs_ml)


# ---------------------------------------------------------------------------
# F-10 Regression Tests: Shared Feature Extraction & Training/Inference Parity
# ---------------------------------------------------------------------------

def test_f10_training_inference_feature_parity():
    """Verify exact 8-feature parity between training example generation and inference."""
    from learnpath_ai.mastery.calculator import calculate_mastery_snapshots
    from learnpath_ai.ml.features import FEATURE_NAMES

    cutoff = "2025-01-10 12:00:00.000000"

    master = {
        "learners": [
            {"learner_id": "LRN-001", "current_course_id": "CRS-001", "starting_level": "intermediate", "status": "active"}
        ],
        "courses": [{"course_id": "CRS-001", "status": "published"}],
        "skills": [
            {"skill_id": "SKL-001", "status": "active"},
            {"skill_id": "SKL-002", "status": "active"},
        ],
        "content_items": [
            {"content_id": "CNT-001", "course_id": "CRS-001", "difficulty": "3.20", "status": "published"},
            {"content_id": "CNT-002", "course_id": "CRS-001", "difficulty": "2.80", "status": "published"},
        ],
        "content_skills": [
            {"content_id": "CNT-001", "skill_id": "SKL-001", "skill_role": "primary", "weight": "1.0"},
            {"content_id": "CNT-002", "skill_id": "SKL-001", "skill_role": "primary", "weight": "1.0"},
        ],
        "skill_prerequisites": [],
    }

    # Historical activities prior to cutoff:
    # Attempt 1: scored 100 on CNT-001 (Skill 1)
    # Attempt 2: in_progress on CNT-001
    # Attempt 3: scored 50 (fail) on CNT-002 (Skill 1)
    activity = {
        "attempts": [
            {
                "attempt_id": "ATT-001", "learner_id": "LRN-001", "content_id": "CNT-001",
                "status": "scored", "started_at": "2025-01-05 10:00:00.000000", "max_score": 100.0,
            },
            {
                "attempt_id": "ATT-002", "learner_id": "LRN-001", "content_id": "CNT-001",
                "status": "in_progress", "started_at": "2025-01-06 10:00:00.000000", "max_score": 100.0,
            },
            {
                "attempt_id": "ATT-003", "learner_id": "LRN-001", "content_id": "CNT-002",
                "status": "scored", "started_at": "2025-01-07 10:00:00.000000", "max_score": 100.0,
            },
            # Candidate attempt occurring exactly at cutoff on CNT-001:
            {
                "attempt_id": "ATT-TARGET", "learner_id": "LRN-001", "content_id": "CNT-001",
                "status": "scored", "started_at": cutoff, "max_score": 100.0,
            },
        ],
        "assessment_results": [
            {"result_id": "RES-001", "attempt_id": "ATT-001", "outcome": "pass", "passed": 1, "score": 100.0, "scored_at": "2025-01-05 10:30:00.000000"},
            {"result_id": "RES-003", "attempt_id": "ATT-003", "outcome": "fail", "passed": 0, "score": 50.0, "scored_at": "2025-01-07 10:30:00.000000"},
            {"result_id": "RES-TARGET", "attempt_id": "ATT-TARGET", "outcome": "pass", "passed": 1, "score": 90.0, "scored_at": "2025-01-10 12:30:00.000000"},
        ],
    }

    # 1. Training features for ATT-TARGET (LRN-001, CNT-001 at cutoff)
    dataset = build_ml_dataset(master, activity)
    target_ex = next(ex for ex in dataset if ex.attempt_id == "ATT-TARGET")
    assert target_ex.learner_id == "LRN-001"
    assert target_ex.content_id == "CNT-001"
    train_vec = target_ex.feature_vector()

    # 2. Inference features at cutoff
    pit_mastery = calculate_mastery_snapshots(master, activity, as_of_timestamp=cutoff, include_cold_start=True)

    mock_lr = LogisticRegression()
    mock_lr.fit(np.zeros((2, 8)), [0, 1])
    artifact = ModelArtifact(
        model=mock_lr,
        feature_names=FEATURE_NAMES,
        model_version="test-v1",
        trained_at="2025-01-10",
        metrics={},
        coefficients={f: 1.0 for f in FEATURE_NAMES},
        intercept=0.0,
    )

    captured_by_content: dict[str, list[float]] = {}
    original_predict = artifact.predict_proba

    def capture_proba(vecs):
        # Parallel to generate_ml_recommendations: one vector per eligible candidate,
        # in the same published-content order used to build metadata_list.
        published = [
            c for c in master["content_items"]
            if str(c.get("status", "")).lower() == "published"
        ]
        assert len(vecs) == len(published), (
            f"Expected one inference vector per published candidate; got {len(vecs)} vectors "
            f"and {len(published)} published items"
        )
        for cand, vec in zip(published, vecs, strict=True):
            captured_by_content[str(cand["content_id"])] = list(vec)
        return original_predict(vecs)

    with unittest.mock.patch.object(artifact, "predict_proba", side_effect=capture_proba):
        recs = generate_ml_recommendations(
            master, activity, pit_mastery, artifact, as_of_timestamp=cutoff, k=5
        )

    assert any(r.learner_id == "LRN-001" and r.content_id == "CNT-001" for r in recs)
    assert "CNT-001" in captured_by_content
    assert "CNT-002" in captured_by_content
    inf_vec = captured_by_content["CNT-001"]
    other_vec = captured_by_content["CNT-002"]
    assert inf_vec != other_vec, (
        "CNT-001 and CNT-002 inference vectors are identical; a wrong-candidate "
        "comparison would not be detected"
    )

    assert len(train_vec) == 8
    assert len(inf_vec) == 8
    for i, (t_val, inf_val) in enumerate(zip(train_vec, inf_vec)):
        assert abs(t_val - inf_val) < 1e-6, (
            f"Feature {FEATURE_NAMES[i]} mismatch for LRN-001/CNT-001: "
            f"train={t_val}, inf={inf_val}"
        )


def test_f10_single_scored_attempt_receives_credibility_shrinkage():
    """Verify that a single 100% scored attempt receives 50% credibility shrinkage in both training and inference."""
    master = {
        "learners": [{"learner_id": "LRN-001", "current_course_id": "CRS-001", "starting_level": "beginner", "status": "active"}],
        "courses": [{"course_id": "CRS-001", "status": "published"}],
        "skills": [{"skill_id": "SKL-001", "status": "active"}],
        "content_items": [{"content_id": "CNT-001", "course_id": "CRS-001", "difficulty": "1.50", "status": "published"}],
        "content_skills": [{"content_id": "CNT-001", "skill_id": "SKL-001", "weight": "1.0"}],
        "skill_prerequisites": [],
    }
    # Exactly one scored attempt (score 100) on 2025-01-01
    activity = {
        "attempts": [
            {"attempt_id": "ATT-001", "learner_id": "LRN-001", "content_id": "CNT-001", "status": "scored", "started_at": "2025-01-01 10:00:00.000000", "max_score": 100.0},
            {"attempt_id": "ATT-002", "learner_id": "LRN-001", "content_id": "CNT-001", "status": "scored", "started_at": "2025-01-01 10:00:01.000000", "max_score": 100.0},
        ],
        "assessment_results": [
            {"result_id": "RES-001", "attempt_id": "ATT-001", "outcome": "pass", "passed": 1, "score": 100.0, "scored_at": "2025-01-01 10:00:00.500000"},
            {"result_id": "RES-002", "attempt_id": "ATT-002", "outcome": "pass", "passed": 1, "score": 100.0, "scored_at": "2025-01-01 10:00:01.500000"},
        ],
    }
    dataset = build_ml_dataset(master, activity)
    ex2 = next(ex for ex in dataset if ex.attempt_id == "ATT-002")
    # At ATT-002 start time, 1 prior attempt (score 100). Credibility = 1 / (1 + 1) = 0.5000.
    # Recency decay across 0.5s is ~1.0. Mastery must equal 0.5000 (not 1.0000 simple average).
    assert abs(ex2.primary_skill_mastery - 0.5000) < 1e-4


def test_f10_unscored_attempts_affect_only_behavioral_counts():
    """Unscored attempts increment behavioral prior attempt counts, but do not affect mastery or passes."""
    master = {
        "learners": [{"learner_id": "LRN-001", "current_course_id": "CRS-001", "starting_level": "beginner", "status": "active"}],
        "courses": [{"course_id": "CRS-001", "status": "published"}],
        "skills": [{"skill_id": "SKL-001", "status": "active"}],
        "content_items": [{"content_id": "CNT-001", "course_id": "CRS-001", "difficulty": "1.50", "status": "published"}],
        "content_skills": [{"content_id": "CNT-001", "skill_id": "SKL-001", "weight": "1.0"}],
        "skill_prerequisites": [],
    }
    activity = {
        "attempts": [
            {"attempt_id": "ATT-001", "learner_id": "LRN-001", "content_id": "CNT-001", "status": "scored", "started_at": "2025-01-01 10:00:00.000000", "max_score": 100.0},
            {"attempt_id": "ATT-INPROG", "learner_id": "LRN-001", "content_id": "CNT-001", "status": "in_progress", "started_at": "2025-01-02 10:00:00.000000", "max_score": 100.0},
            {"attempt_id": "ATT-TARGET", "learner_id": "LRN-001", "content_id": "CNT-001", "status": "scored", "started_at": "2025-01-03 10:00:00.000000", "max_score": 100.0},
        ],
        "assessment_results": [
            {"result_id": "RES-001", "attempt_id": "ATT-001", "outcome": "pass", "passed": 1, "score": 80.0, "scored_at": "2025-01-01 10:30:00.000000"},
            {"result_id": "RES-TARGET", "attempt_id": "ATT-TARGET", "outcome": "pass", "passed": 1, "score": 80.0, "scored_at": "2025-01-03 10:30:00.000000"},
        ],
    }
    dataset = build_ml_dataset(master, activity)
    target = next(ex for ex in dataset if ex.attempt_id == "ATT-TARGET")
    # Behavioral counts count both ATT-001 and ATT-INPROG:
    assert target.learner_prior_attempts_on_content == 2
    assert target.learner_overall_prior_attempts == 2
    # Passes and scored counts reflect only ATT-001:
    assert target.learner_prior_passes_on_content == 1
    assert target.learner_overall_prior_pass_rate == 1.0  # 1 pass / 1 scored attempt


