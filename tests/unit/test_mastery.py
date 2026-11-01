"""Unit tests for Phase 5 learner skill mastery estimation."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from learnpath_ai.config_loader import load_app_config
from learnpath_ai.generate.clock import format_timestamp
from learnpath_ai.ids import format_id, is_valid_id
from learnpath_ai.mastery.calculator import (
    CALCULATOR_VERSION,
    MasteryCalculationError,
    MasterySnapshotRecord,
    calculate_mastery_snapshots,
    calculate_single_mastery,
)
from learnpath_ai.mastery.validate import (
    MasteryValidationError,
    validate_mastery_dataset_dir,
    validate_mastery_snapshots,
)
from learnpath_ai.mastery.writer import (
    write_mastery_dataset,
    write_mastery_manifest,
)

UTC = timezone.utc
SEED = 20261022


# ---------------------------------------------------------------------------
# Fixtures & Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_master():
    return {
        "learners": [
            {"learner_id": "LRN-000001", "starting_level": "beginner"},
            {"learner_id": "LRN-000002", "starting_level": "intermediate"},
            {"learner_id": "LRN-000003", "starting_level": "advanced"},
        ],
        "skills": [
            {"skill_id": "SKL-000001", "domain": "Programming Fundamentals"},
            {"skill_id": "SKL-000002", "domain": "Data Structures"},
            {"skill_id": "SKL-000003", "domain": "Databases"},
        ],
        "content_skills": [
            {"content_id": "CNT-000001", "skill_id": "SKL-000001", "skill_role": "primary", "weight": 1.0},
            {"content_id": "CNT-000002", "skill_id": "SKL-000001", "skill_role": "supporting", "weight": 0.5},
            {"content_id": "CNT-000002", "skill_id": "SKL-000002", "skill_role": "primary", "weight": 1.0},
            {"content_id": "CNT-000003", "skill_id": "SKL-000003", "skill_role": "primary", "weight": 1.0},
        ],
    }


@pytest.fixture
def sample_activity():
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
            },
            {
                "attempt_id": "ATT-LRN-000001-000002",
                "learner_id": "LRN-000001",
                "content_id": "CNT-000002",
                "started_at": "2025-01-15 10:00:00.000000",
                "submitted_at": "2025-01-15 10:45:00.000000",
                "status": "scored",
                "max_score": 100.0,
            },
            {
                "attempt_id": "ATT-LRN-000001-000003",
                "learner_id": "LRN-000001",
                "content_id": "CNT-000001",
                "started_at": "2025-05-01 10:00:00.000000",  # Future attempt relative to Feb 2025
                "submitted_at": "2025-05-01 10:30:00.000000",
                "status": "scored",
                "max_score": 100.0,
            },
        ],
        "assessment_results": [
            {
                "result_id": "RES-000001",
                "attempt_id": "ATT-LRN-000001-000001",
                "score": 80.0,
                "passed": True,
                "outcome": "pass",
                "scored_at": "2025-01-10 10:35:00.000000",
            },
            {
                "result_id": "RES-000002",
                "attempt_id": "ATT-LRN-000001-000002",
                "score": 90.0,
                "passed": True,
                "outcome": "pass",
                "scored_at": "2025-01-15 10:50:00.000000",
            },
            {
                "result_id": "RES-000003",
                "attempt_id": "ATT-LRN-000001-000003",
                "score": 100.0,
                "passed": True,
                "outcome": "pass",
                "scored_at": "2025-05-01 10:35:00.000000",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Test 1: Single mastery calculation formula with known fixture
# ---------------------------------------------------------------------------

def test_single_mastery_calculation_formula():
    as_of = datetime(2025, 2, 1, 0, 0, 0, tzinfo=UTC)
    ev_time = datetime(2025, 2, 1, 0, 0, 0, tzinfo=UTC)

    # 1 evidence with score=1.0, weight=1.0, zero decay (ev_time == as_of)
    # credibility for n=1, k=1 is 1 / (1 + 1) = 0.5
    score, count, latest = calculate_single_mastery(
        [{"event_time": ev_time, "normalized_score": 1.0, "weight": 1.0}],
        as_of,
        credibility_k=1.0,
    )
    assert count == 1
    assert score == 0.5000
    assert latest == ev_time


# ---------------------------------------------------------------------------
# Test 2: Range constraints [scale_min, scale_max]
# ---------------------------------------------------------------------------

def test_mastery_score_range_constraints():
    as_of = datetime(2025, 1, 1, tzinfo=UTC)
    # High score
    score, _, _ = calculate_single_mastery(
        [{"event_time": as_of, "normalized_score": 10.0, "weight": 1.0}],
        as_of,
    )
    assert 0.0 <= score <= 1.0

    # Negative score
    score_neg, _, _ = calculate_single_mastery(
        [{"event_time": as_of, "normalized_score": -1.0, "weight": 1.0}],
        as_of,
    )
    assert 0.0 <= score_neg <= 1.0


# ---------------------------------------------------------------------------
# Test 3: Multiple attempts accumulate credibility
# ---------------------------------------------------------------------------

def test_multiple_attempts_accumulate_credibility():
    as_of = datetime(2025, 1, 1, tzinfo=UTC)
    # 1 attempt with perfect score: cred = 1/(1+1) = 0.5 -> 0.5
    s1, _, _ = calculate_single_mastery(
        [{"event_time": as_of, "normalized_score": 1.0, "weight": 1.0}],
        as_of,
        credibility_k=1.0,
    )
    # 4 attempts with perfect score: cred = 4/(4+1) = 0.8 -> 0.8
    s4, count, _ = calculate_single_mastery(
        [{"event_time": as_of, "normalized_score": 1.0, "weight": 1.0} for _ in range(4)],
        as_of,
        credibility_k=1.0,
    )
    assert count == 4
    assert s4 > s1
    assert s4 == 0.8000


# ---------------------------------------------------------------------------
# Test 4: Multiple content items mapped to the same skill
# ---------------------------------------------------------------------------

def test_multiple_content_items_to_skill(sample_master, sample_activity):
    # LRN-000001 attempted CNT-000001 and CNT-000002, both mapping to SKL-000001
    as_of = "2025-02-01 00:00:00.000000"
    snapshots = calculate_mastery_snapshots(
        sample_master,
        sample_activity,
        as_of_timestamp=as_of,
    )
    skl1_snaps = [s for s in snapshots if s.learner_id == "LRN-000001" and s.skill_id == "SKL-000001"]
    assert len(skl1_snaps) == 1
    snap = skl1_snaps[0]
    # CNT-000001 (score 80, weight 1.0) and CNT-000002 (score 90, weight 0.5)
    assert snap.evidence_attempt_count == 2
    assert snap.mastery_score > 0.0


# ---------------------------------------------------------------------------
# Test 5: Cold-start learners have 0.0000 mastery
# ---------------------------------------------------------------------------

def test_cold_start_learners_have_zero_mastery(sample_master, sample_activity):
    # LRN-000002 and LRN-000003 have no attempts in sample_activity
    as_of = "2025-02-01 00:00:00.000000"
    snapshots = calculate_mastery_snapshots(
        sample_master,
        sample_activity,
        as_of_timestamp=as_of,
        include_cold_start=True,
    )
    cold_snaps = [s for s in snapshots if s.learner_id in ("LRN-000002", "LRN-000003")]
    assert len(cold_snaps) > 0
    for cs in cold_snaps:
        assert cs.mastery_score == 0.0000
        assert cs.evidence_attempt_count == 0


# ---------------------------------------------------------------------------
# Test 6: Unscored / in-progress attempts are excluded from mastery evidence (F-01)
# ---------------------------------------------------------------------------

def test_unscored_attempts_handled(sample_master):
    """Under F-01 policy, unscored/in-progress attempts must not contribute to evidence count
    and must not dilute mastery scores.
    """
    activity_with_in_progress = {
        "attempts": [
            {
                "attempt_id": "ATT-LRN-000001-000001",
                "learner_id": "LRN-000001",
                "content_id": "CNT-000001",
                "started_at": "2025-01-10 10:00:00.000000",
                "submitted_at": "",
                "status": "in_progress",
                "max_score": 100.0,
            }
        ],
        "assessment_results": [],
    }
    snapshots = calculate_mastery_snapshots(
        sample_master,
        activity_with_in_progress,
        as_of_timestamp="2025-01-15 00:00:00.000000",
        include_cold_start=True,
    )
    snap = [s for s in snapshots if s.learner_id == "LRN-000001" and s.skill_id == "SKL-000001"][0]
    # Under F-01: in_progress attempt is excluded from evidence
    assert snap.evidence_attempt_count == 0
    assert snap.mastery_score == 0.0000


# ---------------------------------------------------------------------------
# Test 7: Prerequisite-independent mastery calculation
# ---------------------------------------------------------------------------

def test_prerequisite_independent_mastery(sample_master, sample_activity):
    """Mastery is purely an assessment evidence aggregator; it does NOT gate itself on prerequisites."""
    # Even if SKL-000002 has prerequisites, mastery for SKL-000002 is calculated from direct evidence
    as_of = "2025-02-01 00:00:00.000000"
    snapshots = calculate_mastery_snapshots(
        sample_master,
        sample_activity,
        as_of_timestamp=as_of,
    )
    skl2_snaps = [s for s in snapshots if s.learner_id == "LRN-000001" and s.skill_id == "SKL-000002"]
    assert len(skl2_snaps) == 1
    assert skl2_snaps[0].evidence_attempt_count == 1
    assert skl2_snaps[0].mastery_score > 0.0


# ---------------------------------------------------------------------------
# Test 8: Explicit as_of_timestamp cutoff
# ---------------------------------------------------------------------------

def test_explicit_as_of_timestamp_cutoff(sample_master, sample_activity):
    # ATT 3 is on 2025-05-01.
    # Cutoff at 2025-02-01 should ONLY include ATT 1 and ATT 2 (2 attempts for SKL-000001)
    snap_feb = calculate_mastery_snapshots(
        sample_master,
        sample_activity,
        as_of_timestamp="2025-02-01 00:00:00.000000",
    )
    snap1 = [s for s in snap_feb if s.learner_id == "LRN-000001" and s.skill_id == "SKL-000001"][0]
    assert snap1.evidence_attempt_count == 2

    # Cutoff at 2025-06-01 should include all 3 attempts
    snap_jun = calculate_mastery_snapshots(
        sample_master,
        sample_activity,
        as_of_timestamp="2025-06-01 00:00:00.000000",
    )
    snap2 = [s for s in snap_jun if s.learner_id == "LRN-000001" and s.skill_id == "SKL-000001"][0]
    assert snap2.evidence_attempt_count == 3


# ---------------------------------------------------------------------------
# Test 9: Future attempts strictly excluded (Temporal Leakage Test)
# ---------------------------------------------------------------------------

def test_future_attempts_strictly_excluded(sample_master, sample_activity):
    as_of = "2025-01-12 00:00:00.000000"  # After ATT 1 (Jan 10), before ATT 2 (Jan 15) and ATT 3 (May 1)
    snapshots = calculate_mastery_snapshots(
        sample_master,
        sample_activity,
        as_of_timestamp=as_of,
    )
    snap = [s for s in snapshots if s.learner_id == "LRN-000001" and s.skill_id == "SKL-000001"][0]
    assert snap.evidence_attempt_count == 1
    assert snap.evidence_cutoff_ts <= as_of


# ---------------------------------------------------------------------------
# Test 10: Earlier snapshots unaffected by subsequent future events
# ---------------------------------------------------------------------------

def test_earlier_snapshots_unaffected_by_future_events(sample_master, sample_activity):
    as_of_past = "2025-02-01 00:00:00.000000"

    # Snapshots before future attempts added
    base_activity = {
        "attempts": sample_activity["attempts"][:2],
        "assessment_results": sample_activity["assessment_results"][:2],
    }
    snaps_initial = calculate_mastery_snapshots(sample_master, base_activity, as_of_timestamp=as_of_past)

    # Snapshots calculated with future attempts present in activity database
    snaps_with_future = calculate_mastery_snapshots(sample_master, sample_activity, as_of_timestamp=as_of_past)

    assert len(snaps_initial) == len(snaps_with_future)
    for s1, s2 in zip(snaps_initial, snaps_with_future):
        assert s1.mastery_score == s2.mastery_score
        assert s1.evidence_attempt_count == s2.evidence_attempt_count
        assert s1.evidence_cutoff_ts == s2.evidence_cutoff_ts


# ---------------------------------------------------------------------------
# Test 11: Same-seed / identical inputs produce identical outputs (reproducibility)
# ---------------------------------------------------------------------------

def test_same_seed_reproducibility(sample_master, sample_activity):
    as_of = "2025-02-01 00:00:00.000000"
    res1 = calculate_mastery_snapshots(sample_master, sample_activity, as_of_timestamp=as_of)
    res2 = calculate_mastery_snapshots(sample_master, sample_activity, as_of_timestamp=as_of)

    assert [s.to_dict() for s in res1] == [s.to_dict() for s in res2]


# ---------------------------------------------------------------------------
# Test 12: Invalid input rejection
# ---------------------------------------------------------------------------

def test_invalid_input_rejection():
    # Attempt with event_time after as_of in calculate_single_mastery
    as_of = datetime(2025, 1, 1, tzinfo=UTC)
    future_ev = datetime(2025, 1, 2, tzinfo=UTC)

    with pytest.raises(MasteryCalculationError, match="Future evidence leakage"):
        calculate_single_mastery(
            [{"event_time": future_ev, "normalized_score": 0.8, "weight": 1.0}],
            as_of,
        )


# ---------------------------------------------------------------------------
# Test 13: Mastery snapshot validation against DB schema
# ---------------------------------------------------------------------------

def test_validate_mastery_snapshots_schema(sample_master, sample_activity):
    snaps = calculate_mastery_snapshots(
        sample_master,
        sample_activity,
        as_of_timestamp="2025-02-01 00:00:00.000000",
    )
    rows = [s.to_dict() for s in snaps]
    # Valid rows must pass
    validate_mastery_snapshots(rows, master_tables=sample_master)

    # Invalid ID must fail
    bad_rows = [dict(r) for r in rows]
    bad_rows[0]["snapshot_id"] = "INVALID-ID"
    with pytest.raises(MasteryValidationError, match="Invalid snapshot_id"):
        validate_mastery_snapshots(bad_rows, master_tables=sample_master)

    # Out of range score must fail
    bad_score_rows = [dict(r) for r in rows]
    bad_score_rows[0]["mastery_score"] = "1.5000"
    with pytest.raises(MasteryValidationError, match="out of range"):
        validate_mastery_snapshots(bad_score_rows, master_tables=sample_master)

    # Future leakage in rows must fail
    leak_rows = [dict(r) for r in rows]
    leak_rows[0]["evidence_cutoff_ts"] = "2026-01-01 00:00:00.000000"
    leak_rows[0]["as_of_timestamp"] = "2025-01-01 00:00:00.000000"
    with pytest.raises(MasteryValidationError, match="Future leakage"):
        validate_mastery_snapshots(leak_rows, master_tables=sample_master)


# ---------------------------------------------------------------------------
# Test 14: Manifest and checksum validation
# ---------------------------------------------------------------------------

def test_write_and_validate_mastery_dataset(tmp_path, sample_master, sample_activity):
    snaps = calculate_mastery_snapshots(
        sample_master,
        sample_activity,
        as_of_timestamp="2025-02-01 00:00:00.000000",
    )
    manifest = write_mastery_dataset(
        snaps,
        tmp_path,
        profile="dev",
        seed=SEED,
        overwrite=True,
    )
    assert (tmp_path / "mastery_snapshots.csv").is_file()
    assert (tmp_path / "mastery_manifest.json").is_file()
    assert "mastery_snapshots.csv" in manifest["checksums"]

    # Validate directory
    result = validate_mastery_dataset_dir(tmp_path, expected_profile="dev", master_tables=sample_master)
    assert result["row_counts"]["mastery_snapshots.csv"] == len(snaps)


# ---------------------------------------------------------------------------
# Test 16: F-01 & F-02 Comprehensive Unit Tests
# ---------------------------------------------------------------------------

def test_f01_mastery_evidence_policy_by_status():
    """Verify that only scored attempts with valid pre-cutoff assessment results contribute."""
    master = {
        "learners": [{"learner_id": "LRN-101", "current_course_id": "CRS-101", "status": "active"}],
        "courses": [{"course_id": "CRS-101", "status": "published"}],
        "skills": [{"skill_id": "SKL-101", "status": "active"}],
        "content_items": [{"content_id": "CNT-101", "course_id": "CRS-101", "status": "published"}],
        "content_skills": [{"content_id": "CNT-101", "skill_id": "SKL-101", "weight": 1.0}],
    }
    cutoff = "2025-06-01 00:00:00.000000"

    # 1. Scored attempt with pre-cutoff result -> contributes
    activity_scored = {
        "attempts": [
            {
                "attempt_id": "ATT-001", "learner_id": "LRN-101", "content_id": "CNT-101",
                "status": "scored", "started_at": "2025-05-10 10:00:00.000000", "max_score": 100.0,
            }
        ],
        "assessment_results": [
            {"result_id": "RES-001", "attempt_id": "ATT-001", "score": 90.0, "scored_at": "2025-05-10 10:30:00.000000"}
        ],
    }
    snaps = calculate_mastery_snapshots(master, activity_scored, as_of_timestamp=cutoff)
    snap = next(s for s in snaps if s.learner_id == "LRN-101" and s.skill_id == "SKL-101")
    assert snap.evidence_attempt_count == 1
    # 90% normalized * (1/(1+1)) = 45% = 0.4500
    assert snap.mastery_score == 0.4500

    # 2. Submitted, in_progress, abandoned, invalid attempts -> excluded
    activity_unscored = {
        "attempts": [
            {"attempt_id": "ATT-002", "learner_id": "LRN-101", "content_id": "CNT-101", "status": "submitted", "started_at": "2025-05-11 10:00:00.000000"},
            {"attempt_id": "ATT-003", "learner_id": "LRN-101", "content_id": "CNT-101", "status": "in_progress", "started_at": "2025-05-12 10:00:00.000000"},
            {"attempt_id": "ATT-004", "learner_id": "LRN-101", "content_id": "CNT-101", "status": "abandoned", "started_at": "2025-05-13 10:00:00.000000"},
            {"attempt_id": "ATT-005", "learner_id": "LRN-101", "content_id": "CNT-101", "status": "invalid", "started_at": "2025-05-14 10:00:00.000000"},
        ],
        "assessment_results": [],
    }
    snaps2 = calculate_mastery_snapshots(master, activity_unscored, as_of_timestamp=cutoff, include_cold_start=True)
    snap2 = next(s for s in snaps2 if s.learner_id == "LRN-101" and s.skill_id == "SKL-101")
    assert snap2.evidence_attempt_count == 0
    assert snap2.mastery_score == 0.0000

    # 3. Future-scored attempt -> excluded
    activity_future = {
        "attempts": [
            {
                "attempt_id": "ATT-006", "learner_id": "LRN-101", "content_id": "CNT-101",
                "status": "scored", "started_at": "2025-05-20 10:00:00.000000", "max_score": 100.0,
            }
        ],
        "assessment_results": [
            {"result_id": "RES-006", "attempt_id": "ATT-006", "score": 90.0, "scored_at": "2025-06-15 10:00:00.000000"}  # Scored after cutoff
        ],
    }
    snaps3 = calculate_mastery_snapshots(master, activity_future, as_of_timestamp=cutoff, include_cold_start=True)
    snap3 = next(s for s in snaps3 if s.learner_id == "LRN-101" and s.skill_id == "SKL-101")
    assert snap3.evidence_attempt_count == 0
    assert snap3.mastery_score == 0.0000


def test_f01_unscored_attempts_do_not_dilute_scored_evidence():
    """Unscored attempts must never dilute the mastery score of previously scored attempts."""
    master = {
        "learners": [{"learner_id": "LRN-101", "current_course_id": "CRS-101", "status": "active"}],
        "courses": [{"course_id": "CRS-101", "status": "published"}],
        "skills": [{"skill_id": "SKL-101", "status": "active"}],
        "content_items": [{"content_id": "CNT-101", "course_id": "CRS-101", "status": "published"}],
        "content_skills": [{"content_id": "CNT-101", "skill_id": "SKL-101", "weight": 1.0}],
    }
    cutoff = "2025-06-01 00:00:00.000000"

    # Attempt 1: scored 100
    # Attempt 2: submitted (ungraded)
    # Attempt 3: in_progress
    activity = {
        "attempts": [
            {
                "attempt_id": "ATT-001", "learner_id": "LRN-101", "content_id": "CNT-101",
                "status": "scored", "started_at": "2025-05-01 10:00:00.000000", "max_score": 100.0,
            },
            {
                "attempt_id": "ATT-002", "learner_id": "LRN-101", "content_id": "CNT-101",
                "status": "submitted", "started_at": "2025-05-02 10:00:00.000000", "max_score": 100.0,
            },
            {
                "attempt_id": "ATT-003", "learner_id": "LRN-101", "content_id": "CNT-101",
                "status": "in_progress", "started_at": "2025-05-03 10:00:00.000000", "max_score": 100.0,
            },
        ],
        "assessment_results": [
            {"result_id": "RES-001", "attempt_id": "ATT-001", "score": 100.0, "scored_at": "2025-05-01 10:30:00.000000"}
        ],
    }
    snaps = calculate_mastery_snapshots(master, activity, as_of_timestamp=cutoff)
    snap = next(s for s in snaps if s.learner_id == "LRN-101" and s.skill_id == "SKL-101")
    # Only 1 eligible scored attempt; score should not be diluted by 0s from ATT-002 and ATT-003
    assert snap.evidence_attempt_count == 1
    # 100% * (1/(1+1)) = 50% = 0.5000
    assert snap.mastery_score == 0.5000


def test_f02_cold_start_course_specific_skills():
    """Verify that cold learners receive zero-mastery snapshots for their enrolled course skills,
    without receiving unrelated global skills.
    """
    master = {
        "learners": [
            {"learner_id": "LRN-C1", "current_course_id": "CRS-001", "status": "active"},
            {"learner_id": "LRN-C2", "current_course_id": "CRS-002", "status": "active"},
            {"learner_id": "LRN-WITHDRAWN", "current_course_id": "", "status": "withdrawn"},
            {"learner_id": "LRN-NO-COURSE", "current_course_id": "", "status": "active"},
        ],
        "courses": [
            {"course_id": "CRS-001", "status": "published"},
            {"course_id": "CRS-002", "status": "published"},
        ],
        "skills": [
            {"skill_id": "SKL-A", "status": "active"},
            {"skill_id": "SKL-B", "status": "active"},
            {"skill_id": "SKL-X", "status": "active"},
            {"skill_id": "SKL-Y", "status": "active"},
        ],
        "content_items": [
            {"content_id": "CNT-001", "course_id": "CRS-001", "status": "published"},
            {"content_id": "CNT-002", "course_id": "CRS-002", "status": "published"},
        ],
        "content_skills": [
            {"content_id": "CNT-001", "skill_id": "SKL-A", "weight": 1.0},
            {"content_id": "CNT-001", "skill_id": "SKL-B", "weight": 0.5},
            {"content_id": "CNT-002", "skill_id": "SKL-X", "weight": 1.0},
            {"content_id": "CNT-002", "skill_id": "SKL-Y", "weight": 0.5},
        ],
    }
    activity = {"attempts": [], "assessment_results": []}

    snaps = calculate_mastery_snapshots(master, activity, as_of_timestamp="2025-01-01 00:00:00", include_cold_start=True)

    c1_skills = {s.skill_id for s in snaps if s.learner_id == "LRN-C1"}
    c2_skills = {s.skill_id for s in snaps if s.learner_id == "LRN-C2"}
    withdrawn_snaps = [s for s in snaps if s.learner_id == "LRN-WITHDRAWN"]
    no_course_snaps = [s for s in snaps if s.learner_id == "LRN-NO-COURSE"]

    # LRN-C1 has skills from CRS-001 (SKL-A, SKL-B) and NOT CRS-002 (SKL-X, SKL-Y)
    assert c1_skills == {"SKL-A", "SKL-B"}
    # LRN-C2 has skills from CRS-002 (SKL-X, SKL-Y) and NOT CRS-001 (SKL-A, SKL-B)
    assert c2_skills == {"SKL-X", "SKL-Y"}

    # All baseline snapshots have 0.0000 mastery and 0 attempts
    for s in snaps:
        assert s.mastery_score == 0.0000
        assert s.evidence_attempt_count == 0

    # Withdrawn and no-course learners safely receive no baseline snapshots
    assert len(withdrawn_snaps) == 0
    assert len(no_course_snaps) == 0


def test_f02_partially_active_learner_receives_unattempted_skill_baselines():
    """A partially active learner receives baseline snapshots for unattempted skills in their course,
    while skills with evidence reflect actual scores.
    """
    master = {
        "learners": [{"learner_id": "LRN-PARTIAL", "current_course_id": "CRS-001", "status": "active"}],
        "courses": [{"course_id": "CRS-001", "status": "published"}],
        "skills": [
            {"skill_id": "SKL-ATTEMPTED", "status": "active"},
            {"skill_id": "SKL-UNATTEMPTED", "status": "active"},
        ],
        "content_items": [
            {"content_id": "CNT-001", "course_id": "CRS-001", "status": "published"},
            {"content_id": "CNT-002", "course_id": "CRS-001", "status": "published"},
        ],
        "content_skills": [
            {"content_id": "CNT-001", "skill_id": "SKL-ATTEMPTED", "weight": 1.0},
            {"content_id": "CNT-002", "skill_id": "SKL-UNATTEMPTED", "weight": 1.0},
        ],
    }
    # Only CNT-001 was attempted and scored
    activity = {
        "attempts": [
            {
                "attempt_id": "ATT-001", "learner_id": "LRN-PARTIAL", "content_id": "CNT-001",
                "status": "scored", "started_at": "2025-01-10 10:00:00.000000", "max_score": 100.0,
            }
        ],
        "assessment_results": [
            {"result_id": "RES-001", "attempt_id": "ATT-001", "score": 80.0, "scored_at": "2025-01-10 10:30:00.000000"}
        ],
    }
    snaps = calculate_mastery_snapshots(master, activity, as_of_timestamp="2025-01-15 00:00:00", include_cold_start=True)

    att_snap = next(s for s in snaps if s.skill_id == "SKL-ATTEMPTED")
    unatt_snap = next(s for s in snaps if s.skill_id == "SKL-UNATTEMPTED")

    # Attempted skill reflects score > 0 and count == 1
    assert att_snap.evidence_attempt_count == 1
    assert att_snap.mastery_score > 0.0

    # Unattempted skill in the same course reflects baseline 0.0 and count == 0
    assert unatt_snap.evidence_attempt_count == 0
    assert unatt_snap.mastery_score == 0.0000


def test_f02_baseline_snapshots_cannot_pass_prerequisites():
    """Baseline 0.0000 snapshots strictly fail prerequisite gating (min_mastery >= 0.60)."""
    from learnpath_ai.eligibility import (
        ContentSkillLink,
        content_passes_required_prerequisites,
    )
    from learnpath_ai.prerequisites import PrerequisiteEdge

    content_skills = [ContentSkillLink(content_id="CNT-ADV", skill_id="SKL-ADV")]
    edges = [
        PrerequisiteEdge(
            skill_id="SKL-ADV",
            prerequisite_skill_id="SKL-PREREQ",
            relationship_type="required",
            min_mastery_required=0.60,
        )
    ]
    # Cold-start / baseline mastery map where SKL-PREREQ is 0.0000
    mastery_map = {"SKL-PREREQ": 0.0000}

    assert not content_passes_required_prerequisites("CNT-ADV", content_skills, edges, mastery_map)


