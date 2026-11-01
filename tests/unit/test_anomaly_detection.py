"""Unit tests for Phase 4 anomaly detection (≥14 tests)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from learnpath_ai.config_loader import load_app_config
from learnpath_ai.generate.master import generate_master_tables
from learnpath_ai.generate.activity import generate_activity_tables
from learnpath_ai.generate.writer import write_master_dataset
from learnpath_ai.ids import format_id, is_valid_id
from learnpath_ai.quality.anomalies import (
    ISSUE_TYPES,
    AnomalyFixture,
    AnomalyFixtures,
    generate_anomaly_fixtures,
)
from learnpath_ai.quality.detectors import (
    DetectionResult,
    detect_all_anomalies,
    detect_content_missing_skill,
    detect_mastery_future_attempt,
    detect_path_loop,
    detect_prereq_violation,
    detect_timestamp_order,
)
from learnpath_ai.quality.quarantine import write_quarantine
from learnpath_ai.quality.report import write_quality_report


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

SEED = 20261022


def _small_master(tmp_path: Path):
    """Generate a minimal dev master dataset."""
    cfg = load_app_config()
    tables = generate_master_tables(profile="dev", seed=SEED, config=cfg)
    return tables


def _small_activity(master_tables, tmp_path: Path):
    cfg = load_app_config()
    activity = generate_activity_tables(
        profile="dev",
        seed=SEED,
        config=cfg,
        master_tables=master_tables,
    )
    return activity


def _fixtures(master_tables, activity_tables):
    cfg = load_app_config()
    return generate_anomaly_fixtures(
        master_tables,
        activity_tables,
        profile="dev",
        seed=SEED,
        config=cfg,
    )


# ---------------------------------------------------------------------------
# Test 1: All five anomaly types are present in generated fixtures
# ---------------------------------------------------------------------------

def test_all_five_types_generated(tmp_path):
    master = _small_master(tmp_path)
    activity = _small_activity(master, tmp_path)
    fixtures = _fixtures(master, activity)

    by_type = fixtures.by_type()
    for issue_type in ISSUE_TYPES:
        assert issue_type in by_type, f"Missing fixture type: {issue_type}"
        assert len(by_type[issue_type]) > 0, f"No fixtures for type: {issue_type}"


# ---------------------------------------------------------------------------
# Test 2: All five anomaly types are detected
# ---------------------------------------------------------------------------

def test_all_five_types_detected(tmp_path):
    master = _small_master(tmp_path)
    activity = _small_activity(master, tmp_path)
    fixtures = _fixtures(master, activity)

    results = detect_all_anomalies(fixtures, master)
    detected_types = {r.anomaly_type for r in results}

    for issue_type in ISSUE_TYPES:
        assert issue_type in detected_types, f"Type not detected: {issue_type}"


# ---------------------------------------------------------------------------
# Test 3: Clean baseline data produces zero detections (detectors on empty fixtures)
# ---------------------------------------------------------------------------

def test_clean_data_produces_no_detections(tmp_path):
    """Detectors on empty fixture lists must return zero results."""
    empty = AnomalyFixtures(
        rec_prereq_violations=[],
        attempt_timestamp_orders=[],
        content_missing_skills=[],
        mastery_future_attempts=[],
        path_loop_no_progress=[],
    )
    results = detect_all_anomalies(empty)
    assert results == [], f"Expected no detections on empty fixtures, got {results}"


# ---------------------------------------------------------------------------
# Test 4: Broken foreign key (Anomaly C) detected
# ---------------------------------------------------------------------------

def test_content_missing_skill_detected(tmp_path):
    master = _small_master(tmp_path)
    activity = _small_activity(master, tmp_path)
    fixtures = _fixtures(master, activity)

    known_ids = {s["skill_id"] for s in master.skills}
    results = detect_content_missing_skill(fixtures.content_missing_skills, known_ids)
    assert len(results) > 0, "No CONTENT_MISSING_SKILL anomalies detected"
    for r in results:
        assert r.anomaly_type == "CONTENT_MISSING_SKILL"
        assert r.reason_code == "BROKEN_FOREIGN_KEY"


# ---------------------------------------------------------------------------
# Test 5: Invalid timestamp order (Anomaly B) detected
# ---------------------------------------------------------------------------

def test_timestamp_order_violation_detected(tmp_path):
    master = _small_master(tmp_path)
    activity = _small_activity(master, tmp_path)
    fixtures = _fixtures(master, activity)

    results = detect_timestamp_order(fixtures.attempt_timestamp_orders)
    assert len(results) > 0, "No ATTEMPT_TIMESTAMP_ORDER anomalies detected"
    for r in results:
        assert r.anomaly_type == "ATTEMPT_TIMESTAMP_ORDER"
        assert "submitted_at" in r.description.lower() or "timestamp" in r.description.lower()


# ---------------------------------------------------------------------------
# Test 6: Prerequisite violation (Anomaly A) detected
# ---------------------------------------------------------------------------

def test_prereq_violation_detected(tmp_path):
    master = _small_master(tmp_path)
    activity = _small_activity(master, tmp_path)
    fixtures = _fixtures(master, activity)

    results = detect_prereq_violation(fixtures.rec_prereq_violations)
    assert len(results) > 0, "No REC_PREREQ_VIOLATION anomalies detected"
    for r in results:
        assert r.anomaly_type == "REC_PREREQ_VIOLATION"
        assert r.reason_code == "PREREQ_NOT_SATISFIED"


# ---------------------------------------------------------------------------
# Test 7: Future-data leakage (Anomaly D) detected
# ---------------------------------------------------------------------------

def test_mastery_future_attempt_detected(tmp_path):
    master = _small_master(tmp_path)
    activity = _small_activity(master, tmp_path)
    fixtures = _fixtures(master, activity)

    results = detect_mastery_future_attempt(fixtures.mastery_future_attempts)
    assert len(results) > 0, "No MASTERY_FUTURE_ATTEMPT anomalies detected"
    for r in results:
        assert r.anomaly_type == "MASTERY_FUTURE_ATTEMPT"
        assert r.reason_code == "TEMPORAL_LEAKAGE"


# ---------------------------------------------------------------------------
# Test 8: Learning path loop (Anomaly E) detected
# ---------------------------------------------------------------------------

def test_path_loop_detected(tmp_path):
    master = _small_master(tmp_path)
    activity = _small_activity(master, tmp_path)
    fixtures = _fixtures(master, activity)

    results = detect_path_loop(fixtures.path_loop_no_progress)
    assert len(results) > 0, "No PATH_LOOP_NO_PROGRESS anomalies detected"
    for r in results:
        assert r.anomaly_type == "PATH_LOOP_NO_PROGRESS"
        assert r.reason_code == "LOOP_WITHOUT_PROGRESS"


# ---------------------------------------------------------------------------
# Test 9: Detection is reproducible (same seed → same issue_ids)
# ---------------------------------------------------------------------------

def test_detection_is_reproducible(tmp_path):
    master = _small_master(tmp_path)
    activity = _small_activity(master, tmp_path)

    cfg = load_app_config()
    fixtures1 = generate_anomaly_fixtures(master, activity, profile="dev", seed=SEED, config=cfg)
    fixtures2 = generate_anomaly_fixtures(master, activity, profile="dev", seed=SEED, config=cfg)

    ids1 = sorted(f.issue_id for f in fixtures1.all())
    ids2 = sorted(f.issue_id for f in fixtures2.all())
    assert ids1 == ids2, "Fixtures are not reproducible with the same seed"

    results1 = detect_all_anomalies(fixtures1, master)
    results2 = detect_all_anomalies(fixtures2, master)
    rids1 = sorted(r.issue_id for r in results1)
    rids2 = sorted(r.issue_id for r in results2)
    assert rids1 == rids2, "Detection results are not reproducible"


# ---------------------------------------------------------------------------
# Test 10: Quarantine records preserve source references
# ---------------------------------------------------------------------------

def test_quarantine_preserves_source_references(tmp_path):
    master = _small_master(tmp_path)
    activity = _small_activity(master, tmp_path)
    fixtures = _fixtures(master, activity)

    out_dir = tmp_path / "quarantine"
    write_quarantine(fixtures, out_dir, overwrite=True)

    combined_path = out_dir / "data_quality_issues.csv"
    assert combined_path.is_file(), "Combined quarantine CSV not found"

    with combined_path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) > 0, "Quarantine CSV is empty"
    for row in rows:
        assert row.get("source_table"), f"source_table missing in row {row.get('issue_id')}"
        assert row.get("source_pk"), f"source_pk missing in row {row.get('issue_id')}"
        assert row.get("issue_id", "").startswith("DQI-"), f"Bad issue_id: {row.get('issue_id')}"


# ---------------------------------------------------------------------------
# Test 11: Reports contain correct counts by type
# ---------------------------------------------------------------------------

def test_report_counts_match_fixtures(tmp_path):
    master = _small_master(tmp_path)
    activity = _small_activity(master, tmp_path)
    fixtures = _fixtures(master, activity)
    results = detect_all_anomalies(fixtures, master)

    reports_dir = tmp_path / "reports"
    summary = write_quality_report(
        results, reports_dir, profile="dev",
        total_inspected=len(fixtures.all()),
    )

    assert summary["total_anomalous"] == len(results)
    for atype, count in summary["count_by_type"].items():
        actual = sum(1 for r in results if r.anomaly_type == atype)
        assert count == actual, f"Count mismatch for {atype}: report={count}, actual={actual}"

    json_path = reports_dir / "quality_report_dev.json"
    assert json_path.is_file(), "JSON report not written"
    with json_path.open() as f:
        data = json.load(f)
    assert data["total_anomalous"] == len(results)


# ---------------------------------------------------------------------------
# Test 12: Invalid anomaly fixture is rejected on validate()
# ---------------------------------------------------------------------------

def test_invalid_fixture_rejected():
    bad_fixture = AnomalyFixture(
        issue_id="INVALID-123",           # Bad format
        issue_type="REC_PREREQ_VIOLATION",
        severity="high",
        source_table="recommendation",
        source_pk="REC-000001",
        status="quarantined",
        expected_decision="reject",
        details={},
        detected_at="2025-01-01 00:00:00.000000",
        created_at="2025-01-01 00:00:00.000000",
    )
    with pytest.raises(ValueError, match="issue_id must start with"):
        bad_fixture.validate()

    bad_type = AnomalyFixture(
        issue_id="DQI-000001",
        issue_type="NOT_A_REAL_TYPE",     # Bad type
        severity="high",
        source_table="attempt",
        source_pk="ATT-LRN-000001-000001",
        status="quarantined",
        expected_decision="reject",
        details={},
        detected_at="2025-01-01 00:00:00.000000",
        created_at="2025-01-01 00:00:00.000000",
    )
    with pytest.raises(ValueError, match="Unknown issue_type"):
        bad_type.validate()


# ---------------------------------------------------------------------------
# Test 13: Existing CLI commands (generate master) continue working
# ---------------------------------------------------------------------------

def test_existing_cli_generate_master_still_works(tmp_path):
    from learnpath_ai.cli import main

    out_dir = tmp_path / "master_out"
    rc = main(["generate", "master", "--profile", "dev", "--output-dir", str(out_dir)])
    assert rc == 0, f"generate master returned non-zero: {rc}"
    assert (out_dir / "courses.csv").is_file()
    assert (out_dir / "learners.csv").is_file()


# ---------------------------------------------------------------------------
# Test 14: Phase 2 and Phase 3 baseline datasets remain unchanged
# ---------------------------------------------------------------------------

def test_baseline_datasets_unchanged():
    """Check that data/sample master_manifest.json and activity_manifest.json
    are still present (we do not corrupt them during Phase 4)."""
    sample_dir = Path("data/sample")
    if not sample_dir.is_dir():
        pytest.skip("data/sample not present; skipping baseline integrity check")

    master_manifest = sample_dir / "master_manifest.json"
    activity_manifest = sample_dir / "activity_manifest.json"

    assert master_manifest.is_file(), "master_manifest.json is missing from data/sample"
    assert activity_manifest.is_file(), "activity_manifest.json is missing from data/sample"

    with master_manifest.open() as f:
        mm = json.load(f)
    assert "checksums" in mm or "checksums_sha256" in mm, "Manifest missing checksums"
    assert mm.get("dataset_profile") == "dev", "Unexpected profile in master manifest"

    with activity_manifest.open() as f:
        am = json.load(f)
    assert "checksums" in am or "checksums_sha256" in am, "Activity manifest missing checksums"


# ---------------------------------------------------------------------------
# Test 15 (bonus): DQI ID format is valid
# ---------------------------------------------------------------------------

def test_dqi_id_format():
    assert is_valid_id("DQI-000001", "DQI")
    assert is_valid_id("DQI-999999", "DQI")
    assert not is_valid_id("DQI-00001", "DQI")  # too short
    assert not is_valid_id("dqi-000001", "DQI")  # lower case
    assert format_id("DQI", 1) == "DQI-000001"
    assert format_id("DQI", 999999) == "DQI-999999"


# ---------------------------------------------------------------------------
# Test 16 (bonus): Fixture count matches n_per_type config
# ---------------------------------------------------------------------------

def test_fixture_count_matches_config(tmp_path):
    master = _small_master(tmp_path)
    activity = _small_activity(master, tmp_path)
    cfg = load_app_config()
    expected_n = int(cfg["dataset_profiles"]["dev"]["anomaly_injections_per_type"])

    fixtures = generate_anomaly_fixtures(master, activity, profile="dev", seed=SEED, config=cfg)
    for atype, lst in fixtures.by_type().items():
        assert len(lst) == expected_n, (
            f"Expected {expected_n} fixtures for {atype}, got {len(lst)}"
        )

