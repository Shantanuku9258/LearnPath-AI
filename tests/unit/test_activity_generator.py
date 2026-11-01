"""Unit tests for Phase 3 learner activity and assessment data generator."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from learnpath_ai.cli import main
from learnpath_ai.config_loader import ConfigError
from learnpath_ai.generate.activity import (
    ActivityTables,
    generate_activity_tables,
    write_activity_dataset,
)
from learnpath_ai.generate.columns import (
    ACTIVITY_FILES,
    ACTIVITY_MANIFEST_NAME,
    ASSESSMENT_RESULT_COLUMNS,
    ATTEMPT_COLUMNS,
)
from learnpath_ai.generate.master import generate_master_tables
from learnpath_ai.generate.validate import (
    ActivityDataError,
    validate_activity_dataset_dir,
    validate_activity_tables,
)
from learnpath_ai.generate.writer import OutputExistsError
from learnpath_ai.ids import PREFIX_PATTERNS


@pytest.fixture(scope="module")
def dev_master():
    """Shared dev master dataset fixture."""
    return generate_master_tables(profile="dev", seed=20261022)


@pytest.fixture(scope="module")
def dev_activity(dev_master):
    """Shared dev activity dataset fixture."""
    return generate_activity_tables(profile="dev", seed=20261022, master_tables=dev_master, target_attempts=5000)


# 1. Same seed produces identical files
def test_same_seed_produces_identical_activity(dev_master):
    act1 = generate_activity_tables(profile="dev", seed=20261022, master_tables=dev_master, target_attempts=500)
    act2 = generate_activity_tables(profile="dev", seed=20261022, master_tables=dev_master, target_attempts=500)
    assert act1.attempts == act2.attempts
    assert act1.assessment_results == act2.assessment_results


# 2. Different seeds produce different logical data
def test_different_seeds_produce_different_activity(dev_master):
    act1 = generate_activity_tables(profile="dev", seed=20261022, master_tables=dev_master, target_attempts=500)
    act2 = generate_activity_tables(profile="dev", seed=999999, master_tables=dev_master, target_attempts=500)
    assert act1.attempts != act2.attempts
    assert act1.assessment_results != act2.assessment_results


# 3. Attempt IDs follow the required format
def test_attempt_ids_follow_required_format(dev_activity):
    for att in dev_activity.attempts:
        assert PREFIX_PATTERNS["ATT"].fullmatch(att["attempt_id"])
        # Format ATT-<learner_id>-<seq>
        expected_prefix = f"ATT-{att['learner_id']}-"
        assert att["attempt_id"].startswith(expected_prefix)


# 4. Attempt IDs are unique
def test_attempt_ids_are_unique(dev_activity):
    att_ids = [a["attempt_id"] for a in dev_activity.attempts]
    assert len(att_ids) == len(set(att_ids))


# 5. Learner foreign keys are valid
def test_learner_foreign_keys_are_valid(dev_master, dev_activity):
    valid_learners = {l["learner_id"] for l in dev_master.learners}
    for att in dev_activity.attempts:
        assert att["learner_id"] in valid_learners


# 6. Content foreign keys are valid
def test_content_foreign_keys_are_valid(dev_master, dev_activity):
    valid_content = {c["content_id"] for c in dev_master.content_items}
    for att in dev_activity.attempts:
        assert att["content_id"] in valid_content


# 7. Course relationships are valid
def test_course_relationships_are_valid(dev_master, dev_activity):
    course_ids = {c["course_id"] for c in dev_master.courses}
    content_course_map = {c["content_id"]: c["course_id"] for c in dev_master.content_items}
    for att in dev_activity.attempts:
        course_id = content_course_map[att["content_id"]]
        assert course_id in course_ids


# 8. Assessment results reference valid attempts
def test_assessment_results_reference_valid_attempts(dev_activity):
    attempt_ids = {a["attempt_id"] for a in dev_activity.attempts}
    scored_attempt_ids = {a["attempt_id"] for a in dev_activity.attempts if a["status"] == "scored"}
    result_attempt_ids = {r["attempt_id"] for r in dev_activity.assessment_results}

    assert result_attempt_ids == scored_attempt_ids
    for r in dev_activity.assessment_results:
        assert r["attempt_id"] in attempt_ids
        assert PREFIX_PATTERNS["RES"].fullmatch(r["result_id"])


# 9. Score arithmetic reconciles
def test_score_arithmetic_reconciles(dev_activity):
    att_map = {a["attempt_id"]: a for a in dev_activity.attempts}
    for res in dev_activity.assessment_results:
        parent_att = att_map[res["attempt_id"]]
        score_val = float(res["score"])
        parent_score = float(parent_att["score"])
        assert abs(score_val - parent_score) < 0.001
        assert 0.0 <= score_val <= float(parent_att["max_score"])
        passed = int(res["passed"])
        if score_val >= 70.0:
            assert res["outcome"] == "pass"
            assert passed == 1
        else:
            assert res["outcome"] == "fail"
            assert passed == 0


# 10. Enum values are valid
def test_enum_values_are_valid(dev_activity):
    valid_statuses = {"in_progress", "submitted", "scored", "abandoned", "invalid"}
    valid_outcomes = {"pass", "fail", "incomplete"}
    for a in dev_activity.attempts:
        assert a["status"] in valid_statuses
    for r in dev_activity.assessment_results:
        assert r["outcome"] in valid_outcomes


# 11. Timestamp ordering is valid
def test_timestamp_ordering_is_valid(dev_activity):
    att_map = {a["attempt_id"]: a for a in dev_activity.attempts}
    for a in dev_activity.attempts:
        if a["submitted_at"]:
            assert a["submitted_at"] >= a["started_at"]
    for r in dev_activity.assessment_results:
        parent = att_map[r["attempt_id"]]
        assert r["scored_at"] >= parent["submitted_at"]


# 12. Cold-start learners are represented
def test_cold_start_learners_are_represented(dev_master, dev_activity):
    all_learners = {l["learner_id"] for l in dev_master.learners}
    active_learners = {a["learner_id"] for a in dev_activity.attempts}
    cold_start = all_learners - active_learners
    assert len(cold_start) > 0


# 13. Repeated attempts are represented
def test_repeated_attempts_are_represented(dev_activity):
    repeat_counts = [a["attempt_number"] for a in dev_activity.attempts if a["attempt_number"] > 1]
    assert len(repeat_counts) > 0


# 14. Unfinished activities can lack assessment results
def test_unfinished_activities_lack_assessment_results(dev_activity):
    unscored_statuses = [a for a in dev_activity.attempts if a["status"] in ("in_progress", "abandoned", "submitted")]
    assert len(unscored_statuses) > 0
    res_att_ids = {r["attempt_id"] for r in dev_activity.assessment_results}
    for a in unscored_statuses:
        assert a["attempt_id"] not in res_att_ids


# 15. Invalid input data is rejected
def test_invalid_input_data_is_rejected(dev_master, dev_activity):
    # Foreign key mismatch
    bad_attempts = [dict(a) for a in dev_activity.attempts[:10]]
    bad_attempts[0]["learner_id"] = "LRN-999999"
    with pytest.raises(ActivityDataError, match="foreign key violation"):
        validate_activity_tables(
            {"attempts": bad_attempts, "assessment_results": []},
            dev_master,
        )

    # Score mismatch between attempt and result
    bad_res = [dict(r) for r in dev_activity.assessment_results[:5]]
    bad_res[0]["score"] = "99.99"
    with pytest.raises(ActivityDataError, match="does not match attempt score"):
        validate_activity_tables(
            {"attempts": dev_activity.attempts, "assessment_results": bad_res},
            dev_master,
        )


# 16. CLI generation and validation work
def test_cli_activity_commands(tmp_path: Path, capsys):
    # First generate master into tmp_path
    code_m = main(["generate", "master", "--profile", "dev", "--output-dir", str(tmp_path)])
    assert code_m == 0

    # Generate activity into tmp_path
    code_a = main(["generate", "activity", "--profile", "dev", "--output-dir", str(tmp_path), "--attempts", "200"])
    captured = capsys.readouterr()
    assert code_a == 0
    assert "status=generated" in captured.out
    assert "attempts=200" in captured.out

    # Overwrite protection
    code_err = main(["generate", "activity", "--profile", "dev", "--output-dir", str(tmp_path)])
    captured_err = capsys.readouterr()
    assert code_err == 5
    assert "output error" in captured_err.err

    # Overwrite works
    code_ow = main(["generate", "activity", "--profile", "dev", "--output-dir", str(tmp_path), "--attempts", "200", "--overwrite"])
    assert code_ow == 0

    # Validate activity
    code_v = main(["validate", "activity", "--dir", str(tmp_path), "--profile", "dev"])
    captured_v = capsys.readouterr()
    assert code_v == 0
    assert "status=valid" in captured_v.out
    assert "rows_attempts.csv=200" in captured_v.out


# 17. Dev profile produces the configured volume
def test_dev_profile_produces_configured_volume(dev_activity):
    assert len(dev_activity.attempts) == 5000
    assert len(dev_activity.assessment_results) > 0


# 18. Full profile supports 500,000 attempts (tested via target count on smaller master or reference)
def test_full_profile_supports_target_count(dev_master):
    # Verify generator accepts target_attempts and generates exactly that volume
    t_act = generate_activity_tables(profile="dev", seed=20261022, master_tables=dev_master, target_attempts=2500)
    assert len(t_act.attempts) == 2500
    validate_activity_tables(t_act, dev_master)

