"""Unit tests for evaluate compare orchestration in CLI (F-08 Part 2).

Proves:
- An existing ml-v1 model is loaded rather than retrained.
- Rule-v1 and ML-v1 receive identical point-in-time mastery data.
- Both engines evaluate at the exact same chronological cutoff timestamp.
- The expected chronological evaluation test slice is utilized.
- Evaluation does NOT regenerate datasets or retrain/modify the saved model.
- Evaluation outputs expected report format to stdout without modifying files on disk.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from learnpath_ai import cli
from learnpath_ai.ml.model import ModelArtifact


@pytest.fixture
def mini_eval_env(tmp_path: Path):
    """Create a minimal synthetic sample directory and mock model directory for testing orchestration."""
    source_dir = tmp_path / "sample"
    source_dir.mkdir()
    model_dir = tmp_path / "models"
    model_dir.mkdir()

    # Create dummy master CSVs
    courses = [{"course_id": "CRS-001", "status": "published"}]
    learners = [
        {"learner_id": "LRN-001", "current_course_id": "CRS-001", "status": "active"},
        {"learner_id": "LRN-002", "current_course_id": "CRS-001", "status": "active"},
        {"learner_id": "LRN-003", "current_course_id": "CRS-001", "status": "withdrawn"},
    ]
    skills = [{"skill_id": "SKL-001", "status": "active"}]
    skill_prereqs = []
    content_items = [
        {"content_id": "CNT-001", "course_id": "CRS-001", "status": "published", "difficulty": "1.00"},
        {"content_id": "CNT-002", "course_id": "CRS-001", "status": "published", "difficulty": "1.50"},
    ]
    content_skills = [
        {"content_id": "CNT-001", "skill_id": "SKL-001", "skill_role": "primary", "weight": "1.0"},
        {"content_id": "CNT-002", "skill_id": "SKL-001", "skill_role": "primary", "weight": "1.0"},
    ]

    for fname, data in [
        ("courses.csv", courses),
        ("learners.csv", learners),
        ("skills.csv", skills),
        ("skill_prerequisites.csv", skill_prereqs),
        ("content_items.csv", content_items),
        ("content_skills.csv", content_skills),
    ]:
        with (source_dir / fname).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()) if data else ["skill_id", "prerequisite_skill_id"])
            writer.writeheader()
            if data:
                writer.writerows(data)

    # 4 scored attempts across 4 days: 85% train/val (3 attempts), 15% test slice (1 attempt)
    # Total attempts = 4 -> test_cutoff_idx = int(4 * 0.85) = 3 -> attempt 4 is test attempt
    attempts = [
        {"attempt_id": "ATT-001", "learner_id": "LRN-001", "content_id": "CNT-001", "status": "scored", "started_at": "2025-01-01 10:00:00.000000"},
        {"attempt_id": "ATT-002", "learner_id": "LRN-001", "content_id": "CNT-002", "status": "scored", "started_at": "2025-01-02 10:00:00.000000"},
        {"attempt_id": "ATT-003", "learner_id": "LRN-002", "content_id": "CNT-001", "status": "scored", "started_at": "2025-01-03 10:00:00.000000"},
        {"attempt_id": "ATT-004", "learner_id": "LRN-002", "content_id": "CNT-002", "status": "scored", "started_at": "2025-01-04 10:00:00.000000"},
    ]
    results = [
        {"result_id": "RES-001", "attempt_id": "ATT-001", "outcome": "pass", "passed": 1, "score": 90.0, "scored_at": "2025-01-01 10:30:00.000000"},
        {"result_id": "RES-002", "attempt_id": "ATT-002", "outcome": "fail", "passed": 0, "score": 40.0, "scored_at": "2025-01-02 10:30:00.000000"},
        {"result_id": "RES-003", "attempt_id": "ATT-003", "outcome": "pass", "passed": 1, "score": 85.0, "scored_at": "2025-01-03 10:30:00.000000"},
        {"result_id": "RES-004", "attempt_id": "ATT-004", "outcome": "pass", "passed": 1, "score": 95.0, "scored_at": "2025-01-04 10:30:00.000000"},
    ]

    for fname, data in [("attempts.csv", attempts), ("assessment_results.csv", results)]:
        with (source_dir / fname).open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)

    # Place an existing model artifact in model_dir
    model_file = model_dir / "ml-v1.pkl"
    model_file.write_bytes(b"EXISTING_SERIALIZED_MODEL")

    return {"source_dir": source_dir, "model_dir": model_dir, "model_file": model_file}


def test_evaluate_compare_loads_existing_model_without_retraining(mini_eval_env, capsys):
    """When ml-v1.pkl exists, evaluate compare loads it and never calls _train_ml_model."""
    source_dir = mini_eval_env["source_dir"]
    model_dir = mini_eval_env["model_dir"]
    model_file = mini_eval_env["model_file"]

    args = argparse.Namespace(
        profile="dev",
        source_dir=source_dir,
        model_dir=model_dir,
    )

    fake_artifact = MagicMock(spec=ModelArtifact)
    fake_artifact.model_version = "ml-v1"

    with patch.object(ModelArtifact, "load", return_value=fake_artifact) as mock_load, \
         patch.object(cli, "_train_ml_model") as mock_train:

        code = cli._evaluate_compare(args)
        assert code == 0
        mock_load.assert_called_once_with(model_dir)
        mock_train.assert_not_called()

    # Verify model artifact file on disk was untouched
    assert model_file.read_bytes() == b"EXISTING_SERIALIZED_MODEL"


def test_evaluate_compare_orchestration_identical_cutoff_and_inputs(mini_eval_env, capsys):
    """Rule-v1 and ML-v1 receive identical point-in-time mastery snapshots and the same cutoff timestamp."""
    source_dir = mini_eval_env["source_dir"]
    model_dir = mini_eval_env["model_dir"]

    args = argparse.Namespace(
        profile="dev",
        source_dir=source_dir,
        model_dir=model_dir,
    )

    fake_artifact = MagicMock(spec=ModelArtifact)
    captured_rule_args = {}
    captured_ml_args = {}

    orig_rule_gen = cli.generate_recommendations
    orig_ml_gen = cli.generate_ml_recommendations

    def mock_rule_gen(*args, **kwargs):
        captured_rule_args["as_of"] = kwargs.get("as_of_timestamp")
        captured_rule_args["mastery"] = args[2] if len(args) > 2 else kwargs.get("mastery_snapshots")
        return orig_rule_gen(*args, **kwargs)

    def mock_ml_gen(*args, **kwargs):
        captured_ml_args["as_of"] = kwargs.get("as_of_timestamp")
        captured_ml_args["mastery"] = args[2] if len(args) > 2 else kwargs.get("mastery_snapshots")
        return orig_ml_gen(*args, **kwargs)

    with patch.object(ModelArtifact, "load", return_value=fake_artifact), \
         patch.object(cli, "generate_recommendations", side_effect=mock_rule_gen), \
         patch.object(cli, "generate_ml_recommendations", side_effect=mock_ml_gen):

        code = cli._evaluate_compare(args)
        assert code == 0

    expected_cutoff = "2025-01-04 10:00:00.000000"
    assert captured_rule_args["as_of"] == expected_cutoff
    assert captured_ml_args["as_of"] == expected_cutoff
    # Point-in-time mastery snapshots passed to both engines are identical objects
    assert captured_rule_args["mastery"] is captured_ml_args["mastery"]

    # Verify standard output reported status=evaluated and strategy comparisons
    out = capsys.readouterr().out
    assert "status=evaluated" in out
    assert "Strategy: rule-v1" in out
    assert "Strategy: ml-v1" in out
    assert f"Test Cutoff Timestamp: {expected_cutoff}" in out
    assert "Eligible Learners: 2" in out  # LRN-003 is withdrawn and excluded

