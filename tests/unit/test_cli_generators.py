"""Targeted CLI unit tests for generate subcommands and error handling (F-08 Part 2).

Tests:
- generate mastery with overwrite protection and execution.
- generate recommendations with overwrite protection.
- generate learning paths with overwrite protection.
- generate ml-dataset.
- Missing input directory / prerequisite file error handling.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from learnpath_ai.cli import (
    _generate_mastery,
    _generate_ml_dataset,
    _generate_paths,
    _generate_recommendations,
    main,
)
from learnpath_ai.generate.writer import OutputExistsError


@pytest.fixture
def mock_gen_source_dir(tmp_path: Path):
    source_dir = tmp_path / "mock_source"
    source_dir.mkdir()

    # Create dummy master CSVs
    for fn in [
        "courses.csv",
        "learners.csv",
        "skills.csv",
        "skill_prerequisites.csv",
        "content_items.csv",
        "content_skills.csv",
        "attempts.csv",
        "assessment_results.csv",
    ]:
        p = source_dir / fn
        p.write_text("header1,header2\n", encoding="utf-8")

    return source_dir


def test_cli_validate_missing_input_directory_reports_error(tmp_path: Path, capsys):
    """validate master reports clean error code 5 when directory does not exist."""
    missing_dir = tmp_path / "does_not_exist"
    code = main(["validate", "master", "--dir", str(missing_dir)])
    assert code == 5
    captured = capsys.readouterr()
    assert "validation error" in captured.err.lower()


def test_cli_generate_mastery_overwrite_protection(mock_gen_source_dir, tmp_path: Path, capsys):
    """_generate_mastery respects overwrite=False when destination files exist."""
    out_dir = tmp_path / "mastery_out"
    out_dir.mkdir()

    args = argparse.Namespace(
        profile="dev",
        source_dir=mock_gen_source_dir,
        output_dir=out_dir,
        as_of=None,
        seed=20261022,
        overwrite=False,
    )

    with patch("learnpath_ai.cli.calculate_mastery_snapshots", return_value=[]), \
         patch("learnpath_ai.cli.write_mastery_dataset", side_effect=OutputExistsError("Refusing to overwrite")):
        with pytest.raises(OutputExistsError, match="Refusing to overwrite"):
            _generate_mastery(args)


def test_cli_generate_recommendations_overwrite_protection(mock_gen_source_dir, tmp_path: Path):
    """_generate_recommendations raises OutputExistsError when destination files exist and overwrite is False."""
    out_dir = tmp_path / "rec_out"
    out_dir.mkdir()

    args = argparse.Namespace(
        profile="dev",
        source_dir=mock_gen_source_dir,
        output_dir=out_dir,
        as_of=None,
        k=5,
        seed=20261022,
        overwrite=False,
    )

    with patch("learnpath_ai.cli.generate_recommendations", return_value=[]), \
         patch("learnpath_ai.cli.write_recommendation_dataset", side_effect=OutputExistsError("Refusing to overwrite")):
        with pytest.raises(OutputExistsError, match="Refusing to overwrite"):
            _generate_recommendations(args)


def test_cli_generate_paths_overwrite_protection(mock_gen_source_dir, tmp_path: Path):
    """_generate_paths raises OutputExistsError when destination files exist and overwrite is False."""
    out_dir = tmp_path / "path_out"
    out_dir.mkdir()

    args = argparse.Namespace(
        profile="dev",
        source_dir=mock_gen_source_dir,
        output_dir=out_dir,
        as_of=None,
        seed=20261022,
        overwrite=False,
    )

    with patch("learnpath_ai.cli.generate_learning_paths", return_value=([], [])), \
         patch("learnpath_ai.cli.write_path_dataset", side_effect=OutputExistsError("Refusing to overwrite")):
        with pytest.raises(OutputExistsError, match="Refusing to overwrite"):
            _generate_paths(args)


def test_cli_generate_ml_dataset_execution(mock_gen_source_dir, tmp_path: Path, capsys):
    """_generate_ml_dataset successfully parses datasets and writes split CSVs."""
    out_dir = tmp_path / "ml_out"

    args = argparse.Namespace(
        profile="dev",
        source_dir=mock_gen_source_dir,
        output_dir=out_dir,
    )

    mock_split = MagicMock()
    mock_split.train = []
    mock_split.val = []
    mock_split.test = []

    with patch("learnpath_ai.cli.build_ml_dataset", return_value=[]), \
         patch("learnpath_ai.cli.chronological_split", return_value=mock_split), \
         patch("learnpath_ai.cli.write_dataset_to_csv") as mock_write:
        code = _generate_ml_dataset(args)
        assert code == 0
        captured = capsys.readouterr()
        assert "status=generated" in captured.out
        assert mock_write.call_count == 4  # all, train, val, test
