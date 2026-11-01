from __future__ import annotations

from pathlib import Path

from learnpath_ai.cli import main
from learnpath_ai.settings import REQUIRED_MYSQL_VARS


def test_cli_ping_reports_missing_env(monkeypatch, capsys) -> None:
    for name in REQUIRED_MYSQL_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    code = main(["db", "ping"])
    captured = capsys.readouterr()
    assert code == 2
    assert "configuration error" in captured.err
    assert "MYSQL_HOST" in captured.err


def test_cli_generate_master_and_validate(tmp_path: Path, capsys) -> None:
    out_dir = tmp_path / "cli_test"
    code = main(["generate", "master", "--profile", "dev", "--output-dir", str(out_dir)])
    captured = capsys.readouterr()
    assert code == 0
    assert "status=generated" in captured.out
    assert "courses=8" in captured.out
    assert "learners=100" in captured.out

    # Overwrite protection in CLI
    code_err = main(["generate", "master", "--profile", "dev", "--output-dir", str(out_dir)])
    captured_err = capsys.readouterr()
    assert code_err == 5
    assert "output error" in captured_err.err

    # Overwrite flag works
    code_ow = main(["generate", "master", "--profile", "dev", "--output-dir", str(out_dir), "--overwrite"])
    assert code_ow == 0

    # Validate CLI
    code_val = main(["validate", "master", "--dir", str(out_dir), "--profile", "dev"])
    captured_val = capsys.readouterr()
    assert code_val == 0
    assert "status=valid" in captured_val.out
    assert "manifest=verified" in captured_val.out


def test_cli_validate_reports_error_on_corrupt_dir(tmp_path: Path, capsys) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    code = main(["validate", "master", "--dir", str(empty_dir)])
    captured = capsys.readouterr()
    assert code == 5
    assert "validation error" in captured.err
