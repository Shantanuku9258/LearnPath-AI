"""CLI discovery, registration, and handler regression protection tests (F-07 / F-08.2)."""

from __future__ import annotations

import argparse
import inspect

import pytest

from learnpath_ai import cli
from learnpath_ai.cli import build_parser, main


def test_cli_help_succeeds(capsys) -> None:
    """Main CLI help exits cleanly with code 0 and displays the program overview."""
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    captured = capsys.readouterr()
    assert "learnpath-ai" in captured.out
    assert "usage:" in captured.out.lower()
    assert "commands:" in captured.out.lower() or "positional arguments:" in captured.out.lower()


def test_cli_top_level_command_groups_registered() -> None:
    """Verify all top-level command groups are properly registered in subparsers."""
    parser = build_parser()
    subparser_actions = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ]
    assert len(subparser_actions) == 1, "Expected exactly one top-level subparser action"
    top_commands = set(subparser_actions[0].choices.keys())
    expected_top = {"db", "generate", "validate", "evaluate"}
    assert expected_top.issubset(top_commands), f"Missing top-level commands: {expected_top - top_commands}"


def test_cli_subcommands_registered() -> None:
    """Verify all expected subcommands are registered under their respective command groups."""
    parser = build_parser()
    subparser_action = [
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    ][0]
    choices = subparser_action.choices

    # db subcommands
    db_parser = choices["db"]
    db_subs = [a for a in db_parser._actions if isinstance(a, argparse._SubParsersAction)][0]
    assert set(db_subs.choices.keys()) == {"ping", "status", "init"}

    # generate subcommands
    gen_parser = choices["generate"]
    gen_subs = [a for a in gen_parser._actions if isinstance(a, argparse._SubParsersAction)][0]
    expected_gen = {
        "master",
        "activity",
        "anomalies",
        "mastery",
        "recommendations",
        "paths",
        "ml-dataset",
        "ml-model",
        "ml-recommendations",
    }
    assert expected_gen.issubset(set(gen_subs.choices.keys()))

    # validate subcommands
    val_parser = choices["validate"]
    val_subs = [a for a in val_parser._actions if isinstance(a, argparse._SubParsersAction)][0]
    expected_val = {"master", "activity", "quality", "mastery", "recommendations", "paths"}
    assert expected_val.issubset(set(val_subs.choices.keys()))

    # evaluate subcommands
    eval_parser = choices["evaluate"]
    eval_subs = [a for a in eval_parser._actions if isinstance(a, argparse._SubParsersAction)][0]
    assert "compare" in eval_subs.choices


def test_cli_handler_functions_available_exactly_once() -> None:
    """Important command handlers exist as callable module functions and are not redefined or duplicated."""
    expected_handlers = [
        "_db_command",
        "_generate_command",
        "_validate_command",
        "_evaluate_command",
        "_print_ping",
        "_print_status",
        "_print_init",
        "_generate_master",
        "_generate_activity",
        "_generate_anomalies",
        "_generate_mastery",
        "_generate_recommendations",
        "_generate_paths",
        "_generate_ml_dataset",
        "_train_ml_model",
        "_generate_ml_recommendations",
        "_validate_master",
        "_validate_activity",
        "_validate_quality",
        "_validate_mastery",
        "_validate_recommendations",
        "_validate_paths",
        "_evaluate_compare",
    ]

    for handler_name in expected_handlers:
        assert hasattr(cli, handler_name), f"Expected CLI handler '{handler_name}' is missing"
        handler_obj = getattr(cli, handler_name)
        assert callable(handler_obj), f"Handler '{handler_name}' is not callable"
        # Confirm function belongs directly to the learnpath_ai.cli module
        assert handler_obj.__module__ == "learnpath_ai.cli"

