from __future__ import annotations

import pytest

from learnpath_ai.eligibility import (
    ContentSkillLink,
    IneligibleContentError,
    assert_rankable,
    content_passes_required_prerequisites,
)
from learnpath_ai.prerequisites import PrerequisiteEdge


EDGES = [
    PrerequisiteEdge("SKL-000002", "SKL-000001", "required", 0.6),
    PrerequisiteEdge("SKL-000002", "SKL-000099", "recommended", 0.9),
]
SKILLS = [ContentSkillLink("CNT-000010", "SKL-000002", "primary")]


def test_required_prerequisite_failure_blocks_content() -> None:
    ok = content_passes_required_prerequisites(
        "CNT-000010",
        SKILLS,
        EDGES,
        {"SKL-000001": 0.2},
    )
    assert ok is False


def test_required_prerequisite_success_allows_content() -> None:
    ok = content_passes_required_prerequisites(
        "CNT-000010",
        SKILLS,
        EDGES,
        {"SKL-000001": 0.9},
    )
    assert ok is True


def test_recommended_edge_does_not_block() -> None:
    ok = content_passes_required_prerequisites(
        "CNT-000010",
        SKILLS,
        EDGES,
        {"SKL-000001": 0.9, "SKL-000099": 0.0},
    )
    assert ok is True


def test_ranker_cannot_override_failed_gate() -> None:
    with pytest.raises(IneligibleContentError, match="not allowed to override"):
        assert_rankable(
            ["CNT-000010"],
            SKILLS,
            EDGES,
            {"SKL-000001": 0.1},
        )
