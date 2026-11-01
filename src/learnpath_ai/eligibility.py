"""Prerequisite eligibility hard gate.

Ranking must only receive content that passed this check. A failed gate cannot
be overridden by score, path continuity, or machine-learning probability.
Mastery estimation itself is not implemented here; callers supply as-of scores.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from learnpath_ai.prerequisites import PrerequisiteEdge


class IneligibleContentError(ValueError):
    """Content failed the prerequisite gate and must not be ranked."""


@dataclass(frozen=True)
class ContentSkillLink:
    content_id: str
    skill_id: str
    skill_role: str = "primary"


def required_prerequisites_for_content(
    content_id: str,
    content_skills: Sequence[ContentSkillLink],
    edges: Sequence[PrerequisiteEdge],
) -> list[PrerequisiteEdge]:
    taught = {link.skill_id for link in content_skills if link.content_id == content_id}
    return [
        edge
        for edge in edges
        if edge.is_required() and edge.skill_id in taught
    ]


def content_passes_required_prerequisites(
    content_id: str,
    content_skills: Sequence[ContentSkillLink],
    edges: Sequence[PrerequisiteEdge],
    mastery_by_skill: Mapping[str, float],
    *,
    missing_mastery: float = 0.0,
) -> bool:
    """Return True only if every required prerequisite skill meets min mastery."""
    for edge in required_prerequisites_for_content(content_id, content_skills, edges):
        current = mastery_by_skill.get(edge.prerequisite_skill_id, missing_mastery)
        if current < edge.min_mastery_required:
            return False
    return True


def assert_rankable(
    content_ids: Sequence[str],
    content_skills: Sequence[ContentSkillLink],
    edges: Sequence[PrerequisiteEdge],
    mastery_by_skill: Mapping[str, float],
) -> None:
    """Guard for future rankers: refuse to rank any ineligible content."""
    failed = [
        content_id
        for content_id in content_ids
        if not content_passes_required_prerequisites(
            content_id, content_skills, edges, mastery_by_skill
        )
    ]
    if failed:
        raise IneligibleContentError(
            "Prerequisite eligibility failed for content: "
            + ", ".join(failed)
            + ". Ranking is not allowed to override this gate."
        )
