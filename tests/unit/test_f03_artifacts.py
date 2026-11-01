"""On-disk F-03 regression tests for generated master artifacts.

These checks load `data/sample` (dev) and `data/reference` (full) using the
same CSV + manifest validation path as the CLI. Foundational eligibility is
derived from skill mappings and required-prerequisite edges, never from list
position.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest

from learnpath_ai.eligibility import (
    ContentSkillLink,
    content_passes_required_prerequisites,
)
from learnpath_ai.generate.columns import FILE_COLUMNS, MASTER_FILES
from learnpath_ai.generate.validate import validate_master_dataset_dir
from learnpath_ai.prerequisites import PrerequisiteEdge

ON_DISK_PROFILES = (
    ("dev", Path("data/sample")),
    ("full", Path("data/reference")),
)

_TABLE_KEY_MAP = {
    "courses.csv": "courses",
    "learners.csv": "learners",
    "skills.csv": "skills",
    "skill_prerequisites.csv": "skill_prerequisites",
    "content_items.csv": "content_items",
    "content_skills.csv": "content_skills",
}


def _load_tables(directory: Path) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = {}
    for fname in MASTER_FILES:
        fpath = directory / fname
        with fpath.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            expected_cols = list(FILE_COLUMNS[fname])
            assert reader.fieldnames == expected_cols, (
                f"CSV column mismatch in {fpath}. Expected {expected_cols}, got {reader.fieldnames}"
            )
            tables[_TABLE_KEY_MAP[fname]] = list(reader)
    return tables


@pytest.fixture(scope="module", params=ON_DISK_PROFILES, ids=["dev-sample", "full-reference"])
def on_disk_master(request) -> dict[str, Any]:
    profile, directory = request.param
    assert directory.is_dir(), (
        f"Profile {profile}: expected on-disk master directory {directory} to exist"
    )
    summary = validate_master_dataset_dir(
        directory,
        expected_profile=profile,
        verify_manifest=True,
    )
    assert summary["status"] == "valid", (
        f"Profile {profile}: validate_master_dataset_dir status={summary.get('status')!r}"
    )
    assert summary.get("profile") == profile, (
        f"Profile {profile}: manifest dataset_profile={summary.get('profile')!r}"
    )
    tables = _load_tables(directory)
    return {
        "profile": profile,
        "directory": directory,
        "tables": tables,
        "summary": summary,
    }


def _prerequisite_edges(tables: dict[str, list[dict[str, Any]]]) -> list[PrerequisiteEdge]:
    return [
        PrerequisiteEdge(
            skill_id=str(row["skill_id"]),
            prerequisite_skill_id=str(row["prerequisite_skill_id"]),
            relationship_type=str(row.get("relationship_type", "required")),
            min_mastery_required=float(row["min_mastery_required"]),
        )
        for row in tables["skill_prerequisites"]
    ]


def _content_skill_links(tables: dict[str, list[dict[str, Any]]]) -> list[ContentSkillLink]:
    return [
        ContentSkillLink(
            content_id=str(row["content_id"]),
            skill_id=str(row["skill_id"]),
            skill_role=str(row.get("skill_role", "primary")),
        )
        for row in tables["content_skills"]
    ]


def _skills_without_required_prerequisites(
    skills: list[dict[str, Any]],
    edges: list[PrerequisiteEdge],
) -> set[str]:
    skills_with_required = {edge.skill_id for edge in edges if edge.is_required()}
    return {str(row["skill_id"]) for row in skills if str(row["skill_id"]) not in skills_with_required}


def _published_items_for_course(
    tables: dict[str, list[dict[str, Any]]],
    course_id: str,
) -> list[dict[str, Any]]:
    return [
        item
        for item in tables["content_items"]
        if str(item["course_id"]) == course_id
        and str(item.get("status", "")).strip().lower() == "published"
    ]


def test_on_disk_f03_every_course_has_published_root_skill_content(on_disk_master) -> None:
    """Every course has published content mapped to a skill with no required prerequisites."""
    profile = on_disk_master["profile"]
    tables = on_disk_master["tables"]
    edges = _prerequisite_edges(tables)
    links = _content_skill_links(tables)
    root_skills = _skills_without_required_prerequisites(tables["skills"], edges)

    skills_by_content: dict[str, set[str]] = {}
    for link in links:
        skills_by_content.setdefault(link.content_id, set()).add(link.skill_id)

    course_ids = sorted({str(row["course_id"]) for row in tables["courses"]})
    assert course_ids, f"Profile {profile}: courses.csv is empty"

    missing: list[str] = []
    for course_id in course_ids:
        published = _published_items_for_course(tables, course_id)
        foundational = []
        for item in published:
            content_id = str(item["content_id"])
            mapped = skills_by_content.get(content_id, set())
            if not mapped:
                continue
            if mapped & root_skills:
                assert str(item["course_id"]) == course_id, (
                    f"Profile {profile}: content {content_id} mapped as foundational for "
                    f"{course_id} but course_id={item['course_id']!r}"
                )
                foundational.append(item)

        if not foundational:
            missing.append(course_id)

    assert not missing, (
        f"Profile {profile}: courses missing published content mapped to a skill "
        f"with no required prerequisites: {missing}"
    )


def test_on_disk_f03_foundational_content_belongs_to_its_course(on_disk_master) -> None:
    """Foundational items (root-skill mapped, published) are owned by a real course."""
    profile = on_disk_master["profile"]
    tables = on_disk_master["tables"]
    edges = _prerequisite_edges(tables)
    links = _content_skill_links(tables)
    root_skills = _skills_without_required_prerequisites(tables["skills"], edges)
    known_courses = {str(row["course_id"]) for row in tables["courses"]}

    skills_by_content: dict[str, set[str]] = {}
    for link in links:
        skills_by_content.setdefault(link.content_id, set()).add(link.skill_id)

    by_course: dict[str, list[str]] = {course_id: [] for course_id in known_courses}
    misowned: list[str] = []
    for item in tables["content_items"]:
        if str(item.get("status", "")).strip().lower() != "published":
            continue
        content_id = str(item["content_id"])
        mapped = skills_by_content.get(content_id, set())
        root_mapped = mapped & root_skills
        if not root_mapped:
            continue
        course_id = str(item["course_id"])
        if course_id not in known_courses:
            misowned.append(f"{content_id}->{course_id}")
            continue
        by_course[course_id].append(content_id)

    assert not misowned, (
        f"Profile {profile}: foundational content pointing at unknown course_id: {sorted(misowned)}"
    )
    empty_courses = sorted(course_id for course_id, items in by_course.items() if not items)
    assert not empty_courses, (
        f"Profile {profile}: courses with no foundational published content of their own: {empty_courses}"
    )
    for course_id, content_ids in by_course.items():
        for content_id in content_ids:
            item = next(row for row in tables["content_items"] if str(row["content_id"]) == content_id)
            assert str(item["course_id"]) == course_id, (
                f"Profile {profile}: foundational content {content_id} listed for {course_id} "
                f"but course_id={item['course_id']!r}"
            )


def test_on_disk_f03_cold_start_passes_hard_gate_in_every_course(on_disk_master) -> None:
    """A zero-mastery learner can pass the hard gate for at least one published mapped item per course."""
    profile = on_disk_master["profile"]
    tables = on_disk_master["tables"]
    edges = _prerequisite_edges(tables)
    links = _content_skill_links(tables)
    mapped_content_ids = {link.content_id for link in links}

    cold_mastery: dict[str, float] = {}
    course_ids = sorted({str(row["course_id"]) for row in tables["courses"]})
    blocked: list[str] = []

    for course_id in course_ids:
        passed_items: list[str] = []
        for item in _published_items_for_course(tables, course_id):
            content_id = str(item["content_id"])
            if content_id not in mapped_content_ids:
                continue
            if content_passes_required_prerequisites(
                content_id,
                links,
                edges,
                cold_mastery,
                missing_mastery=0.0,
            ):
                passed_items.append(content_id)

        if not passed_items:
            blocked.append(course_id)

    assert not blocked, (
        f"Profile {profile}: cold-start learner (missing_mastery=0.0) cannot pass the "
        f"hard prerequisite gate for any published mapped content in: {blocked}"
    )
