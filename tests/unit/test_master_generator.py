"""Unit tests for the master data generator and validator (Phase 2)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from learnpath_ai.config_loader import ConfigError, load_app_config
from learnpath_ai.generate.columns import FILE_COLUMNS, MANIFEST_NAME, MASTER_FILES
from learnpath_ai.generate.master import generate_master_tables
from learnpath_ai.generate.validate import (
    MasterDataError,
    validate_master_dataset_dir,
    validate_master_tables,
)
from learnpath_ai.generate.writer import OutputExistsError, write_master_dataset
from learnpath_ai.ids import PREFIX_PATTERNS


@pytest.fixture(scope="module")
def dev_tables():
    """Shared dev dataset fixture to avoid redundant generation."""
    return generate_master_tables(profile="dev", seed=20261022)


# 1. Same seed produces identical datasets
def test_same_seed_produces_identical_datasets():
    t1 = generate_master_tables(profile="dev", seed=20261022)
    t2 = generate_master_tables(profile="dev", seed=20261022)
    assert t1.courses == t2.courses
    assert t1.learners == t2.learners
    assert t1.skills == t2.skills
    assert t1.skill_prerequisites == t2.skill_prerequisites
    assert t1.content_items == t2.content_items
    assert t1.content_skills == t2.content_skills


# 2. Different seeds can produce different logical data
def test_different_seeds_produce_different_data(dev_tables):
    t_other = generate_master_tables(profile="dev", seed=999999)
    assert dev_tables.learners != t_other.learners
    assert dev_tables.content_items != t_other.content_items


# 3. IDs follow the required formats
def test_ids_follow_required_format(dev_tables):
    for course in dev_tables.courses:
        assert PREFIX_PATTERNS["CRS"].fullmatch(course["course_id"])
    for learner in dev_tables.learners:
        assert PREFIX_PATTERNS["LRN"].fullmatch(learner["learner_id"])
    for skill in dev_tables.skills:
        assert PREFIX_PATTERNS["SKL"].fullmatch(skill["skill_id"])
    for item in dev_tables.content_items:
        assert PREFIX_PATTERNS["CNT"].fullmatch(item["content_id"])


# 4. IDs are unique
def test_ids_are_unique(dev_tables):
    course_ids = [c["course_id"] for c in dev_tables.courses]
    assert len(course_ids) == len(set(course_ids))

    learner_ids = [l["learner_id"] for l in dev_tables.learners]
    assert len(learner_ids) == len(set(learner_ids))

    skill_ids = [s["skill_id"] for s in dev_tables.skills]
    assert len(skill_ids) == len(set(skill_ids))

    content_ids = [c["content_id"] for c in dev_tables.content_items]
    assert len(content_ids) == len(set(content_ids))

    prereq_pairs = [(p["skill_id"], p["prerequisite_skill_id"]) for p in dev_tables.skill_prerequisites]
    assert len(prereq_pairs) == len(set(prereq_pairs))

    content_skill_pairs = [(cs["content_id"], cs["skill_id"]) for cs in dev_tables.content_skills]
    assert len(content_skill_pairs) == len(set(content_skill_pairs))


# 5. All foreign keys reference valid parents
def test_foreign_keys_reference_valid_parents(dev_tables):
    course_ids = {c["course_id"] for c in dev_tables.courses}
    skill_ids = {s["skill_id"] for s in dev_tables.skills}
    content_ids = {c["content_id"] for c in dev_tables.content_items}

    for learner in dev_tables.learners:
        cid = learner.get("current_course_id")
        if cid:
            assert cid in course_ids

    for item in dev_tables.content_items:
        assert item["course_id"] in course_ids

    for prereq in dev_tables.skill_prerequisites:
        assert prereq["skill_id"] in skill_ids
        assert prereq["prerequisite_skill_id"] in skill_ids

    for cs in dev_tables.content_skills:
        assert cs["content_id"] in content_ids
        assert cs["skill_id"] in skill_ids


# 6. Prerequisite graph is acyclic
def test_prerequisite_graph_is_acyclic(dev_tables):
    # Base tables must validate cleanly
    validate_master_tables(dev_tables)

    # Injecting a cycle must trigger MasterDataError
    invalid_prereqs = list(dev_tables.skill_prerequisites)
    s1 = dev_tables.skills[0]["skill_id"]
    s2 = dev_tables.skills[1]["skill_id"]
    invalid_prereqs.append({
        "skill_id": s1,
        "prerequisite_skill_id": s2,
        "relationship_type": "required",
        "min_mastery_required": "0.6000",
        "created_at": "2024-01-01 00:00:00.000000",
        "effective_from": "2024-01-01 00:00:00.000000",
        "effective_to": "",
    })
    invalid_prereqs.append({
        "skill_id": s2,
        "prerequisite_skill_id": s1,
        "relationship_type": "required",
        "min_mastery_required": "0.6000",
        "created_at": "2024-01-01 00:00:00.000000",
        "effective_from": "2024-01-01 00:00:00.000000",
        "effective_to": "",
    })
    bad_tables = {
        "courses": dev_tables.courses,
        "learners": dev_tables.learners,
        "skills": dev_tables.skills,
        "skill_prerequisites": invalid_prereqs,
        "content_items": dev_tables.content_items,
        "content_skills": dev_tables.content_skills,
    }
    with pytest.raises(MasterDataError, match="cycle"):
        validate_master_tables(bad_tables)


# 7. No duplicate content-skill mappings
def test_no_duplicate_content_skills(dev_tables):
    cs_pairs = [(r["content_id"], r["skill_id"]) for r in dev_tables.content_skills]
    assert len(cs_pairs) == len(set(cs_pairs))

    # Duplicate injection must fail
    dup = list(dev_tables.content_skills)
    dup.append(dict(dup[0]))
    bad_tables = {
        "courses": dev_tables.courses,
        "learners": dev_tables.learners,
        "skills": dev_tables.skills,
        "skill_prerequisites": dev_tables.skill_prerequisites,
        "content_items": dev_tables.content_items,
        "content_skills": dup,
    }
    with pytest.raises(MasterDataError, match="Duplicate content_skill mapping"):
        validate_master_tables(bad_tables)


# 8. Valid enums and required fields
def test_valid_enums_and_required_fields(dev_tables):
    validate_master_tables(dev_tables)

    # Invalid enum on course status
    bad_courses = [dict(c) for c in dev_tables.courses]
    bad_courses[0]["status"] = "nonexistent_status"
    bad_tables = {
        "courses": bad_courses,
        "learners": dev_tables.learners,
        "skills": dev_tables.skills,
        "skill_prerequisites": dev_tables.skill_prerequisites,
        "content_items": dev_tables.content_items,
        "content_skills": dev_tables.content_skills,
    }
    with pytest.raises(MasterDataError, match="invalid status"):
        validate_master_tables(bad_tables)

    # Missing required field on learner
    bad_learners = [dict(l) for l in dev_tables.learners]
    bad_learners[0]["display_name"] = ""
    bad_tables2 = {
        "courses": dev_tables.courses,
        "learners": bad_learners,
        "skills": dev_tables.skills,
        "skill_prerequisites": dev_tables.skill_prerequisites,
        "content_items": dev_tables.content_items,
        "content_skills": dev_tables.content_skills,
    }
    with pytest.raises(MasterDataError, match="missing required 'display_name'"):
        validate_master_tables(bad_tables2)


# 9. Development profile produces expected counts
def test_dev_profile_expected_counts(dev_tables):
    counts = dev_tables.row_counts()
    assert counts["courses"] == 8
    assert counts["learners"] == 100
    assert counts["skills"] == 50
    assert counts["content_items"] == 100
    assert counts["skill_prerequisites"] >= 1
    assert counts["content_skills"] >= 100


# 10. Full profile supports the required counts
def test_full_profile_supports_required_counts():
    full_tables = generate_master_tables(profile="full", seed=20261022)
    counts = full_tables.row_counts()
    assert counts["courses"] == 40
    assert counts["learners"] == 10000
    assert counts["skills"] == 500
    assert counts["content_items"] == 2000
    assert counts["skill_prerequisites"] >= 500
    assert counts["content_skills"] >= 2000
    validate_master_tables(full_tables, profile="full")


# 11. Invalid configurations fail clearly
def test_invalid_configurations_fail_clearly():
    # Unknown profile
    with pytest.raises(ConfigError, match="Unknown dataset profile"):
        generate_master_tables(profile="nonexistent")

    # Invalid custom counts
    cfg = load_app_config()
    with pytest.raises(ConfigError):
        generate_master_tables(counts={"courses": 0, "learners": 10, "skills": 10, "content_items": 10})

    # Invalid content_type_weights
    with pytest.raises(ConfigError, match="content_type_weights"):
        generate_master_tables(
            profile="dev",
            generation_overrides={"content_type_weights": [1.0, 1.0]}  # 2 weights for 5 types
        )


# 12. CSV exports preserve schema and column order
def test_csv_exports_preserve_schema_and_column_order(dev_tables, tmp_path: Path):
    out_dir = tmp_path / "export_test"
    manifest = write_master_dataset(dev_tables, out_dir, overwrite=False)

    assert (out_dir / MANIFEST_NAME).is_file()
    assert (out_dir / "manifest.json").is_file()

    for fname in MASTER_FILES:
        fpath = out_dir / fname
        assert fpath.is_file()
        with fpath.open("r", encoding="utf-8") as handle:
            reader = csv.reader(handle)
            header = next(reader)
            assert header == list(FILE_COLUMNS[fname])

    # Validation from disk passes
    res = validate_master_dataset_dir(out_dir, expected_profile="dev")
    assert res["status"] == "valid"

    # Overwrite protection
    with pytest.raises(OutputExistsError, match="Refusing to overwrite"):
        write_master_dataset(dev_tables, out_dir, overwrite=False)

    # Overwrite works when enabled
    write_master_dataset(dev_tables, out_dir, overwrite=True)


# 13. F-03: Every generated course has at least one published zero-required-prerequisite content item
def test_f03_every_course_has_published_zero_prerequisite_content():
    from learnpath_ai.eligibility import ContentSkillLink, required_prerequisites_for_content
    from learnpath_ai.prerequisites import PrerequisiteEdge

    for profile in ["dev", "full"]:
        tables = generate_master_tables(profile=profile, seed=20261022)
        edges = [
            PrerequisiteEdge(
                skill_id=r["skill_id"],
                prerequisite_skill_id=r["prerequisite_skill_id"],
                relationship_type=r["relationship_type"],
                min_mastery_required=float(r["min_mastery_required"]),
            )
            for r in tables.skill_prerequisites
        ]
        links = [
            ContentSkillLink(
                content_id=r["content_id"],
                skill_id=r["skill_id"],
                skill_role=r["skill_role"],
            )
            for r in tables.content_skills
        ]

        pub_items = [c for c in tables.content_items if c["status"] == "published"]
        zero_prereq_items = [
            c for c in pub_items
            if len(required_prerequisites_for_content(c["content_id"], links, edges)) == 0
        ]
        courses_with_zero = {c["course_id"] for c in zero_prereq_items}
        all_courses = {c["course_id"] for c in tables.courses}

        assert courses_with_zero == all_courses, (
            f"Profile {profile} missing zero-prereq published content in: {all_courses - courses_with_zero}"
        )


def _f03_tables(profile: str):
    return generate_master_tables(profile=profile, seed=20261022)


def _f03_root_skills(tables) -> set[str]:
    skills_with_required = {
        str(row["skill_id"])
        for row in tables.skill_prerequisites
        if str(row.get("relationship_type", "required")).strip().lower() == "required"
    }
    return {str(row["skill_id"]) for row in tables.skills if str(row["skill_id"]) not in skills_with_required}


def _f03_skills_by_content(tables) -> dict[str, list[dict[str, Any]]]:
    from collections import defaultdict

    by_cid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tables.content_skills:
        by_cid[str(row["content_id"])].append(row)
    return by_cid


def _f03_foundational_items(tables) -> list[dict[str, Any]]:
    """Identify foundational catalog items from mappings and edges, not list position.

    A foundational item is published, owned by a known course, maps to exactly one
    root skill (no required prerequisites) as primary weight 1.0, and is beginner
    difficulty in [1.00, 1.50].
    """
    known_courses = {str(row["course_id"]) for row in tables.courses}
    root_skills = _f03_root_skills(tables)
    skills_by_content = _f03_skills_by_content(tables)
    found = []
    for item in tables.content_items:
        if str(item.get("status", "")).strip().lower() != "published":
            continue
        course_id = str(item["course_id"])
        if course_id not in known_courses:
            continue
        links = skills_by_content.get(str(item["content_id"]), [])
        if len(links) != 1:
            continue
        link = links[0]
        if str(link.get("skill_role", "primary")) != "primary":
            continue
        if float(link.get("weight", 1.0)) != 1.0:
            continue
        if str(link["skill_id"]) not in root_skills:
            continue
        difficulty = float(item["difficulty"])
        if 1.00 <= difficulty <= 1.50:
            found.append(item)
    return found


@pytest.mark.parametrize("profile", ["dev", "full"])
def test_f03_foundational_items_map_to_single_root_skill(profile):
    tables = _f03_tables(profile)
    root_skills = _f03_root_skills(tables)
    skills_by_content = _f03_skills_by_content(tables)
    foundational = _f03_foundational_items(tables)
    known_courses = {str(row["course_id"]) for row in tables.courses}

    assert foundational, f"Profile {profile}: no foundational items identified from mappings/edges"
    courses_with_foundational = {str(item["course_id"]) for item in foundational}
    assert courses_with_foundational == known_courses, (
        f"Profile {profile} missing foundational root-skill content in: "
        f"{sorted(known_courses - courses_with_foundational)}"
    )

    for item in foundational:
        cid = str(item["content_id"])
        links = skills_by_content[cid]
        assert len(links) == 1, (
            f"Profile {profile}: foundational item {cid} must have exactly 1 skill link, got {len(links)}"
        )
        assert links[0]["skill_role"] == "primary"
        assert float(links[0]["weight"]) == 1.0000
        assert links[0]["skill_id"] in root_skills, (
            f"Profile {profile}: skill {links[0]['skill_id']} on {cid} must be a root skill"
        )
        assert str(item["course_id"]) in known_courses


@pytest.mark.parametrize("profile", ["dev", "full"])
def test_f03_foundational_items_have_beginner_difficulty(profile):
    tables = _f03_tables(profile)
    foundational = _f03_foundational_items(tables)
    assert foundational, f"Profile {profile}: no foundational items identified"

    for item in foundational:
        diff = float(item["difficulty"])
        assert 1.00 <= diff <= 1.50, (
            f"Profile {profile}: foundational item {item['content_id']} difficulty {diff} "
            f"outside [1.00, 1.50]"
        )
        assert item["status"] == "published"
        assert str(item["course_id"]) in {str(row["course_id"]) for row in tables.courses}


@pytest.mark.parametrize("profile", ["dev", "full"])
def test_f03_cold_start_passes_hard_prerequisite_gate_in_every_course(profile):
    from learnpath_ai.eligibility import ContentSkillLink, content_passes_required_prerequisites
    from learnpath_ai.prerequisites import PrerequisiteEdge

    tables = _f03_tables(profile)
    edges = [
        PrerequisiteEdge(
            skill_id=r["skill_id"],
            prerequisite_skill_id=r["prerequisite_skill_id"],
            relationship_type=r["relationship_type"],
            min_mastery_required=float(r["min_mastery_required"]),
        )
        for r in tables.skill_prerequisites
    ]
    links = [
        ContentSkillLink(
            content_id=r["content_id"],
            skill_id=r["skill_id"],
            skill_role=r["skill_role"],
        )
        for r in tables.content_skills
    ]
    mapped = {link.content_id for link in links}

    cold_mastery: dict[str, float] = {}
    all_courses = {c["course_id"] for c in tables.courses}
    passed_courses: set[str] = set()
    for item in tables.content_items:
        if str(item.get("status", "")).strip().lower() != "published":
            continue
        cid = str(item["content_id"])
        if cid not in mapped:
            continue
        if content_passes_required_prerequisites(
            cid, links, edges, cold_mastery, missing_mastery=0.0
        ):
            passed_courses.add(str(item["course_id"]))

    assert passed_courses == all_courses, (
        f"Profile {profile}: cold-start cannot pass prerequisites in: "
        f"{sorted(all_courses - passed_courses)}"
    )


