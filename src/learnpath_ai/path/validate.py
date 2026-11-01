"""Validation for learning-path dataset (Phase 7)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from learnpath_ai.eligibility import (
    ContentSkillLink,
    content_passes_required_prerequisites,
)
from learnpath_ai.generate.clock import parse_iso_utc
from learnpath_ai.ids import is_valid_id
from learnpath_ai.path.writer import (
    LEARNING_PATH_COLUMNS,
    LEARNING_PATH_ITEM_COLUMNS,
    PATH_MANIFEST_NAME,
    _sha256,
)
from learnpath_ai.prerequisites import PrerequisiteEdge

VALID_PATH_STATUSES = {"draft", "active", "completed", "abandoned"}
VALID_ITEM_STATUSES = {"planned", "in_progress", "completed", "skipped"}


class PathValidationError(ValueError):
    """Raised when learning-path dataset violates schema or integrity constraints."""


def validate_path_tables(
    paths: list[dict[str, Any]],
    items: list[dict[str, Any]],
    master_tables: Any | None = None,
) -> None:
    """Validate in-memory path and item records."""
    known_learners: set[str] | None = None
    known_courses: set[str] | None = None
    known_content: set[str] | None = None
    known_skills: set[str] | None = None
    prereq_edges: list[PrerequisiteEdge] = []
    content_links: list[ContentSkillLink] = []

    if master_tables is not None:
        def _get(key: str) -> list[dict[str, Any]]:
            if hasattr(master_tables, key):
                return list(getattr(master_tables, key))
            if isinstance(master_tables, dict):
                return list(master_tables.get(key, []))
            return []

        known_learners = {str(r["learner_id"]) for r in _get("learners")}
        known_courses = {str(r["course_id"]) for r in _get("courses")}
        known_content = {str(r["content_id"]) for r in _get("content_items")}
        known_skills = {str(r["skill_id"]) for r in _get("skills")}

        for p in _get("skill_prerequisites"):
            sid = str(p["skill_id"])
            psid = str(p["prerequisite_skill_id"])
            if known_skills and sid in known_skills and psid in known_skills:
                prereq_edges.append(
                    PrerequisiteEdge(
                        skill_id=sid,
                        prerequisite_skill_id=psid,
                        relationship_type=str(p.get("relationship_type", "required")),
                        min_mastery_required=float(p.get("min_mastery_required", 0.60)),
                    )
                )
        for cs in _get("content_skills"):
            cid = str(cs["content_id"])
            sid = str(cs["skill_id"])
            if known_skills and sid in known_skills:
                content_links.append(
                    ContentSkillLink(
                        content_id=cid,
                        skill_id=sid,
                        skill_role=str(cs.get("skill_role", "primary")),
                    )
                )

    # ----- Validate learning_paths -----
    seen_path_ids: set[str] = set()
    active_learner_course: set[tuple[str, str]] = set()
    path_ids: set[str] = set()

    for idx, row in enumerate(paths):
        pid = row.get("path_id", "")

        # ID format: PATH-YYYY-000001
        import re
        if not re.fullmatch(r"^PATH-[0-9]{4}-[0-9]{6}$", pid):
            raise PathValidationError(
                f"Path row {idx}: Invalid path_id format '{pid}'."
            )
        if pid in seen_path_ids:
            raise PathValidationError(f"Path row {idx}: Duplicate path_id '{pid}'.")
        seen_path_ids.add(pid)
        path_ids.add(pid)

        lid = row.get("learner_id", "")
        if not is_valid_id(lid, "LRN"):
            raise PathValidationError(f"Path row {idx}: Invalid learner_id '{lid}'.")
        if known_learners and lid not in known_learners:
            raise PathValidationError(f"Path row {idx}: Unknown learner_id '{lid}'.")

        crs = row.get("course_id", "")
        if not is_valid_id(crs, "CRS"):
            raise PathValidationError(f"Path row {idx}: Invalid course_id '{crs}'.")
        if known_courses and crs not in known_courses:
            raise PathValidationError(f"Path row {idx}: Unknown course_id '{crs}'.")

        status = row.get("status", "")
        if status not in VALID_PATH_STATUSES:
            raise PathValidationError(f"Path row {idx}: Invalid status '{status}'.")

        # Uniqueness: at most one active path per (learner_id, course_id)
        if status == "active":
            pair = (lid, crs)
            if pair in active_learner_course:
                raise PathValidationError(
                    f"Path row {idx}: Duplicate active path for learner '{lid}' course '{crs}'."
                )
            active_learner_course.add(pair)

        # Temporal: activated_at must be set if status == 'active'
        activated_at = row.get("activated_at", "")
        if status == "active" and not activated_at:
            raise PathValidationError(
                f"Path row {idx}: 'active' path missing activated_at (path_id={pid})."
            )

    # ----- Validate learning_path_items -----
    seen_item_ids: set[str] = set()
    seen_path_sequence: set[tuple[str, int]] = set()

    for idx, row in enumerate(items):
        iid = row.get("path_item_id", "")
        if not is_valid_id(iid, "LPI"):
            raise PathValidationError(
                f"Item row {idx}: Invalid path_item_id format '{iid}'."
            )
        if iid in seen_item_ids:
            raise PathValidationError(f"Item row {idx}: Duplicate path_item_id '{iid}'.")
        seen_item_ids.add(iid)

        pid = row.get("path_id", "")
        if pid not in path_ids:
            raise PathValidationError(
                f"Item row {idx}: path_id '{pid}' references a non-existent path."
            )

        try:
            seq = int(row.get("sequence_no", 0))
        except (TypeError, ValueError):
            raise PathValidationError(f"Item row {idx}: sequence_no is not an integer.")
        if seq < 1:
            raise PathValidationError(
                f"Item row {idx}: sequence_no {seq} < 1 (path_item_id={iid})."
            )
        ps_key = (pid, seq)
        if ps_key in seen_path_sequence:
            raise PathValidationError(
                f"Item row {idx}: Duplicate (path_id, sequence_no)={ps_key}."
            )
        seen_path_sequence.add(ps_key)

        cid = row.get("content_id", "")
        if not is_valid_id(cid, "CNT"):
            raise PathValidationError(f"Item row {idx}: Invalid content_id '{cid}'.")
        if known_content and cid not in known_content:
            raise PathValidationError(f"Item row {idx}: Unknown content_id '{cid}'.")

        item_status = row.get("status", "")
        if item_status not in VALID_ITEM_STATUSES:
            raise PathValidationError(f"Item row {idx}: Invalid item status '{item_status}'.")


def validate_path_dataset_dir(
    dir_path: Path,
    *,
    expected_profile: str | None = None,
    verify_manifest: bool = True,
    master_tables: Any | None = None,
) -> dict[str, Any]:
    """Validate learning_paths.csv and learning_path_items.csv in a directory."""
    if not dir_path.is_dir():
        raise PathValidationError(f"Directory not found: {dir_path}")

    paths_csv = dir_path / "learning_paths.csv"
    items_csv = dir_path / "learning_path_items.csv"

    if not paths_csv.is_file():
        raise PathValidationError(f"learning_paths.csv not found in {dir_path}")
    if not items_csv.is_file():
        raise PathValidationError(f"learning_path_items.csv not found in {dir_path}")

    with paths_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != LEARNING_PATH_COLUMNS:
            raise PathValidationError(
                f"learning_paths.csv columns mismatch. "
                f"Expected {LEARNING_PATH_COLUMNS}, got {reader.fieldnames}"
            )
        paths = list(reader)

    with items_csv.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != LEARNING_PATH_ITEM_COLUMNS:
            raise PathValidationError(
                f"learning_path_items.csv columns mismatch. "
                f"Expected {LEARNING_PATH_ITEM_COLUMNS}, got {reader.fieldnames}"
            )
        items = list(reader)

    validate_path_tables(paths, items, master_tables=master_tables)

    manifest_info: dict[str, Any] = {}
    if verify_manifest:
        manifest_path = dir_path / PATH_MANIFEST_NAME
        if not manifest_path.is_file():
            raise PathValidationError(f"{PATH_MANIFEST_NAME} not found in {dir_path}")
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest_info = json.load(fh)

        stored = manifest_info.get("checksums_sha256") or manifest_info.get("checksums", {})
        for fname, csv_path in [
            ("learning_paths.csv", paths_csv),
            ("learning_path_items.csv", items_csv),
        ]:
            actual = _sha256(csv_path)
            expected = stored.get(fname)
            if actual != expected:
                raise PathValidationError(
                    f"Checksum mismatch for {fname}: expected {expected}, got {actual}"
                )

        if expected_profile:
            actual_profile = manifest_info.get("dataset_profile")
            if actual_profile != expected_profile:
                raise PathValidationError(
                    f"Profile mismatch: expected {expected_profile}, manifest says {actual_profile}"
                )

    return {
        "directory": str(dir_path),
        "profile": manifest_info.get("dataset_profile") or expected_profile,
        "row_counts": {
            "learning_paths.csv": len(paths),
            "learning_path_items.csv": len(items),
        },
        "manifest": manifest_info,
    }

