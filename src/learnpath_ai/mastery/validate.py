"""Validation logic for mastery snapshots dataset (Phase 5)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from learnpath_ai.generate.clock import parse_iso_utc
from learnpath_ai.generate.columns import (
    MASTERY_FILES,
    MASTERY_MANIFEST_NAME,
    MASTERY_SNAPSHOT_COLUMNS,
)
from learnpath_ai.ids import is_valid_id
from learnpath_ai.mastery.writer import _sha256


class MasteryValidationError(ValueError):
    """Raised when mastery snapshot dataset violates schema, constraints or checksums."""


def validate_mastery_snapshots(
    snapshots: list[dict[str, Any]],
    master_tables: Any | None = None,
    *,
    expected_profile: str | None = None,
) -> None:
    """Validate a list of mastery snapshot dictionaries against schema rules."""
    if not snapshots:
        raise MasteryValidationError("Mastery snapshot dataset is empty.")

    # Foreign key sets if available
    known_learners: set[str] | None = None
    known_skills: set[str] | None = None

    if master_tables is not None:
        def _get_ids(key: str, id_col: str) -> set[str]:
            if hasattr(master_tables, key):
                return {str(r[id_col]) for r in getattr(master_tables, key)}
            if isinstance(master_tables, dict) and key in master_tables:
                return {str(r[id_col]) for r in master_tables[key]}
            return set()

        known_learners = _get_ids("learners", "learner_id")
        known_skills = _get_ids("skills", "skill_id")

    seen_ids: set[str] = set()
    seen_unique_keys: set[tuple[str, str, str]] = set()

    for idx, row in enumerate(snapshots):
        # 1. Snapshot ID format and uniqueness
        sid = row.get("snapshot_id", "")
        if not is_valid_id(sid, "MST"):
            raise MasteryValidationError(
                f"Row {idx}: Invalid snapshot_id format '{sid}'. Must match '^MST-[0-9]{{6}}$'."
            )
        if sid in seen_ids:
            raise MasteryValidationError(f"Row {idx}: Duplicate snapshot_id '{sid}'.")
        seen_ids.add(sid)

        # 2. Learner ID & Skill ID formats
        lid = row.get("learner_id", "")
        if not is_valid_id(lid, "LRN"):
            raise MasteryValidationError(f"Row {idx}: Invalid learner_id format '{lid}'.")
        if known_learners and lid not in known_learners:
            raise MasteryValidationError(f"Row {idx}: Unknown learner_id foreign key '{lid}'.")

        skid = row.get("skill_id", "")
        if not is_valid_id(skid, "SKL"):
            raise MasteryValidationError(f"Row {idx}: Invalid skill_id format '{skid}'.")
        if known_skills and skid not in known_skills:
            raise MasteryValidationError(f"Row {idx}: Unknown skill_id foreign key '{skid}'.")

        # 3. Unique key (learner_id, skill_id, as_of_timestamp)
        as_of_raw = row.get("as_of_timestamp", "")
        if not as_of_raw:
            raise MasteryValidationError(f"Row {idx}: Missing as_of_timestamp.")
        key = (lid, skid, as_of_raw)
        if key in seen_unique_keys:
            raise MasteryValidationError(
                f"Row {idx}: Duplicate unique key (learner, skill, as_of_timestamp): {key}"
            )
        seen_unique_keys.add(key)

        # 4. Score range [0.0, 1.0]
        try:
            score = float(row.get("mastery_score", -1))
        except (ValueError, TypeError):
            raise MasteryValidationError(f"Row {idx}: Unparseable mastery_score '{row.get('mastery_score')}'.")
        if score < 0.0 or score > 1.0:
            raise MasteryValidationError(
                f"Row {idx}: mastery_score {score} out of range [0.0, 1.0]."
            )

        # 5. Evidence attempt count >= 0
        try:
            count = int(row.get("evidence_attempt_count", -1))
        except (ValueError, TypeError):
            raise MasteryValidationError(f"Row {idx}: Unparseable evidence_attempt_count.")
        if count < 0:
            raise MasteryValidationError(f"Row {idx}: evidence_attempt_count must be >= 0, got {count}.")

        # 6. Temporal constraint: evidence_cutoff_ts <= as_of_timestamp
        cutoff_raw = row.get("evidence_cutoff_ts", "")
        if not cutoff_raw:
            raise MasteryValidationError(f"Row {idx}: Missing evidence_cutoff_ts.")

        as_of_dt = parse_iso_utc(as_of_raw)
        cutoff_dt = parse_iso_utc(cutoff_raw)
        if cutoff_dt > as_of_dt:
            raise MasteryValidationError(
                f"Row {idx}: Future leakage! evidence_cutoff_ts ({cutoff_raw}) > as_of_timestamp ({as_of_raw})."
            )


def validate_mastery_dataset_dir(
    dir_path: Path,
    *,
    expected_profile: str | None = None,
    verify_manifest: bool = True,
    master_tables: Any | None = None,
) -> dict[str, Any]:
    """Validate mastery_snapshots.csv in directory and optionally verify mastery_manifest.json."""
    if not dir_path.is_dir():
        raise MasteryValidationError(f"Directory not found: {dir_path}")

    csv_path = dir_path / "mastery_snapshots.csv"
    if not csv_path.is_file():
        raise MasteryValidationError(f"mastery_snapshots.csv not found in {dir_path}")

    with csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if tuple(reader.fieldnames or ()) != MASTERY_SNAPSHOT_COLUMNS:
            raise MasteryValidationError(
                f"Columns in {csv_path.name} do not match schema. Expected {MASTERY_SNAPSHOT_COLUMNS}, got {reader.fieldnames}"
            )
        rows = list(reader)

    validate_mastery_snapshots(rows, master_tables=master_tables, expected_profile=expected_profile)

    manifest_info = {}
    if verify_manifest:
        manifest_path = dir_path / MASTERY_MANIFEST_NAME
        if not manifest_path.is_file():
            raise MasteryValidationError(f"Manifest {MASTERY_MANIFEST_NAME} not found in {dir_path}")
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest_info = json.load(fh)

        # Check checksums
        stored_checksums = manifest_info.get("checksums_sha256") or manifest_info.get("checksums", {})
        actual_checksum = _sha256(csv_path)
        expected_checksum = stored_checksums.get("mastery_snapshots.csv")
        if actual_checksum != expected_checksum:
            raise MasteryValidationError(
                f"Checksum mismatch for mastery_snapshots.csv: expected {expected_checksum}, got {actual_checksum}"
            )

        if expected_profile:
            actual_profile = manifest_info.get("dataset_profile")
            if actual_profile != expected_profile:
                raise MasteryValidationError(
                    f"Profile mismatch: expected {expected_profile}, manifest specifies {actual_profile}"
                )

    return {
        "directory": str(dir_path),
        "profile": manifest_info.get("dataset_profile") or expected_profile,
        "row_counts": {"mastery_snapshots.csv": len(rows)},
        "manifest": manifest_info,
    }

