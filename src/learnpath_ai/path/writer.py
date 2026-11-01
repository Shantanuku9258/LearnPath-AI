"""CSV and manifest writer for learning paths (Phase 7)."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from learnpath_ai.generate.writer import OutputExistsError
from learnpath_ai.path.engine import LearningPathItemRecord, LearningPathRecord

LEARNING_PATH_COLUMNS = (
    "path_id",
    "learner_id",
    "course_id",
    "status",
    "generated_at",
    "activated_at",
    "completed_at",
    "generator_version",
    "created_at",
    "updated_at",
)

LEARNING_PATH_ITEM_COLUMNS = (
    "path_item_id",
    "path_id",
    "sequence_no",
    "content_id",
    "recommendation_id",
    "status",
    "planned_at",
    "completed_at",
)

PATH_FILES = ("learning_paths.csv", "learning_path_items.csv")
PATH_MANIFEST_NAME = "learning_path_manifest.json"

TIMESTAMP_POLICY = (
    "generated_at / activated_at / planned_at all equal the as_of_timestamp used for path generation. "
    "This is the Phase 6 recommendation cutoff: no attempt, assessment or mastery information "
    "after that instant is used. All timestamps are deterministic UTC."
)


def write_path_dataset(
    paths: list[LearningPathRecord],
    items: list[LearningPathItemRecord],
    output_dir: Path,
    *,
    overwrite: bool = False,
    profile: str = "dev",
    seed: int = 20261022,
    as_of_timestamp: str = "",
    generator_version: str = "path-v1",
) -> dict[str, Any]:
    """Write learning_paths.csv, learning_path_items.csv and manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)

    paths_csv = output_dir / "learning_paths.csv"
    items_csv = output_dir / "learning_path_items.csv"
    manifest_path = output_dir / PATH_MANIFEST_NAME

    planned = [paths_csv, items_csv, manifest_path]
    existing = [p for p in planned if p.exists()]
    if existing and not overwrite:
        names = ", ".join(p.name for p in existing)
        raise OutputExistsError(
            f"Refusing to overwrite existing path files in {output_dir}: {names}. "
            "Pass --overwrite to replace them."
        )

    # Write learning_paths.csv
    with paths_csv.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(LEARNING_PATH_COLUMNS),
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for p in paths:
            writer.writerow(p.to_dict())

    # Write learning_path_items.csv
    with items_csv.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(LEARNING_PATH_ITEM_COLUMNS),
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for it in items:
            writer.writerow(it.to_dict())

    checksums = {
        "learning_paths.csv": _sha256(paths_csv),
        "learning_path_items.csv": _sha256(items_csv),
    }
    row_counts = {
        "learning_paths.csv": len(paths),
        "learning_path_items.csv": len(items),
    }

    manifest: dict[str, Any] = {
        "generator_version": generator_version,
        "dataset_profile": profile,
        "seed": seed,
        "as_of_timestamp": as_of_timestamp,
        "timestamp_policy": TIMESTAMP_POLICY,
        "row_counts": row_counts,
        "checksums": checksums,
        "checksums_sha256": checksums,
        "files": list(PATH_FILES),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)

    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()

