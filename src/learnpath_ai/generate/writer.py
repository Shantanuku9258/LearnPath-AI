"""Deterministic CSV + JSON manifest writers."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from learnpath_ai.generate.columns import FILE_COLUMNS, MANIFEST_NAME, MASTER_FILES
from learnpath_ai.generate.master import MasterTables

TIMESTAMP_POLICY = (
    "Record timestamps are UTC instants sampled from [calendar_start, calendar_end] "
    "using numpy.random.Generator with the configured seed. Wall-clock time is not used."
)


class OutputExistsError(FileExistsError):
    """Refuse to overwrite generated files unless explicitly allowed."""


def write_master_dataset(
    tables: MasterTables,
    output_dir: Path,
    *,
    overwrite: bool = False,
    calendar_start: str = "2024-01-01T00:00:00Z",
    calendar_end: str = "2026-10-22T23:59:59Z",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    planned = [output_dir / name for name in MASTER_FILES]
    planned.append(output_dir / MANIFEST_NAME)
    planned.append(output_dir / "manifest.json")
    existing = [path for path in planned if path.exists()]
    if existing and not overwrite:
        names = ", ".join(path.name for path in existing)
        raise OutputExistsError(
            f"Refusing to overwrite existing files in {output_dir}: {names}. "
            "Pass --overwrite to replace them."
        )

    mapping = {
        "courses.csv": tables.courses,
        "learners.csv": tables.learners,
        "skills.csv": tables.skills,
        "skill_prerequisites.csv": tables.skill_prerequisites,
        "content_items.csv": tables.content_items,
        "content_skills.csv": tables.content_skills,
    }
    checksums: dict[str, str] = {}
    row_counts: dict[str, int] = {}
    for filename, rows in mapping.items():
        path = output_dir / filename
        columns = FILE_COLUMNS[filename]
        _write_csv(path, columns, rows)
        checksums[filename] = _sha256(path)
        row_counts[filename] = len(rows)

    manifest = {
        "generator_version": tables.generator_version,
        "seed": tables.seed,
        "dataset_profile": tables.profile,
        "timestamp_policy": TIMESTAMP_POLICY,
        "calendar_start": calendar_start,
        "calendar_end": calendar_end,
        "row_counts": row_counts,
        "checksums": checksums,
        "checksums_sha256": checksums,
        "files": list(MASTER_FILES),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (output_dir / MANIFEST_NAME).write_bytes(manifest_bytes)
    (output_dir / "manifest.json").write_bytes(manifest_bytes)
    return manifest


def _write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()
