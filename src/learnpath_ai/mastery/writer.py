"""CSV and manifest writer for mastery snapshots (Phase 5)."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from learnpath_ai.generate.columns import MASTERY_FILES, MASTERY_MANIFEST_NAME, MASTERY_SNAPSHOT_COLUMNS
from learnpath_ai.generate.writer import OutputExistsError
from learnpath_ai.mastery.calculator import MasterySnapshotRecord

TIMESTAMP_POLICY = (
    "as_of_timestamp represents the point in time of decision. "
    "evidence_cutoff_ts represents the latest event timestamp included (evidence_cutoff_ts <= as_of_timestamp). "
    "All timestamps are deterministic UTC."
)


def write_mastery_dataset(
    snapshots: list[MasterySnapshotRecord],
    output_dir: Path,
    *,
    overwrite: bool = False,
    profile: str = "dev",
    seed: int = 20261022,
    as_of_timestamp: str | None = None,
    generator_version: str = "mastery-v1",
) -> dict[str, Any]:
    """Write mastery snapshots to mastery_snapshots.csv and write mastery_manifest.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "mastery_snapshots.csv"
    manifest_path = output_dir / MASTERY_MANIFEST_NAME

    planned = [csv_path, manifest_path]
    existing = [p for p in planned if p.exists()]
    if existing and not overwrite:
        names = ", ".join(p.name for p in existing)
        raise OutputExistsError(
            f"Refusing to overwrite existing mastery files in {output_dir}: {names}. "
            "Pass --overwrite to replace them."
        )

    rows = [s.to_dict() for s in snapshots]
    with csv_path.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(MASTERY_SNAPSHOT_COLUMNS),
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    checksum = _sha256(csv_path)
    checksums = {"mastery_snapshots.csv": checksum}
    row_counts = {"mastery_snapshots.csv": len(rows)}

    manifest = write_mastery_manifest(
        output_dir,
        row_counts=row_counts,
        checksums=checksums,
        profile=profile,
        seed=seed,
        as_of_timestamp=as_of_timestamp or (snapshots[0].as_of_timestamp if snapshots else ""),
        generator_version=generator_version,
        overwrite=True,
    )
    return manifest


def write_mastery_manifest(
    output_dir: Path,
    *,
    row_counts: dict[str, int],
    checksums: dict[str, str],
    profile: str = "dev",
    seed: int = 20261022,
    as_of_timestamp: str = "",
    generator_version: str = "mastery-v1",
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest_path = output_dir / MASTERY_MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        raise OutputExistsError(f"Refusing to overwrite existing manifest: {manifest_path}")

    manifest: dict[str, Any] = {
        "generator_version": generator_version,
        "dataset_profile": profile,
        "seed": seed,
        "as_of_timestamp": as_of_timestamp,
        "timestamp_policy": TIMESTAMP_POLICY,
        "row_counts": row_counts,
        "checksums": checksums,
        "checksums_sha256": checksums,
        "files": list(MASTERY_FILES),
    }

    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()

