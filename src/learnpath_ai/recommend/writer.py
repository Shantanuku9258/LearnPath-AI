"""CSV and manifest writer for recommendations (Phase 6)."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from learnpath_ai.generate.columns import (
    RECOMMENDATION_COLUMNS,
    RECOMMENDATION_FILES,
    RECOMMENDATION_MANIFEST_NAME,
)
from learnpath_ai.generate.writer import OutputExistsError
from learnpath_ai.recommend.engine import RecommendationRecord

TIMESTAMP_POLICY = (
    "as_of_timestamp represents the feature cutoff point. "
    "recommended_at represents the issuance time (recommended_at >= as_of_timestamp). "
    "All timestamps are deterministic UTC."
)


def write_recommendation_dataset(
    recommendations: list[RecommendationRecord],
    output_dir: Path,
    *,
    overwrite: bool = False,
    profile: str = "dev",
    seed: int = 20261022,
    as_of_timestamp: str | None = None,
    generator_version: str = "rule-v1",
) -> dict[str, Any]:
    """Write recommendations to recommendations.csv and write recommendation_manifest.json."""
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "recommendations.csv"
    manifest_path = output_dir / RECOMMENDATION_MANIFEST_NAME

    planned = [csv_path, manifest_path]
    existing = [p for p in planned if p.exists()]
    if existing and not overwrite:
        names = ", ".join(p.name for p in existing)
        raise OutputExistsError(
            f"Refusing to overwrite existing recommendation files in {output_dir}: {names}. "
            "Pass --overwrite to replace them."
        )

    rows = [r.to_dict() for r in recommendations]
    with csv_path.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=list(RECOMMENDATION_COLUMNS),
            lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    checksum = _sha256(csv_path)
    checksums = {"recommendations.csv": checksum}
    row_counts = {"recommendations.csv": len(rows)}

    manifest = write_recommendation_manifest(
        output_dir,
        row_counts=row_counts,
        checksums=checksums,
        profile=profile,
        seed=seed,
        as_of_timestamp=as_of_timestamp or (recommendations[0].as_of_timestamp if recommendations else ""),
        generator_version=generator_version,
        overwrite=True,
    )
    return manifest


def write_recommendation_manifest(
    output_dir: Path,
    *,
    row_counts: dict[str, int],
    checksums: dict[str, str],
    profile: str = "dev",
    seed: int = 20261022,
    as_of_timestamp: str = "",
    generator_version: str = "rule-v1",
    overwrite: bool = False,
) -> dict[str, Any]:
    manifest_path = output_dir / RECOMMENDATION_MANIFEST_NAME
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
        "files": list(RECOMMENDATION_FILES),
    }

    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    return manifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()

