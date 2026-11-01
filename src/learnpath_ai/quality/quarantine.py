"""Quarantine writer — persists anomaly fixtures to data/quarantine/.

Each anomaly type gets its own CSV in the quarantine directory.
A combined data_quality_issues.csv and a quality_manifest.json are also written.
Ground-truth labels are written separately to data/ground_truth/.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from learnpath_ai.generate.clock import format_timestamp
from learnpath_ai.generate.columns import DATA_QUALITY_ISSUE_COLUMNS, QUALITY_MANIFEST_NAME
from learnpath_ai.generate.writer import OutputExistsError
from learnpath_ai.quality.anomalies import AnomalyFixture, AnomalyFixtures

UTC = timezone.utc

# File name per anomaly type
_TYPE_TO_FILE = {
    "REC_PREREQ_VIOLATION": "rec_prereq_violations.csv",
    "ATTEMPT_TIMESTAMP_ORDER": "attempt_timestamp_orders.csv",
    "CONTENT_MISSING_SKILL": "content_missing_skills.csv",
    "MASTERY_FUTURE_ATTEMPT": "mastery_future_attempts.csv",
    "PATH_LOOP_NO_PROGRESS": "path_loop_no_progress.csv",
}

# Columns written to ground_truth CSV (labels separate from features)
_GROUND_TRUTH_COLUMNS = (
    "issue_id",
    "issue_type",
    "source_table",
    "source_pk",
    "expected_decision",
)


def write_quarantine(
    fixtures: AnomalyFixtures,
    output_dir: Path,
    *,
    overwrite: bool = False,
    ground_truth_dir: Path | None = None,
    profile: str = "dev",
    seed: int = 20261022,
    generator_version: str = "quality-v1",
) -> dict[str, Any]:
    """Write all anomaly fixtures to quarantine CSVs and return a manifest dict.

    Parameters
    ----------
    fixtures:
        Generated AnomalyFixtures container.
    output_dir:
        Destination for quarantine CSVs (e.g. data/quarantine/dev/).
    overwrite:
        If False, raise OutputExistsError when files already exist.
    ground_truth_dir:
        Where to write ground_truth_anomalies.csv (e.g. data/ground_truth/).
        If None, no ground truth file is written.
    profile:
        'dev' or 'full', stored in manifest.
    seed:
        RNG seed, stored in manifest.
    generator_version:
        Version tag, stored in manifest.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_fixtures = fixtures.all()
    by_type = fixtures.by_type()

    # -----------------------------------------------------------------------
    # Check for existing files
    # -----------------------------------------------------------------------
    planned_files = list(_TYPE_TO_FILE.values()) + ["data_quality_issues.csv", QUALITY_MANIFEST_NAME]
    existing = [output_dir / fn for fn in planned_files if (output_dir / fn).exists()]
    if existing and not overwrite:
        names = ", ".join(p.name for p in existing)
        raise OutputExistsError(
            f"Refusing to overwrite existing quarantine files in {output_dir}: {names}. "
            "Pass --overwrite to replace them."
        )

    now_str = format_timestamp(datetime.now(UTC))
    checksums: dict[str, str] = {}
    row_counts: dict[str, int] = {}

    # -----------------------------------------------------------------------
    # Per-type CSVs
    # -----------------------------------------------------------------------
    for issue_type, filename in _TYPE_TO_FILE.items():
        type_fixtures = by_type.get(issue_type, [])
        rows = [f.as_db_row() for f in type_fixtures]
        path = output_dir / filename
        _write_csv(path, DATA_QUALITY_ISSUE_COLUMNS, rows)
        checksums[filename] = _sha256(path)
        row_counts[filename] = len(rows)

    # -----------------------------------------------------------------------
    # Combined CSV
    # -----------------------------------------------------------------------
    combined_path = output_dir / "data_quality_issues.csv"
    all_rows = [f.as_db_row() for f in all_fixtures]
    _write_csv(combined_path, DATA_QUALITY_ISSUE_COLUMNS, all_rows)
    checksums["data_quality_issues.csv"] = _sha256(combined_path)
    row_counts["data_quality_issues.csv"] = len(all_rows)

    # -----------------------------------------------------------------------
    # Ground truth labels (separate file, separate directory)
    # -----------------------------------------------------------------------
    if ground_truth_dir is not None:
        ground_truth_dir.mkdir(parents=True, exist_ok=True)
        gt_path = ground_truth_dir / "ground_truth_anomalies.csv"
        gt_rows = [
            {col: f.as_db_row().get(col, "") for col in _GROUND_TRUTH_COLUMNS}
            for f in all_fixtures
        ]
        _write_csv(gt_path, _GROUND_TRUTH_COLUMNS, gt_rows)

    # -----------------------------------------------------------------------
    # Manifest
    # -----------------------------------------------------------------------
    manifest = write_quarantine_manifest(
        output_dir,
        row_counts=row_counts,
        checksums=checksums,
        profile=profile,
        seed=seed,
        generator_version=generator_version,
        generated_at=now_str,
        overwrite=True,  # Manifest always overwritten if we got this far
    )
    return manifest


def write_quarantine_manifest(
    output_dir: Path,
    *,
    row_counts: dict[str, int],
    checksums: dict[str, str],
    profile: str = "dev",
    seed: int = 20261022,
    generator_version: str = "quality-v1",
    generated_at: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Write quality_manifest.json and return the manifest dict."""
    manifest_path = output_dir / QUALITY_MANIFEST_NAME
    if manifest_path.exists() and not overwrite:
        raise OutputExistsError(
            f"Refusing to overwrite existing manifest: {manifest_path}. "
            "Pass overwrite=True to replace it."
        )

    if generated_at is None:
        generated_at = format_timestamp(datetime.now(UTC))

    manifest: dict[str, Any] = {
        "generator_version": generator_version,
        "dataset_profile": profile,
        "seed": seed,
        "generated_at": generated_at,
        "row_counts": row_counts,
        "checksums_sha256": checksums,
        "files": list(row_counts.keys()),
    }
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    return manifest


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _write_csv(
    path: Path,
    columns: tuple[str, ...],
    rows: list[dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(columns),
            lineterminator="\n",
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()

