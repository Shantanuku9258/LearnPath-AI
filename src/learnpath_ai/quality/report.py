"""Quality report writer — JSON, CSV and plain-text summary."""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from learnpath_ai.generate.clock import format_timestamp
from learnpath_ai.quality.detectors import DetectionResult

UTC = timezone.utc

_REPORT_COLUMNS = (
    "issue_id",
    "anomaly_type",
    "severity",
    "source_table",
    "source_pk",
    "entity_type",
    "entity_id",
    "detected_at",
    "reason_code",
    "description",
    "detection_status",
)


def write_quality_report(
    detection_results: list[DetectionResult],
    output_dir: Path,
    *,
    profile: str = "dev",
    total_inspected: int = 0,
    total_valid: int = 0,
) -> dict[str, Any]:
    """Write JSON, CSV and plain-text quality reports.

    Parameters
    ----------
    detection_results:
        List of DetectionResult from detect_all_anomalies().
    output_dir:
        Directory for report files (e.g. reports/).
    profile:
        'dev' or 'full' — used in filenames.
    total_inspected:
        Total records scanned (informational).
    total_valid:
        Records without detected issues (informational).

    Returns
    -------
    dict
        Report summary dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    total_anomalous = len(detection_results)
    count_by_type: dict[str, int] = Counter(r.anomaly_type for r in detection_results)
    count_by_severity: dict[str, int] = Counter(r.severity for r in detection_results)

    now_str = format_timestamp(datetime.now(UTC))

    summary: dict[str, Any] = {
        "profile": profile,
        "generated_at": now_str,
        "total_inspected": total_inspected,
        "total_valid": total_valid,
        "total_anomalous": total_anomalous,
        "count_by_type": dict(count_by_type),
        "count_by_severity": dict(count_by_severity),
        "detection_status": "complete",
        "issues": [_result_to_dict(r) for r in detection_results],
    }

    # JSON report
    json_path = output_dir / f"quality_report_{profile}.json"
    json_path.write_bytes(
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )

    # CSV report
    csv_path = output_dir / f"quality_report_{profile}.csv"
    with csv_path.open("w", encoding="utf-8", newline="\n") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(_REPORT_COLUMNS), lineterminator="\n")
        writer.writeheader()
        for r in detection_results:
            writer.writerow(_result_to_csv_row(r))

    # Plain-text summary
    txt_path = output_dir / f"quality_summary_{profile}.txt"
    txt_path.write_text(_build_summary_text(summary, profile), encoding="utf-8")

    return summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _result_to_dict(r: DetectionResult) -> dict[str, Any]:
    return {
        "issue_id": r.issue_id,
        "anomaly_type": r.anomaly_type,
        "severity": r.severity,
        "source_table": r.source_table,
        "source_pk": r.source_pk,
        "entity_type": r.entity_type,
        "entity_id": r.entity_id,
        "detected_at": r.detected_at,
        "reason_code": r.reason_code,
        "description": r.description,
        "detection_status": r.detection_status,
        "affected_record_reference": r.affected_record_reference,
        "details": r.details,
    }


def _result_to_csv_row(r: DetectionResult) -> dict[str, str]:
    return {
        "issue_id": r.issue_id,
        "anomaly_type": r.anomaly_type,
        "severity": r.severity,
        "source_table": r.source_table,
        "source_pk": r.source_pk,
        "entity_type": r.entity_type,
        "entity_id": r.entity_id,
        "detected_at": r.detected_at,
        "reason_code": r.reason_code,
        "description": r.description,
        "detection_status": r.detection_status,
    }


def _build_summary_text(summary: dict[str, Any], profile: str) -> str:
    lines = [
        "LearnPath AI — Data Quality Report",
        "=" * 50,
        f"Profile       : {profile}",
        f"Generated at  : {summary['generated_at']}",
        "",
        "Inspection Summary",
        "-" * 30,
        f"  Total inspected  : {summary['total_inspected']}",
        f"  Total valid      : {summary['total_valid']}",
        f"  Total anomalous  : {summary['total_anomalous']}",
        "",
        "Count by Anomaly Type",
        "-" * 30,
    ]
    for atype, count in sorted(summary["count_by_type"].items()):
        lines.append(f"  {atype:<40} {count}")
    lines.append("")
    lines.append("Count by Severity")
    lines.append("-" * 30)
    for sev in ("critical", "high", "medium", "low"):
        count = summary["count_by_severity"].get(sev, 0)
        lines.append(f"  {sev:<10} {count}")
    lines.append("")
    lines.append(f"Detection status: {summary['detection_status']}")
    return "\n".join(lines) + "\n"

